import json
import os
from typing import Dict, Callable, Literal
from typing_extensions import TypedDict, Annotated

from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from llm.qwen_llm import QwenLLM


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class GraphAgent:
    """
    使用 LangGraph 框架实现的 ReAct Agent
    以 StateGraph 形式清晰展示"思考-行动-观察"循环结构
    对比 plain/raw_agent.py 可以看到：图的节点和边 == 手写的 while 循环
    """

    def __init__(self, role: str, goal: str, backstory: str, tools: Dict[str, Callable]):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools
        self.llm = QwenLLM(model="qwen-turbo", api_key=os.getenv("QWEN_API_KEY"))
        self.graph = self._build_graph()

    def _build_system_prompt(self) -> str:
        tools_map = ""
        tools_name = ""
        for name, func in self.tools.items():
            desc = func.__doc__ or "无描述"
            tools_map += f"Tool Name: {name}\nTool Description: {desc}\n\n"
            tools_name += f"{name}, "

        return (
            f"你是 {self.role}. 你的背景是: {self.backstory}\n"
            f"你的目标是: {self.goal}\n"
            f"你只能访问以下工具，且不应该使用列表中未提到的工具:\n{tools_map}\n"
            "重要提示：使用以下格式进行响应：\n"
            "Thought: you should always think about what to do\n"
            f"Action: the action to take, only one name of [{tools_name.rstrip(', ')}], just the name, exactly as it's written.\n"
            'Action Input: the input to the action, just a simple JSON object, enclosed in curly braces, using " to wrap keys and values.\n'
            "Observation: the result of the action\n\n"
            "收集到所有必要的信息后，返回以下格式：\n"
            "Thought: I now know the final answer\n"
            "Final Answer: the final answer to the original input question"
        )

    def _messages_to_api_format(self, messages: list[BaseMessage]) -> list[dict]:
        """Convert LangChain messages to Qwen API format.
        ToolMessages are folded into the preceding assistant message as Observation text,
        matching the pattern used by plain/raw_agent.py."""
        result = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, ToolMessage):
                if result and result[-1]["role"] == "assistant":
                    result[-1]["content"] += f"\nObservation:{msg.content}"
                else:
                    result.append({"role": "assistant", "content": f"Observation:{msg.content}"})
        return result

    def _agent_node(self, state: AgentState) -> dict:
        """LLM 节点：将当前对话历史发给 LLM，获取下一步动作或最终答案"""
        api_messages = self._messages_to_api_format(state["messages"])
        response = self.llm.call(api_messages, stop=["Observation:"])
        return {"messages": [AIMessage(content=response)]}

    def _tool_node(self, state: AgentState) -> dict:
        """工具节点：解析 LLM 输出中的工具调用，执行工具，将结果作为 Observation 返回"""
        last_msg = state["messages"][-1]
        response = last_msg.content

        tool_name = self._parse_field(response, "Action: ")
        tool_input = self._parse_field(response, "Action Input: ")
        result = self._execute_tool(tool_name, tool_input)

        return {"messages": [ToolMessage(content=result, tool_call_id="0")]}

    def _should_continue(self, state: AgentState) -> Literal["tools", "__end__"]:
        """条件边：LLM 给出 Final Answer 则结束，否则继续调用工具"""
        last_msg = state["messages"][-1]
        content = last_msg.content
        if "Final Answer:" in content:
            return "__end__"
        if "Action:" in content:
            return "tools"
        return "__end__"

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", self._tool_node)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent",
            self._should_continue,
            {"tools": "tools", "__end__": END},
        )
        graph.add_edge("tools", "agent")
        return graph.compile()

    def _parse_field(self, text: str, prefix: str) -> str:
        if prefix not in text:
            raise ValueError(f"响应中未找到 '{prefix}' 字段。响应内容：\n{text}")
        return text.split(prefix)[1].split("\n")[0].strip()

    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        if tool_name not in self.tools:
            return f"错误：工具 '{tool_name}' 不存在。可用工具：{list(self.tools.keys())}"
        try:
            params = json.loads(tool_input) if tool_input.strip() else {}
        except json.JSONDecodeError as e:
            return f"错误：无法解析工具输入参数：{tool_input}。错误：{e}"
        try:
            result = self.tools[tool_name](**params) if isinstance(params, dict) else self.tools[tool_name](params)
            return str(result)
        except Exception as e:
            return f"错误：执行工具 '{tool_name}' 时发生异常：{str(e)}"

    def run(self, description: str, expected_output: str) -> str:
        system_prompt = self._build_system_prompt()
        user_prompt = (
            f"当前任务: {description}\n\n"
            f"这是你最终答案的预期标准: {expected_output}\n"
            "你必须返回实际完整的内容作为最终答案，而不是摘要。\n\n"
            "Thought:"
        )

        initial_state: AgentState = {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        }

        final_state = self.graph.invoke(initial_state)

        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage) and "Final Answer:" in msg.content:
                return msg.content.split("Final Answer:")[1].strip()

        return final_state["messages"][-1].content
