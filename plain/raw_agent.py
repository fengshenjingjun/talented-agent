import json
import os
from typing import Dict, Callable

from llm.qwen_llm import QwenLLM


class RawAgent:
    """
    不用任何框架的最朴素的agent，用于厘清agent本质
    """

    """
        初始化Agent，设置角色、目标、背景故事和工具

        Args:
            role (str): Agent的角色
            goal (str): Agent的目标
            backstory (str): Agent的背景故事
            tools (Dict[str, Callable]): 可用工具的字典，键为工具名称，值为工具函数
    """
    def __init__(self, role: str, goal: str, backstory: str, tools: Dict[str, Callable]):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools


    def generate_system_prompt(self) -> str:
        # 读取提示词模板文件（相对于当前文件所在目录）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        template_path = os.path.join(current_dir, "system_prompt.txt")

        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()

        # 构建工具信息字符串
        tools_map = ""  # 工具详细描述
        tools_name = ""  # 工具名称列表
        for name, tool_func in self.tools.items():
            # 获取工具的描述（如果有 docstring）
            tool_desc = tool_func.__doc__ or "无描述"
            tools_map += f"Tool Name: {name}\nTool Description: {tool_desc}\n\n"
            tools_name += f"{name}, "

        # 格式化模板，填充变量
        return template.format(
            role=self.role,
            goal=self.goal,
            backstory=self.backstory,
            tools_map=tools_map,
            tools_name=tools_name.rstrip(", ")  # 去掉最后的逗号和空格
        )

    def generate_user_prompt(self, description: str, expected_output: str) -> str:
        # 读取提示词模板文件
        current_dir = os.path.dirname(os.path.abspath(__file__))
        template_path = os.path.join(current_dir, "user_prompt.txt")

        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()

        return template.format(description=description, expected_output=expected_output)

    def parse_tool_name(self, response: str) -> str:
        if "Action: " not in response:
            raise ValueError(f"响应中未找到 Action 字段。响应内容：\n{response}")

        # 提取 "Action: " 之后到换行符之前的内容
        tool_name = response.split("Action: ")[1].split("\n")[0].strip()
        return tool_name

    def parse_tool_input(self, response: str) -> str:
        if "Action Input: " not in response:
            raise ValueError(f"响应中未找到 Action Input 字段。响应内容：\n{response}")

        # 提取 "Action Input: " 之后到换行符之前的内容
        tool_input = response.split("Action Input: ")[1].split("\n")[0].strip()
        return tool_input

    def execute_tool(self, tool_name: str, tool_input: str) -> str:
        # 1. 检查工具是否存在
        if tool_name not in self.tools:
            return f"错误：工具 '{tool_name}' 不存在。可用工具：{list(self.tools.keys())}"

        # 2. 解析 JSON 格式的输入参数
        try:
            # 尝试解析 JSON
            if tool_input.strip():
                params = json.loads(tool_input)
            else:
                params = {}  # 空字符串表示无参数
        except json.JSONDecodeError as e:
            return f"错误：无法解析工具输入参数（JSON 格式错误）：{tool_input}。错误：{e}"

        # 3. 获取工具函数
        tool_func = self.tools[tool_name]

        # 4. 执行工具函数
        try:
            # 如果参数是字典，使用 ** 展开为关键字参数
            if isinstance(params, dict):
                result = tool_func(**params)
            else:
                # 如果参数不是字典，直接传递
                result = tool_func(params)

            # 将结果转换为字符串
            return str(result)
        except Exception as e:
            return f"错误：执行工具 '{tool_name}' 时发生异常：{str(e)}"

    def extract_final_answer(self, response: str) -> str:
        if "Final Answer:" not in response:
            raise ValueError(f"响应中未找到 Final Answer 字段。响应内容：\n{response}")

        # 提取 "Final Answer: " 之后的内容
        final_answer = response.split("Final Answer: ")[1].strip()
        return final_answer

    def run(self, description: str, expected_output: str) -> str:

        # 1. 生成系统提示词和用户提示词
        system_prompt = self.generate_system_prompt()
        user_prompt = self.generate_user_prompt(description, expected_output)

        # 2. 初始化消息列表（对话历史）
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # 3. 初始化 llm
        llm = QwenLLM(
            model="qwen-turbo",
            api_key=os.getenv("QWEN_API_KEY")
        )

        # 4. 核心循环：不断调用 llm，直到得到 Final Answer
        response = llm.call(messages, stop=["Observation:"])

        while "Final Answer:" not in response:
            # 4.1 解析 llm 返回的 Action（工具名称和输入）
            tool_name = self.parse_tool_name(response)
            tool_input = self.parse_tool_input(response)

            # 4.2 执行工具，获取结果
            tool_result = self.execute_tool(tool_name, tool_input)

            # 4.3 将工具执行结果作为 Observation 添加到对话历史
            # 格式：之前的 response + "\nObservation:" + 工具结果
            content = response + "\nObservation:" + tool_result
            messages.append({"role": "assistant", "content": content})

            # 4.4 再次调用 llm，传入包含 Observation 的完整对话历史
            response = llm.call(messages, stop=["Observation:"])

        # 5. 提取并返回最终答案
        final_answer = self.extract_final_answer(response)
        return final_answer