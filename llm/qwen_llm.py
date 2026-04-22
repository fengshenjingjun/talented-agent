import os
from typing import Dict, List, Optional
import requests

class QwenLLM:
    """
    阿里云千问大模型调用封装类
    支持 qwen-turbo, qwen-plus, qwen-max 等模型
    """

    def __init__(
            self,
            model: str = "qwen-turbo",
            api_key: Optional[str] = None,
            base_url: str = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    ):
        """
        初始化千问 LLM

        Args:
            model: 模型名称，可选 qwen-turbo, qwen-plus, qwen-max 等
            api_key: API Key，默认从环境变量 QWEN_API_KEY 读取
            base_url: API 地址
        """
        self.model = model
        self.api_key = api_key or os.getenv("QWEN_API_KEY")
        self.base_url = base_url

        if not self.api_key:
            raise ValueError("API Key 不能为空，请设置 QWEN_API_KEY 环境变量或直接传入")

    def call(
            self,
            messages: List[Dict[str, str]],
            temperature: float = 0.7,
            max_tokens: Optional[int] = None,
            stop: Optional[List[str]] = None
    ) -> str:
        """
        调用千问模型

        Args:
            messages: 对话消息列表，格式为 [{"role": "user", "content": "你好"}]
            temperature: 采样温度，0-2之间，越大输出越随机
            max_tokens: 最大生成 token 数
            stop: 停止词列表

        Returns:
            模型生成的文本
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model,
            "input": {
                "messages": messages
            },
            "parameters": {
                "temperature": temperature,
                "result_format": "message"
            }
        }

        if max_tokens:
            payload["parameters"]["max_tokens"] = max_tokens
        if stop:
            payload["parameters"]["stop"] = stop

        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()

        result = response.json()

        # 解析响应
        if "output" in result and "choices" in result["output"]:
            return result["output"]["choices"][0]["message"]["content"]
        elif "output" in result and "text" in result["output"]:
            return result["output"]["text"]
        else:
            raise ValueError(f"Unexpected response format: {result}")

    def chat(self, user_message: str, system_message: Optional[str] = None) -> str:
        """
        简单的单轮对话接口

        Args:
            user_message: 用户输入
            system_message: 系统提示词（可选）

        Returns:
            模型回复
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})

        return self.call(messages)
