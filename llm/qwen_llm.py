import json
import logging
import os
from typing import Dict, List, Optional, Any
import requests
from crewai import BaseLLM

def _get_logger():
    """获取模块级 logger。"""
    logger = logging.getLogger("llm.aliyun_llm")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


logger = _get_logger()

class QwenLLM(BaseLLM):
    """
    阿里云千问大模型调用封装类
    支持 qwen-turbo, qwen-plus, qwen-max 等模型
    """

    ENDPOINTS = {
        "cn": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    }

    def __init__(
            self,
            model: str = "qwen-turbo",
            api_key: Optional[str] = None,
            image_model: str | None = None,
            base_url: str = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
            region: str = "cn",
            temperature: float | None = None,
            timeout: int = 600,
            retry_count: int | None = None,
    ):
        super().__init__(model=model, temperature=temperature)

        self.api_key = api_key or os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API Key 未提供。请通过 api_key 传入或设置环境变量 QWEN_API_KEY 或 DASHSCOPE_API_KEY"
            )

        if region not in self.ENDPOINTS:
            raise ValueError(f"不支持的地域: {region}，支持: {list(self.ENDPOINTS.keys())}")
        self.endpoint = self.ENDPOINTS[region]
        self.region = region
        self.timeout = timeout
        self.image_model = image_model
        if not self.image_model:
            self.image_model = "qwen3-vl-plus"
        _rc = retry_count
        if _rc is None and os.getenv("LLM_RETRY_COUNT") is not None:
            try:
                _rc = int(os.getenv("LLM_RETRY_COUNT", "2"))
            except ValueError:
                _rc = 2
        self.retry_count = _rc if _rc is not None else 2

    def call(
            self,
            messages: str | list[dict[str, Any]],
            tools: list[dict] | None = None,
            callbacks: list[Any] | None = None,
            available_functions: dict[str, Any] | None = None,
            max_iterations: int = 10,
            _retry_on_empty: bool = True,
            **kwargs: Any,
    ) -> str | Any:
        """
        调用阿里云 LLM API，支持 Function Calling、多模态消息、重试与空内容重试。

        Args:
            messages: 消息列表或单字符串；content 可为字符串或多模态数组
            tools: 工具定义（Function Calling）
            callbacks: 回调列表
            available_functions: 可执行函数映射
            max_iterations: Function Calling 最大迭代次数
            _retry_on_empty: 是否在返回空内容时自动重试
            **kwargs: 兼容 CrewAI 额外参数（如 from_task）
        Returns:
            LLM 返回的文本内容
        """
        if max_iterations <= 0:
            raise RuntimeError("Function calling 达到最大迭代次数，可能存在无限循环")

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        messages, flag = self._normalize_multimodal_tool_result(messages)
        logger.info("normalized_multimodal_tool_result flag=%s messages=%s", flag,
                    json.dumps(messages, ensure_ascii=False, indent=2))
        self._validate_messages(messages)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if flag:
            payload["model"] = self.image_model
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.stop and self.supports_stop_words():
            stop_value = self._prepare_stop_words(self.stop)
            if stop_value:
                payload["stop"] = stop_value
        if tools and self.supports_function_calling():
            payload["tools"] = tools

        if callbacks:
            for cb in callbacks:
                if hasattr(cb, "on_llm_start"):
                    try:
                        cb.on_llm_start(messages)
                    except Exception:
                        pass

        logger.info("发送 LLM API 请求 endpoint=%s model=%s", self.endpoint, payload.get("model"))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("发送 LLM API 请求 payload=%s", json.dumps(payload, ensure_ascii=False, indent=2))
        last_exception: BaseException | None = None
        result: dict[str, Any] = {}
        for attempt in range(self.retry_count + 1):
            try:
                response = requests.post(
                    self.endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
                status_code = response.status_code
                if status_code >= 500:
                    if attempt < self.retry_count:
                        logger.warning(
                            "llm_server_error_retry status_code=%s attempt=%s max=%s",
                            status_code,
                            attempt + 1,
                            self.retry_count + 1,
                        )
                        last_exception = RuntimeError(
                            f"LLM 服务器错误 {status_code}: {response.text[:200]}"
                        )
                        continue
                    response.raise_for_status()
                elif status_code == 429:
                    if attempt < self.retry_count:
                        logger.warning(
                            "llm_rate_limit_retry attempt=%s max=%s",
                            attempt + 1,
                            self.retry_count + 1,
                        )
                        last_exception = RuntimeError(f"LLM 请求限流: {response.text[:200]}")
                        continue
                    response.raise_for_status()
                elif status_code >= 400:
                    err_body = response.text[:500] if response.text else ""
                    logger.error(
                        "llm_request_4xx status_code=%s url=%s body=%s",
                        status_code,
                        response.url,
                        err_body,
                    )
                    response.raise_for_status()

                result = response.json()
                if attempt > 0:
                    logger.info(
                        "llm_request_success_after_retry attempt=%s total=%s",
                        attempt + 1,
                        self.retry_count + 1,
                    )
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Response result: %s", json.dumps(result, ensure_ascii=False, indent=2))
                break

            except requests.Timeout:
                last_exception = TimeoutError(f"LLM 请求超时（{self.timeout} 秒）")
                if attempt < self.retry_count:
                    logger.warning(
                        "llm_timeout_retry timeout=%s attempt=%s max=%s",
                        self.timeout,
                        attempt + 1,
                        self.retry_count + 1,
                    )
                    continue
                logger.error(
                    "llm_timeout_final timeout=%s total_attempts=%s",
                    self.timeout,
                    self.retry_count + 1,
                )
                raise last_exception
            except requests.RequestException as e:
                last_exception = RuntimeError(f"LLM 请求失败: {e}")
                if attempt < self.retry_count:
                    logger.warning(
                        "llm_request_error_retry error=%s attempt=%s max=%s",
                        str(e),
                        attempt + 1,
                        self.retry_count + 1,
                    )
                    continue
                logger.exception("llm_request_failed error=%s total_attempts=%s", str(e), self.retry_count + 1)
                raise last_exception
        else:
            if last_exception:
                raise last_exception
            raise RuntimeError("LLM 请求失败：未知错误")

        if callbacks:
            for cb in callbacks:
                if hasattr(cb, "on_llm_end"):
                    try:
                        cb.on_llm_end(result)
                    except Exception:
                        pass
        logger.info("收到 LLM API 响应  result=%s", result)
        if "choices" not in result or not result["choices"]:
            raise ValueError("响应中未找到 choices 字段")

        message = result["choices"][0].get("message", {})
        if "tool_calls" in message:
            if available_functions:
                return self._handle_function_calls(
                    message["tool_calls"],
                    messages,
                    tools,
                    available_functions,
                    max_iterations - 1,
                )
            # CrewAI 会故意传 available_functions=None，让 LLM 只返回原始 tool_calls，
            # 由 executor 的 _handle_native_tool_calls 执行。此处直接返回 tool_calls 列表。
            return message["tool_calls"]

        content = message.get("content")
        if content is None:
            raise ValueError("响应中未找到 content 字段")

        if isinstance(content, str) and not content.strip():
            if _retry_on_empty:
                max_empty_retries = 2
                empty_retry_count = kwargs.get("_empty_retry_count", 0)
                if empty_retry_count >= max_empty_retries:
                    raise ValueError(
                        f"LLM 连续 {max_empty_retries + 1} 次返回空内容，可能是模型限流或异常，请稍后重试或检查 API 配额"
                    )
                logger.warning(
                    "llm_empty_content_retry model=%s retry_count=%s max_retries=%s",
                    self.model,
                    empty_retry_count + 1,
                    max_empty_retries,
                )
                return self.call(
                    messages,
                    tools=tools,
                    callbacks=callbacks,
                    available_functions=available_functions,
                    max_iterations=max_iterations,
                    _retry_on_empty=False,
                    _empty_retry_count=empty_retry_count + 1,
                    **kwargs,
                )
            raise ValueError(
                "LLM 返回空内容，可能是模型限流或偶发异常，请稍后重试或检查 API 配额"
            )

        return content

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

    def _normalize_multimodal_tool_result(self, messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
        """
        将 CrewAI 对 AddImageTool/AddImageToolLocal 的 stringify 结果还原为多模态 user 消息，
        否则 DashScope 会因消息格式或体积返回 400。

        兼容两种 Agent 调用模式：
        - 旧版 ReAct：base64/URL 在 assistant 消息的 content 字符串中（Thought/Action/Observation）
        - 新版 Function Calling：base64 在 role=tool 消息的 content 中，图片合并到紧随其后的 user
          消息。假设 user 消息紧跟在所有 tool 消息之后（CrewAI 当前行为）；若消息列表以 tool 结尾
          则自动 flush 一条合成 user 消息。

        Returns:
            tuple[list[dict[str, Any]], bool]: 处理后的消息列表 + 是否需要切换多模态模型
        """
        out: list[dict[str, Any]] = []
        flag = False
        # function calling 模式：累积待注入到下一条 user 消息的图片 URL 列表
        # 用列表支持同一轮多个 tool 返回图片的情况
        pending_images: list[str] = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            # ── 新版 Function Calling：tool 消息 content 含 base64 data URL ───
            if role == "tool" and isinstance(content, str) and "data:image/" in content and ";base64," in content:
                data_url_start = content.find("data:image/")
                prefix = content[:data_url_start]
                data_url = content[data_url_start:]
                pending_images.append(data_url)
                new_msg = dict(msg)
                new_msg["content"] = (prefix + "图片内容已加载") if prefix else "图片内容已加载"
                out.append(new_msg)
                flag = True
                logger.info("normalized_multimodal_tool_result: function calling mode, base64 in tool message")
                continue

            # ── 新版 Function Calling：把图片合并到紧随其后的 user 消息 ──────
            # 注：仅在紧跟 tool 消息的第一条 user 消息时触发，与 CrewAI 当前消息结构一致
            if pending_images and role == "user":
                text = content if isinstance(content, str) else ""
                image_blocks = [{"type": "image_url", "image_url": {"url": u}} for u in pending_images]
                out.append({
                    "role": "user",
                    "content": [{"type": "text", "text": text}] + image_blocks,
                })
                pending_images = []
                logger.info("normalized_multimodal_tool_result: injected %d image(s) into user message (function calling)", len(image_blocks))
                continue

            # ── 旧版 ReAct：assistant 消息 content 字符串含 Observation base64 ─
            # content 可能是 None（tool_calls 消息）或 list（多模态 assistant），均跳过
            if role == "assistant" and isinstance(content, str):
                s = content
                if "Add image to content Local" in s and "data:image/" in s and ";base64," in s:
                    logger.info("normalized_multimodal_tool_result: ReAct mode, base64 in assistant message")
                    data_url_start = s.find("data:image/")
                    data_url = s[data_url_start:]
                    text = s[:data_url_start] + "图片内容已加载"
                    out.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    })
                    flag = True
                    continue
                elif "Add image to content Local" in s and "Observation: http" in s:
                    obs_start = s.find("Observation: http")
                    http_start = s.find("http", obs_start)
                    data_url = s[http_start:]
                    text = s[:obs_start] + "图片内容已加载"
                    out.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text},
                            {"type": "image", "image": data_url},
                        ],
                    })
                    flag = True
                    continue

            out.append(msg)

        # ── Flush：tool 消息是最后一条（后面没有 user 消息）时合成一条 user 消息 ─
        if pending_images:
            image_blocks = [{"type": "image_url", "image_url": {"url": u}} for u in pending_images]
            out.append({
                "role": "user",
                "content": [{"type": "text", "text": "请分析上面工具返回的图片内容。"}] + image_blocks,
            })
            logger.warning(
                "normalized_multimodal_tool_result: flushed %d pending image(s) — tool message was last in list",
                len(image_blocks),
            )

        return out, flag

    def _validate_messages(self, messages: list[dict[str, Any]]) -> None:
        """校验消息格式（含多模态 content）。"""
        valid_roles = {"system", "user", "assistant", "tool"}
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise ValueError(f"消息 {i} 必须是字典: {msg}")
            if "role" not in msg or msg["role"] not in valid_roles:
                raise ValueError(f"消息 {i} 缺少或无效的 role: {msg}")
            if msg["role"] == "tool":
                if "tool_call_id" not in msg or "content" not in msg:
                    raise ValueError(f"tool 消息 {i} 缺少 tool_call_id/content: {msg}")
            elif "content" not in msg and msg.get("tool_calls") is None:
                raise ValueError(f"消息 {i} 缺少 content 且无 tool_calls: {msg}")

    def _prepare_stop_words(
        self, stop: str | list[str | int]
    ) -> str | list[str | int] | None:
        """准备 stop 参数。"""
        if not stop:
            return None
        if isinstance(stop, str):
            return stop
        if isinstance(stop, list) and stop:
            return stop
        return None

    def get_context_window_size(self) -> int:
        """根据模型名返回上下文窗口大小（Token 数）。"""
        m = self.model.lower()
        if "long" in m:
            return 200_000
        if "max" in m or "plus" in m or "turbo" in m or "flash" in m:
            return 8192
        return 8192