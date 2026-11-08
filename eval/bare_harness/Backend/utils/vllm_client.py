"""
VLLM客户端
功能：调用VLLM API生成文本回答
"""
import logging
import requests
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class VLLMClient:
    """VLLM API客户端"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化VLLM客户端

        参数:
            config: VLLM配置字典，包含以下键：
                - api_url: VLLM API地址
                - model: 模型名称
                - api_key: API密钥
                - max_tokens: 最大生成token数
                - temperature: 温度参数
                - top_p: top_p参数
        """
        self.api_url = config['vllm']['api_url']
        self.model = config['vllm']['model']
        self.max_tokens = config['vllm']['max_tokens']
        self.temperature = config['vllm']['temperature']
        self.top_p = config['vllm']['top_p']

        # 构建请求头
        self.headers = {
            'Content-Type': 'application/json'
        }

        logger.info(
            f"VLLMClient initialized: "
            f"model={self.model}, "
            f"api_url={self.api_url}, "
            f"max_tokens={self.max_tokens}"
        )

    def generate(
            self,
            messages: List[Dict[str, str]],
            temperature: Optional[float] = None,
            enable_thinking: bool = False,
            max_tokens: Optional[int] = None,
            top_p: Optional[float] = None
    ) -> str:
        """
        生成回答

        参数:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            temperature: 温度参数（可选，默认使用配置值）
            enable_thinking: 是否启用思维链（可选，默认False）
            max_tokens: 最大生成token数（可选，默认使用配置值）
            top_p: top_p参数（可选，默认使用配置值）

        返回:
            str: 生成的文本内容

        异常:
            requests.exceptions.RequestException: 请求失败
            KeyError: 响应格式错误
        """
        # 使用提供的参数或默认配置
        if temperature is None:
            temperature = self.temperature
        if max_tokens is None:
            max_tokens = self.max_tokens
        if top_p is None:
            top_p = self.top_p

        # 构建请求payload
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }

        # 添加chat_template_kwargs（如果需要）
        if enable_thinking:
            payload["chat_template_kwargs"] = {
                "temperature": temperature,
                "enable_thinking": enable_thinking,
                "top_p": top_p
            }

        logger.info(
            f"Calling VLLM API: "
            f"messages_count={len(messages)}, "
            f"temperature={temperature}, "
            f"enable_thinking={enable_thinking}"
        )

        try:
            # 发送请求
            response = requests.post(
                self.api_url,
                json=payload,
                headers=self.headers,
                timeout=120  # 2分钟超时
            )

            # 检查响应状态
            response.raise_for_status()

            # 解析响应
            response_data = response.json()
            content = response_data["choices"][0]["message"]["content"]

            logger.info(f"VLLM API call successful, response length: {len(content)}")
            return content

        except requests.exceptions.Timeout:
            logger.error("VLLM API request timeout")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"VLLM API request failed: {str(e)}")
            if hasattr(e.response, 'text'):
                logger.error(f"Response content: {e.response.text}")
            raise
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to parse VLLM API response: {str(e)}")
            logger.error(f"Response data: {response_data if 'response_data' in locals() else 'N/A'}")
            raise

    def generate_with_retry(
            self,
            messages: List[Dict[str, str]],
            temperature: Optional[float] = None,
            enable_thinking: bool = False,
            max_retries: int = 3
    ) -> str:
        """
        带重试机制的生成方法

        参数:
            messages: 消息列表
            temperature: 温度参数
            enable_thinking: 是否启用思维链
            max_retries: 最大重试次数

        返回:
            str: 生成的文本内容
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                return self.generate(
                    messages=messages,
                    temperature=temperature,
                    enable_thinking=enable_thinking
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    f"VLLM API call failed (attempt {attempt + 1}/{max_retries}): {str(e)}"
                )
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)  # 指数退避

        logger.error(f"All {max_retries} attempts failed")
        raise last_error

    def test_connection(self) -> bool:
        """
        测试VLLM API连接

        返回:
            bool: 连接是否成功
        """
        try:
            test_messages = [{"role": "user", "content": "Hello"}]
            self.generate(test_messages, max_tokens=10)
            logger.info("VLLM API connection test successful")
            return True
        except Exception as e:
            logger.error(f"VLLM API connection test failed: {str(e)}")
            return False