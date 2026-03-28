"""
LLM 调用模块
"""
import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

import openai
from openai import OpenAI

from .logger import get_logger
from .structures import LLMResponse


class LLMClient:
    """
    LLM 客户端，封装 OpenAI 兼容 API 调用。
    
    支持 OpenAI、Azure、sglang、vllm 等兼容接口。
    
    特性:
    - 支持 JSON 模式
    - 自动重试
    - 通过 extra_params 支持任意 API 参数
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4",
        timeout: int = 120,
        retry_times: int = 3,
        retry_delay: int = 5,
        **extra_params,
    ):
        """
        Args:
            api_key: API 密钥
            base_url: API 地址
            model: 默认模型
            timeout: 超时时间
            retry_times: 重试次数
            retry_delay: 重试间隔
            **extra_params: 其他 API 参数 (temperature, top_p, seed, max_tokens 等)
        """
        self.model = model
        self.timeout = timeout
        self.retry_times = retry_times
        self.retry_delay = retry_delay
        self.extra_params = extra_params  # 存储所有额外参数
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        
        self.logger = get_logger()
        # 使用线程本地存储，支持多 worker 并行
        self._thread_local = threading.local()
    
    def set_log_dir(self, log_dir: Optional[Path]):
        """
        设置日志目录，用于记录每次 LLM 调用。
        使用线程本地存储，支持多 worker 并行。
        
        Args:
            log_dir: 日志目录路径，如果为 None 则禁用日志记录
        """
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self._thread_local.log_dir = log_dir
            self._thread_local.call_counter = 0  # 每个线程独立的计数器
        else:
            self._thread_local.log_dir = None
            self._thread_local.call_counter = 0
    
    def _get_log_dir(self) -> Optional[Path]:
        """获取当前线程的日志目录"""
        return getattr(self._thread_local, 'log_dir', None)
    
    def _get_call_counter(self) -> int:
        """获取当前线程的调用计数器"""
        return getattr(self._thread_local, 'call_counter', 0)
    
    def _increment_call_counter(self):
        """增加当前线程的调用计数器"""
        if not hasattr(self._thread_local, 'call_counter'):
            self._thread_local.call_counter = 0
        self._thread_local.call_counter += 1
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        model: Optional[str] = None,
        call_name: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        发送聊天请求。
        
        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            json_mode: 是否要求 JSON 格式返回
            model: 覆盖默认模型
            call_name: 调用名称（用于日志文件名）
            **kwargs: 覆盖默认的 API 参数
            
        Returns:
            LLMResponse 对象
        """
        # 合并参数：默认参数 < extra_params < kwargs
        params = {
            "model": model or self.model,
            "messages": messages,
            **self.extra_params,
            **kwargs,
        }
        
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        
        last_error = None
        for attempt in range(self.retry_times):
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(**params)
                call_time = time.time() - start_time
                
                content = response.choices[0].message.content or ""
                
                # 处理 usage 可能为 None 的情况
                usage = {}
                if response.usage:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens or 0,
                        "completion_tokens": response.usage.completion_tokens or 0,
                        "total_tokens": response.usage.total_tokens or 0,
                    }
                
                result = LLMResponse(
                    content=content,
                    raw_response=response,
                    usage=usage,
                    model=params["model"],
                    call_time=call_time,
                )
                
                self.logger.log_llm_call({
                    "model": params["model"],
                    "usage": usage,
                    "call_time": call_time,
                })
                
                # 记录到文件（使用线程本地存储）
                log_dir = self._get_log_dir()
                if log_dir:
                    self._log_call_to_file(messages, result, call_name, attempt)
                
                return result
                
            except Exception as e:
                last_error = e
                self.logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{self.retry_times}): {e}"
                )
                if attempt < self.retry_times - 1:
                    time.sleep(self.retry_delay)
        
        raise last_error
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        call_name: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        简化的生成接口。
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            json_mode: 是否要求 JSON 格式
            call_name: 调用名称（用于日志文件名）
            **kwargs: 传递给 chat 方法的其他参数
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        return self.chat(messages, json_mode=json_mode, call_name=call_name, **kwargs)
    
    def extract_tags(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        call_name: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        从 LLM 响应中提取 XML 标签格式的字段。
        
        例如: <name>xxx</name> <motivation>yyy</motivation>
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            call_name: 调用名称（用于日志文件名）
            **kwargs: 传递给 generate 方法的其他参数
        
        Returns:
            解析后的字典，键为标签名，值为标签内容
            
        Raises:
            ValueError: 无法提取任何标签
        """
        response = self.generate(prompt, system_prompt, json_mode=False, call_name=call_name, **kwargs)
        content = response.content.strip()
        
        # 使用正则表达式提取所有 <tag>content</tag> 格式的标签
        # 使用更智能的匹配方式：先找到所有标签对，然后提取内容
        # 这样可以正确处理代码中包含 < 和 > 的情况
        result = {}
        
        # 找到所有开始标签的位置
        tag_pattern = r'<(\w+)>'
        pos = 0
        while True:
            match = re.search(tag_pattern, content[pos:])
            if not match:
                break
            
            tag_name = match.group(1)
            tag_start = pos + match.end()
            
            # 查找对应的结束标签 </tag_name>
            end_tag = f'</{tag_name}>'
            end_pos = content.find(end_tag, tag_start)
            
            if end_pos == -1:
                # 没有找到结束标签，跳过这个开始标签
                pos = tag_start
                continue
            
            # 提取标签内容
            tag_content = content[tag_start:end_pos].strip()
            result[tag_name] = tag_content
            
            # 继续查找下一个标签
            pos = end_pos + len(end_tag)
        
        if not result:
            self.logger.error(f"Failed to extract tags from response")
            self.logger.error(f"Response content (full, {len(content)} chars):\n{content[:1000]}...")
            raise ValueError("No valid tags found in LLM response")
        
        self.logger.debug(f"Extracted {len(result)} tags: {list(result.keys())}")
        return result
    
    def _log_call_to_file(
        self,
        messages: List[Dict[str, str]],
        response: LLMResponse,
        call_name: Optional[str],
        attempt: int,
    ):
        """将 LLM 调用记录到文件（使用线程本地存储）"""
        log_dir = self._get_log_dir()
        if not log_dir:
            return
        
        self._increment_call_counter()
        call_counter = self._get_call_counter()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 毫秒精度
        
        # 生成文件名
        if call_name:
            filename = f"llm_call_{call_counter:03d}_{call_name}_{timestamp}.json"
        else:
            filename = f"llm_call_{call_counter:03d}_{timestamp}.json"
        
        log_file = log_dir / filename
        
        # 准备日志数据
        log_data = {
            "call_number": call_counter,
            "timestamp": datetime.now().isoformat(),
            "call_name": call_name,
            "attempt": attempt + 1,
            "model": response.model,
            "usage": response.usage,
            "call_time": response.call_time,
            "messages": messages,
            "response": {
                "content": response.content,
                "finish_reason": response.raw_response.choices[0].finish_reason if response.raw_response.choices else None,
            },
        }
        
        # 写入文件
        try:
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to write LLM call log: {e}")


def create_llm_client(config: Dict[str, Any]) -> LLMClient:
    """从配置创建 LLM 客户端"""
    api_config = config.get("api", {})
    
    # 框架层参数
    framework_keys = {"provider", "base_url", "api_key", "model", "timeout", "retry_times", "retry_delay"}
    
    # 提取框架参数
    framework_params = {
        "api_key": api_config.get("api_key", "EMPTY"),
        "base_url": api_config.get("base_url", "https://api.openai.com/v1"),
        "model": api_config.get("model", "default"),
        "timeout": api_config.get("timeout", 120),
        "retry_times": api_config.get("retry_times", 3),
        "retry_delay": api_config.get("retry_delay", 5),
    }
    
    # 其余参数作为 API 参数透传
    extra_params = {k: v for k, v in api_config.items() if k not in framework_keys}
    
    return LLMClient(**framework_params, **extra_params)
