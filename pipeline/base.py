"""
Agent 基类
"""
from abc import ABC, abstractmethod
from typing import Any, Dict

from ..utils.llm import LLMClient
from ..utils.prompt import PromptManager
from ..utils.logger import get_logger


class BaseAgent(ABC):
    """
    Agent 基类。
    
    所有 Agent 都继承此类，实现 run 方法。
    """
    
    def __init__(
        self,
        llm: LLMClient,
        prompt_manager: PromptManager,
        name: str = "agent",
    ):
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.name = name
        self.logger = get_logger()
        self.step_dir = None  # Step 目录，用于记录 LLM 调用日志
    
    def set_step_dir(self, step_dir):
        """
        设置 step 目录，用于记录 LLM 调用日志。
        
        Args:
            step_dir: Step 目录路径
        """
        from pathlib import Path
        self.step_dir = Path(step_dir) if step_dir else None
        if self.step_dir:
            # 创建 llm_logs 子目录
            log_dir = self.step_dir / "llm_logs"
            self.llm.set_log_dir(log_dir)
    
    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        执行 Agent 任务。
        
        Returns:
            包含结果的字典
        """
        pass
    
    def get_prompt(self, template_name: str, **context) -> str:
        """获取渲染后的 prompt（优先用户目录，回退到 utils/prompts/）"""
        if self.prompt_manager.has_template(template_name):
            return self.prompt_manager.render(template_name, **context)
        raise ValueError(f"No prompt template found for: {template_name}")
