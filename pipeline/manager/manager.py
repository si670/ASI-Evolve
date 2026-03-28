"""
Manager Agent
=============
Meta Prompter，自动生成其他 Agent 的 prompt。
"""
from pathlib import Path
from typing import Any, Dict

from ..base import BaseAgent


class Manager(BaseAgent):
    """
    管理者 Agent (Meta Prompter)。
    
    根据任务需求和评估标准，自动生成其他 Agent 的 prompt。
    仅在第一轮实验时执行，生成的 prompt 保存到实验目录。
    """
    
    def __init__(self, llm, prompt_manager):
        super().__init__(llm, prompt_manager, name="manager")
    
    def run(
        self,
        task_description: str,
        eval_criteria: str,
        prompt_dir: Path,
        **kwargs,
    ) -> Dict[str, Any]:
        self.logger.info("[Manager] Generating agent prompts")
        
        prompt_dir = Path(prompt_dir)
        prompt_dir.mkdir(parents=True, exist_ok=True)
        
        prompt = self.get_prompt(
            "manager",
            task_description=task_description,
            eval_criteria=eval_criteria,
        )
        
        result = self.llm.extract_tags(prompt, call_name="manager")
        
        prompts = {}
        
        for agent_name in ["researcher", "analyzer"]:
            key = f"{agent_name}_prompt"
            if key in result:
                content = result[key]
                prompts[key] = content
                
                template_file = prompt_dir / f"{agent_name}.jinja2"
                template_file.write_text(content, encoding="utf-8")
                self.logger.info(f"[Manager] Saved {agent_name} prompt to {template_file}")
        
        return prompts
