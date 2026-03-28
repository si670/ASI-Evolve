"""
Researcher Agent
================
负责根据上下文生成代码。
"""
from typing import Any, Dict, List, Optional

from ..base import BaseAgent
from ...utils.structures import Node, CognitionItem
from ...utils.diff import extract_diffs, apply_diff, parse_full_rewrite, format_diff_summary


class Researcher(BaseAgent):
    """
    研究员 Agent。
    
    根据历史实验节点和相关知识，生成新的代码方案。
    支持两种模式:
    1. diff_based_evolution: 基于已有代码进行增量修改（默认）
    2. full_rewrite: 完整重新生成代码
    """
    
    def __init__(self, llm, prompt_manager, config: Optional[Dict] = None):
        super().__init__(llm, prompt_manager, name="researcher")
        
        # 加载 researcher 配置
        self.config = config or {}
        self.diff_based = self.config.get("diff_based_evolution", True)
        self.diff_pattern = self.config.get(
            "diff_pattern", 
            r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"
        )
        self.max_code_length = self.config.get("max_code_length", 10000)
    
    def run(
        self,
        task_description: str,
        context_nodes: List[Node],
        cognition_items: List[CognitionItem],
        base_code: Optional[str] = None,  # 新增：用于 diff 模式的基础代码
        **kwargs,
    ) -> Dict[str, Any]:
        self.logger.info(
            f"[Researcher] Starting with {len(context_nodes)} context nodes, "
            f"mode={'diff' if self.diff_based else 'full_rewrite'}"
        )
        
        # 如果是 diff 模式但没有提供 base_code，从 context_nodes 中选择第一个作为 base
        if self.diff_based and not base_code and context_nodes:
            base_code = context_nodes[0].code
            self.logger.info(f"[Researcher] Using first context node as base: {context_nodes[0].name}")
        
        # 生成代码
        if self.diff_based and base_code:
            result = self._generate_diff(
                task_description, context_nodes, cognition_items, base_code
            )
        else:
            prompt = self.get_prompt(
                "researcher",
                task_description=task_description,
                context_nodes=[n.to_dict() for n in context_nodes],
                cognition_items=[c.to_dict() for c in cognition_items],
                base_code=None,
                diff_based=False,
            )
            result = self._generate_full(prompt)
        
        # 检查代码长度
        if len(result["code"]) > self.max_code_length:
            self.logger.warning(
                f"[Researcher] Generated code exceeds max length "
                f"({len(result['code'])} > {self.max_code_length})"
            )
            result["code"] = result["code"][:self.max_code_length]
        
        self.logger.info(f"[Researcher] Generated: {result.get('name', 'unnamed')}")
        
        return result
    
    def _generate_diff(
        self,
        task_description: str,
        context_nodes: List[Node],
        cognition_items: List[CognitionItem],
        base_code: str,
    ) -> Dict[str, Any]:
        """
        使用 diff 模式生成代码。
        
        Args:
            task_description: 任务描述
            context_nodes: 上下文节点
            cognition_items: 知识项
            base_code: 基础代码
            
        Returns:
            包含 name, motivation, code, changes 的字典
        """
        # 生成 diff 模式的 prompt
        prompt = self.get_prompt(
            "researcher",
            task_description=task_description,
            context_nodes=[n.to_dict() for n in context_nodes],
            cognition_items=[c.to_dict() for c in cognition_items],
            base_code=base_code,
            diff_based=True,
        )
        
        # 让 LLM 生成 diff
        response = self.llm.generate(prompt, call_name="researcher_diff")
        response_text = response.content if hasattr(response, "content") else str(response)
        
        # 提取 diff 块（config 里可能是字面 \n，需转成真换行）
        pattern = self.diff_pattern.replace("\\n", "\n") if isinstance(self.diff_pattern, str) else self.diff_pattern
        diff_blocks = extract_diffs(response_text, pattern)
        
        if not diff_blocks:
            # 输出完整响应用于 debug
            self.logger.error(f"[Researcher] No diff blocks found. Full response ({len(response_text)} chars):\n{response_text}")
            # Fallback: 重新生成 full rewrite 的 prompt
            full_prompt = self.get_prompt(
                "researcher",
                task_description=task_description,
                context_nodes=[n.to_dict() for n in context_nodes],
                cognition_items=[c.to_dict() for c in cognition_items],
                base_code=None,
                diff_based=False,
            )
            return self._generate_full(full_prompt)
        
        # 应用 diff
        try:
            new_code = apply_diff(base_code, response_text, pattern)
            changes_summary = format_diff_summary(diff_blocks)
            
            # 从响应开头提取 XML 标签（name 和 motivation）
            import re
            
            name = "diff_modification"
            motivation = changes_summary
            
            # 提取 <name> 和 <motivation> 标签
            name_match = re.search(r'<name>(.*?)</name>', response_text, re.DOTALL)
            if name_match:
                name = name_match.group(1).strip()
            
            motivation_match = re.search(r'<motivation>(.*?)</motivation>', response_text, re.DOTALL)
            if motivation_match:
                motivation = motivation_match.group(1).strip()
            
            return {
                "name": name,
                "motivation": motivation,
                "code": new_code,
                "changes": changes_summary,
            }
            
        except ValueError as e:
            self.logger.error(f"[Researcher] Failed to apply diff: {e}")
            # 回退到完整重写模式：重新生成 full rewrite 的 prompt
            full_prompt = self.get_prompt(
                "researcher",
                task_description=task_description,
                context_nodes=[n.to_dict() for n in context_nodes],
                cognition_items=[c.to_dict() for c in cognition_items],
                base_code=None,
                diff_based=False,
            )
            return self._generate_full(full_prompt)
    
    def _generate_full(self, prompt: str) -> Dict[str, Any]:
        """
        使用完整重写模式生成代码。
        
        Args:
            prompt: 提示词
            
        Returns:
            包含 name, motivation, code 的字典
        """
        try:
            result = self.llm.extract_tags(prompt, call_name="researcher_full")
        except ValueError as e:
            # 如果 XML 标签提取失败，尝试获取原始响应用于 debug
            response = self.llm.generate(prompt, call_name="researcher_full_debug")
            response_text = response.content if hasattr(response, "content") else str(response)
            self.logger.error(f"[Researcher] Full rewrite tag extraction failed. Full response ({len(response_text)} chars):\n{response_text}")
            raise
        
        return {
            "name": result.get("name", ""),
            "motivation": result.get("motivation", ""),
            "code": result.get("code", ""),
        }
