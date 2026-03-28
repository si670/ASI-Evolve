"""
Analyzer Agent
==============
负责分析实验结果。

注意：Analyzer 只负责提供分析文本，不负责打分。
打分由以下方式产生：
1. eval 脚本返回的 score（主要来源）
2. LLM Judge 打分（可选，通过 config 开启）
"""
import json
from typing import Any, Dict

from ..base import BaseAgent


class Analyzer(BaseAgent):
    """
    分析师 Agent。
    
    根据代码和实验结果，分析实验的优缺点和经验教训。
    只返回分析文本，不负责打分。
    """
    
    def __init__(self, llm, prompt_manager):
        super().__init__(llm, prompt_manager, name="analyzer")
    
    def run(
        self,
        code: str,
        results: Dict[str, Any],
        task_description: str,
        best_sampled_node=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        分析实验结果。
        
        Args:
            code: 实验代码
            results: 实验结果字典（包含temp字段）
            task_description: 任务描述
            best_sampled_node: 这一轮sample出来的节点中分数最高的那个（用于比较分析）
            
        Returns:
            包含 analysis 的字典
        """
        self.logger.info("[Analyzer] Starting analysis")
        
        # 将results格式化为JSON字符串，保持可读性
        results_str = json.dumps(results, indent=2, ensure_ascii=False)
        
        # 准备最高分节点的信息
        best_node_info = None
        if best_sampled_node:
            # 将results格式化为JSON字符串，保持可读性
            best_results_str = json.dumps(best_sampled_node.results, indent=2, ensure_ascii=False)
            best_node_info = {
                "name": best_sampled_node.name,
                "score": best_sampled_node.score,
                "motivation": best_sampled_node.motivation,
                "code": best_sampled_node.code,
                "results": best_results_str,
                "analysis": best_sampled_node.analysis,
            }
        
        prompt = self.get_prompt(
            "analyzer",
            code=code,
            results=results_str,
            task_description=task_description,
            best_sampled_node=best_node_info,
        )
        
        result = self.llm.extract_tags(prompt, call_name="analyzer")
        
        analysis = result.get("analysis", "")
        
        self.logger.info("[Analyzer] Analysis completed")
        
        return {
            "analysis": analysis,
        }
