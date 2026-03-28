"""
UCB1 采样算法
"""
import math
import random
from typing import List, TYPE_CHECKING

from .base import BaseSampler

if TYPE_CHECKING:
    from ...utils.structures import Node


class UCB1Sampler(BaseSampler):
    """
    UCB1 采样算法。
    
    UCB1 = normalized_score + c * sqrt(ln(N) / n_i)
    
    平衡探索（未访问节点优先）与利用（高分节点优先）。
    """
    
    def __init__(self, c: float = 1.414):
        """
        Args:
            c: 探索参数，越大越倾向于探索
        """
        self.c = c
    
    def sample(self, nodes: List["Node"], n: int) -> List["Node"]:
        if not nodes:
            return []
        
        n = min(n, len(nodes))
        
        # 计算总访问次数
        total_visits = sum(node.visit_count for node in nodes)
        if total_visits == 0:
            # 全部未访问，随机选择
            return random.sample(nodes, n)
        
        # 计算分数范围用于归一化
        scores = [node.score for node in nodes if node.visit_count > 0]
        if not scores:
            return random.sample(nodes, n)
        
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score if max_score != min_score else 1.0
        
        # 计算 UCB1 值
        ucb1_values = []
        for node in nodes:
            if node.visit_count == 0:
                # 未访问节点优先级最高
                ucb1 = float('inf')
            else:
                # 归一化分数到 [0, 1]
                normalized_score = (node.score - min_score) / score_range
                # UCB1 公式
                exploration = self.c * math.sqrt(math.log(total_visits) / node.visit_count)
                ucb1 = normalized_score + exploration
            
            ucb1_values.append((node, ucb1))
        
        # 按 UCB1 值降序排序
        ucb1_values.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前 n 个，并增加访问计数
        selected = [node for node, _ in ucb1_values[:n]]
        for node in selected:
            node.visit_count += 1
        
        return selected
