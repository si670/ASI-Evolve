"""
贪心采样算法
"""
from typing import List, TYPE_CHECKING

from .base import BaseSampler

if TYPE_CHECKING:
    from ...utils.structures import Node


class GreedySampler(BaseSampler):
    """贪心采样（按分数降序）"""
    
    def sample(self, nodes: List["Node"], n: int) -> List["Node"]:
        if not nodes:
            return []
        sorted_nodes = sorted(nodes, key=lambda x: x.score, reverse=True)
        return sorted_nodes[:n]
