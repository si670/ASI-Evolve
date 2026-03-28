"""
随机采样算法
"""
import random
from typing import List, TYPE_CHECKING

from .base import BaseSampler

if TYPE_CHECKING:
    from ...utils.structures import Node


class RandomSampler(BaseSampler):
    """随机采样"""
    
    def sample(self, nodes: List["Node"], n: int) -> List["Node"]:
        if not nodes:
            return []
        n = min(n, len(nodes))
        return random.sample(nodes, n)
