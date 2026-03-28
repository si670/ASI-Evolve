"""
采样器工厂函数

根据算法名称创建对应的采样器实例。
"""
from typing import Any

from .base import BaseSampler
from .random import RandomSampler
from .greedy import GreedySampler
from .ucb1 import UCB1Sampler
from .island import IslandSampler


def get_sampler(algorithm: str, **kwargs) -> BaseSampler:
    """
    获取采样器。
    
    Args:
        algorithm: 算法名称 (ucb1, random, greedy, island)
        **kwargs: 算法参数
            - ucb1: c (探索参数)
            - island: num_islands, migration_interval, migration_rate,
                     exploration_ratio, exploitation_ratio, c
        
    Returns:
        采样器实例
        
    Raises:
        ValueError: 如果算法名称未知
    """
    # 过滤掉 None 值的参数
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    
    samplers: dict[str, type[BaseSampler]] = {
        "ucb1": UCB1Sampler,
        "random": RandomSampler,
        "greedy": GreedySampler,
        "island": IslandSampler,
    }
    
    if algorithm not in samplers:
        raise ValueError(
            f"Unknown sampling algorithm: {algorithm}. "
            f"Available: {list(samplers.keys())}"
        )
    
    return samplers[algorithm](**kwargs)
