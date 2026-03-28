"""
采样器基类
"""
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..utils.structures import Node


class BaseSampler(ABC):
    """采样器基类"""
    
    @abstractmethod
    def sample(self, nodes: List["Node"], n: int) -> List["Node"]:
        """
        从节点列表中采样。
        
        Args:
            nodes: 所有节点
            n: 需要采样的数量
            
        Returns:
            采样的节点列表
        """
        pass
    
    def on_node_added(self, node: "Node") -> None:
        """
        当新节点被添加时的回调。
        
        子类可以重写此方法来处理新节点的逻辑（如岛屿分配）。
        
        Args:
            node: 新添加的节点
        """
        pass
    
    def on_node_removed(self, node: "Node") -> None:
        """
        当节点被移除时的回调。
        
        Args:
            node: 被移除的节点
        """
        pass
