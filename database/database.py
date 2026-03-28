"""
数据库主模块
"""
import json
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional

from ..utils.structures import Node
from .algorithms import BaseSampler, get_sampler
from .faiss_index import FAISSIndex
from .embedding import EmbeddingService


class Database:
    """
    实验数据库。
    
    功能:
    - 节点的增删查
    - 基于算法的采样 (UCB1, random, greedy, island)
    - 向量相似度搜索
    - 本地持久化存储
    
    接口:
    - sample(n, algorithm=None, **kwargs): 采样 n 个节点
    - add(node) / add_batch(nodes): 添加节点
    - remove(node_id) / remove_batch(node_ids): 删除节点
    - get(node_id): 根据 ID 获取节点
    - reset(): 清空数据库
    """
    
    def __init__(
        self,
        storage_dir: Path,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        sampling_algorithm: str = "ucb1",
        sampling_kwargs: Optional[Dict[str, Any]] = None,
        faiss_index_type: str = "IP",
        max_size: Optional[int] = None,
    ):
        """
        Args:
            storage_dir: 存储目录
            embedding_model: Embedding 模型名称
            embedding_dim: Embedding 维度
            sampling_algorithm: 采样算法名称
            sampling_kwargs: 采样算法参数字典
            faiss_index_type: FAISS 索引类型
            max_size: 数据库大小上限，如果设置了此值，当节点数超过此值时会自动移除 score 最低的节点
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.lock = RLock()
        
        # 数据存储
        self.nodes: Dict[int, Node] = {}
        self.next_id = 0
        self.max_size = max_size
        
        # Embedding 服务
        self.embedding = EmbeddingService(model_name=embedding_model)
        
        # FAISS 索引
        self.faiss = FAISSIndex(
            dimension=embedding_dim,
            index_type=faiss_index_type,
            storage_path=self.storage_dir / "faiss",
        )
        
        # 采样器
        self.sampling_algorithm = sampling_algorithm
        self.sampling_kwargs = sampling_kwargs or {}
        self.default_sampler = get_sampler(sampling_algorithm, **self.sampling_kwargs)
        
        # 加载已有数据
        self._load()
    
    def sample(
        self,
        n: int,
        algorithm: Optional[str] = None,
        **kwargs,
    ) -> List[Node]:
        """
        采样节点。
        
        Args:
            n: 采样数量
            algorithm: 可选，指定采样算法（不指定则使用默认算法）
            **kwargs: 传递给采样算法的额外参数
            
        Returns:
            采样的节点列表
        """
        with self.lock:
            nodes = list(self.nodes.values())
            
            if algorithm:
                sampler = get_sampler(algorithm, **kwargs)
            else:
                sampler = self.default_sampler
            
            selected = sampler.sample(nodes, n)
            self._save()
            
            return selected
    
    def add(self, node: Node) -> int:
        """
        添加节点到数据库。
        
        如果设置了 max_size 且当前节点数已达到上限，会自动移除 score 最低的节点。
        如果多个节点 score 相同，则移除 id 最小的（先加入的）节点。
        
        Args:
            node: 要添加的节点
            
        Returns:
            节点 ID
        """
        with self.lock:
            # 如果设置了上限且已达到上限，需要先移除一个节点
            if self.max_size is not None and len(self.nodes) >= self.max_size:
                self._remove_worst_node()
            
            node.id = self.next_id
            self.next_id += 1
            
            self.nodes[node.id] = node
            
            # 通知采样器新节点被添加（采样器内部处理自己的逻辑）
            self.default_sampler.on_node_added(node)
            
            text = node.get_context_text()
            if text:
                vector = self.embedding.encode(text)
                self.faiss.add(node.id, vector)
            
            self._save()
            return node.id
    
    def add_batch(self, nodes: List[Node]) -> List[int]:
        return [self.add(node) for node in nodes]
    
    def remove(self, node_id: int) -> bool:
        with self.lock:
            if node_id not in self.nodes:
                return False
            
            node = self.nodes[node_id]
            
            # 通知采样器节点被移除
            self.default_sampler.on_node_removed(node)
            
            del self.nodes[node_id]
            self.faiss.remove(node_id)
            self._save()
            return True
    
    def remove_batch(self, node_ids: List[int]) -> int:
        return sum(1 for nid in node_ids if self.remove(nid))
    
    def _remove_worst_node(self):
        """
        移除 score 最低的节点。
        如果多个节点 score 相同，则移除 id 最小的（先加入的）节点。
        """
        if not self.nodes:
            return
        
        # 找到 score 最低的节点，如果 score 相同则选择 id 最小的
        worst_node_id = min(
            self.nodes.keys(),
            key=lambda node_id: (self.nodes[node_id].score, node_id)
        )
        
        self.remove(worst_node_id)
    
    def get(self, node_id: int) -> Optional[Node]:
        return self.nodes.get(node_id)
    
    def get_all(self) -> List[Node]:
        return list(self.nodes.values())
    
    def search_similar(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Node]:
        query_vector = self.embedding.encode(query)
        results = self.faiss.search(query_vector, top_k, score_threshold)
        
        nodes = []
        for node_id, score in results:
            node = self.nodes.get(node_id)
            if node:
                nodes.append(node)
        
        return nodes
    
    def reset(self):
        """清空数据库"""
        with self.lock:
            # 通知采样器重置（采样器自己处理内部状态）
            if hasattr(self.default_sampler, 'reset'):
                self.default_sampler.reset()
            
            self.nodes.clear()
            self.next_id = 0
            self.faiss.reset()
            
            data_file = self.storage_dir / "nodes.json"
            if data_file.exists():
                data_file.unlink()
    
    def get_sampler_stats(self) -> Optional[Dict[str, Any]]:
        """
        获取采样器的统计信息（如果支持）。
        
        不同的采样器可能提供不同的统计信息：
        - IslandSampler: 岛屿种群、代数、迁移等信息
        - UCB1Sampler: 可能提供访问计数等信息
        
        Returns:
            统计信息字典，如果采样器不支持则返回 None
        """
        # 岛屿算法的特殊方法
        if hasattr(self.default_sampler, 'get_island_stats'):
            with self.lock:
                nodes = list(self.nodes.values())
                return self.default_sampler.get_island_stats(nodes)
        
        # 其他采样器可以扩展自己的 get_stats 方法
        if hasattr(self.default_sampler, 'get_stats'):
            with self.lock:
                nodes = list(self.nodes.values())
                return self.default_sampler.get_stats(nodes)
        
        return None
    
    def call_sampler_method(self, method_name: str, *args, **kwargs) -> Any:
        """
        调用采样器的自定义方法（如果存在）。
        
        这是一个通用接口，允许访问采样器的特殊功能，例如：
        - IslandSampler.sample_from_island(island_id, n)
        
        Args:
            method_name: 方法名称
            *args, **kwargs: 传递给方法的参数
            
        Returns:
            方法返回值
            
        Raises:
            AttributeError: 如果方法不存在
        """
        if not hasattr(self.default_sampler, method_name):
            raise AttributeError(
                f"Sampler '{self.sampling_algorithm}' does not have method '{method_name}'"
            )
        
        method = getattr(self.default_sampler, method_name)
        
        # 如果方法需要节点列表，自动传入
        with self.lock:
            # 检查方法签名，如果需要 nodes 参数，自动传入
            import inspect
            sig = inspect.signature(method)
            if 'nodes' in sig.parameters:
                kwargs['nodes'] = list(self.nodes.values())
            
            return method(*args, **kwargs)
    
    def _save(self):
        """保存数据库状态"""
        data_file = self.storage_dir / "nodes.json"
        
        data = {
            "next_id": self.next_id,
            "nodes": {str(k): v.to_dict() for k, v in self.nodes.items()},
        }
        
        # 保存采样器状态（采样器自己决定保存什么）
        if hasattr(self.default_sampler, 'get_state'):
            data["sampler_state"] = self.default_sampler.get_state()
        
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        self.faiss.save()
    
    def _load(self):
        """加载数据库状态"""
        data_file = self.storage_dir / "nodes.json"
        
        if not data_file.exists():
            return
        
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.next_id = data.get("next_id", 0)
        
        for node_id, node_data in data.get("nodes", {}).items():
            node = Node.from_dict(node_data)
            self.nodes[int(node_id)] = node
        
        # 恢复采样器状态（采样器自己决定如何恢复）
        if hasattr(self.default_sampler, 'load_state') and "sampler_state" in data:
            self.default_sampler.load_state(data["sampler_state"])
        
        # 重建采样器内部结构（从已有节点）
        if hasattr(self.default_sampler, 'rebuild_from_nodes'):
            self.default_sampler.rebuild_from_nodes(list(self.nodes.values()))
    
    @property
    def size(self) -> int:
        return len(self.nodes)
    
    def __len__(self) -> int:
        return self.size
