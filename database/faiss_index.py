"""
FAISS 向量索引管理
"""
import pickle
from pathlib import Path
from threading import RLock
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class FAISSIndex:
    """
    FAISS 向量索引管理器。
    
    用于高效的向量相似度搜索。
    """
    
    def __init__(
        self,
        dimension: int = 384,
        index_type: str = "IP",
        storage_path: Optional[Path] = None,
    ):
        """
        Args:
            dimension: 向量维度
            index_type: 索引类型，IP (内积/cosine) 或 L2
            storage_path: 持久化存储路径
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not installed. Run: pip install faiss-cpu")
        
        self.dimension = dimension
        self.index_type = index_type
        self.storage_path = Path(storage_path) if storage_path else None
        self.lock = RLock()
        
        # 创建索引
        if index_type == "IP":
            self.index = faiss.IndexFlatIP(dimension)
        else:
            self.index = faiss.IndexFlatL2(dimension)
        
        # ID 映射
        self.id_to_idx: Dict[int, int] = {}  # node_id -> faiss_idx
        self.idx_to_id: Dict[int, int] = {}  # faiss_idx -> node_id
        self.next_idx = 0
        
        # 尝试加载已有索引
        if self.storage_path:
            self._load()
    
    def add(self, node_id: int, vector: np.ndarray):
        """
        添加向量。
        
        Args:
            node_id: 节点 ID
            vector: 向量 (需要归一化用于 cosine 相似度)
        """
        with self.lock:
            if node_id in self.id_to_idx:
                return  # 已存在
            
            vector = self._normalize(vector.reshape(1, -1).astype(np.float32))
            self.index.add(vector)
            
            self.id_to_idx[node_id] = self.next_idx
            self.idx_to_id[self.next_idx] = node_id
            self.next_idx += 1
    
    def add_batch(self, node_ids: List[int], vectors: np.ndarray):
        """批量添加向量"""
        with self.lock:
            for i, node_id in enumerate(node_ids):
                if node_id not in self.id_to_idx:
                    self.add(node_id, vectors[i])
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Tuple[int, float]]:
        """
        搜索相似向量。
        
        Args:
            query_vector: 查询向量
            top_k: 返回数量
            score_threshold: 分数阈值
            
        Returns:
            [(node_id, score), ...] 列表
        """
        with self.lock:
            if self.index.ntotal == 0:
                return []
            
            query_vector = self._normalize(
                query_vector.reshape(1, -1).astype(np.float32)
            )
            
            k = min(top_k, self.index.ntotal)
            scores, indices = self.index.search(query_vector, k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                if score < score_threshold:
                    continue
                node_id = self.idx_to_id.get(idx)
                if node_id is not None:
                    results.append((node_id, float(score)))
            
            return results
    
    def remove(self, node_id: int):
        """
        标记删除（FAISS 不支持直接删除，使用映射表标记）
        """
        with self.lock:
            if node_id in self.id_to_idx:
                idx = self.id_to_idx.pop(node_id)
                self.idx_to_id.pop(idx, None)
    
    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2 归一化（用于 cosine 相似度）"""
        if self.index_type == "IP":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            return vectors / norms
        return vectors
    
    def save(self):
        """保存索引到文件"""
        if not self.storage_path:
            return
        
        with self.lock:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
            # 保存 FAISS 索引
            index_file = self.storage_path / "faiss.index"
            faiss.write_index(self.index, str(index_file))
            
            # 保存映射
            meta_file = self.storage_path / "faiss_meta.pkl"
            meta = {
                "id_to_idx": self.id_to_idx,
                "idx_to_id": self.idx_to_id,
                "next_idx": self.next_idx,
            }
            with open(meta_file, "wb") as f:
                pickle.dump(meta, f)
    
    def _load(self):
        """从文件加载索引"""
        if not self.storage_path:
            return
        
        index_file = self.storage_path / "faiss.index"
        meta_file = self.storage_path / "faiss_meta.pkl"
        
        if not index_file.exists() or not meta_file.exists():
            return
        
        with self.lock:
            self.index = faiss.read_index(str(index_file))
            
            with open(meta_file, "rb") as f:
                meta = pickle.load(f)
            
            self.id_to_idx = meta["id_to_idx"]
            self.idx_to_id = meta["idx_to_id"]
            self.next_idx = meta["next_idx"]
    
    def reset(self):
        """重置索引"""
        with self.lock:
            if self.index_type == "IP":
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                self.index = faiss.IndexFlatL2(self.dimension)
            
            self.id_to_idx.clear()
            self.idx_to_id.clear()
            self.next_idx = 0
            
            # 删除持久化文件
            if self.storage_path:
                for f in self.storage_path.glob("faiss*"):
                    f.unlink()
    
    @property
    def size(self) -> int:
        """有效向量数量"""
        return len(self.id_to_idx)
