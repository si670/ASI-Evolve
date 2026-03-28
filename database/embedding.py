"""
Embedding 服务
"""
from typing import List, Union

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


class EmbeddingService:
    """
    本地 Embedding 服务。
    
    使用 sentence-transformers 进行文本向量化。
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        """
        Args:
            model_name: 模型名称
            device: 运行设备 (cpu/cuda)
        """
        if not ST_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
        
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        将文本编码为向量。
        
        Args:
            texts: 单个文本或文本列表
            normalize: 是否 L2 归一化
            
        Returns:
            向量数组 (n, dimension)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        
        return np.array(embeddings, dtype=np.float32)
    
    def get_dimension(self) -> int:
        """获取向量维度"""
        return self.dimension
