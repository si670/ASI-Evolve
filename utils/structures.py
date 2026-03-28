"""
数据结构定义
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime


@dataclass
class Node:
    """
    数据库中的实验节点。
    
    Attributes:
        name: 节点名字
        created_at: 创建时间戳
        parent: 来自于哪些节点的索引列表
        motivation: 动机分析以及期待的效果
        code: 具体代码
        results: 具体的结果字典
        analysis: 分析结果，有什么可以汲取的经验
        meta_info: 用于后续添加杂项记录
        id: 节点唯一标识（由数据库分配）
        visit_count: 访问次数（用于 UCB1）
        score: 评分
    """
    name: str = ""
    created_at: str = ""
    parent: List[int] = field(default_factory=list)
    motivation: str = ""
    code: str = ""
    results: Dict[str, Any] = field(default_factory=dict)
    analysis: str = ""
    meta_info: Dict[str, Any] = field(default_factory=dict)
    id: Optional[int] = None
    visit_count: int = 0
    score: float = 0.0
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "parent": self.parent,
            "motivation": self.motivation,
            "code": self.code,
            "results": self.results,
            "analysis": self.analysis,
            "meta_info": self.meta_info,
            "visit_count": self.visit_count,
            "score": self.score,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Node":
        """从字典创建"""
        return cls(
            id=data.get("id"),
            name=data.get("name", ""),
            created_at=data.get("created_at", ""),
            parent=data.get("parent", []),
            motivation=data.get("motivation", ""),
            code=data.get("code", ""),
            results=data.get("results", {}),
            analysis=data.get("analysis", ""),
            meta_info=data.get("meta_info", {}),
            visit_count=data.get("visit_count", 0),
            score=data.get("score", 0.0),
        )
    
    def get_context_text(self) -> str:
        """获取用于 embedding 的文本"""
        parts = [self.name, self.motivation, self.analysis]
        return " ".join(p for p in parts if p)


@dataclass
class CognitionItem:
    """
    Cognition 知识库中的条目。
    
    Attributes:
        id: 唯一标识
        content: 内容
        source: 来源（如论文标题、URL等）
        metadata: 元信息
    """
    content: str
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CognitionItem":
        return cls(
            id=data.get("id"),
            content=data.get("content", ""),
            source=data.get("source", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str
    experiment_dir: Path = None
    input_file: Optional[str] = None  # 任务描述文件
    eval_script: Optional[str] = None  # 评估脚本
    run_script: Optional[str] = None   # 运行脚本
    
    def __post_init__(self):
        from pathlib import Path
        if self.experiment_dir is None:
            base_dir = Path(__file__).parent.parent / "experiments"
            self.experiment_dir = base_dir / self.name
        elif isinstance(self.experiment_dir, str):
            self.experiment_dir = Path(self.experiment_dir)


@dataclass  
class LLMResponse:
    """LLM 响应结构"""
    content: str
    raw_response: Any = None
    usage: Dict[str, int] = field(default_factory=dict)
    model: str = ""
    call_time: float = 0.0
