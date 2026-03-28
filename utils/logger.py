"""
日志模块
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class EvolveLogger:
    """
    Evolve 框架的日志管理器。
    
    支持:
    - Console 输出
    - 文件日志
    - WandB 集成
    """
    
    def __init__(
        self,
        name: str = "evolve",
        log_dir: Optional[Path] = None,
        level: str = "INFO",
        console: bool = True,
        wandb_config: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.log_dir = Path(log_dir) if log_dir else None
        self.level = getattr(logging, level.upper(), logging.INFO)
        self.console = console
        self.wandb_config = wandb_config
        
        # 创建 logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        self.logger.handlers.clear()
        
        # 格式
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_file = self.log_dir / "evolve.log"
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # WandB
        self.wandb_run = None
        if wandb_config and wandb_config.get("enabled") and WANDB_AVAILABLE:
            self._init_wandb(wandb_config)
        
        # 统计信息
        self.stats = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_calls": 0,
            "total_time": 0.0,
        }
    
    def _init_wandb(self, config: Dict[str, Any]):
        """初始化 WandB"""
        import os
        
        # 设置 offline 模式
        if config.get("offline", False):
            os.environ["WANDB_MODE"] = "offline"
        
        try:
            self.wandb_run = wandb.init(
                project=config.get("project", "evolve"),
                entity=config.get("entity"),
                name=config.get("run_name"),  # 会在 pipeline 中设置为 experiment_name
                config=config.get("config", {}),
                dir=str(self.log_dir) if self.log_dir else None,  # wandb 文件存到实验日志目录
                resume="allow",  # 支持 resume
            )
            self.logger.info(f"WandB initialized: {self.wandb_run.name} (mode: {wandb.run.settings.mode})")
        except Exception as e:
            self.logger.warning(f"Failed to initialize wandb: {e}")
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def debug(self, msg: str):
        self.logger.debug(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)
    
    def log_llm_call(self, call_info: Dict[str, Any]):
        """记录 LLM 调用信息"""
        usage = call_info.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        
        self.stats["total_calls"] += 1
        self.stats["prompt_tokens"] += prompt_tokens
        self.stats["completion_tokens"] += completion_tokens
        self.stats["total_tokens"] += prompt_tokens + completion_tokens
        self.stats["total_time"] += call_info.get("call_time", 0)
        
        self.debug(
            f"LLM Call: model={call_info.get('model')}, "
            f"tokens={prompt_tokens}+{completion_tokens}, "
            f"time={call_info.get('call_time', 0):.2f}s"
        )
        
        # WandB logging - 注意：不单独记录，避免干扰 step 计数
        # LLM 调用统计会在每个 step 结束时一起记录
    
    def log_experiment_step(self, step: int, metrics: Dict[str, Any]):
        """记录实验步骤"""
        self.info(f"Step {step}: {metrics}")
        
        if self.wandb_run:
            # 注意：不使用 wandb 的内部 step 参数，避免并行时 step 乱序报错
            # 把逻辑 step 作为一个普通标量字段记录，由用户在面板中手动选择 x 轴
            wandb.log({"pipeline/step": step, **metrics})
    
    def log_node(self, node: "Node", step: int, database: Optional[Any] = None):
        """
        记录新节点。
        
        自动从 node.results 中提取所有数值型指标，无需针对特定实验修改。
        
        Args:
            node: 要记录的节点
            step: 当前 step
            database: 可选的 Database 实例，用于计算历史最大 score
        """
        self.info(f"New node: {node.name} (score={node.score:.4f})")
        
        if self.wandb_run:
            log_data = {
                # 逻辑上的 pipeline step，作为普通字段记录
                "pipeline/step": step,
                "node/score": node.score,
                "node/code_length": len(node.code) if node.code else 0,
                # 添加 LLM 统计信息
                "llm/total_calls": self.stats["total_calls"],
                "llm/total_tokens": self.stats["total_tokens"],
                "llm/prompt_tokens": self.stats["prompt_tokens"],
                "llm/completion_tokens": self.stats["completion_tokens"],
            }
            
            # 计算历史最大 score（包括当前节点和所有历史节点）
            if database is not None:
                all_nodes = database.get_all()
                if all_nodes:
                    max_score = max(n.score for n in all_nodes)
                    log_data["best/max_score"] = max_score
                else:
                    log_data["best/max_score"] = node.score
            
            # 自动提取 results 中的所有数值型指标
            if node.results:
                self._extract_metrics(node.results, "results", log_data)
            
            # 自动提取 meta_info 中的数值型指标
            if node.meta_info:
                self._extract_metrics(node.meta_info, "meta", log_data)
            
            # 不显式传 step，避免并行时 step 乱序触发 wandb 限制
            # 内部全局 step 由 wandb 按调用顺序自增
            wandb.log(log_data)
    
    def _extract_metrics(
        self, 
        data: Any, 
        prefix: str, 
        output: Dict[str, Any],
        max_depth: int = 3,
        _depth: int = 0,
    ):
        """
        递归提取字典中的所有数值型指标。
        
        Args:
            data: 要提取的数据
            prefix: 指标名前缀
            output: 输出字典
            max_depth: 最大递归深度
            _depth: 当前深度
        """
        if _depth >= max_depth:
            return
        
        if isinstance(data, dict):
            for key, value in data.items():
                # 跳过太长的字符串值（如 error traceback）
                if isinstance(value, str) and len(value) > 200:
                    continue
                
                new_prefix = f"{prefix}/{key}"
                self._extract_metrics(value, new_prefix, output, max_depth, _depth + 1)
        
        elif isinstance(data, (int, float)) and not isinstance(data, bool):
            # 数值型指标直接记录
            output[prefix] = data
        
        elif isinstance(data, bool):
            # 布尔值转为 0/1
            output[prefix] = 1 if data else 0
        
        elif isinstance(data, list) and len(data) > 0 and all(isinstance(x, (int, float)) for x in data):
            # 数值列表记录平均值
            output[f"{prefix}/mean"] = sum(data) / len(data)
            output[f"{prefix}/max"] = max(data)
            output[f"{prefix}/min"] = min(data)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
    
    def finish(self):
        """结束日志"""
        self.info(f"Total stats: {self.stats}")
        if self.wandb_run:
            wandb.finish()


# 全局 logger 实例
_logger: Optional[EvolveLogger] = None


def get_logger() -> EvolveLogger:
    """获取全局 logger"""
    global _logger
    if _logger is None:
        _logger = EvolveLogger()
    return _logger


def init_logger(
    name: str = "evolve",
    log_dir: Optional[Path] = None,
    level: str = "INFO",
    console: bool = True,
    wandb_config: Optional[Dict[str, Any]] = None,
) -> EvolveLogger:
    """初始化全局 logger"""
    global _logger
    _logger = EvolveLogger(
        name=name,
        log_dir=log_dir,
        level=level,
        console=console,
        wandb_config=wandb_config,
    )
    return _logger
