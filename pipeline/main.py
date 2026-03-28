"""
Pipeline 主模块
===============
整合所有 Agent，实现完整的实验流程。

支持并行 worker 模式和错误恢复机制。
"""
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from ..utils.config import load_config
from ..utils.llm import create_llm_client
from ..utils.logger import init_logger
from ..utils.prompt import PromptManager
from ..utils.structures import Node, CognitionItem
from ..utils import BestSnapshotManager
from ..database import Database
from ..cognition import Cognition

from .researcher import Researcher
from .engineer import Engineer
from .analyzer import Analyzer
from .manager import Manager


class Pipeline:
    """
    Evolve 主 Pipeline。
    
    流程:
    1. [可选] Manager 生成 prompts（仅第一轮）
    2. 从 Database 采样历史节点
    3. 从 Cognition 检索相关知识
    4. Researcher 生成代码
    5. Engineer 运行实验
    6. Analyzer 分析结果
    7. 将结果存入 Database
    
    支持 Resume:
    - 当使用已存在的 experiment_name 时，自动恢复状态
    - Database 和 Cognition 从本地文件加载
    - step 计数器从最后完成的 step 继续
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ):
        # 先确定实验名称（需要从默认配置或参数获取）
        if experiment_name is None:
            # 尝试从默认配置获取实验名称
            from ..utils.config import load_config as _load_config
            temp_config = _load_config(config_path=config_path)
            experiment_name = temp_config.get("experiment_name", "default")
        
        self.experiment_name = experiment_name
        
        # 加载配置（支持实验目录配置优先）
        self.config = load_config(config_path=config_path, experiment_name=experiment_name)
        self.config["experiment_name"] = experiment_name
        
        # 设置实验目录
        base_dir = Path(__file__).parent.parent / "experiments"
        self.experiment_dir = base_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # 统一管理 steps 目录，并在实验开始时创建 best 目录
        self.steps_dir = self.experiment_dir / "steps"
        self.steps_dir.mkdir(parents=True, exist_ok=True)
        
        # 状态文件路径
        self.state_file = self.experiment_dir / "pipeline_state.json"
        
        # 初始化日志
        log_config = self.config.get("logging", {})
        wandb_config = log_config.get("wandb", {})
        if wandb_config:
            wandb_config = wandb_config.copy()
            wandb_config["run_name"] = self.experiment_name
            wandb_config["config"] = self.config
        
        self.logger = init_logger(
            name="evolve",
            log_dir=self.experiment_dir / "logs",
            level=log_config.get("level", "INFO"),
            console=log_config.get("console", True),
            wandb_config=wandb_config,
        )
        
        # 初始化 LLM
        self.llm = create_llm_client(self.config)
        
        # 初始化 Prompt Manager
        prompt_dir = self.experiment_dir / "prompts"
        self.prompt_manager = PromptManager(prompt_dir)
        
        # 初始化 Database
        db_config = self.config.get("database", {})
        sampling_config = db_config.get("sampling", {})
        algorithm = sampling_config.get("algorithm", "ucb1")
        
        # 根据算法类型构建参数字典
        sampling_kwargs = {}
        if algorithm == "ucb1":
            sampling_kwargs["c"] = sampling_config.get("ucb1_c", 1.414)
        elif algorithm.startswith("island"):
            island_config = sampling_config.get(algorithm, sampling_config.get("island", {}))
            sampling_kwargs = {
                "num_islands": island_config.get("num_islands", 5),
                "migration_interval": island_config.get("migration_interval", 10),
                "migration_rate": island_config.get("migration_rate", 0.1),
                "exploration_ratio": island_config.get("exploration_ratio", 0.2),
                "exploitation_ratio": island_config.get("exploitation_ratio", 0.3),
                "feature_dimensions": island_config.get("feature_dimensions", []),
                "feature_bins": island_config.get("feature_bins", 10),
            }
        # random 和 greedy 不需要额外参数
        
        self.database = Database(
            storage_dir=self.experiment_dir / db_config.get("storage_dir", "database_data"),
            embedding_model=db_config.get("embedding", {}).get(
                "model", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            embedding_dim=db_config.get("embedding", {}).get("dimension", 384),
            sampling_algorithm=algorithm,
            sampling_kwargs=sampling_kwargs,
            max_size=db_config.get("max_size"),
        )
        
        # 初始化 Cognition
        cog_config = self.config.get("cognition", {})
        self.cognition = Cognition(
            storage_dir=self.experiment_dir / cog_config.get("storage_dir", "cognition_data"),
            embedding_model=cog_config.get("embedding", {}).get(
                "model", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            embedding_dim=cog_config.get("embedding", {}).get("dimension", 384),
            retrieval_top_k=cog_config.get("retrieval", {}).get("top_k", 5),
            score_threshold=cog_config.get("retrieval", {}).get("score_threshold", 0.5),
        )
        
        # 初始化 Agents
        pipeline_config = self.config.get("pipeline", {})
        agents_config = pipeline_config.get("agents", {})
        
        self.use_manager = agents_config.get("manager", False)
        self.use_researcher = agents_config.get("researcher", True)
        self.use_engineer = agents_config.get("engineer", True)
        self.use_analyzer = agents_config.get("analyzer", True)
        
        # Researcher 配置
        self.researcher_config = pipeline_config.get("researcher", {})
        
        self.manager = Manager(self.llm, self.prompt_manager) if self.use_manager else None
        self.researcher = Researcher(self.llm, self.prompt_manager, self.researcher_config) if self.use_researcher else None
        self.engineer = Engineer(self.llm, self.prompt_manager) if self.use_engineer else None
        self.analyzer = Analyzer(self.llm, self.prompt_manager) if self.use_analyzer else None
        
        # 重试配置
        self.max_retries = pipeline_config.get("max_retries", {})
        
        # Judge 配置
        judge_config = pipeline_config.get("judge", {})
        self.judge_enabled = judge_config.get("enabled", False)
        self.judge_ratio = judge_config.get("ratio", 0.2)
        
        # 并行配置
        parallel_config = pipeline_config.get("parallel", {})
        self.num_workers = parallel_config.get("num_workers", 1)
        self.step_lock = Lock()  # 用于保护 step 计数器
        
        # 超时配置
        self.engineer_timeout = pipeline_config.get("engineer_timeout", 3600)  # 默认 1 小时
        
        # 采样配置
        self.sample_n = pipeline_config.get("sample_n", 3)  # 每个 step 从历史节点中采样的数量
        
        # 状态 - 尝试从文件恢复
        self.step = 0
        self.manager_initialized = False
        self._load_state()
        
        # 检查是否为 resume
        self.is_resume = self.step > 0 or len(self.database) > 0
        if self.is_resume:
            self.logger.info(
                f"Resuming experiment '{self.experiment_name}' from step {self.step} "
                f"(database: {len(self.database)} nodes, cognition: {len(self.cognition)} items)"
            )
        else:
            self.logger.info(f"Starting new experiment: {self.experiment_name}")
        
        # 初始节点标记（用于只创建一次 initial_program 节点）
        self.initial_node_created = False

        # 初始化 best 快照管理（resume 时会恢复历史最高分）
        self.best_snapshot = BestSnapshotManager(self.steps_dir, logger=self.logger)
        self.best_snapshot.init_from_nodes(self.database.get_all())
    
    def _load_state(self):
        """从文件加载 Pipeline 状态"""
        import json
        
        if not self.state_file.exists():
            if len(self.database) > 0:
                max_id = max(n.id for n in self.database.get_all() if n.id is not None)
                self.step = max_id + 1
                prompt_dir = self.experiment_dir / "prompts"
                if prompt_dir.exists() and any(prompt_dir.glob("*.jinja2")):
                    self.manager_initialized = True
            return
        
        with open(self.state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
        
        self.step = state.get("step", 0)
        self.manager_initialized = state.get("manager_initialized", False)
    
    def _save_state(self):
        """保存 Pipeline 状态到文件"""
        state = {
            "step": self.step,
            "manager_initialized": self.manager_initialized,
            "last_updated": datetime.now().isoformat(),
        }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    
    def run_step(
        self,
        task_description: Optional[str] = None,
        eval_script: Optional[str] = None,
        sample_n: Optional[int] = None,
    ) -> Optional[Node]:
        """
        运行一个evolution step。
        
        包含完整的错误处理和恢复机制。
        每个 worker 获得独立的 step 编号，确保并行安全。
        
        Returns:
            成功返回 Node，失败返回 None
        """
        # 原子操作：分配独立的 step 编号给当前 worker
        # 这确保每个 worker 的 step 是唯一的，不会冲突
        with self.step_lock:
            self.step += 1
            current_step = self.step
            # 保存状态，确保 step 计数持久化
            self._save_state()
        
        self.logger.info(f"=== Step {current_step} ===")
        
        # 使用配置中的 sample_n 作为默认值
        if sample_n is None:
            sample_n = self.sample_n
        
        try:
            # 读取任务描述
            if task_description is None:
                input_file = self.experiment_dir / "input.md"
                if input_file.exists():
                    task_description = input_file.read_text(encoding="utf-8")
                else:
                    self.logger.error("No task description provided")
                    return None
            
            # 1. Manager（仅第一轮且启用时）
            if self.use_manager and not self.manager_initialized:
                self._run_manager(task_description)
                self.manager_initialized = True
                self._save_state()
                self.prompt_manager = PromptManager(self.experiment_dir / "prompts")
                if self.researcher:
                    self.researcher.prompt_manager = self.prompt_manager
                if self.analyzer:
                    self.analyzer.prompt_manager = self.prompt_manager
            
            # 2. 采样历史节点（线程安全）
            context_nodes = self.database.sample(sample_n)
            parent_ids = [n.id for n in context_nodes if n.id is not None]
            self.logger.info(f"Sampled {len(context_nodes)} context nodes")
            
            # 3. 检索相关知识（线程安全）
            cognition_items = []
            if context_nodes:
                for node in context_nodes:
                    if node.analysis:
                        items = self.cognition.search(node.analysis, top_k=2)
                        cognition_items.extend(items)
                    else:
                        items = self.cognition.search(node.motivation, top_k=2)
                        cognition_items.extend(items)
            self.logger.info(f"Retrieved {len(cognition_items)} cognition items")
            
            # 4. Researcher 生成代码
            if not self.researcher:
                self.logger.error("Researcher not enabled")
                return None
            
            # 创建 step 目录（提前创建，用于记录 LLM 日志）
            step_dir = self.steps_dir / f"step_{current_step}"
            step_dir.mkdir(parents=True, exist_ok=True)
            
            # 设置 step_dir 给各个 agent（用于记录 LLM 调用日志）
            if self.researcher:
                self.researcher.set_step_dir(step_dir)
            if self.analyzer:
                self.analyzer.set_step_dir(step_dir)
            if self.engineer:
                self.engineer.set_step_dir(step_dir)
            
            # 准备 base_code (如果是 diff 模式)
            base_code = None
            if self.researcher_config.get("diff_based_evolution", True) and context_nodes:
                # 使用第一个采样节点作为 base
                base_code = context_nodes[0].code
                self.logger.info(f"Using base code from: {context_nodes[0].name}")
            
            try:
                researcher_result = self.researcher.run(
                    task_description=task_description,
                    context_nodes=context_nodes,
                    cognition_items=cognition_items,
                    base_code=base_code,
                )
            except Exception as e:
                self.logger.error(f"Researcher failed: {type(e).__name__}: {e}")
                self.logger.error(traceback.format_exc())
                return None
            
            # 创建节点
            node = Node(
                name=researcher_result.get("name", f"node_{current_step}"),
                created_at=datetime.now().isoformat(),
                parent=parent_ids,
                motivation=researcher_result.get("motivation", ""),
                code=researcher_result.get("code", ""),
            )
            
            # 初始化engineer_result，以便在异常情况下analyzer也能使用
            engineer_result = {}
            
            # 5. Engineer 运行实验
            if self.engineer and (eval_script or self.judge_enabled):
                try:
                    engineer_result = self.engineer.run(
                        code=node.code,
                        experiment_dir=step_dir,
                        eval_script=eval_script,
                        timeout=self.engineer_timeout,
                        task_description=task_description,
                        judge_enabled=self.judge_enabled,
                        judge_ratio=self.judge_ratio,
                    )
                    
                    # engineer_result包含完整的json (含temp字段)
                    # 分离temp字段：node.results不含temp，analyzer使用完整的
                    node.results = {k: v for k, v in engineer_result.items() if k != "temp"}
                    
                    # Score 从 Engineer 获取
                    node.score = engineer_result.get("score", 0.0)
                    node.meta_info["runtime"] = engineer_result.get("runtime")
                    node.meta_info["success"] = engineer_result.get("success")
                    node.meta_info["eval_score"] = engineer_result.get("eval_score", 0.0)
                    if self.judge_enabled:
                        node.meta_info["judge_score"] = engineer_result.get("judge_score")
                    
                    if not engineer_result.get("success"):
                        node.meta_info["error"] = engineer_result.get("error")
                        
                except Exception as e:
                    self.logger.error(f"Engineer failed: {type(e).__name__}: {e}")
                    self.logger.error(traceback.format_exc())
                    node.meta_info["success"] = False
                    node.meta_info["error"] = str(e)
                    node.score = 0.0
                    engineer_result = {}
            
            # 6. Analyzer 分析结果（使用完整的engineer_result，包含temp字段）
            if self.analyzer:
                try:
                    # 找到这一轮sample出来的节点中分数最高的那个
                    best_sampled_node = None
                    if context_nodes:
                        best_sampled_node = max(context_nodes, key=lambda n: n.score)
                        self.logger.info(f"Best sampled node for comparison: {best_sampled_node.name} (score={best_sampled_node.score:.4f})")
                    
                    analyzer_result = self.analyzer.run(
                        code=node.code,
                        results=engineer_result,
                        task_description=task_description,
                        best_sampled_node=best_sampled_node,
                    )
                    node.analysis = analyzer_result.get("analysis", "")
                except Exception as e:
                    self.logger.error(f"Analyzer failed: {type(e).__name__}: {e}")
                    self.logger.error(traceback.format_exc())
                    node.analysis = f"Analysis failed: {e}"
            else:
                # 没有开analyzer模式，把temp里的内容也加到results
                if "temp" in engineer_result:
                    node.results["temp"] = engineer_result["temp"]
            
            # 7. 存入 Database（线程安全）
            node_id = self.database.add(node)
            self.logger.info(f"Added node {node_id}: {node.name} (score={node.score:.4f})")
            
            # 记录到日志（传入 database 以计算历史最大 score）
            self.logger.log_node(node, current_step, database=self.database)

            # 更新 best 快照（保存当前 max_score 对应的 code 和 results.json）
            self.best_snapshot.update_if_better(
                node,
                step_name=f"step_{current_step}",
                source_step_dir=step_dir,
            )
            
            return node
            
        except Exception as e:
            # 最外层错误捕获
            self.logger.error(f"Step {current_step} failed with unexpected error:")
            self.logger.error(f"{type(e).__name__}: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    def _run_manager(self, task_description: str):
        self.logger.info("[Manager] Generating prompts...")
        
        eval_file = self.experiment_dir / "eval_criteria.md"
        eval_criteria = ""
        if eval_file.exists():
            eval_criteria = eval_file.read_text(encoding="utf-8")
        
        self.manager.run(
            task_description=task_description,
            eval_criteria=eval_criteria,
            prompt_dir=self.experiment_dir / "prompts",
        )
    
    def run(
        self,
        max_steps: int = 10,
        task_description: Optional[str] = None,
        eval_script: Optional[str] = None,
        sample_n: Optional[int] = None,
    ):
        """
        运行 Pipeline。
        
        支持单worker（num_workers=1）和多worker并行模式。
        每个worker独立采样、进化、更新数据库。
        """
        # 使用配置中的 sample_n 作为默认值
        if sample_n is None:
            sample_n = self.sample_n
        
        # 如果不是 resume，并且存在 initial_program，则先 formalize 一个初始节点
        if not self.is_resume and not self.initial_node_created:
            self._create_initial_node(task_description, eval_script)
        
        if self.num_workers == 1:
            # 单线程模式
            self._run_sequential(max_steps, task_description, eval_script, sample_n)
        else:
            # 多线程并行模式
            self._run_parallel(max_steps, task_description, eval_script, sample_n)

    def _create_initial_node(
        self,
        task_description: Optional[str],
        eval_script: Optional[str],
    ) -> None:
        """
        如果用户在实验目录下提供了 initial_program，
        则在正式进化前用它跑一遍完整实验，形成一个标准的初始节点写入 Database。
        """
        initial_program_file = self.experiment_dir / "initial_program"
        if not initial_program_file.exists():
            return
        
        self.logger.info("Found initial_program, creating initial node before evolution steps")
        
        # 读取任务描述（与 run_step 中逻辑保持一致）
        if task_description is None:
            input_file = self.experiment_dir / "input.md"
            if input_file.exists():
                task_description = input_file.read_text(encoding="utf-8")
            else:
                task_description = ""
        
        initial_code = initial_program_file.read_text(encoding="utf-8")
        
        # 创建一个专门的 step 目录，例如 step_0_initial
        step_dir = self.steps_dir / "step_0_initial"
        step_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置 step_dir 给各个 agent（用于记录 LLM 调用日志）
        if self.researcher:
            self.researcher.set_step_dir(step_dir)
        if self.analyzer:
            self.analyzer.set_step_dir(step_dir)
        if self.engineer:
            self.engineer.set_step_dir(step_dir)
        
        # 构造初始节点（没有父节点）
        node = Node(
            name="initial_program",
            created_at=datetime.now().isoformat(),
            parent=[],
            motivation="Initial program provided by user",
            code=initial_code,
        )
        
        engineer_result: Dict[str, Any] = {}
        
        # 运行实验
        if self.engineer and (eval_script or self.judge_enabled):
            try:
                engineer_result = self.engineer.run(
                    code=node.code,
                    experiment_dir=step_dir,
                    eval_script=eval_script,
                    timeout=self.engineer_timeout,
                    task_description=task_description or "",
                    judge_enabled=self.judge_enabled,
                    judge_ratio=self.judge_ratio,
                )
                
                # engineer_result 包含完整 json（含 temp）
                node.results = {k: v for k, v in engineer_result.items() if k != "temp"}
                
                node.score = engineer_result.get("score", 0.0)
                node.meta_info["runtime"] = engineer_result.get("runtime")
                node.meta_info["success"] = engineer_result.get("success")
                node.meta_info["eval_score"] = engineer_result.get("eval_score", 0.0)
                if self.judge_enabled:
                    node.meta_info["judge_score"] = engineer_result.get("judge_score")
                
                if not engineer_result.get("success"):
                    node.meta_info["error"] = engineer_result.get("error")
            
            except Exception as e:
                self.logger.error(f"Initial Engineer failed: {type(e).__name__}: {e}")
                self.logger.error(traceback.format_exc())
                node.meta_info["success"] = False
                node.meta_info["error"] = str(e)
                node.score = 0.0
                engineer_result = {}
        
        # Analyzer 分析（使用完整 engineer_result，含 temp）
        if self.analyzer:
            try:
                analyzer_result = self.analyzer.run(
                    code=node.code,
                    results=engineer_result,
                    task_description=task_description or "",
                )
                node.analysis = analyzer_result.get("analysis", "")
            except Exception as e:
                self.logger.error(f"Initial Analyzer failed: {type(e).__name__}: {e}")
                self.logger.error(traceback.format_exc())
                node.analysis = f"Analysis failed: {e}"
        
        # 写入 Database
        node_id = self.database.add(node)
        self.logger.info(f"Added initial node {node_id}: {node.name} (score={node.score:.4f})")
        
        # 记录日志（传入 database 以计算历史最大 score）
        self.logger.log_node(node, 0, database=self.database)

        # 初始节点也参与 best 快照更新
        self.best_snapshot.update_if_better(
            node,
            step_name="step_0_initial",
            source_step_dir=step_dir,
        )
        
        self.initial_node_created = True
    
    def _run_sequential(
        self,
        max_steps: int,
        task_description: Optional[str],
        eval_script: Optional[str],
        sample_n: int,
    ):
        """单线程顺序执行"""
        self.logger.info(f"Starting sequential pipeline for {max_steps} steps")
        
        for _ in range(max_steps):
            node = self.run_step(
                task_description=task_description,
                eval_script=eval_script,
                sample_n=sample_n,
            )
            
            if node is None:
                self.logger.warning("Step failed, continuing to next step...")
        
        self.logger.info("Pipeline completed")
        self.logger.finish()
    
    def _run_parallel(
        self,
        max_steps: int,
        task_description: Optional[str],
        eval_script: Optional[str],
        sample_n: int,
    ):
        """多线程并行执行"""
        self.logger.info(f"Starting parallel pipeline with {self.num_workers} workers for {max_steps} steps")
        
        completed_steps = 0
        failed_steps = 0
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # 提交所有任务
            futures = []
            for _ in range(max_steps):
                future = executor.submit(
                    self.run_step,
                    task_description=task_description,
                    eval_script=eval_script,
                    sample_n=sample_n,
                )
                futures.append(future)
            
            # 收集结果
            for future in as_completed(futures):
                try:
                    node = future.result()
                    if node is not None:
                        completed_steps += 1
                    else:
                        failed_steps += 1
                        self.logger.warning("Step failed, worker will continue with next task...")
                except Exception as e:
                    failed_steps += 1
                    self.logger.error(f"Worker encountered unexpected error: {type(e).__name__}: {e}")
                    self.logger.error(traceback.format_exc())
        
        self.logger.info(f"Parallel pipeline completed: {completed_steps} successful, {failed_steps} failed")
        self.logger.finish()
    
    def get_best_node(self) -> Optional[Node]:
        nodes = self.database.get_all()
        if not nodes:
            return None
        return max(nodes, key=lambda n: n.score)
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "total_steps": self.step,
            "total_nodes": len(self.database),
            "total_cognition": len(self.cognition),
            "manager_initialized": self.manager_initialized,
            "llm_stats": self.logger.get_stats(),
        }
