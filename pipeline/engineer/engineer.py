"""
Engineer Agent
==============
负责运行实验并收集结果。

职责：
1. 将代码写入文件
2. 运行用户提供的实验脚本
3. 收集评估结果（包含 eval 脚本返回的 score）
4. 可选：运行 LLM Judge 进行额外打分
"""
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ..base import BaseAgent


class Engineer(BaseAgent):
    """
    工程师 Agent。
    
    负责:
    1. 将代码写入文件
    2. 运行用户提供的实验脚本
    3. 收集评估结果（包含 score）
    4. 可选：LLM Judge 打分
    """
    
    def __init__(self, llm, prompt_manager):
        super().__init__(llm, prompt_manager, name="engineer")
    
    def run(
        self,
        code: str,
        experiment_dir: Path,
        eval_script: Optional[str] = None,
        timeout: int = 3600,
        task_description: str = "",
        judge_enabled: bool = False,
        judge_ratio: float = 0.2,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        运行实验并收集结果。
        
        Args:
            code: 实验代码
            experiment_dir: 实验目录
            eval_script: 评估脚本路径
            timeout: 超时时间（秒）
            task_description: 任务描述（用于 judge）
            judge_enabled: 是否启用 LLM Judge
            judge_ratio: Judge 分数权重（最终分数 = (1-ratio)*eval_score + ratio*judge_score）
            
        Returns:
            包含用户json的所有字段 + score/eval_score/judge_score/runtime/success/error
            注意：包含temp字段（如果用户json中有）
        """
        self.logger.info("[Engineer] Starting experiment")
        
        experiment_dir = Path(experiment_dir)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 写入代码文件（不带后缀，便于各种语言的实验通用）
        code_file = experiment_dir / "code"
        code_file.write_text(code, encoding="utf-8")
        self.logger.info(f"[Engineer] Code written to {code_file}")
        
        results = {}
        error = None
        success = True
        start_time = time.time()
        
        # 运行评估脚本
        if eval_script:
            eval_result = self._run_script(eval_script, experiment_dir, timeout)
            
            # 脚本执行成功才读取结果，超时就不要结果
            if eval_result["success"]:
                results = self._parse_results(experiment_dir)
                
                # 将脚本执行日志放入temp字段
                if "temp" not in results:
                    results["temp"] = {}
                results["temp"]["stdout"] = eval_result.get("stdout", "")
                results["temp"]["stderr"] = eval_result.get("stderr", "")
                
                # 检查results的success字段
                if not results.get("success", False):
                    success = False
                    error = results.get("temp", {}).get("error") or results.get("error", "Eval returned success=False")
                    self.logger.error(f"[Engineer] Eval failed: {error}")
                
                # 脚本执行成功且读取到结果，才检查eval_score
                assert "eval_score" in results, "eval results must contain 'eval_score' field"
            else:
                # 脚本执行失败（包括超时）
                success = False
                error = eval_result.get("error", "Eval script failed")
                self.logger.error(f"[Engineer] Eval failed: {error}")
                # 保存失败的日志
                results = {
                    "temp": {
                        "stdout": eval_result.get("stdout", ""),
                        "stderr": eval_result.get("stderr", ""),
                        "error": error
                    }
                }
        
        runtime = time.time() - start_time
        
        # 从用户json中获取eval_score（如果脚本失败/超时，默认为0）
        eval_score = results.get("eval_score", 0.0)
        self.logger.info(f"[Engineer] Eval score: {eval_score:.4f}")
        
        # 可选：运行 LLM Judge
        judge_score = None
        if judge_enabled and success:
            judge_score = self._run_judge(
                code=code,
                results=results,
                task_description=task_description,
            )
            self.logger.info(f"[Engineer] Judge score: {judge_score:.4f}")
        
        # 计算最终 score
        if judge_enabled and judge_score is not None:
            final_score = (1 - judge_ratio) * eval_score + judge_ratio * judge_score
        else:
            final_score = eval_score
        
        self.logger.info(f"[Engineer] Completed in {runtime:.2f}s, success={success}, final_score={final_score:.4f}")
        
        # 构建返回结果：用户json的所有字段 + 覆盖score + 添加judge_score/runtime/success/error
        result = {
            **results,  # 包含用户json的所有字段（包括temp、eval_score）
            "score": final_score,  # 覆盖用户的score字段（如果有）
            "runtime": runtime,
            "success": success,
        }
        
        if judge_enabled:
            result["judge_score"] = judge_score
        
        return result
    
    def _run_script(
        self,
        script_path: str,
        cwd: Path,
        timeout: int,
    ) -> Dict[str, Any]:
        """运行脚本并返回结果,确保超时后进程被彻底清理"""
        process = None
        try:
            # 使用 Popen 以便更好地控制进程
            process = subprocess.Popen(
                ["bash", script_path],
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                # 创建新的进程组,以便可以一次性杀死整个进程树
                start_new_session=True,
            )
            
            # 等待进程完成或超时
            stdout, stderr = process.communicate(timeout=timeout)
            
            return {
                "success": process.returncode == 0,
                "stdout": stdout,
                "stderr": stderr,
                "error": stderr if process.returncode != 0 else None,
            }
            
        except subprocess.TimeoutExpired:
            # 超时处理:彻底清理进程树
            self.logger.warning(f"[Engineer] Script timeout after {timeout}s, terminating process tree...")
            
            stdout = ""
            stderr = ""
            
            if process:
                try:
                    # 尝试使用 psutil 清理整个进程树
                    try:
                        import psutil
                        parent = psutil.Process(process.pid)
                        children = parent.children(recursive=True)
                    except ImportError:
                        # 如果没有 psutil,只能清理主进程
                        children = []
                        self.logger.warning("[Engineer] psutil not available, may not kill all subprocesses")
                    
                    # 先尝试 SIGTERM(优雅终止)
                    process.terminate()
                    try:
                        # 等待最多 5 秒让进程自己退出
                        stdout, stderr = process.communicate(timeout=5)
                        self.logger.info(f"[Engineer] Process terminated gracefully")
                    except subprocess.TimeoutExpired:
                        # 如果进程不响应,强制 SIGKILL
                        self.logger.warning(f"[Engineer] Process not responding, force killing...")
                        process.kill()
                        
                        # 同时杀死所有子进程
                        for child in children:
                            try:
                                child.kill()
                            except Exception:
                                pass
                        
                        # 等待进程真正退出
                        try:
                            stdout, stderr = process.communicate(timeout=2)
                        except:
                            pass
                    
                    # 确保所有子进程都已退出
                    for child in children:
                        try:
                            child.wait(timeout=1)
                        except Exception:
                            pass
                    
                except Exception as e:
                    self.logger.error(f"[Engineer] Error during process cleanup: {e}")
            
            # 等待一小段时间,确保文件系统操作完成
            # 防止进程被杀死前最后一刻写入的数据还在缓冲区
            time.sleep(0.5)
            
            return {
                "success": False,
                "timeout": True,
                "stdout": stdout if isinstance(stdout, str) else (stdout.decode() if stdout else ""),
                "stderr": stderr if isinstance(stderr, str) else (stderr.decode() if stderr else ""),
                "error": f"Timeout after {timeout}s"
            }
            
        except Exception as e:
            self.logger.error(f"[Engineer] Script execution error: {e}")
            # 确保进程被清理
            if process and process.poll() is None:
                try:
                    process.kill()
                    process.wait(timeout=2)
                except:
                    pass
            return {"success": False, "error": str(e)}
    
    def _parse_results(self, experiment_dir: Path) -> Dict[str, Any]:
        """解析实验结果文件,处理可能的文件损坏"""
        results_file = experiment_dir / "results.json"
        if results_file.exists():
            try:
                with open(results_file, "r") as f:
                    results = json.load(f)
                # 验证结果的基本结构
                if not isinstance(results, dict):
                    self.logger.warning("[Engineer] results.json is not a dict, ignoring")
                    return {}
                return results
            except json.JSONDecodeError as e:
                # 文件可能在写入时被中断,导致JSON不完整
                self.logger.warning(f"[Engineer] results.json is corrupted: {e}, ignoring")
                return {}
            except Exception as e:
                self.logger.error(f"[Engineer] Failed to read results.json: {e}")
                return {}
        
        results_txt = experiment_dir / "results.txt"
        if results_txt.exists():
            try:
                return {"raw": results_txt.read_text()}
            except Exception as e:
                self.logger.error(f"[Engineer] Failed to read results.txt: {e}")
                return {}
        
        return {}
    
    def _run_judge(
        self,
        code: str,
        results: Dict[str, Any],
        task_description: str,
    ) -> float:
        """
        运行 LLM Judge 进行打分。
        
        Args:
            code: 实验代码
            results: 实验结果
            task_description: 任务描述
            
        Returns:
            Judge 打分（0-100）
        """
        try:
            # 尝试获取 judge prompt
            prompt = self.get_prompt(
                "judge",
                code=code,
                results=str(results),
                task_description=task_description,
            )
            
            result = self.llm.extract_tags(prompt, call_name="engineer_judge")
            
            judge_score = result.get("score", 0.0)
            try:
                judge_score = float(judge_score)
                # 确保分数在合理范围内
                judge_score = max(0.0, min(100.0, judge_score))
            except (ValueError, TypeError):
                self.logger.warning("[Engineer] Failed to parse judge score, using 0.0")
                judge_score = 0.0
            
            # 记录 judge 的理由（如果有的话）
            judge_reason = result.get("reason", result.get("reasoning", ""))
            if judge_reason:
                self.logger.info(f"[Engineer] Judge reasoning: {judge_reason[:200]}...")
            
            return judge_score
            
        except Exception as e:
            self.logger.warning(f"[Engineer] Judge failed: {e}, using eval_score only")
            return 0.0
