"""
配置加载模块

支持配置优先级:
1. 命令行指定的 config_path
2. 实验目录下的 config.yaml (experiments/<exp_name>/config.yaml)
3. 项目根目录的 config.yaml (默认配置)

实验目录的配置会深度合并到默认配置上，允许只覆盖部分字段。
"""
import os
import yaml
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    深度合并两个字典。
    
    override 中的值会覆盖 base 中的对应值。
    对于嵌套字典，会递归合并而不是直接替换。
    
    Args:
        base: 基础字典
        override: 覆盖字典
        
    Returns:
        合并后的新字典
    """
    result = deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # 递归合并嵌套字典
            result[key] = deep_merge(result[key], value)
        else:
            # 直接覆盖
            result[key] = deepcopy(value)
    
    return result


def load_config(
    config_path: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    加载配置文件，支持实验目录配置优先。
    
    优先级 (从高到低):
    1. config_path 指定的配置文件
    2. 实验目录下的 config.yaml
    3. 项目根目录的 config.yaml (默认配置)
    
    Args:
        config_path: 配置文件路径 (最高优先级)
        experiment_name: 实验名称，用于定位实验目录配置
        
    Returns:
        合并后的配置字典
    """
    project_root = Path(__file__).parent.parent
    default_config_path = project_root / "config.yaml"
    
    # 1. 加载默认配置
    if default_config_path.exists():
        with open(default_config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}
    
    # 2. 如果指定了 experiment_name，尝试加载实验目录配置
    if experiment_name:
        exp_config_path = project_root / "experiments" / experiment_name / "config.yaml"
        if exp_config_path.exists():
            with open(exp_config_path, "r", encoding="utf-8") as f:
                exp_config = yaml.safe_load(f) or {}
            config = deep_merge(config, exp_config)
    
    # 3. 如果指定了 config_path，加载并合并 (最高优先级)
    if config_path:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            custom_config = yaml.safe_load(f) or {}
        config = deep_merge(config, custom_config)
    
    # 处理环境变量
    config = _resolve_env_vars(config)
    
    return config


def load_experiment_config(experiment_name: str) -> Dict[str, Any]:
    """
    便捷函数：加载指定实验的配置。
    
    等价于 load_config(experiment_name=experiment_name)
    
    Args:
        experiment_name: 实验名称
        
    Returns:
        合并后的配置字典
    """
    return load_config(experiment_name=experiment_name)


def _resolve_env_vars(obj: Any) -> Any:
    """递归解析配置中的环境变量 (${VAR_NAME} 格式)"""
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_env_vars(item) for item in obj]
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        var_name = obj[2:-1]
        return os.environ.get(var_name, "")
    return obj
