#!/usr/bin/env python3
"""
Evolve Framework
================
自动化实验进化框架的主入口。

Usage:
    # 使用默认配置运行
    python main.py
    
    # 指定实验名和步数
    python main.py --experiment my_exp --steps 20
    
    # 指定配置文件
    python main.py --config path/to/config.yaml
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录的父目录到 path，使 Evolve 可以作为包导入
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from Evolve.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Evolve Framework - Automated Experiment Evolution"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: config.yaml)",
    )
    
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name (overrides config)",
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of evolution steps (default: 10)",
    )
    
    parser.add_argument(
        "--sample-n",
        type=int,
        default=3,
        help="Number of nodes to sample per step (default: 3)",
    )
    
    parser.add_argument(
        "--eval-script",
        type=str,
        default=None,
        help="Path to evaluation script",
    )
    
    args = parser.parse_args()
    
    # 初始化 Pipeline
    pipeline = Pipeline(
        config_path=args.config,
        experiment_name=args.experiment,
    )
    
    # 运行
    pipeline.run(
        max_steps=args.steps,
        eval_script=args.eval_script,
        sample_n=args.sample_n,
    )
    
    # 输出统计
    stats = pipeline.get_stats()
    print("\n=== Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 输出最佳节点
    best = pipeline.get_best_node()
    if best:
        print(f"\n=== Best Node ===")
        print(f"Name: {best.name}")
        print(f"Score: {best.score:.4f}")
        print(f"Motivation: {best.motivation[:200]}...")


if __name__ == "__main__":
    main()
