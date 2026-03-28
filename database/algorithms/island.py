"""
岛屿算法采样器

实现多岛屿进化模型：
- 维护多个独立进化的岛屿（子种群）
- 支持岛屿间的周期性迁移
- 每个岛屿可以使用不同的采样策略（探索、开发、加权）
- 支持 MAP-Elites 特征映射（每个岛屿独立维护特征网格）
- 支持 diversity 计算（与 openevolve 对齐）

参考 OpenEvolve 的 MAP-Elites + Island 混合架构。
"""
import random
import time
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from .base import BaseSampler

if TYPE_CHECKING:
    from ...utils.structures import Node


class IslandSampler(BaseSampler):
    """
    岛屿算法采样器。
    
    实现多岛屿进化模型：
    - 维护多个独立进化的岛屿（子种群）
    - 支持岛屿间的周期性迁移
    - 每个岛屿可以使用不同的采样策略（探索、开发、加权）
    - 支持 MAP-Elites 特征映射（每个岛屿独立维护特征网格）
    
    参考 OpenEvolve 的 MAP-Elites + Island 混合架构。
    """
    
    def __init__(
        self,
        num_islands: int = 5,
        migration_interval: int = 10,
        migration_rate: float = 0.1,
        exploration_ratio: float = 0.2,
        exploitation_ratio: float = 0.3,
        c: float = 1.414,
        feature_dimensions: Optional[List[str]] = None,
        feature_bins: int = 10,
    ):
        """
        Args:
            num_islands: 岛屿数量
            migration_interval: 迁移间隔（每隔多少代进行一次迁移）
            migration_rate: 迁移比例（每次迁移的程序比例）
            exploration_ratio: 探索采样的概率
            exploitation_ratio: 开发采样的概率（剩余为加权采样）
            c: UCB 探索参数（目前未使用，保留用于未来可能的 UCB 加权采样变体）
            feature_dimensions: MAP-Elites 特征维度列表（如 ['complexity', 'diversity']）
            feature_bins: 每个特征维度的网格数量
        """
        self.num_islands = num_islands
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        self.exploration_ratio = exploration_ratio
        self.exploitation_ratio = exploitation_ratio
        self.c = c  # 目前未使用，加权采样仅基于分数
        
        # 岛屿数据结构
        self.islands: List[Set[int]] = [set() for _ in range(num_islands)]
        self.island_generations: List[int] = [0] * num_islands
        self.island_best_nodes: List[Optional[int]] = [None] * num_islands
        
        # MAP-Elites 特征映射（每个岛屿独立）
        self.feature_dimensions = feature_dimensions or []
        self.feature_bins = feature_bins
        self.island_feature_maps: List[Dict[Tuple, int]] = [{} for _ in range(num_islands)]
        
        # 特征统计（用于归一化）
        self.feature_stats: Dict[str, Dict[str, Any]] = {}
        
        # 迁移追踪
        self.last_migration_generation = 0
        
        # 当前采样的岛屿（轮询）
        self.current_island = 0
        
        # 精英档案
        self.archive: Set[int] = set()
        self.archive_size = 100
        
        # Diversity 计算基础设施（与 openevolve 对齐）
        self.diversity_cache: Dict[int, Dict[str, Any]] = {}  # hash -> {"value": float, "timestamp": float}
        self.diversity_cache_size: int = 1000  # LRU cache size
        self.diversity_reference_set: List[str] = []  # Reference program codes for consistent diversity
        self.diversity_reference_size: int = 20  # Size of reference set
        
        # 维护所有节点的字典（用于 diversity 计算）
        self.all_nodes: Dict[int, "Node"] = {}  # node_id -> Node
    
    def sample(self, nodes: List["Node"], n: int) -> List["Node"]:
        """
        从当前岛屿采样节点。
        
        采样策略按概率分为三种：
        1. 探索（exploration_ratio）：随机采样，增加多样性
        2. 开发（exploitation_ratio）：从精英档案采样
        3. 加权（剩余概率）：基于适应度的加权采样
        """
        if not nodes:
            return []
        
        # 同步更新节点字典（用于 diversity 计算）
        for node in nodes:
            if node.id is not None:
                self.all_nodes[node.id] = node
        
        n = min(n, len(nodes))
        
        # 检查是否需要迁移
        if self._should_migrate():
            self._migrate(nodes)
        
        # 确保 current_island 在有效范围内（防止状态恢复时的不匹配）
        if self.current_island >= self.num_islands:
            self.current_island = 0
        
        # 确保 island_generations 长度匹配 num_islands（防止配置变更）
        if len(self.island_generations) != self.num_islands:
            old_len = len(self.island_generations)
            if old_len < self.num_islands:
                # 扩展列表
                self.island_generations.extend([0] * (self.num_islands - old_len))
                self.island_best_nodes.extend([None] * (self.num_islands - old_len))
                self.islands.extend([set() for _ in range(self.num_islands - old_len)])
            else:
                # 截断列表
                self.island_generations = self.island_generations[:self.num_islands]
                self.island_best_nodes = self.island_best_nodes[:self.num_islands]
                self.islands = self.islands[:self.num_islands]
        
        # 获取当前岛屿的节点
        island_nodes = self._get_island_nodes(self.current_island, nodes)
        
        # 如果当前岛屿为空，从所有节点采样
        if not island_nodes:
            island_nodes = nodes
        
        # 增加当前岛屿的代数
        self.island_generations[self.current_island] += 1
        
        # 轮询到下一个岛屿
        self.current_island = (self.current_island + 1) % self.num_islands
        
        # 根据概率选择采样策略
        selected = []
        for _ in range(n):
            rand_val = random.random()
            
            if rand_val < self.exploration_ratio:
                # 探索：随机采样
                node = self._sample_random(island_nodes)
            elif rand_val < self.exploration_ratio + self.exploitation_ratio:
                # 开发：从精英档案采样
                node = self._sample_from_archive(nodes)
                if node is None:
                    node = self._sample_weighted(island_nodes)
            else:
                # 加权：基于适应度采样
                node = self._sample_weighted(island_nodes)
            
            if node and node not in selected:
                selected.append(node)
                node.visit_count += 1
        
        return selected
    
    def sample_from_island(
        self, 
        island_id: int, 
        nodes: List["Node"], 
        n: int
    ) -> List["Node"]:
        """
        从指定岛屿采样。
        
        Args:
            island_id: 岛屿 ID
            nodes: 所有节点
            n: 采样数量
            
        Returns:
            采样的节点列表
        """
        island_id = island_id % self.num_islands
        island_nodes = self._get_island_nodes(island_id, nodes)
        
        if not island_nodes:
            island_nodes = nodes
        
        n = min(n, len(island_nodes))
        selected = []
        
        for _ in range(n):
            rand_val = random.random()
            
            if rand_val < self.exploration_ratio:
                node = self._sample_random(island_nodes)
            elif rand_val < self.exploration_ratio + self.exploitation_ratio:
                node = self._sample_from_archive(nodes)
                if node is None:
                    node = self._sample_weighted(island_nodes)
            else:
                node = self._sample_weighted(island_nodes)
            
            if node and node not in selected:
                selected.append(node)
                node.visit_count += 1
        
        return selected
    
    def on_node_added(self, node: "Node") -> None:
        """当新节点被添加时，分配到相应的岛屿，并更新 MAP-Elites 特征网格"""
        if node.id is None:
            return
        
        # 添加到节点字典（用于 diversity 计算）
        self.all_nodes[node.id] = node
        
        # 确定目标岛屿
        island_id = node.meta_info.get("island")
        
        if island_id is None:
            # 如果有父节点，继承父节点的岛屿
            if node.parent and len(node.parent) > 0:
                # 尝试从父节点获取岛屿信息
                # 这里我们假设第一个父节点的岛屿信息最重要
                # 实际实现中，需要通过 database 来获取父节点信息
                island_id = self.current_island
            else:
                island_id = self.current_island
        
        island_id = island_id % self.num_islands
        node.meta_info["island"] = island_id
        
        # MAP-Elites: 计算特征坐标并更新特征网格
        if self.feature_dimensions:
            feature_coords = self._calculate_feature_coords(node)
            if feature_coords is not None:
                feature_key = tuple(feature_coords)
                feature_map = self.island_feature_maps[island_id]
                
                # 检查该特征格子是否已有节点
                should_replace = feature_key not in feature_map
                
                if not should_replace:
                    # 格子已有节点，比较分数决定是否替换
                    existing_node_id = feature_map[feature_key]
                    if existing_node_id in self.all_nodes:
                        existing_node = self.all_nodes[existing_node_id]
                        if node.score > existing_node.score:
                            should_replace = True
                
                if should_replace:
                    feature_map[feature_key] = node.id
        
        # 添加到岛屿
        self.islands[island_id].add(node.id)
        
        # 更新岛屿最优节点
        if self.island_best_nodes[island_id] is None:
            self.island_best_nodes[island_id] = node.id
        
        # 更新精英档案
        self._update_archive(node)
        
        # 如果添加了新节点且使用 diversity 特征，标记需要更新 reference set
        # （不立即更新，延迟到实际需要计算时）
        if "diversity" in self.feature_dimensions:
            # 如果 reference set 为空或太小，标记需要更新
            if len(self.diversity_reference_set) < self.diversity_reference_size:
                self.diversity_reference_set = []  # 标记需要重新计算
    
    def on_node_removed(self, node: "Node") -> None:
        """当节点被移除时，从岛屿和特征网格中删除"""
        if node.id is None:
            return
        
        # 从节点字典中移除
        self.all_nodes.pop(node.id, None)
        
        island_id = node.meta_info.get("island")
        if island_id is not None and island_id < len(self.islands):
            self.islands[island_id].discard(node.id)
            
            # 从特征网格中移除
            if self.feature_dimensions:
                feature_coords = self._calculate_feature_coords(node)
                if feature_coords is not None:
                    feature_key = tuple(feature_coords)
                    feature_map = self.island_feature_maps[island_id]
                    if feature_key in feature_map and feature_map[feature_key] == node.id:
                        del feature_map[feature_key]
        
        self.archive.discard(node.id)
        
        # 如果移除了节点且使用 diversity 特征，标记需要更新 reference set
        if "diversity" in self.feature_dimensions:
            # 如果被移除的节点在 reference set 中，需要重新计算
            if node.code and node.code in self.diversity_reference_set:
                self.diversity_reference_set = []  # 标记需要重新计算
    
    def _get_island_nodes(self, island_id: int, all_nodes: List["Node"]) -> List["Node"]:
        """获取指定岛屿的所有节点"""
        island_node_ids = self.islands[island_id]
        return [n for n in all_nodes if n.id in island_node_ids]
    
    def _sample_random(self, nodes: List["Node"]) -> Optional["Node"]:
        """随机采样"""
        if not nodes:
            return None
        return random.choice(nodes)
    
    def _sample_weighted(self, nodes: List["Node"]) -> Optional["Node"]:
        """基于适应度的加权采样"""
        if not nodes:
            return None
        
        # 计算权重（基于分数）
        weights = []
        for node in nodes:
            # 添加小的 epsilon 避免零权重
            weights.append(max(node.score, 0.001))
        
        # 归一化
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(nodes)] * len(nodes)
        
        # 加权采样
        return random.choices(nodes, weights=weights, k=1)[0]
    
    def _sample_from_archive(self, all_nodes: List["Node"]) -> Optional["Node"]:
        """从精英档案采样"""
        if not self.archive:
            return None
        
        archive_nodes = [n for n in all_nodes if n.id in self.archive]
        if not archive_nodes:
            return None
        
        return random.choice(archive_nodes)
    
    def _update_archive(self, node: "Node") -> None:
        """更新精英档案"""
        if node.id is None:
            return
        
        self.archive.add(node.id)
        
        # 如果档案超出限制，移除分数最低的
        if len(self.archive) > self.archive_size:
            # 需要访问所有节点来排序，这里简单处理
            # 实际使用中可能需要维护一个排序的数据结构
            pass
    
    def _calculate_feature_coords(self, node: "Node") -> Optional[List[int]]:
        """
        计算节点的特征坐标（用于 MAP-Elites）。
        
        从 node.results 中提取特征值，或计算内置特征（complexity, diversity），
        进行归一化并映射到网格坐标。
        
        Args:
            node: 节点对象
            
        Returns:
            特征坐标列表，如果无法计算则返回 None
        """
        if not self.feature_dimensions:
            return None
        
        coords = []
        for dim in self.feature_dimensions:
            # 处理内置特征
            if dim == "complexity":
                # 使用代码长度作为复杂度（与 openevolve 对齐）
                feature_value = len(node.code) if node.code else 0
            elif dim == "diversity":
                # 使用缓存的 diversity 计算（与 openevolve 对齐）
                if len(self.all_nodes) < 2:
                    feature_value = 0.0
                else:
                    feature_value = self._get_cached_diversity(node)
            else:
                # 从 node.results 中获取特征值
                if not node.results or dim not in node.results:
                    # 特征不存在，无法计算坐标
                    return None
                
                feature_value = node.results[dim]
            
            # 确保是数值类型
            if not isinstance(feature_value, (int, float)):
                return None
            
            # 更新特征统计
            self._update_feature_stats(dim, feature_value)
            
            # 归一化特征值到 [0, 1]
            scaled_value = self._scale_feature_value(dim, feature_value)
            
            # 映射到网格坐标
            bin_idx = int(scaled_value * self.feature_bins)
            bin_idx = max(0, min(self.feature_bins - 1, bin_idx))
            coords.append(bin_idx)
        
        return coords
    
    def _update_feature_stats(self, feature: str, value: float) -> None:
        """更新特征统计信息（用于归一化）"""
        if feature not in self.feature_stats:
            self.feature_stats[feature] = {
                "min": value,
                "max": value,
                "values": []
            }
        
        stats = self.feature_stats[feature]
        stats["min"] = min(stats["min"], value)
        stats["max"] = max(stats["max"], value)
        stats["values"].append(value)
        
        # 限制存储的值数量，避免内存过大
        if len(stats["values"]) > 1000:
            stats["values"] = stats["values"][-1000:]
    
    def _scale_feature_value(self, feature: str, value: float) -> float:
        """
        归一化特征值到 [0, 1]。
        
        使用 min-max 归一化。
        
        Args:
            feature: 特征名称
            value: 特征值
            
        Returns:
            归一化后的值 [0, 1]
        """
        if feature not in self.feature_stats:
            return 0.5
        
        stats = self.feature_stats[feature]
        min_val = stats["min"]
        max_val = stats["max"]
        
        # 避免除零
        if max_val - min_val < 1e-10:
            return 0.5
        
        # Min-max 归一化
        scaled = (value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, scaled))
    
    def _fast_code_diversity(self, code1: str, code2: str) -> float:
        """
        快速代码多样性近似计算（与 openevolve 对齐）。
        
        使用简单指标来近似代码多样性。
        
        Args:
            code1: 第一个代码
            code2: 第二个代码
            
        Returns:
            多样性分数（越高越不同）
        """
        if code1 == code2:
            return 0.0
        
        # 长度差异（缩放到合理范围）
        len1, len2 = len(code1), len(code2)
        length_diff = abs(len1 - len2)
        
        # 行数差异
        lines1 = code1.count("\n")
        lines2 = code2.count("\n")
        line_diff = abs(lines1 - lines2)
        
        # 简单字符集差异
        chars1 = set(code1)
        chars2 = set(code2)
        char_diff = len(chars1.symmetric_difference(chars2))
        
        # 组合指标（缩放到匹配原始编辑距离范围）
        diversity = length_diff * 0.1 + line_diff * 10 + char_diff * 0.5
        
        return diversity
    
    def _get_cached_diversity(self, node: "Node") -> float:
        """
        获取节点的 diversity 分数（使用缓存和参考集合）。
        
        与 openevolve 对齐的实现。
        
        Args:
            node: 要计算 diversity 的节点
            
        Returns:
            Diversity 分数（缓存或新计算的）
        """
        if not node.code:
            return 0.0
        
        code_hash = hash(node.code)
        
        # 先检查缓存
        if code_hash in self.diversity_cache:
            return self.diversity_cache[code_hash]["value"]
        
        # 更新参考集合（如果需要）
        if (
            not self.diversity_reference_set
            or len(self.diversity_reference_set) < self.diversity_reference_size
        ):
            self._update_diversity_reference_set()
        
        # 与参考集合计算 diversity
        diversity_scores = []
        for ref_code in self.diversity_reference_set:
            if ref_code != node.code:  # 不与自身比较
                diversity_scores.append(self._fast_code_diversity(node.code, ref_code))
        
        diversity = (
            sum(diversity_scores) / max(1, len(diversity_scores)) if diversity_scores else 0.0
        )
        
        # 缓存结果（LRU 淘汰）
        self._cache_diversity_value(code_hash, diversity)
        
        return diversity
    
    def _update_diversity_reference_set(self) -> None:
        """更新 diversity 计算的参考集合（与 openevolve 对齐）"""
        if len(self.all_nodes) == 0:
            return
        
        # 选择多样化的程序作为参考集合
        all_programs = list(self.all_nodes.values())
        
        if len(all_programs) <= self.diversity_reference_size:
            self.diversity_reference_set = [p.code for p in all_programs if p.code]
        else:
            # 选择最大多样性的程序
            selected = []
            remaining = all_programs.copy()
            
            # 从随机程序开始
            if remaining:
                first_idx = random.randint(0, len(remaining) - 1)
                selected.append(remaining.pop(first_idx))
            
            # 贪心地添加最大化多样性的程序到选中集合
            while len(selected) < self.diversity_reference_size and remaining:
                max_diversity = -1
                best_idx = -1
                
                for i, candidate in enumerate(remaining):
                    if not candidate.code:
                        continue
                    # 计算与已选中程序的最小 diversity
                    min_div = float("inf")
                    for selected_prog in selected:
                        if not selected_prog.code:
                            continue
                        div = self._fast_code_diversity(candidate.code, selected_prog.code)
                        min_div = min(min_div, div)
                    
                    if min_div > max_diversity:
                        max_diversity = min_div
                        best_idx = i
                
                if best_idx >= 0:
                    selected.append(remaining.pop(best_idx))
            
            self.diversity_reference_set = [p.code for p in selected if p.code]
    
    def _cache_diversity_value(self, code_hash: int, diversity: float) -> None:
        """缓存 diversity 值（LRU 淘汰）"""
        # 检查缓存是否已满
        if len(self.diversity_cache) >= self.diversity_cache_size:
            # 移除最旧的条目
            oldest_hash = min(self.diversity_cache.items(), key=lambda x: x[1]["timestamp"])[0]
            del self.diversity_cache[oldest_hash]
        
        # 添加新条目
        self.diversity_cache[code_hash] = {"value": diversity, "timestamp": time.time()}
    
    def _invalidate_diversity_cache(self) -> None:
        """当程序发生显著变化时，使 diversity 缓存失效"""
        self.diversity_cache.clear()
        self.diversity_reference_set = []
    
    def _should_migrate(self) -> bool:
        """检查是否应该进行迁移"""
        max_generation = max(self.island_generations)
        return (max_generation - self.last_migration_generation) >= self.migration_interval
    
    def _migrate(self, all_nodes: List["Node"]) -> None:
        """
        执行岛屿间迁移。
        
        采用环形拓扑，每个岛屿向相邻的岛屿迁移最优个体。
        """
        if self.num_islands < 2:
            return
        
        self.last_migration_generation = max(self.island_generations)
        
        for island_id in range(self.num_islands):
            island_nodes = self._get_island_nodes(island_id, all_nodes)
            if not island_nodes:
                continue
            
            # 按分数排序，选择最优的进行迁移
            island_nodes.sort(key=lambda n: n.score, reverse=True)
            num_to_migrate = max(1, int(len(island_nodes) * self.migration_rate))
            migrants = island_nodes[:num_to_migrate]
            
            # 环形拓扑：迁移到相邻岛屿
            target_islands = [
                (island_id + 1) % self.num_islands,
                (island_id - 1) % self.num_islands,
            ]
            
            for migrant in migrants:
                # 防止已迁移的节点再次迁移
                if migrant.meta_info.get("migrant", False):
                    continue
                
                for target_island in target_islands:
                    # 检查目标岛屿是否已有相同代码
                    target_nodes = self._get_island_nodes(target_island, all_nodes)
                    has_duplicate = any(n.code == migrant.code for n in target_nodes)
                    
                    if has_duplicate:
                        continue
                    
                    # 将节点添加到目标岛屿（作为引用，不创建副本）
                    # 在实际使用中，可能需要创建副本
                    self.islands[target_island].add(migrant.id)
                    migrant.meta_info["migrant"] = True
    
    def get_island_stats(self, all_nodes: List["Node"]) -> Dict[str, Any]:
        """获取岛屿统计信息"""
        stats = {
            "num_islands": self.num_islands,
            "island_populations": [],
            "island_generations": self.island_generations.copy(),
            "archive_size": len(self.archive),
            "current_island": self.current_island,
            "last_migration_generation": self.last_migration_generation,
            "feature_dimensions": self.feature_dimensions,
            "feature_stats": self.feature_stats.copy(),
        }
        
        for island_id in range(self.num_islands):
            island_nodes = self._get_island_nodes(island_id, all_nodes)
            feature_map_size = len(self.island_feature_maps[island_id]) if self.feature_dimensions else 0
            
            stats["island_populations"].append({
                "island_id": island_id,
                "size": len(island_nodes),
                "best_score": max((n.score for n in island_nodes), default=0.0),
                "avg_score": sum(n.score for n in island_nodes) / len(island_nodes) if island_nodes else 0.0,
                "feature_map_coverage": feature_map_size,
            })
        
        return stats
    
    def reset(self) -> None:
        """重置岛屿算法状态"""
        self.islands = [set() for _ in range(self.num_islands)]
        self.island_generations = [0] * self.num_islands
        self.island_best_nodes = [None] * self.num_islands
        self.island_feature_maps = [{} for _ in range(self.num_islands)]
        self.feature_stats.clear()
        self.last_migration_generation = 0
        self.current_island = 0
        self.archive.clear()
    
    def get_state(self) -> Dict[str, Any]:
        """获取采样器状态（用于持久化）"""
        return {
            "island_generations": self.island_generations,
            "last_migration_generation": self.last_migration_generation,
            "current_island": self.current_island,
            "archive": list(self.archive),
            "island_best_nodes": self.island_best_nodes,
            "island_feature_maps": [
                {str(k): v for k, v in feature_map.items()}
                for feature_map in self.island_feature_maps
            ],
            "feature_stats": self.feature_stats,
            "diversity_cache": {str(k): v for k, v in self.diversity_cache.items()},
            "diversity_reference_set": self.diversity_reference_set,
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """加载采样器状态（从持久化恢复）"""
        loaded_generations = state.get("island_generations", [])
        loaded_best_nodes = state.get("island_best_nodes", [])
        
        # 确保状态长度与当前 num_islands 匹配
        if len(loaded_generations) != self.num_islands:
            if len(loaded_generations) < self.num_islands:
                # 扩展：添加新的岛屿
                loaded_generations.extend([0] * (self.num_islands - len(loaded_generations)))
                if len(loaded_best_nodes) < self.num_islands:
                    loaded_best_nodes.extend([None] * (self.num_islands - len(loaded_best_nodes)))
            else:
                # 截断：移除多余的岛屿
                loaded_generations = loaded_generations[:self.num_islands]
                loaded_best_nodes = loaded_best_nodes[:self.num_islands]
        
        self.island_generations = loaded_generations
        self.last_migration_generation = state.get("last_migration_generation", 0)
        self.current_island = state.get("current_island", 0) % self.num_islands  # 确保在有效范围内
        self.archive = set(state.get("archive", []))
        self.island_best_nodes = loaded_best_nodes if len(loaded_best_nodes) == self.num_islands else [None] * self.num_islands
        
        # 加载特征网格
        loaded_feature_maps = state.get("island_feature_maps", [])
        if loaded_feature_maps and len(loaded_feature_maps) == self.num_islands:
            self.island_feature_maps = [
                {eval(k): v for k, v in feature_map.items()}
                for feature_map in loaded_feature_maps
            ]
        else:
            self.island_feature_maps = [{} for _ in range(self.num_islands)]
        
        # 加载特征统计
        self.feature_stats = state.get("feature_stats", {})
        
        # 加载 diversity 相关状态
        loaded_diversity_cache = state.get("diversity_cache", {})
        self.diversity_cache = {int(k): v for k, v in loaded_diversity_cache.items()}
        self.diversity_reference_set = state.get("diversity_reference_set", [])
    
    def rebuild_from_nodes(self, nodes: List["Node"]) -> None:
        """从节点列表重建岛屿结构"""
        # 清空现有岛屿
        self.islands = [set() for _ in range(self.num_islands)]
        self.island_feature_maps = [{} for _ in range(self.num_islands)]
        
        # 重建节点字典
        self.all_nodes = {node.id: node for node in nodes if node.id is not None}
        
        # 使 diversity 缓存失效（因为节点集合可能已改变）
        if "diversity" in self.feature_dimensions:
            self._invalidate_diversity_cache()
        
        # 根据节点的 meta_info 重建岛屿结构
        for node in nodes:
            if node.id is None:
                continue
            
            island_id = node.meta_info.get("island")
            if island_id is not None:
                island_id = island_id % self.num_islands
                self.islands[island_id].add(node.id)
                
                # 重建特征网格
                if self.feature_dimensions:
                    feature_coords = self._calculate_feature_coords(node)
                    if feature_coords is not None:
                        feature_key = tuple(feature_coords)
                        self.island_feature_maps[island_id][feature_key] = node.id
                
                # 更新精英档案
                if node.id in self.archive or node.score > 0:
                    self._update_archive(node)
