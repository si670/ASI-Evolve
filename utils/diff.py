"""
代码 diff 工具模块
==================
提供代码差异提取和应用功能,支持增量修改模式。
"""
import re
from typing import List, Optional, Tuple


def extract_diffs(
    diff_text: str, 
    diff_pattern: str = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"
) -> List[Tuple[str, str]]:
    """
    从文本中提取 diff 块。
    
    Args:
        diff_text: 包含 diff 的文本
        diff_pattern: diff 的正则表达式模式
        
    Returns:
        [(search_text, replace_text), ...] 的列表
    """
    diff_blocks = re.findall(diff_pattern, diff_text, re.DOTALL)
    return [(match[0].rstrip(), match[1].rstrip()) for match in diff_blocks]


def apply_diff(
    original_code: str,
    diff_text: str,
    diff_pattern: str = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE",
) -> str:
    """
    将 diff 应用到原始代码上。
    
    Args:
        original_code: 原始代码
        diff_text: 包含 diff 的文本
        diff_pattern: diff 的正则表达式模式
        
    Returns:
        应用 diff 后的代码
        
    Raises:
        ValueError: 如果找不到匹配的代码块
    """
    # 提取所有 diff 块
    diff_blocks = extract_diffs(diff_text, diff_pattern)
    
    if not diff_blocks:
        raise ValueError("No diff blocks found in the input text")
    
    result_code = original_code
    
    # 依次应用每个 diff 块
    for i, (search_text, replace_text) in enumerate(diff_blocks):
        # 检查 search_text 是否在代码中
        if search_text not in result_code:
            raise ValueError(
                f"Diff block {i+1}: Search text not found in code.\n"
                f"Search text:\n{search_text[:200]}..."
            )
        
        # 替换第一次出现的位置
        result_code = result_code.replace(search_text, replace_text, 1)
    
    return result_code


def apply_diff_blocks(
    original_code: str,
    diff_blocks: List[Tuple[str, str]],
) -> Tuple[str, int]:
    """
    将多个 diff 块应用到原始代码上。
    
    Args:
        original_code: 原始代码
        diff_blocks: [(search_text, replace_text), ...] 的列表
        
    Returns:
        (修改后的代码, 成功应用的 diff 数量)
    """
    result_code = original_code
    applied_count = 0
    
    for search_text, replace_text in diff_blocks:
        if search_text in result_code:
            result_code = result_code.replace(search_text, replace_text, 1)
            applied_count += 1
    
    return result_code, applied_count


def parse_full_rewrite(llm_response: str, language: str = "python") -> Optional[str]:
    """
    从 LLM 响应中提取完整重写的代码。
    
    Args:
        llm_response: LLM 的响应文本
        language: 编程语言
        
    Returns:
        提取的代码,如果没找到则返回 None
    """
    # 尝试匹配指定语言的代码块
    code_block_pattern = rf"```{language}\n(.*?)```"
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # 回退:匹配任意代码块
    code_block_pattern = r"```(?:\w+\n)?(.*?)```"
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    return None


def format_diff_summary(diff_blocks: List[Tuple[str, str]]) -> str:
    """
    创建 diff 的可读摘要。
    
    Args:
        diff_blocks: [(search_text, replace_text), ...] 的列表
        
    Returns:
        摘要字符串
    """
    if not diff_blocks:
        return "No changes"
    
    summary = []
    
    for i, (search_text, replace_text) in enumerate(diff_blocks):
        search_lines = search_text.strip().split("\n")
        replace_lines = replace_text.strip().split("\n")
        
        # 创建简短摘要
        if len(search_lines) == 1 and len(replace_lines) == 1:
            summary.append(f"修改 {i+1}: '{search_lines[0][:50]}...' → '{replace_lines[0][:50]}...'")
        else:
            summary.append(
                f"修改 {i+1}: 替换 {len(search_lines)} 行 → {len(replace_lines)} 行"
            )
    
    return "\n".join(summary)
