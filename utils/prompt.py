"""
Prompt 管理模块
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional

from jinja2 import Environment, FileSystemLoader, Template


class PromptManager:
    """
    Prompt 模板管理器。
    
    支持:
    - Jinja2 模板渲染
    - 从实验目录加载自定义 prompt
    - 从 utils/prompts 加载默认 prompt
    """
    
    def __init__(self, prompt_dir: Optional[Path] = None):
        """
        Args:
            prompt_dir: Prompt 模板目录，通常是 experiments/{name}/prompts/
        """
        self.prompt_dir = Path(prompt_dir) if prompt_dir else None
        self.env = None
        self.templates: Dict[str, Template] = {}
        
        # 默认模板目录
        self.default_prompt_dir = Path(__file__).parent / "prompts"
        
        if self.prompt_dir and self.prompt_dir.exists():
            self.env = Environment(
                loader=FileSystemLoader(str(self.prompt_dir)),
                trim_blocks=True,
                lstrip_blocks=True,
            )
    
    def get_template(self, name: str) -> Optional[Template]:
        """
        获取模板。
        
        优先级:
        1. 用户实验目录的模板
        2. 默认模板目录 (utils/prompts/)
        
        Args:
            name: 模板名称（不含 .jinja2 后缀）
            
        Returns:
            Template 对象，如果不存在返回 None
        """
        if name in self.templates:
            return self.templates[name]
        
        # 优先从用户目录加载
        if self.env:
            try:
                template = self.env.get_template(f"{name}.jinja2")
                self.templates[name] = template
                return template
            except Exception:
                pass
        
        # 回退到默认模板目录
        default_template_path = self.default_prompt_dir / f"{name}.jinja2"
        if default_template_path.exists():
            try:
                template_content = default_template_path.read_text(encoding="utf-8")
                template = Template(template_content)
                self.templates[name] = template
                return template
            except Exception:
                pass
        
        return None
    
    def get_default_template(self, name: str) -> Optional[Template]:
        """
        仅从默认模板目录 (utils/prompts/) 加载模板，不从用户目录加载。
        用于 diff 模式：researcher_diff 必须用框架默认模板，用户内容通过 user_prompt 注入。
        """
        default_template_path = self.default_prompt_dir / f"{name}.jinja2"
        if not default_template_path.exists():
            return None
        try:
            template_content = default_template_path.read_text(encoding="utf-8")
            return Template(template_content)
        except Exception:
            return None
    
    def render(self, name: str, **context) -> str:
        """
        渲染模板。
        
        Args:
            name: 模板名称
            **context: 模板变量
            
        Returns:
            渲染后的字符串
        """
        # 如果是 researcher 模板且有 diff_based 参数
        if name == "researcher" and context.get("diff_based", False):
            # 用户提供的 researcher_diff.jinja2 仅作为 context 注入，不当作主模板
            user_prompt = self._render_user_template("researcher_diff", context)
            if user_prompt:
                context["user_prompt"] = user_prompt
            
            # Diff 模式必须使用框架默认的 researcher_diff 模板（含 base_code、SEARCH/REPLACE、XML 格式说明）
            template = self.get_default_template("researcher_diff")
            if template:
                return template.render(**context)
            
            raise ValueError(f"Default template 'researcher_diff' not found")
        
        # 非 diff 模式：正常加载模板
        template = self.get_template(name)
        if template:
            return template.render(**context)
        
        raise ValueError(f"Template '{name}' not found")
    
    def _render_user_template(self, name: str, context: Dict) -> Optional[str]:
        """
        渲染用户自定义模板（如果存在）。
        
        在 diff 模式下使用，将用户的 researcher.jinja2 渲染为 prefix。
        
        Args:
            name: 模板名称
            context: 模板上下文变量
            
        Returns:
            渲染后的内容，如果模板不存在返回 None
        """
        if not self.prompt_dir or not self.prompt_dir.exists():
            return None
        
        template_path = self.prompt_dir / f"{name}.jinja2"
        if not template_path.exists():
            return None
        
        try:
            template_content = template_path.read_text(encoding="utf-8")
            template = Template(template_content)
            return template.render(**context).strip()
        except Exception as e:
            # 静默失败，返回 None
            return None
    
    def has_template(self, name: str) -> bool:
        """检查模板是否存在"""
        if name in self.templates:
            return True
        
        # 检查用户目录
        if self.prompt_dir:
            template_path = self.prompt_dir / f"{name}.jinja2"
            if template_path.exists():
                return True
        
        # 检查默认目录
        default_template_path = self.default_prompt_dir / f"{name}.jinja2"
        return default_template_path.exists()
    
    def list_templates(self) -> list:
        """列出所有可用模板"""
        templates = set()
        
        # 用户目录
        if self.prompt_dir and self.prompt_dir.exists():
            templates.update(f.stem for f in self.prompt_dir.glob("*.jinja2"))
        
        # 默认目录
        if self.default_prompt_dir.exists():
            templates.update(f.stem for f in self.default_prompt_dir.glob("*.jinja2"))
        
        return list(templates)
    
    def save_template(self, name: str, content: str):
        """
        保存模板。
        
        Args:
            name: 模板名称
            content: 模板内容
        """
        if not self.prompt_dir:
            raise ValueError("No prompt directory configured")
        
        self.prompt_dir.mkdir(parents=True, exist_ok=True)
        template_path = self.prompt_dir / f"{name}.jinja2"
        template_path.write_text(content, encoding="utf-8")
        
        # 清除缓存
        if name in self.templates:
            del self.templates[name]

