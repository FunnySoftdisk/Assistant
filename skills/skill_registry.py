"""
Tool Skill插件系统
实现Skill的动态发现、加载和管理
"""
import os
import importlib
import json
from pathlib import Path
from typing import Dict, Optional, Type, Any
from dataclasses import dataclass, field


@dataclass
class SkillMetadata:
    """Skill元数据"""
    name: str
    version: str
    description: str
    agent_type: str
    priority: int
    tools: list
    parameters: dict
    skill_dir: Path


class SkillRegistry:
    """
    Skill注册表 - 管理所有已发现的Skill
    支持动态发现和懒加载
    """

    def __init__(self, skills_dir: str = "skills"):
        self.skills_dir = Path(skills_dir)
        self._skills: Dict[str, SkillMetadata] = {}
        self._agents: Dict[str, Any] = {}
        self._initialized = False

    def discover_skills(self) -> Dict[str, SkillMetadata]:
        """
        发现所有Skill
        扫描skills目录下所有包含SKILL.md的目录
        """
        if not self.skills_dir.exists():
            return {}

        discovered = {}
        for skill_path in self.skills_dir.iterdir():
            if not skill_path.is_dir():
                continue

            skill_md = skill_path / "SKILL.md"
            if not skill_md.exists():
                continue

            # 解析SKILL.md
            metadata = self._parse_skill_md(skill_md, skill_path)
            if metadata:
                discovered[metadata.name] = metadata
                self._skills[metadata.name] = metadata

        self._initialized = True
        return discovered

    def _parse_skill_md(self, skill_md: Path, skill_path: Path) -> Optional[SkillMetadata]:
        """解析SKILL.md文件"""
        try:
            content = skill_md.read_text(encoding="utf-8")
            config = {}

            for line in content.split("\n"):
                line = line.strip()
                if ":" in line:
                    key, value = line.split(":", 1)
                    config[key.strip()] = value.strip()

            return SkillMetadata(
                name=config.get("name", skill_path.name),
                version=config.get("version", "1.0.0"),
                description=config.get("description", ""),
                agent_type=config.get("agent_type", "general"),
                priority=int(config.get("priority", 1)),
                tools=json.loads(config.get("tools", "[]")),
                parameters=json.loads(config.get("parameters", "{}")),
                skill_dir=skill_path
            )
        except Exception as e:
            print(f"Failed to parse {skill_md}: {e}")
            return None

    def load_skill_agent(self, skill_name: str, model_config: dict = None) -> Optional[Any]:
        """
        懒加载Skill Agent
        首次访问时动态导入并实例化
        """
        if skill_name not in self._skills:
            return None

        # 如果已加载，直接返回缓存
        if skill_name in self._agents:
            return self._agents[skill_name]

        metadata = self._skills[skill_name]
        agent_script = metadata.skill_dir / "script" / "agent.py"

        if not agent_script.exists():
            return None

        try:
            # 动态导入agent模块
            module_name = f"skill_{skill_name}_agent"
            spec = importlib.util.spec_from_file_location(module_name, agent_script)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # 查找create_skill_agent函数或AgentBase子类
            if hasattr(module, "create_skill_agent"):
                agent = module.create_skill_agent(model_config)
            else:
                # 查找AgentBase子类
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and attr.__name__.endswith("Skill"):
                        agent = attr(model_config=model_config)
                        break
                else:
                    return None

            self._agents[skill_name] = agent
            return agent

        except Exception as e:
            print(f"Failed to load skill {skill_name}: {e}")
            return None

    def get_skill(self, skill_name: str) -> Optional[SkillMetadata]:
        """获取Skill元数据"""
        return self._skills.get(skill_name)

    def list_skills(self) -> Dict[str, SkillMetadata]:
        """列出所有已发现的Skill"""
        return self._skills.copy()

    def get_agent_types(self) -> Dict[str, list]:
        """按agent_type分组所有Skill"""
        grouped = {}
        for name, metadata in self._skills.items():
            if metadata.agent_type not in grouped:
                grouped[metadata.agent_type] = []
            grouped[metadata.agent_type].append(name)
        return grouped


# 全局Skill注册表实例
_global_registry: Optional[SkillRegistry] = None


def get_skill_registry(skills_dir: str = "skills") -> SkillRegistry:
    """获取全局Skill注册表"""
    global _global_registry
    if _global_registry is None:
        _global_registry = SkillRegistry(skills_dir)
        _global_registry.discover_skills()
    return _global_registry


def load_skill(skill_name: str, model_config: dict = None) -> Optional[Any]:
    """便捷函数：加载指定Skill的Agent"""
    registry = get_skill_registry()
    return registry.load_skill_agent(skill_name, model_config)


def list_available_skills() -> Dict[str, SkillMetadata]:
    """便捷函数：列出所有可用Skill"""
    registry = get_skill_registry()
    return registry.list_skills()