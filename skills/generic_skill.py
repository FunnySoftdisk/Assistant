"""
通用Skill加载器 - 加载标准格式的Skill

"""
import os
import re
import json
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass


@dataclass
class SkillTool:
    """Skill工具定义"""
    name: str
    description: str
    parameters: Dict[str, Any]
    script_path: str  # 调用的脚本路径


@dataclass
class GenericSkill:
    """通用Skill"""
    name: str
    version: str
    description: str
    skill_dir: Path
    tools: List[SkillTool]
    trigger_keywords: List[str]  # 触发关键词


class GenericSkillLoader:
    """
    通用Skill加载器
    扫描 skills/ 目录，加载所有标准格式的Skill
    无需 agent.py，直接暴露 tools
    """

    def __init__(self, skills_dir: str = "skills"):
        self.skills_dir = Path(skills_dir)
        self._skills: Dict[str, GenericSkill] = {}
        self._tools_by_keyword: Dict[str, List[GenericSkill]] = {}

    def discover_skills(self) -> Dict[str, GenericSkill]:
        """发现并加载所有标准Skill"""
        if not self.skills_dir.exists():
            return {}

        for skill_path in self.skills_dir.iterdir():
            if not skill_path.is_dir() or skill_path.name.startswith("__"):
                continue

            skill_md = skill_path / "SKILL.md"
            if not skill_md.exists():
                continue

            skill = self._load_skill(skill_path)
            if skill:
                self._skills[skill.name] = skill
                self._register_triggers(skill)

        return self._skills

    def _load_skill(self, skill_path: Path) -> Optional[GenericSkill]:
        """加载单个Skill"""
        try:
            content = (skill_path / "SKILL.md").read_text(encoding="utf-8")
            metadata = self._parse_frontmatter(content)

            name = metadata.get("name", skill_path.name)
            description = metadata.get("description", "")

            # 解析 tools
            tools = []
            tools_text = metadata.get("tools", "[]")
            if isinstance(tools_text, str):
                try:
                    tools_list = json.loads(tools_text)
                except:
                    tools_list = [t.strip() for t in tools_text.strip("[]").split(",") if t.strip()]
            else:
                tools_list = tools_text or []

            # 解析 functions（通常和tools相同）
            functions_text = metadata.get("functions", "[]")
            if isinstance(functions_text, str):
                try:
                    functions_list = json.loads(functions_text)
                except:
                    functions_list = [f.strip() for f in functions_text.strip("[]").split(",") if f.strip()]
            else:
                functions_list = functions_text or []

            # 合并 tools 和 functions
            all_tools = list(set(tools_list + functions_list))

            # 查找可执行脚本
            scripts_dir = skill_path / "scripts"
            script_files = {}
            if scripts_dir.exists():
                for script in scripts_dir.iterdir():
                    if script.suffix in [".py", ".sh"]:
                        script_files[script.stem] = str(script)

            # 为每个 tool 创建 SkillTool
            skill_tools = []

            # 如果没有显式定义 tools，但有 scripts，则使用 scripts 中的脚本作为隐式工具
            if not all_tools and script_files:
                # 使用第一个脚本作为默认工具，工具名与skill名关联
                for tool_name, script_path in script_files.items():
                    skill_tools.append(SkillTool(
                        name=tool_name,
                        description=f"{name}: {tool_name}",
                        parameters={},
                        script_path=script_path
                    ))
            else:
                # 为每个显式定义的 tool 查找对应脚本
                for tool_name in all_tools:
                    # 查找对应脚本
                    script_path = script_files.get(tool_name, "")
                    if not script_path and (scripts_dir / f"{tool_name}.py").exists():
                        script_path = str(scripts_dir / f"{tool_name}.py")

                    skill_tools.append(SkillTool(
                        name=tool_name,
                        description=f"{name}: {tool_name}",
                        parameters={},
                        script_path=script_path
                    ))

            # 解析触发关键词（从description和workflow中提取）
            trigger_keywords = self._extract_triggers(content, description)

            return GenericSkill(
                name=name,
                version=metadata.get("version", "1.0.0"),
                description=description,
                skill_dir=skill_path,
                tools=skill_tools,
                trigger_keywords=trigger_keywords
            )

        except Exception as e:
            print(f"Failed to load skill {skill_path}: {e}")
            return None

    def _parse_frontmatter(self, content: str) -> Dict[str, str]:
        """解析YAML frontmatter"""
        metadata = {}

        # 匹配 --- 包裹的frontmatter
        match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        if match:
            frontmatter = match.group(1)
            for line in frontmatter.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip()

        return metadata

    def _extract_triggers(self, content: str, description: str) -> List[str]:
        """从SKILL.md中提取触发关键词"""
        triggers = set()

        # 从description提取
        words = re.findall(r"[\u4e00-\u9fff]+", description)
        triggers.update(words)

        # 从触发条件部分提取
        trigger_section = re.search(r"#?触发条件?\s*(.*?)(?:\n#|\Z)", content, re.DOTALL | re.IGNORECASE)
        if trigger_section:
            triggers_text = trigger_section.group(1)
            triggers.update(re.findall(r"[\u4e00-\u9fff]+", triggers_text))
            # 也提取英文关键词
            triggers.update(re.findall(r"\b\w+\b", triggers_text.lower()))

        # 过滤太短的词
        return [t for t in triggers if len(t) >= 2]

    def _register_triggers(self, skill: GenericSkill):
        """注册触发关键词"""
        for keyword in skill.trigger_keywords:
            if keyword not in self._tools_by_keyword:
                self._tools_by_keyword[keyword] = []
            self._tools_by_keyword[keyword].append(skill)

    def get_skill(self, name: str) -> Optional[GenericSkill]:
        """获取Skill"""
        return self._skills.get(name)

    def get_skill_by_keyword(self, keyword: str) -> List[GenericSkill]:
        """通过关键词查找Skill"""
        return self._tools_by_keyword.get(keyword, [])

    def match_skills(self, query: str) -> List[GenericSkill]:
        """根据查询内容匹配相关Skills"""
        query_lower = query.lower()

        # 提取查询词：中文按单字+按空格/英文词边界
        query_words = set()
        # 中文单字
        query_words.update(re.findall(r"[\u4e00-\u9fff]", query_lower))
        # 英文词
        query_words.update(re.findall(r"\b[a-z0-9]+\b", query_lower))

        matched = []
        seen = set()

        # 精确匹配触发词
        for keyword, skills in self._tools_by_keyword.items():
            if keyword in query_lower:
                for skill in skills:
                    if skill.name not in seen:
                        matched.append(skill)
                        seen.add(skill.name)

        # 如果没有通过触发词匹配到，尝试通过描述匹配
        if not matched:
            for name, skill in self._skills.items():
                if name in seen:
                    continue

                # 检查描述词与查询词的重叠
                desc_words = set(skill.trigger_keywords)
                if not desc_words:
                    # 从description提取
                    desc_words = set(re.findall(r"[\u4e00-\u9fff]", skill.description.lower()))
                    desc_words.update(re.findall(r"\b[a-z0-9]+\b", skill.description.lower()))

                # 重叠的单字/词数量
                overlap = query_words & desc_words
                # 也检查完整匹配（短词组）
                for t in skill.trigger_keywords:
                    if len(t) >= 2 and t in query_lower:
                        overlap.add(t)

                if overlap and len(overlap) >= 2:
                    matched.append(skill)
                    seen.add(name)

        # 按匹配度排序
        matched.sort(key=lambda s: len(set(s.trigger_keywords) & set(query_lower)), reverse=True)

        return matched

    async def invoke_tool(self, tool: SkillTool, parameters: Dict = None) -> Dict:
        """调用Skill工具"""
        params = parameters or {}

        if not tool.script_path:
            return {"error": f"Tool {tool.name} has no script"}

        script_path = Path(tool.script_path)
        if not script_path.exists():
            return {"error": f"Script not found: {tool.script_path}"}

        try:
            if script_path.suffix == ".py":
                cmd = ["python", str(script_path)]
            else:
                cmd = [str(script_path)]

            # 添加参数
            for key, value in params.items():
                cmd.extend([f"--{key}", str(value)])

            # 执行脚本
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                try:
                    return json.loads(stdout.decode("utf-8"))
                except:
                    return {"result": stdout.decode("utf-8").strip()}
            else:
                return {"error": stderr.decode("utf-8").strip() or f"Exit code: {result.returncode}"}

        except Exception as e:
            return {"error": str(e)}

    def list_skills(self) -> Dict[str, GenericSkill]:
        """列出所有已加载的Skill"""
        return self._skills.copy()


# 全局实例
_global_loader: Optional[GenericSkillLoader] = None


def get_generic_skill_loader(skills_dir: str = "skills") -> GenericSkillLoader:
    """获取全局GenericSkillLoader"""
    global _global_loader
    if _global_loader is None:
        _global_loader = GenericSkillLoader(skills_dir)
        _global_loader.discover_skills()
    return _global_loader
