"""
GTSOPS-CANN Skills Loading Service
从GTSOPS-CANN-skills知识库加载指定类型的skill内容
"""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional


class LoadCannSkillsService:
    """加载CANN Skills知识的服务"""

    # Skills目录路径
    SKILLS_BASE_PATH = Path(__file__).parent.parent / "knowledge_base" / "skills"

    # 支持的skill类型
    SUPPORTED_SKILLS = {
        "gtsops-op-scaffold": {
            "name": "gtsops-op-scaffold",
            "description": "GTSOPS算子各模块开发要点。说明在每个文件中需要考虑和完善的内容，标注哪些知识可以从CANN-skills复用，哪些是GTSOPS特有的差异。",
            "file": "SKILL.md"
        },
        "gtsops-kernel": {
            "name": "gtsops-kernel",
            "description": "GTSOPS Kernel实现要点。说明GTSOPS特有的类型处理方式，以及哪些知识可以从CANN-skills的cann-kernel直接复用。",
            "file": "SKILL.md"
        },
        "gtsops-tiling": {
            "name": "gtsops-tiling",
            "description": "GTSOPS Tiling实现要点。说明如何使用Tiling1d辅助类，以及哪些tiling知识可以从CANN-skills的cann-tiling直接复用。",
            "file": "SKILL.md"
        },
        "gtsops-debugging": {
            "name": "gtsops-debugging",
            "description": "GTSOPS算子调试指南。说明常见错误类型和调试方法，引用CANN-skills中的调试知识。",
            "file": "SKILL.md"
        }
    }

    def __init__(self):
        """初始化服务。

        历史上这里会在目录缺失时 raise,导致 backend 整个进程起不来。
        Phase 1 (K-Search 搜索路径)不依赖 skills 服务,所以缺失目录只是
        warning,不再阻塞 backend 启动。LLM 真的调用 load_skills 时,
        load_skill_content 内部自然会返回 error。
        """
        import logging
        logger = logging.getLogger(__name__)
        if not self.SKILLS_BASE_PATH.exists():
            logger.warning(
                "Skills directory not found: %s — load_cann_skills calls "
                "will return error responses but backend will start normally.",
                self.SKILLS_BASE_PATH,
            )

    def load_skill_content(self, skill_name: str) -> Dict[str, Any]:
        """
        加载单个skill的内容

        Args:
            skill_name: skill名称

        Returns:
            包含skill内容的字典
        """
        if skill_name not in self.SUPPORTED_SKILLS:
            return {
                "success": False,
                "error": f"Unknown skill: {skill_name}",
                "available_skills": list(self.SUPPORTED_SKILLS.keys())
            }

        skill_info = self.SUPPORTED_SKILLS[skill_name]
        skill_path = self.SKILLS_BASE_PATH / skill_name / skill_info["file"]

        if not skill_path.exists():
            return {
                "success": False,
                "error": f"Skill file not found: {skill_path}"
            }

        try:
            with open(skill_path, "r", encoding="utf-8") as f:
                content = f.read()

            return {
                "success": True,
                "skill_name": skill_name,
                "description": skill_info["description"],
                "content": content,
                "file_path": str(skill_path)
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to read skill file: {str(e)}"
            }

    def load_skills(self, skills: List[str], context: Optional[str] = None) -> Dict[str, Any]:
        """
        加载多个skills的内容

        Args:
            skills: 要加载的skill名称列表
            context: 可选的上下文描述

        Returns:
            包含所有skill内容的字典
        """
        if not skills:
            return {
                "success": False,
                "error": "No skills specified"
            }

        if len(skills) < 1 or len(skills) > 4:
            return {
                "success": False,
                "error": "Skills list must contain 1-4 items"
            }

        # 验证所有skill名称
        invalid_skills = [s for s in skills if s not in self.SUPPORTED_SKILLS]
        if invalid_skills:
            return {
                "success": False,
                "error": f"Invalid skill names: {invalid_skills}",
                "available_skills": list(self.SUPPORTED_SKILLS.keys())
            }

        # 加载所有skills
        loaded_skills = []
        errors = []

        for skill_name in skills:
            result = self.load_skill_content(skill_name)
            if result["success"]:
                loaded_skills.append({
                    "skill_name": result["skill_name"],
                    "description": result["description"],
                    "content": result["content"]
                })
            else:
                errors.append({
                    "skill_name": skill_name,
                    "error": result["error"]
                })

        # 构建响应
        response = {
            "success": len(loaded_skills) > 0,
            "loaded_count": len(loaded_skills),
            "total_requested": len(skills),
            "skills": loaded_skills
        }

        if context:
            response["context"] = context

        if errors:
            response["errors"] = errors

        # 添加skill摘要信息
        response["skill_summary"] = self._generate_skill_summary(loaded_skills)

        return response

    def _generate_skill_summary(self, loaded_skills: List[Dict]) -> str:
        """
        生成已加载skills的摘要

        Args:
            loaded_skills: 已加载的skill列表

        Returns:
            摘要字符串
        """
        if not loaded_skills:
            return "No skills loaded."

        summary_parts = []
        for skill in loaded_skills:
            summary_parts.append(f"- **{skill['skill_name']}**: {skill['description']}")

        return "\n".join(summary_parts)

    def get_available_skills(self) -> Dict[str, Any]:
        """
        获取所有可用的skills列表

        Returns:
            可用skills的信息
        """
        skills_info = []
        for skill_name, skill_data in self.SUPPORTED_SKILLS.items():
            skill_path = self.SKILLS_BASE_PATH / skill_name / skill_data["file"]
            skills_info.append({
                "name": skill_name,
                "description": skill_data["description"],
                "available": skill_path.exists()
            })

        return {
            "success": True,
            "skills": skills_info,
            "total_count": len(skills_info)
        }


# 测试代码
if __name__ == "__main__":
    service = LoadCannSkillsService()

    # 测试获取可用skills
    print("Available skills:")
    print(service.get_available_skills())

    # 测试加载单个skill
    print("\n\nLoading gtsops-op-scaffold:")
    result = service.load_skills(["gtsops-op-scaffold"], "测试上下文")
    print(f"Success: {result['success']}")
    print(f"Loaded: {result['loaded_count']}")
    print(f"Summary:\n{result['skill_summary']}")

    # 测试加载多个skills
    print("\n\nLoading multiple skills:")
    result = service.load_skills(
        ["gtsops-op-scaffold", "gtsops-kernel"],
        "正在开发一个Add算子"
    )
    print(f"Success: {result['success']}")
    print(f"Loaded: {result['loaded_count']}")
    print(f"Summary:\n{result['skill_summary']}")
