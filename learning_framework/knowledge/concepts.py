"""Concept loading and registry management"""

import json
import yaml
import re
import os
from pathlib import Path
from typing import Dict, Any, List, Optional


def load_concept(concept_slug: str, base_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load concept from content.md (preferred) or concept.json.
    If content.md exists, the old concept.json will be removed.

    Args:
        concept_slug: Concept identifier (e.g., 'backpropagation')
        base_path: Base data directory (default: data/)

    Returns:
        Concept dictionary

    Raises:
        FileNotFoundError: If neither content.md nor concept.json exists
    """
    if base_path is None:
        base_path = Path.cwd() / 'data'

    concept_dir = base_path / concept_slug
    md_path = concept_dir / 'content.md'
    json_path = concept_dir / 'concept.json'

    # 1. 优先尝试 Markdown
    if md_path.exists():
        content = md_path.read_text(encoding='utf-8')
        
        # 分离 Frontmatter 和 正文
        pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
        match = re.match(pattern, content, re.DOTALL)
        
        if match:
            yaml_text = match.group(1)
            markdown_body = match.group(2)
            
            try:
                data = yaml.safe_load(yaml_text) or {}
                data['explanation_md'] = markdown_body.strip() 
                
                if 'explanation' not in data:
                    data['explanation'] = "Please view content in Markdown mode."
                
                if 'slug' not in data:
                    data['slug'] = concept_slug
                
                # --- 策略执行：如果 MD 加载成功且 JSON 还在，则删除 JSON ---
                if json_path.exists():
                    try:
                        os.remove(json_path)
                    except OSError:
                        pass # 忽略删除失败，不影响运行
                    
                return data
            except yaml.YAMLError:
                pass 

    # 2. 回退到 JSON
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    raise FileNotFoundError(f"Concept not found: {concept_slug}")


class ConceptRegistry:
    """Registry of all concepts in the system"""

    def __init__(self, base_path: Optional[Path] = None):
        if base_path is None:
            base_path = Path.cwd() / 'data'

        self.base_path = Path(base_path)
        self.registry_path = self.base_path / 'concepts.json'
        self._concepts = self._load_registry()

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        if not self.registry_path.exists():
            return {'version': '0.2.0', 'concepts': {}, 'topics': {}}

        with open(self.registry_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save(self):
        self.base_path.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(self._concepts, f, indent=2, ensure_ascii=False)

    def register(self, concept_slug: str, topic: str, difficulty: str, status: str = 'skeleton'):
        if 'concepts' not in self._concepts:
            self._concepts['concepts'] = {}
        self._concepts['concepts'][concept_slug] = {'status': status, 'topic': topic, 'difficulty': difficulty}
        if 'topics' not in self._concepts:
            self._concepts['topics'] = {}
        if topic not in self._concepts['topics']:
            self._concepts['topics'][topic] = []
        if concept_slug not in self._concepts['topics'][topic]:
            self._concepts['topics'][topic].append(concept_slug)

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        return self._concepts.get('concepts', {})

    def get_by_topic(self, topic: str) -> List[str]:
        return self._concepts.get('topics', {}).get(topic, [])

    def get_topics(self) -> List[str]:
        return list(self._concepts.get('topics', {}).keys())
