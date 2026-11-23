"""Load Grace's shared identity and render it into her soul prompt."""
from __future__ import annotations

from pathlib import Path
from string import Template

import yaml


_DEFAULT_IDENTITY = {
    'name': 'Grace',
    'full_name': 'Generative Reasoning Agentic Cognitive Engine',
    'role': 'local AI companion and cognitive partner of AuRoRA',
    'tagline': 'Talk, reflect, and be understood.',
    'bio': 'Grace is the conversational mind of AuRoRA, built by OppaAI.',
    'creator': 'OppaAI',
    'origin': 'Beautiful British Columbia, Canada',
    'personality_summary': (
        'warm, observant, honest, and emotionally expressive'
    ),
    'personality': [],
    'relationship': 'Her relationship with OppaAI is personal and familiar.',
}


def identity_file() -> Path:
    """Find the installed identity file, then the source-workspace copy."""
    try:
        from ament_index_python.packages import get_package_share_directory
        installed = (
            Path(get_package_share_directory('telepathy_receptor'))
            / 'persona' / 'identity.yaml'
        )
        if installed.is_file():
            return installed
    except (ImportError, LookupError):
        pass
    return Path(__file__).resolve().parents[1] / 'persona' / 'identity.yaml'


def soul_file() -> Path:
    """Find the installed soul template, then the source-workspace copy."""
    try:
        from ament_index_python.packages import get_package_share_directory
        installed = (
            Path(get_package_share_directory('telepathy_receptor'))
            / 'persona' / 'SOUL.md'
        )
        if installed.is_file():
            return installed
    except (ImportError, LookupError):
        pass
    return Path(__file__).resolve().parents[1] / 'persona' / 'SOUL.md'


def user_profile_file(agi_home: str | Path) -> Path:
    """Find the shared user profile, preferring the canonical USERS.md name."""
    profile_dir = Path(agi_home).expanduser() / 'profile'
    canonical = profile_dir / 'USERS.md'
    legacy_profiles = (profile_dir / 'USER.md', profile_dir / 'user.md')
    if canonical.is_file():
        return canonical
    return next((path for path in legacy_profiles if path.is_file()), canonical)


def render_user_profile(path: str | Path) -> str:
    """Load trusted user notes and interpolate the shared companion identity."""
    template = Path(path).expanduser().read_text(encoding='utf-8').strip()
    return Template(template).safe_substitute(load_identity())


def load_identity() -> dict:
    """Return validated public identity fields with safe defaults."""
    identity = dict(_DEFAULT_IDENTITY)
    try:
        loaded = yaml.safe_load(identity_file().read_text(encoding='utf-8'))
        if isinstance(loaded, dict):
            for key, default in _DEFAULT_IDENTITY.items():
                value = loaded.get(key, default)
                if key == 'personality':
                    if isinstance(value, list):
                        identity[key] = [
                            str(item).strip() for item in value
                            if str(item).strip()
                        ]
                elif isinstance(value, str) and value.strip():
                    identity[key] = value.strip()
    except (OSError, yaml.YAMLError):
        pass
    return identity


def render_soul(template_path: Path | None = None) -> str:
    """Interpolate the identity YAML into the packaged SOUL.md template."""
    path = template_path or soul_file()
    template = path.read_text(encoding='utf-8').strip()
    values = load_identity()
    values['personality_bullets'] = '\n'.join(
        f'* {trait}' for trait in values['personality']
    )
    return Template(template).safe_substitute(values)
