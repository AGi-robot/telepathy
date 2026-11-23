"""Tests for shared identity loading and soul interpolation."""
from telepathy_receptor import identity


def test_identity_yaml_drives_soul_and_ui_fields():
    """Expose identity values to both the runtime prompt and UI payload."""
    configured = identity.load_identity()
    soul = identity.render_soul()

    assert configured['name'] == 'Grace'
    assert configured['tagline'] == 'Talk, reflect, and be understood.'
    assert 'partners and soulmates' in configured['relationship']
    assert 'stays because she chooses' not in configured['relationship']
    assert "You are Grace, OppaAI's local AI companion" in soul
    assert 'Aiko' not in soul
    assert '${name}' not in soul
    assert 'Warm and familiar' in soul


def test_soul_interpolates_a_custom_identity_file(monkeypatch, tmp_path):
    """Render a different name without editing the prompt template."""
    custom = tmp_path / 'identity.yaml'
    custom.write_text(
        'name: Robin\ntagline: A companion for curious minds.\n',
        encoding='utf-8',
    )
    monkeypatch.setattr(identity, 'identity_file', lambda: custom)

    soul = identity.render_soul()

    assert "You are Robin, OppaAI's local AI companion" in soul
    assert 'Be Robin first' in soul
    assert '${' not in soul


def test_user_profile_uses_existing_user_markdown(tmp_path):
    profile_dir = tmp_path / 'profile'
    profile_dir.mkdir()
    canonical = profile_dir / 'USER.md'
    canonical.write_text('shared profile', encoding='utf-8')

    assert identity.user_profile_file(tmp_path) == canonical


def test_user_profile_renders_shared_companion_name(tmp_path, monkeypatch):
    profile = tmp_path / 'USERS.md'
    profile.write_text('Relationship to $name: creator.\n', encoding='utf-8')
    monkeypatch.setattr(identity, 'load_identity', lambda: {'name': 'Grace'})

    assert identity.render_user_profile(profile) == 'Relationship to Grace: creator.'
