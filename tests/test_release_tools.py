from __future__ import annotations

from scripts.package_demo import ROOT, should_include


def test_package_excludes_runtime_and_sensitive_files() -> None:
    excluded = [
        ROOT / ".env",
        ROOT / ".env.local",
        ROOT / ".streamlit" / "secrets.toml",
        ROOT / "runs" / "train" / "weights.pt",
        ROOT / "data" / "demo.mp4",
        ROOT / "outputs" / "classroom" / "db" / "events.db",
        ROOT / ".venv" / "pyvenv.cfg",
    ]
    assert all(not should_include(path) for path in excluded)


def test_private_demo_keeps_reference_faces_and_never_embeds_destination() -> None:
    portrait = ROOT / "data" / "students" / "CE182206" / "reference_01.png"
    destination = ROOT / "release" / "helmet_classroom_demo.zip"
    assert should_include(portrait)
    assert not should_include(destination, excluded_paths={destination.resolve()})
