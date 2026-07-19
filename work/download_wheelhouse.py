from __future__ import annotations

import email
import hashlib
import importlib.metadata
import json
import sys
import urllib.request
import zipfile
from collections import defaultdict, deque
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version


WHEEL_DIR = Path(__file__).resolve().parent / "wheels"
WHEEL_DIR.mkdir(parents=True, exist_ok=True)
SUPPORTED_TAGS = list(sys_tags())
TAG_RANK = {tag: index for index, tag in enumerate(SUPPORTED_TAGS)}
MARKER_ENV = default_environment()
MARKER_ENV["extra"] = ""


def current_version(name: str) -> Version | None:
    try:
        return Version(importlib.metadata.version(name))
    except importlib.metadata.PackageNotFoundError:
        return None


def get_json(name: str) -> dict:
    url = f"https://pypi.org/pypi/{name}/json"
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def best_wheel(name: str, specs: list[SpecifierSet]) -> tuple[Version, dict]:
    data = get_json(name)
    candidates: list[tuple[Version, int, dict]] = []
    for raw_version, files in data.get("releases", {}).items():
        try:
            version = Version(raw_version)
        except Exception:
            continue
        if version.is_prerelease or version.is_devrelease:
            continue
        if not all(version in spec for spec in specs):
            continue
        for item in files:
            filename = item.get("filename", "")
            if item.get("packagetype") != "bdist_wheel" or not filename.endswith(".whl"):
                continue
            requires_python = item.get("requires_python")
            if requires_python and Version(".".join(map(str, sys.version_info[:3]))) not in SpecifierSet(requires_python):
                continue
            try:
                _, _, _, wheel_tags = parse_wheel_filename(filename)
            except Exception:
                continue
            ranks = [TAG_RANK[tag] for tag in wheel_tags if tag in TAG_RANK]
            if ranks:
                candidates.append((version, -min(ranks), item))
    if not candidates:
        joined = ", ".join(str(spec) for spec in specs)
        raise RuntimeError(f"No compatible wheel for {name} ({joined})")
    version = max(item[0] for item in candidates)
    same_version = [item for item in candidates if item[0] == version]
    _, _, wheel = max(same_version, key=lambda item: item[1])
    return version, wheel


def download(item: dict) -> Path:
    destination = WHEEL_DIR / item["filename"]
    expected = item["digests"]["sha256"]
    if destination.exists():
        actual = hashlib.sha256(destination.read_bytes()).hexdigest()
        if actual == expected:
            return destination
        destination.unlink()
    print(f"Downloading {destination.name}", flush=True)
    with urllib.request.urlopen(item["url"], timeout=120) as response, destination.open("wb") as stream:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            stream.write(chunk)
    actual = hashlib.sha256(destination.read_bytes()).hexdigest()
    if actual != expected:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"SHA256 mismatch for {destination.name}")
    return destination


def wheel_requirements(path: Path) -> list[Requirement]:
    with zipfile.ZipFile(path) as archive:
        metadata_name = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        message = email.message_from_bytes(archive.read(metadata_name))
    requirements: list[Requirement] = []
    for raw in message.get_all("Requires-Dist", []):
        req = Requirement(raw)
        if req.marker is None or req.marker.evaluate(MARKER_ENV):
            requirements.append(req)
    return requirements


def main() -> None:
    queue: deque[Requirement] = deque()
    raw_requirements = sys.argv[1:]
    if not raw_requirements:
        raw_requirements = importlib.metadata.distribution("streamlit").requires or []
    for raw in raw_requirements:
        req = Requirement(raw)
        if req.marker is None or req.marker.evaluate(MARKER_ENV):
            queue.append(req)

    constraints: dict[str, list[SpecifierSet]] = defaultdict(list)
    display_names: dict[str, str] = {}
    selected: dict[str, Version] = {}

    while queue:
        req = queue.popleft()
        key = canonicalize_name(req.name)
        display_names.setdefault(key, req.name)
        constraints[key].append(req.specifier)

        installed = current_version(req.name)
        if installed is not None and all(installed in spec for spec in constraints[key]):
            print(f"Using installed {req.name} {installed}", flush=True)
            selected[key] = installed
            continue

        if key in selected and all(selected[key] in spec for spec in constraints[key]):
            continue

        version, item = best_wheel(req.name, constraints[key])
        path = download(item)
        selected[key] = version
        for child in wheel_requirements(path):
            queue.append(child)

    print(f"Wheelhouse ready: {len(list(WHEEL_DIR.glob('*.whl')))} wheels", flush=True)


if __name__ == "__main__":
    main()
