"""Pinned Hugging Face file catalogs; metadata only, never dataset payloads."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import re
import time
import tempfile
import urllib.error
import urllib.parse
import urllib.request

REPOS = {"raw": "brownu/deform360", "processed": "brownu/deform360_processed"}
STREAMS = ("tactilel_left", "tactilel_right", "tactiler_left", "tactiler_right")
REQUIRED = ("metadata.json", "cameras.tar", "splatfacto.tar", "robot/robot.npy")


def safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"Unsafe name: {value!r}")
    return value


def safe_relative(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Invalid relative file path")
    parts = value.split("/")
    if any(not p or p in (".", "..") or ":" in p for p in parts) or PurePosixPath(value).is_absolute():
        raise ValueError(f"Unsafe relative file path: {value}")
    return value


def save_json(path, value):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Refusing to replace a linked journal")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="." + path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        if os.name == "posix":
            descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class Hub:
    def __init__(self, timeout=30, attempts=3):
        self.timeout, self.attempts = timeout, attempts

    def get(self, url):
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or parsed.netloc != "huggingface.co" or not parsed.path.startswith("/api/datasets/"):
            raise ValueError("Only official dataset metadata API URLs are accepted")
        for attempt in range(self.attempts):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "t4d-dataset-build/1"})
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.load(response), response.headers.get("Link", "")
            except urllib.error.HTTPError as exc:
                if exc.code not in (408, 429, 500, 502, 503, 504) or attempt + 1 == self.attempts:
                    raise
            except (OSError, ValueError):
                if attempt + 1 == self.attempts:
                    raise
            time.sleep(2 ** attempt)
        raise RuntimeError("Request failed")

    def pin(self, repo):
        data, _ = self.get("https://huggingface.co/api/datasets/" + repo)
        sha = data.get("sha", "")
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError("Missing immutable repository revision")
        return sha

    def tree(self, repo, revision, path, recursive=False):
        url = f"https://huggingface.co/api/datasets/{repo}/tree/{revision}/{path}?limit=1000"
        if recursive:
            url += "&recursive=true"
        seen_urls, seen_paths, entries = set(), set(), []
        while url:
            if url in seen_urls:
                raise ValueError("Repeated pagination link")
            seen_urls.add(url)
            page, links = self.get(url)
            if not isinstance(page, list):
                raise ValueError("Invalid tree response")
            for item in page:
                name = safe_relative(item.get("path"))
                if not name.startswith(path + "/") or name in seen_paths:
                    raise ValueError("Duplicate or out-of-scope catalog entry")
                seen_paths.add(name)
                if item.get("type") not in ("file", "directory"):
                    raise ValueError("Unknown catalog entry type")
                if item["type"] == "file" and (type(item.get("size")) is not int or item["size"] < 0):
                    raise ValueError("Invalid file size")
                entries.append(item)
            next_links = re.findall(r'<([^>]+)>;\s*rel="next"', links)
            if len(next_links) > 1:
                raise ValueError("Ambiguous pagination")
            url = next_links[0] if next_links else None
        return entries


def file_record(item, repo, revision):
    lfs = item.get("lfs") or {}
    sha = lfs.get("oid")
    git_oid = None if lfs else item.get("oid")
    if sha is not None and not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError("Invalid official SHA256")
    if git_oid is not None and not re.fullmatch(r"[0-9a-f]{40}", git_oid):
        raise ValueError("Invalid Git blob identity")
    if sha is None and git_oid is None:
        raise ValueError(f"No official content identity: {item['path']}")
    return {"path": safe_relative(item["path"]), "size": item["size"], "repo": repo,
            "revision": revision, "sha256": sha, "git_oid": git_oid}


def index_processed(entries, revision):
    scenes = {}
    aliases = {}
    for item in entries:
        parts = item["path"].split("/")
        scene = safe_name(parts[1])
        scenes.setdefault(scene, {})
        if len(parts) < 3 or not re.fullmatch(r"episode_\d+", parts[2]):
            continue
        episode = int(parts[2][8:])
        key = scene, episode
        if key in aliases and aliases[key] != parts[2]:
            raise ValueError("Duplicate padded/unpadded Episode identity")
        aliases[key] = parts[2]
        entry = scenes[scene].setdefault(episode, {"directory": parts[2], "files": {}})
        if item["type"] == "file" and len(parts) > 3:
            entry["files"]["/".join(parts[3:])] = file_record(item, REPOS["processed"], revision)
    return scenes


def raw_inputs(entries, scene, revision):
    chosen, streams = [], {}
    for item in entries:
        if item["type"] != "file":
            continue
        suffix = item["path"][len(f"raw/{scene}/"):]
        if (suffix == "metadata.json" or suffix.startswith("calibration_refined/")
                or (suffix.startswith("brics-odroid_tactile") and suffix.endswith((".npy", ".txt")))):
            chosen.append(file_record(item, REPOS["raw"], revision))
    for stream in STREAMS:
        prefix = f"raw/{scene}/brics-odroid_{stream}/"
        names = [Path(f["path"]).name for f in chosen if f["path"].startswith(prefix)]
        streams[stream] = {"recordings": sum(n.endswith(".npy") and not n.startswith("median_") for n in names),
                           "timestamps": sum(n.endswith(".txt") for n in names),
                           "medians": sorted(n for n in names if n.startswith("median_") and n.endswith(".npy"))}
    return chosen, streams


def build_catalog(hub, revisions, selected=(), cache_dir=None):
    raw_tree = hub.tree(REPOS["raw"], revisions["raw"], "raw")
    raw_scenes = {safe_name(e["path"].split("/")[1]) for e in raw_tree if e["type"] == "directory"}
    processed = index_processed(hub.tree(REPOS["processed"], revisions["processed"], "processed", True), revisions["processed"])
    requested = set(selected) if selected else raw_scenes | processed.keys()
    if requested - (raw_scenes | processed.keys()):
        raise ValueError("Requested scenes are absent from both repositories")
    result = []
    for scene in sorted(requested):
        safe_name(scene)
        print(f"Catalog: {scene}", flush=True)
        if scene in raw_scenes:
            cache = Path(cache_dir) / f"{scene}.json" if cache_dir else None
            if cache and cache.exists():
                saved = json.loads(cache.read_text(encoding="utf-8"))
                if saved["revision"] != revisions["raw"]:
                    raise ValueError("Cached metadata revision mismatch")
                raw_files, streams = saved["files"], saved["streams"]
            else:
                raw_files, streams = raw_inputs(hub.tree(REPOS["raw"], revisions["raw"], "raw/" + scene, True), scene, revisions["raw"])
                if cache:
                    save_json(cache, {"revision": revisions["raw"], "files": raw_files, "streams": streams})
        else:
            raw_files, streams = [], {}
        episodes = []
        for number, info in sorted(processed.get(scene, {}).items()):
            missing = [n for n in REQUIRED if n not in info["files"] or info["files"][n]["size"] == 0]
            reasons = (["raw_scene_missing"] if scene not in raw_scenes else [])
            reasons += ["missing_required_file:" + n for n in missing]
            episodes.append({"episode": number, "directory": info["directory"], "download_ready": not reasons,
                             "reasons": reasons, "files": [info["files"][n] for n in REQUIRED if n in info["files"]]})
        result.append({"scene": scene, "raw_files": raw_files, "streams": streams, "episodes": episodes,
                       "note": "Episode counts and multiple median candidates do not exclude downloads"})
    return {"schema": "t4d-build-catalog-v1", "revisions": revisions, "repos": REPOS,
            "created_utc": datetime.now(timezone.utc).isoformat(), "scenes": result,
            "content_verified": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--scenes", nargs="*", default=[])
    args = p.parse_args()
    if args.output.exists():
        p.error("Refusing to overwrite an existing catalog")
    hub = Hub()
    revisions = {k: hub.pin(repo) for k, repo in REPOS.items()}
    catalog = build_catalog(hub, revisions, args.scenes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, catalog)


if __name__ == "__main__":
    main()
