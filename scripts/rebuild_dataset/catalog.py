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
OFFICIAL_ENDPOINT = "https://huggingface.co"
MIRROR_ENDPOINT = "https://hf-mirror.com"
STREAMS = ("tactilel_left", "tactilel_right", "tactiler_left", "tactiler_right")
REQUIRED = ("metadata.json", "cameras.tar", "splatfacto.tar", "robot/robot.npy")


def normalize_endpoint(endpoint):
    if not isinstance(endpoint, str):
        raise ValueError("Hub endpoint must be a URL")
    endpoint = endpoint.rstrip("/")
    if endpoint not in (OFFICIAL_ENDPOINT, MIRROR_ENDPOINT):
        raise ValueError("Choose the official Hub or the explicitly supported HTTPS mirror")
    return endpoint


def resolve_proxy(proxy):
    if proxy == "auto":
        proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
        if not proxy:
            raise RuntimeError("Explicit proxy requested but unavailable; use proxy=null for direct access")
    if proxy is not None:
        if not isinstance(proxy, str):
            raise ValueError("Invalid proxy")
        parsed = urllib.parse.urlsplit(proxy)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username is not None:
            raise ValueError("Use an HTTP/HTTPS proxy without embedded credentials")
    return proxy


def route_api_url(url, endpoint):
    endpoint = normalize_endpoint(endpoint)
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc not in ("huggingface.co", "hf-mirror.com")
            or not parsed.path.startswith("/api/datasets/") or parsed.fragment):
        raise ValueError("Only Hub dataset metadata API URLs are accepted")
    # Official pagination links returned by the mirror remain on the selected
    # endpoint. This is a host substitution, never a proxy/official fallback.
    return endpoint + parsed.path + ("?" + parsed.query if parsed.query else "")


class MetadataRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, endpoint):
        self.endpoint = endpoint

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = route_api_url(urllib.parse.urljoin(req.full_url, newurl), self.endpoint)
        return super().redirect_request(req, fp, code, msg, headers, target)


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
    def __init__(self, timeout=30, attempts=3, *, endpoint=OFFICIAL_ENDPOINT, proxy=None):
        self.timeout, self.attempts = timeout, attempts
        self.endpoint = normalize_endpoint(endpoint)
        self.proxy = resolve_proxy(proxy)
        proxies = {} if self.proxy is None else {"http": self.proxy, "https": self.proxy}
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler(proxies), MetadataRedirect(self.endpoint))
        self.network_info = {"endpoint": self.endpoint, "proxy_mode": "direct" if self.proxy is None else "explicit_proxy",
                             "proxy_fallback": False, "tls_verification": True}

    def get(self, url):
        url = route_api_url(url, self.endpoint)
        for attempt in range(self.attempts):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "t4d-dataset-build/1"})
                with self.opener.open(request, timeout=self.timeout) as response:
                    return json.load(response), response.headers.get("Link", "")
            except urllib.error.HTTPError as exc:
                if exc.code not in (403, 408, 429, 500, 502, 503, 504) or attempt + 1 == self.attempts:
                    raise
            except (OSError, ValueError):
                if attempt + 1 == self.attempts:
                    raise
            time.sleep(2 ** attempt)
        raise RuntimeError("Request failed")

    def pin(self, repo):
        query = urllib.parse.urlencode({"expand[]": "sha"})
        data, _ = self.get("https://huggingface.co/api/datasets/" + repo + "?" + query)
        sha = data.get("sha", "")
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError("Missing immutable repository revision")
        return sha

    def tree(self, repo, revision, path, recursive=False):
        url = f"https://huggingface.co/api/datasets/{repo}/tree/{revision}/{path}?limit=1000"
        if recursive:
            url += "&recursive=true"
        url = route_api_url(url, self.endpoint)
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
            url = route_api_url(urllib.parse.urljoin(url, next_links[0]), self.endpoint) if next_links else None
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


def parse_scene_ids(value):
    """Explicit numbered selection; an empty or ambiguous selection never means all."""
    if isinstance(value, str):
        numbers = []
        for token in value.split(","):
            match = re.fullmatch(r"\s*(\d{1,3})(?:\s*-\s*(\d{1,3}))?\s*", token)
            if not match:
                raise ValueError("Use scene IDs such as 001-020 or 001,008,020")
            start, end = int(match[1]), int(match[2] or match[1])
            if not 1 <= start <= end <= 999:
                raise ValueError("Scene IDs must be between 001 and 999, with ascending ranges")
            numbers.extend(range(start, end + 1))
    elif isinstance(value, list) and all(type(n) is int and 1 <= n <= 999 for n in value):
        numbers = value
    else:
        raise ValueError("Scene IDs require a range string or integer list")
    if not numbers or len(numbers) != len(set(numbers)):
        raise ValueError("Scene IDs must be nonempty and distinct")
    return sorted(numbers)


def resolve_scene_ids(ids, published):
    selected, missing = [], []
    for number in parse_scene_ids(ids):
        matches = sorted(n for n in published if n.startswith(f"{number:03d}-"))
        if len(matches) > 1:
            raise ValueError(f"Scene ID {number:03d} has multiple names: {matches}")
        if matches:
            selected.append(matches[0])
        else:
            missing.append({"scene_id": number, "reason": "scene_id_not_published"})
    return selected, missing


def build_catalog(hub, revisions, selected=(), cache_dir=None, *, scene_ids=None):
    raw_tree = hub.tree(REPOS["raw"], revisions["raw"], "raw")
    raw_scenes = {safe_name(e["path"].split("/")[1]) for e in raw_tree if e["type"] == "directory"}
    request_failures, selection_errors, published = {}, [], set()
    if scene_ids is not None or selected:
        # A one-Episode smoke does not need every object's full file catalog.
        top = hub.tree(REPOS["processed"], revisions["processed"], "processed")
        published = {e["path"].split("/")[1] for e in top if e["type"] == "directory"}
        if scene_ids is not None:
            if selected:
                raise ValueError("Choose either scene names or scene IDs, not both")
            selected, selection_errors = resolve_scene_ids(scene_ids, raw_scenes | published)
        entries = []
        for scene in sorted(set(selected) & published):
            cache = Path(cache_dir) / f"processed-{scene}.json" if cache_dir else None
            if cache and cache.exists():
                saved = json.loads(cache.read_text(encoding="utf-8"))
                if saved["revision"] != revisions["processed"]:
                    raise ValueError("Cached processed metadata revision mismatch")
                page = saved["entries"]
            else:
                try:
                    page = hub.tree(REPOS["processed"], revisions["processed"], "processed/" + safe_name(scene), True)
                except (OSError, urllib.error.URLError) as exc:
                    request_failures.setdefault(scene, []).append("processed_request_failed:" + type(exc).__name__)
                    continue
                if cache:
                    save_json(cache, {"revision": revisions["processed"], "entries": page})
            entries.extend(page)
        processed = index_processed(entries, revisions["processed"])
    else:
        processed = index_processed(hub.tree(REPOS["processed"], revisions["processed"], "processed", True), revisions["processed"])
    requested = set(selected) if selected or scene_ids is not None else raw_scenes | processed.keys()
    if requested - (raw_scenes | processed.keys() | published):
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
                try:
                    page = hub.tree(REPOS["raw"], revisions["raw"], "raw/" + scene, True)
                except (OSError, urllib.error.URLError) as exc:
                    request_failures.setdefault(scene, []).append("raw_request_failed:" + type(exc).__name__)
                    page = None
                raw_files, streams = raw_inputs(page, scene, revisions["raw"]) if page is not None else ([], {})
                if cache and page is not None:
                    save_json(cache, {"revision": revisions["raw"], "files": raw_files, "streams": streams})
        else:
            raw_files, streams = [], {}
        episodes = []
        raw_reasons = []
        if scene not in raw_scenes:
            raw_reasons.append("raw_scene_missing")
        elif scene not in request_failures:
            for name, info in streams.items():
                for key in ("recordings", "timestamps", "medians"):
                    if not info[key]:
                        raw_reasons.append(f"missing_tactile_resource:{name}:{key}")
        for number, info in sorted(processed.get(scene, {}).items()):
            missing = [n for n in REQUIRED if n not in info["files"] or info["files"][n]["size"] == 0]
            reasons = list(raw_reasons)
            reasons += ["missing_required_file:" + n for n in missing]
            failures = request_failures.get(scene, [])
            status = "request_failed" if failures else ("unavailable" if reasons else "candidate")
            episodes.append({"episode": number, "directory": info["directory"], "download_ready": status == "candidate",
                             "audit_status": status, "request_errors": failures,
                             "reasons": reasons, "files": [info["files"][n] for n in REQUIRED if n in info["files"]]})
        scene_status = ("request_failed" if scene in request_failures else
                        "candidate" if any(e["download_ready"] for e in episodes) else "unavailable")
        result.append({"scene": scene, "raw_files": raw_files, "streams": streams, "episodes": episodes,
                       "audit_status": scene_status, "request_errors": request_failures.get(scene, []),
                       "reasons": raw_reasons + (["no_processed_episodes"] if not episodes and scene not in request_failures else []),
                       "note": "Episode counts and multiple median candidates do not exclude downloads"})
    return {"schema": "t4d-build-catalog-v1", "revisions": revisions, "repos": REPOS,
            "network": getattr(hub, "network_info", {"endpoint": OFFICIAL_ENDPOINT, "proxy_mode": "offline_fixture"}),
            "created_utc": datetime.now(timezone.utc).isoformat(), "scenes": result,
            "selection_errors": selection_errors,
            "content_verified": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--scenes", nargs="*", default=[])
    p.add_argument("--endpoint", default=MIRROR_ENDPOINT, choices=(OFFICIAL_ENDPOINT, MIRROR_ENDPOINT))
    p.add_argument("--proxy", default=None, help="Explicit proxy only; default is direct, ignoring environment proxies")
    args = p.parse_args()
    if args.output.exists():
        p.error("Refusing to overwrite an existing catalog")
    hub = Hub(endpoint=args.endpoint, proxy=args.proxy)
    revisions = {k: hub.pin(repo) for k, repo in REPOS.items()}
    catalog = build_catalog(hub, revisions, args.scenes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, catalog)


if __name__ == "__main__":
    main()
