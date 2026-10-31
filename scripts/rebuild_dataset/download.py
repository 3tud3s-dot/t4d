"""Bounded aria2c downloads, official identities, and owned-file cleanup."""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import threading
import urllib.parse

try:
    from .catalog import REPOS, OFFICIAL_ENDPOINT, normalize_endpoint, resolve_proxy, safe_relative, save_json
except ImportError:
    from catalog import REPOS, OFFICIAL_ENDPOINT, normalize_endpoint, resolve_proxy, safe_relative, save_json


class SpaceBlocked(RuntimeError):
    pass


def bounded_path(root, relative):
    root = Path(root).absolute()
    safe_relative(relative)
    path = root / relative
    for parent in (root, *root.parents):
        if parent.is_symlink():
            raise ValueError("Linked source root is forbidden")
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError("Linked input or output is forbidden")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Path escapes managed root")
    return path


def identities(path):
    path = Path(path)
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha256.update(block)
            blob.update(block)
    return {"size": size, "sha256": sha256.hexdigest(), "git_oid": blob.hexdigest()}


def verify_file(path, record):
    if not Path(path).is_file():
        raise ValueError("Downloaded file is missing")
    actual = identities(path)
    if actual["size"] != record["size"]:
        raise ValueError("Downloaded file size mismatch")
    if record.get("sha256"):
        valid = actual["sha256"] == record["sha256"]
    elif record.get("git_oid"):
        valid = actual["git_oid"] == record["git_oid"]
    else:
        raise ValueError("Official content identity is required")
    if not valid:
        raise ValueError("Downloaded file content identity mismatch")
    return actual


def available_bytes(root, quota_limit_gib):
    free = shutil.disk_usage(root).free
    # df reports shared filesystem space, not this user's quota.
    command = subprocess.run(["quota", "-w", "-v"], capture_output=True, text=True, check=False, timeout=30)
    if command.returncode != 0:
        raise SpaceBlocked("Cannot inspect personal quota")
    usages = []
    pending_device = None
    target_device = Path(root).stat().st_dev
    mounts = []
    for line in Path("/proc/mounts").read_text().splitlines():
        fields = line.split()
        try:
            if Path(fields[1]).stat().st_dev == target_device:
                mounts.append(fields[0])
        except OSError:
            continue
    for line in command.stdout.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0].startswith("/"):
            pending_device = fields.pop(0)
        if pending_device in mounts and len(fields) >= 3:
            try:
                used, soft, hard = [int(x.rstrip("*")) * 1024 for x in fields[:3]]
            except ValueError:
                continue
            limits = [x for x in (soft, hard, int(quota_limit_gib * 2**30)) if x > 0]
            usages.append(min(limits) - used)
            pending_device = None
    if not usages:
        raise SpaceBlocked("No matching personal quota entry; refusing to rely on shared df space")
    return max(0, min(free, *usages))


def check_space(free_bytes, download_bytes, output_bytes, reserve_bytes):
    required = int(download_bytes * 1.5) + output_bytes + reserve_bytes
    if free_bytes < required:
        raise SpaceBlocked(f"Insufficient quota: need {required} bytes, available {free_bytes}")
    return required


class Downloader:
    def __init__(self, root, journal_path, log_path, config):
        self.root, self.journal_path, self.log_path = Path(root), Path(journal_path), Path(log_path)
        self.config = config
        self.concurrent_files = config.get("concurrent_files", 1)
        if type(self.concurrent_files) is not int or not 1 <= self.concurrent_files <= 8:
            raise ValueError("Concurrent files must be an integer between 1 and 8")
        self._journal_lock = threading.RLock()
        self._operation_lock = threading.RLock()
        self.endpoint = normalize_endpoint(config.get("endpoint", OFFICIAL_ENDPOINT))
        self.proxy = resolve_proxy(config["proxy"])
        self.network_info = {"endpoint": self.endpoint, "proxy_mode": "direct" if self.proxy is None else "explicit_proxy",
                             "proxy_fallback": False, "aria2_config_disabled": True}
        import json
        self.journal = json.loads(self.journal_path.read_text(encoding="utf-8")) if self.journal_path.exists() else {}

    def save(self):
        with self._journal_lock:
            save_json(self.journal_path, self.journal)

    def fetch_many(self, records):
        """One admitted Episode only; never prefetch additional Episodes.

        Persist each file's ownership before starting its transfer. Stop scheduling
        after a failure and join in-flight transfers before returning to cleanup
        or recovery. A single journal writer prevents lost ownership records.
        """
        with self._operation_lock:
            unique = {}
            for record in records:
                if record["path"] in unique and unique[record["path"]] != record:
                    raise ValueError("Conflicting identities in download batch")
                unique[record["path"]] = record
            ordered = list(unique.values())
            if self.concurrent_files == 1:
                return [self._fetch(r) for r in ordered]
            # Start long transfers early; small files can use remaining slots.
            ordered.sort(key=lambda r: r["size"], reverse=True)
            iterator = iter(ordered)
            with ThreadPoolExecutor(max_workers=self.concurrent_files) as pool:
                pending = {}
                for _ in range(min(self.concurrent_files, len(ordered))):
                    record = next(iterator)
                    pending[pool.submit(self._fetch, record)] = record["path"]
                paths = {}
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    try:
                        # Observe all completed errors before scheduling more.
                        for future in done:
                            paths[pending.pop(future)] = future.result()
                    except BaseException:
                        for future in pending:
                            future.cancel()
                        raise
                    for _ in done:
                        record = next(iterator, None)
                        if record is not None:
                            pending[pool.submit(self._fetch, record)] = record["path"]
            return [paths[r["path"]] for r in unique.values()]

    def missing_bytes(self, records):
        # Deliberately conservative: do not assume a partial file is valid.
        return sum(r["size"] for r in records if self.journal.get(r["path"], {}).get("status") != "verified"
                   or not bounded_path(self.root, r["path"]).is_file())

    def fetch(self, record):
        with self._operation_lock:
            return self._fetch(record)

    def _fetch(self, record):
        path = bounded_path(self.root, record["path"])
        if record["repo"] not in REPOS.values() or not re.fullmatch(r"[0-9a-f]{40}", record["revision"]):
            raise ValueError("Unpinned or unknown download source")
        previous = self.journal.get(record["path"])
        if previous and previous["record"] != record:
            raise ValueError("Input identity changed inside a run")
        if path.exists() and not previous:
            actual = verify_file(path, record)
            with self._journal_lock:
                self.journal[record["path"]] = {"record": record, "owned": False, "status": "verified", "actual": actual,
                                                 "network": {"mode": "preexisting_file"}}
                self.save()
            return path
        if path.exists() and previous and previous["status"] == "verified":
            verify_file(path, record)
            return path
        if not previous:
            if path.with_name(path.name + ".aria2").exists():
                raise ValueError("Unowned partial download exists")
            with self._journal_lock:
                self.journal[record["path"]] = {"record": record, "owned": True, "status": "downloading", "network": self.network_info}
                self.save()  # Persist ownership before creating the file.
        elif previous["status"] in ("deleted", "delete_pending"):
            raise ValueError("Unexpected request for a previously cleaned input")
        path.parent.mkdir(parents=True, exist_ok=True)
        proxy = self.proxy
        args = ["aria2c", "--no-conf=true", "--continue=true", "--check-integrity=true", "--auto-file-renaming=false",
                "--allow-overwrite=false", "--max-tries=1", "--retry-wait=3", "--connect-timeout=30",
                "--timeout=60", "--file-allocation=none", "--max-connection-per-server=8", "--split=8",
                "--min-split-size=5M", "--summary-interval=30", "--console-log-level=notice",
                "--dir=" + str(path.parent), "--out=" + path.name]
        args.append("--all-proxy=" + (proxy or ""))
        if record.get("sha256"):
            args.append("--checksum=sha-256=" + record["sha256"])
        args.append(f"{self.endpoint}/datasets/{record['repo']}/resolve/{record['revision']}/"
                    + urllib.parse.quote(record["path"], safe="/"))
        env = {k: v for k, v in os.environ.items() if k.lower() not in ("all_proxy", "http_proxy", "https_proxy")}
        env["HF_HUB_DISABLE_XET"] = "1"
        for attempt in range(self.config["attempts"]):
            with self.log_path.open("a", encoding="utf-8") as log:
                log.write(f"\nDownload {record['path']} attempt {attempt + 1}\n")
                log.write("Network policy: " + str(self.network_info) + "\n")
                log.flush()
                result = subprocess.run(args, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            if result.returncode == 0:
                if path.with_name(path.name + ".aria2").exists():
                    raise ValueError("aria2 control file remains after reported completion")
                actual = verify_file(path, record)
                with self._journal_lock:
                    self.journal[record["path"]].update(status="verified", actual=actual)
                    self.save()
                return path
            if result.returncode in (3, 9, 24) or attempt + 1 == self.config["attempts"]:
                raise RuntimeError(f"aria2 failed: {record['path']}; exit={result.returncode}")
            time.sleep(3)
        raise RuntimeError("Download failed")

    def cleanup(self, records, *, validation_passed):
        with self._operation_lock:
            return self._cleanup(records, validation_passed=validation_passed)

    def _cleanup(self, records, *, validation_passed):
        if not validation_passed:
            raise ValueError("Cleanup requires successful V1 validation")
        for record in records:
            item = self.journal.get(record["path"])
            if not item or not item["owned"]:
                continue  # Never delete pre-existing or another run's inputs.
            if item["record"] != record:
                raise ValueError("Cleanup identity mismatch")
            path = bounded_path(self.root, record["path"])
            if item["status"] == "deleted":
                if path.exists():
                    raise ValueError("A file appeared after cleanup")
                continue
            if not path.exists() and item["status"] == "delete_pending":
                item["status"] = "deleted"
                self.save()
                continue
            if item["status"] not in ("verified", "delete_pending"):
                raise ValueError("Cannot clean an incomplete download")
            actual = verify_file(path, record)
            if actual != item["actual"]:
                raise ValueError("Input changed since validation")
            item["status"] = "delete_pending"
            self.save()
            path.unlink()  # Enumerated, owned regular files only; no recursive deletion.
            item["status"] = "deleted"
            self.save()
