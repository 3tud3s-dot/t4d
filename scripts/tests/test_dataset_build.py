"""Offline CPU tests. No real Hub calls, Slurm jobs, or dataset conversion."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

from scripts.rebuild_dataset import catalog as c, download as d, pipeline as p


def entry(path, payload=b"test", lfs=False):
    item = {"path": path, "type": "file", "size": len(payload),
            "oid": hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()}
    if lfs:
        item["lfs"] = {"oid": hashlib.sha256(payload).hexdigest()}
    return item


def record(path="processed/001-rope/episode_0/splatfacto.tar", payload=b"test", lfs=False):
    return c.file_record(entry(path, payload, lfs), c.REPOS["processed"], "a" * 40)


class CatalogTests(unittest.TestCase):
    def test_pin_requests_revision_only_not_full_repository_file_list(self):
        hub = c.Hub(endpoint=c.MIRROR_ENDPOINT)
        with patch.object(hub, "get", return_value=({"sha": "a" * 40}, "")) as get:
            self.assertEqual(hub.pin(c.REPOS["raw"]), "a" * 40)
            self.assertIn("expand%5B%5D=sha", get.call_args.args[0])

    def test_mirror_get_ignores_environment_proxy_and_uses_selected_endpoint(self):
        real_builder = urllib.request.build_opener
        with patch.dict(c.os.environ, {"HTTP_PROXY": "http://vpn.invalid:9", "HTTPS_PROXY": "http://vpn.invalid:9", "ALL_PROXY": "socks5://vpn.invalid:9"}), \
                patch.object(c.urllib.request, "build_opener", side_effect=real_builder) as builder:
            hub = c.Hub(endpoint=c.MIRROR_ENDPOINT, proxy=None)
            self.assertEqual(builder.call_args.args[0].proxies, {})
        response = io.BytesIO(b'{"sha":"fixture"}')
        response.headers = {}
        with patch.object(hub.opener, "open", return_value=response) as opened, \
                patch.object(c.urllib.request, "urlopen", side_effect=AssertionError("global opener forbidden")):
            self.assertEqual(hub.get(c.OFFICIAL_ENDPOINT + "/api/datasets/brownu/deform360")[0], {"sha": "fixture"})
            self.assertEqual(opened.call_args[0][0].full_url, c.MIRROR_ENDPOINT + "/api/datasets/brownu/deform360")
        self.assertEqual(hub.network_info["proxy_mode"], "direct")
        self.assertFalse(hub.network_info["proxy_fallback"])

    def test_official_pagination_link_stays_on_mirror(self):
        hub = c.Hub(endpoint=c.MIRROR_ENDPOINT)
        next_url = c.OFFICIAL_ENDPOINT + "/api/datasets/r/tree/sha/raw?cursor=next"
        with patch.object(hub, "get", side_effect=[([entry("raw/a")], f'<{next_url}>; rel="next"'), ([entry("raw/b")], "")]) as get:
            self.assertEqual(len(hub.tree("r", "sha", "raw")), 2)
            self.assertTrue(get.call_args_list[1].args[0].startswith(c.MIRROR_ENDPOINT + "/api/"))

    def test_metadata_redirect_cannot_downgrade_or_escape(self):
        redirect = c.MetadataRedirect(c.MIRROR_ENDPOINT)
        request = urllib.request.Request(c.MIRROR_ENDPOINT + "/api/datasets/r")
        redirected = redirect.redirect_request(request, None, 302, "Found", {}, c.OFFICIAL_ENDPOINT + "/api/datasets/r?cursor=next")
        self.assertTrue(redirected.full_url.startswith(c.MIRROR_ENDPOINT))
        for url in ("http://hf-mirror.com/api/datasets/r", "https://evil.invalid/api/datasets/r", c.MIRROR_ENDPOINT + "/datasets/r/file.tar"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                redirect.redirect_request(request, None, 302, "Found", {}, url)

    def test_mirror_403_retry_is_bounded_and_does_not_fallback(self):
        hub = c.Hub(attempts=2, endpoint=c.MIRROR_ENDPOINT, proxy=None)
        denied = urllib.error.HTTPError(c.MIRROR_ENDPOINT + "/api/datasets/r", 403, "Forbidden", {}, None)
        with patch.object(hub.opener, "open", side_effect=denied) as opened, patch.object(c.time, "sleep"):
            with self.assertRaises(urllib.error.HTTPError):
                hub.get(c.OFFICIAL_ENDPOINT + "/api/datasets/r")
            self.assertEqual(opened.call_count, 2)
            self.assertTrue(all(x.args[0].full_url.startswith(c.MIRROR_ENDPOINT) for x in opened.call_args_list))

    def test_selected_scene_catalog_does_not_query_other_object_payloads(self):
        calls = []
        class FakeHub:
            def tree(self, repo, rev, path, recursive=False):
                calls.append((path, recursive))
                if path in ("raw", "processed"):
                    return [{"type": "directory", "path": path + "/001-rope"}, {"type": "directory", "path": path + "/other"}]
                if path == "processed/001-rope":
                    return [entry("processed/001-rope/episode_2/" + n) for n in c.REQUIRED]
                if path == "raw/001-rope":
                    return []
                raise AssertionError("Unexpected scene query")
        data = c.build_catalog(FakeHub(), {"raw": "a" * 40, "processed": "b" * 40}, ["001-rope"])
        self.assertEqual(data["scenes"][0]["episodes"][0]["episode"], 2)
        self.assertNotIn(("processed", True), calls)
        self.assertFalse(any("/other" in path for path, _ in calls))

    def test_partial_episode_set_is_kept(self):
        files = [entry("processed/001-rope/episode_7/" + name) for name in c.REQUIRED]
        raw = [entry(f"raw/001-rope/brics-odroid_{s}/{name}") for s in c.STREAMS
               for name in (f"brics-odroid_{s}_1.npy", f"brics-odroid_{s}_1.txt", "median_1.npy", "median_2.npy")]
        class FakeHub:
            def tree(self, repo, rev, path, recursive=False):
                if path == "raw":
                    return [{"type": "directory", "path": "raw/001-rope"}]
                return files if path == "processed" else raw
        result = c.build_catalog(FakeHub(), {"raw": "a" * 40, "processed": "b" * 40})
        scene = result["scenes"][0]
        self.assertEqual([e["episode"] for e in scene["episodes"]], [7])
        self.assertTrue(scene["episodes"][0]["download_ready"])
        self.assertEqual(len(scene["streams"]["tactilel_left"]["medians"]), 2)
        self.assertFalse(result["content_verified"])

    def test_missing_required_file_is_explicit(self):
        files = [entry("processed/001-rope/episode_0/metadata.json")]
        class FakeHub:
            def tree(self, repo, rev, path, recursive=False):
                return [] if path == "raw" else files
        result = c.build_catalog(FakeHub(), {"raw": "a" * 40, "processed": "b" * 40})
        item = result["scenes"][0]["episodes"][0]
        self.assertFalse(item["download_ready"])
        self.assertIn("raw_scene_missing", item["reasons"])
        self.assertIn("missing_required_file:splatfacto.tar", item["reasons"])

    def test_numeric_episode_aliases_rejected(self):
        with self.assertRaises(ValueError):
            c.index_processed([entry("processed/s/episode_0/a"), entry("processed/s/episode_0000/b")], "a" * 40)

    def test_all_pages_are_used(self):
        next_url = "https://huggingface.co/api/datasets/r/tree/sha/raw?cursor=next"
        h = c.Hub()
        with patch.object(h, "get", side_effect=[([entry("raw/a")], f'<{next_url}>; rel="next"'), ([entry("raw/b")], "")]):
            self.assertEqual(len(h.tree("r", "sha", "raw")), 2)

    def test_repeated_page_rejected(self):
        url = "https://huggingface.co/api/datasets/r/tree/sha/raw?limit=1000"
        with patch.object(c.Hub, "get", return_value=([], f'<{url}>; rel="next"')):
            with self.assertRaises(ValueError):
                c.Hub().tree("r", "sha", "raw")

    def test_path_escape_or_duplicate_rejected(self):
        for rows in ([entry("raw/a"), entry("raw/a")], [entry("raw/../a")], [entry("other/a")]):
            with self.subTest(rows=rows), patch.object(c.Hub, "get", return_value=(rows, "")):
                with self.assertRaises(ValueError):
                    c.Hub().tree("r", "sha", "raw")

    def test_only_metadata_api_allowed(self):
        for url in ("https://other.test/api/datasets/r", "http://huggingface.co/api/datasets/r",
                    "https://huggingface.co/datasets/r/resolve/main/a.tar"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                c.Hub().get(url)

    def test_lfs_and_git_blob_identity_are_distinct(self):
        item = entry("raw/a", lfs=True)
        result = c.file_record(item, "r", "a" * 40)
        self.assertIsNone(result["git_oid"])
        self.assertEqual(result["sha256"], item["lfs"]["oid"])
        self.assertIsNotNone(c.file_record(entry("raw/a"), "r", "a" * 40)["git_oid"])

    def test_invalid_or_missing_content_identity_rejected(self):
        for item in ({"path": "raw/a", "size": 1}, {**entry("raw/a"), "lfs": {"oid": "wrong"}}):
            with self.subTest(item=item), self.assertRaises(ValueError):
                c.file_record(item, "r", "a" * 40)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.loader = d.Downloader(self.root, self.root / "ledger.json", self.root / "download.log", {"proxy": None, "attempts": 3})

    def batch_loader(self, concurrent=2):
        return d.Downloader(self.root, self.root / "batch-ledger.json", self.root / "batch-download.log",
                            {"endpoint": c.MIRROR_ENDPOINT, "proxy": None, "attempts": 1,
                             "concurrent_files": concurrent})

    def test_batch_parallel_limit_and_atomic_ownership_journal(self):
        loader = self.batch_loader()
        records = [record(f"raw/s/file_{i}.npy", lfs=True) for i in range(8)]
        barrier = threading.Barrier(2)
        guard = threading.Lock()
        active = peak = 0
        def transfer(args, **kwargs):
            nonlocal active, peak
            name = next(a[6:] for a in args if a.startswith("--out="))
            directory = Path(next(a[6:] for a in args if a.startswith("--dir=")))
            with guard:
                active += 1
                peak = max(peak, active)
            # Ownership must be durably recorded before each transfer starts.
            with loader._journal_lock:
                journal = json.loads(loader.journal_path.read_text())
            self.assertTrue(journal['raw/s/' + name]['owned'])
            barrier.wait(timeout=5)
            (directory / name).write_bytes(b'test')
            with guard:
                active -= 1
            return SimpleNamespace(returncode=0)
        with patch.object(d.subprocess, 'run', side_effect=transfer) as transfer_mock:
            result = loader.fetch_many(records)
        self.assertEqual(peak, 2)
        self.assertEqual(transfer_mock.call_count, 8)
        self.assertEqual(result, [self.root / r['path'] for r in records])
        journal = json.loads(loader.journal_path.read_text())
        self.assertEqual(len(journal), 8)
        self.assertTrue(all(v['status']=='verified' for v in journal.values()))

    def test_batch_failure_joins_active_transfer_and_stops_new_work(self):
        loader = self.batch_loader()
        records = [record(f'raw/s/{i}.npy') for i in range(4)]
        active = threading.Event()
        failed = threading.Event()
        release = threading.Event()
        errors = []
        started = []
        def transfer(rec):
            started.append(rec['path'])
            if rec == records[0]:
                self.assertTrue(active.wait(5))
                failed.set()
                raise RuntimeError('injected network error')
            self.assertEqual(rec, records[1])
            active.set()
            self.assertTrue(release.wait(5))
            return self.root / rec['path']
        def batch():
            try:
                loader.fetch_many(records)
            except Exception as exc:
                errors.append(exc)
        with patch.object(loader, '_fetch', side_effect=transfer):
            worker = threading.Thread(target=batch)
            worker.start()
            try:
                self.assertTrue(failed.wait(5))
                self.assertTrue(worker.is_alive())
            finally:
                release.set()
                worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertEqual(set(started), {r['path'] for r in records[:2]})

    def test_batch_deduplicates_identical_paths_and_rejects_conflict_before_transfer(self):
        loader = self.batch_loader()
        rec = record()
        with patch.object(loader, '_fetch', return_value=self.root / rec['path']) as fetch:
            self.assertEqual(len(loader.fetch_many([rec, deepcopy(rec)])), 1)
            fetch.assert_called_once_with(rec)
        other = {**rec, 'size': rec['size'] + 1}
        with patch.object(loader, '_fetch') as fetch:
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                loader.fetch_many([rec, other])
            fetch.assert_not_called()
        self.assertEqual(loader.fetch_many([]), [])

    def test_batch_single_worker_preserves_original_file_order(self):
        loader = self.batch_loader(1)
        records = [record(f'raw/s/{i}.npy', b'x'*i) for i in [1, 3, 2]]
        with patch.object(loader, '_fetch', side_effect=lambda r:self.root/r['path']) as fetch:
            loader.fetch_many(records)
        self.assertEqual([call.args[0] for call in fetch.call_args_list], records)

    def test_batch_resume_verified_files_never_redownloads(self):
        loader = self.batch_loader()
        records = [record(f'raw/s/{i}.npy') for i in range(3)]
        def transfer(args, **kwargs):
            directory=Path(next(a[6:] for a in args if a.startswith('--dir=')))
            name=next(a[6:] for a in args if a.startswith('--out='))
            (directory/name).write_bytes(b'test')
            return SimpleNamespace(returncode=0)
        with patch.object(d.subprocess, 'run', side_effect=transfer):
            loader.fetch_many(records)
        resumed=self.batch_loader()
        with patch.object(d.subprocess, 'run') as transfer_mock:
            resumed.fetch_many(records)
        transfer_mock.assert_not_called()
        self.assertTrue(all(v['owned'] for v in resumed.journal.values()))

    def test_batch_corrupt_file_retained_and_never_marked_verified(self):
        loader=self.batch_loader()
        rec=record(lfs=True)
        def transfer(args, **kwargs):
            (self.root/rec['path']).write_bytes(b'bad!')
            return SimpleNamespace(returncode=0)
        with patch.object(d.subprocess, 'run', side_effect=transfer):
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                loader.fetch_many([rec])
        self.assertEqual(loader.journal[rec['path']]['status'], 'downloading')
        self.assertTrue((self.root/rec['path']).exists())

    def test_batch_preexisting_files_still_not_owned_or_deleted(self):
        loader=self.batch_loader()
        records=[record(f'raw/s/{i}.npy') for i in range(3)]
        for rec in records:
            path=self.root/rec['path']; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'test')
        with patch.object(d.subprocess,'run') as transfer_mock:
            loader.fetch_many(records)
        transfer_mock.assert_not_called()
        loader.cleanup(records,validation_passed=True)
        self.assertTrue(all((self.root/r['path']).exists() for r in records))

    def test_invalid_concurrency_rejected_by_downloader_and_config(self):
        for value in [0, 9, True, 2.5, '4']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.batch_loader(value)
            config=json.loads((p.HERE/'build_config.json').read_text())
            config['download']['concurrent_files']=value
            file=self.root/'config.json'; file.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'Concurrent'):
                p.load_config(file)

    def test_sha_and_git_blob_checked(self):
        path = self.root / "a"
        path.write_bytes(b"test")
        for lfs in (False, True):
            self.assertEqual(d.verify_file(path, record(lfs=lfs))["size"], 4)
        path.write_bytes(b"bad!")
        with self.assertRaises(ValueError):
            d.verify_file(path, record())

    def test_mirror_download_disables_all_proxy_inheritance_and_aria_config(self):
        rec = record(lfs=True)
        config = {"endpoint": c.MIRROR_ENDPOINT, "proxy": None, "attempts": 2}
        loader = d.Downloader(self.root, self.root / "mirror-ledger.json", self.root / "mirror-download.log", config)
        def download(args, **kwargs):
            self.assertIn("--no-conf=true", args)
            self.assertIn("--all-proxy=", args)
            self.assertTrue(args[-1].startswith(c.MIRROR_ENDPOINT + "/datasets/"))
            self.assertFalse(any(k.lower() in ("http_proxy", "https_proxy", "all_proxy") for k in kwargs["env"]))
            (self.root / rec["path"]).write_bytes(b"test")
            return SimpleNamespace(returncode=0)
        with patch.dict(d.os.environ, {"http_proxy": "http://vpn.invalid:9", "HTTPS_PROXY": "http://vpn.invalid:9", "all_proxy": "socks5://vpn.invalid:9"}), \
                patch.object(d.subprocess, "run", side_effect=download):
            loader.fetch(rec)
        network = loader.journal[rec["path"]]["network"]
        self.assertEqual(network["proxy_mode"], "direct")
        self.assertFalse(network["proxy_fallback"])
        self.assertTrue(network["aria2_config_disabled"])

    def test_mirror_download_failure_never_tries_vpn_or_official_endpoint(self):
        loader = d.Downloader(self.root, self.root / "mirror-ledger.json", self.root / "mirror-download.log",
                              {"endpoint": c.MIRROR_ENDPOINT, "proxy": None, "attempts": 2})
        with patch.object(d.subprocess, "run", return_value=SimpleNamespace(returncode=6)) as run, patch.object(d.time, "sleep"):
            with self.assertRaises(RuntimeError):
                loader.fetch(record())
            self.assertEqual(run.call_count, 2)
            self.assertTrue(all(x.args[0][-1].startswith(c.MIRROR_ENDPOINT) and "--all-proxy=" in x.args[0] for x in run.call_args_list))

    def test_quota_includes_conversion_and_reserve(self):
        self.assertEqual(d.check_space(300, 100, 100, 20), 270)
        with self.assertRaises(d.SpaceBlocked):
            d.check_space(269, 100, 100, 20)

    def test_personal_quota_limits_shared_filesystem_space(self):
        stdout = '/dev/test 1000 2000 2500 0 0 0\n'
        mounts = f'/dev/test {self.root.as_posix()} ext4 rw 0 0\n'
        with patch.object(d.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=stdout)), \
                patch.object(Path, "read_text", return_value=mounts), \
                patch.object(d.shutil, "disk_usage", return_value=SimpleNamespace(free=10**15)):
            self.assertEqual(d.available_bytes(self.root, 1000), 1000 * 1024)

    def test_unreadable_quota_blocks_instead_of_trusting_df(self):
        with patch.object(d.subprocess, "run", return_value=SimpleNamespace(returncode=1)):
            with self.assertRaises(d.SpaceBlocked):
                d.available_bytes(self.root, 1000)

    def test_verified_but_missing_file_is_budgeted_again(self):
        rec = record()
        self.loader.journal[rec["path"]] = {"status": "verified"}
        self.assertEqual(self.loader.missing_bytes([rec]), rec["size"])

    def test_preexisting_matching_file_not_owned_or_deleted(self):
        rec = record()
        path = d.bounded_path(self.root, rec["path"])
        path.parent.mkdir(parents=True)
        path.write_bytes(b"test")
        self.loader.fetch(rec)
        self.assertFalse(self.loader.journal[rec["path"]]["owned"])
        self.loader.cleanup([rec], validation_passed=True)
        self.assertTrue(path.exists())

    def test_unowned_partial_not_touched(self):
        rec = record()
        path = d.bounded_path(self.root, rec["path"])
        path.parent.mkdir(parents=True)
        path.with_name(path.name + ".aria2").write_text("partial")
        with self.assertRaises(ValueError):
            self.loader.fetch(rec)

    def test_real_aria_command_has_eight_connections_and_checksum(self):
        rec = record(lfs=True)
        def fake_download(args, **kwargs):
            path = self.root / rec["path"]
            path.write_bytes(b"test")
            self.assertIn("--split=8", args)
            self.assertIn("--max-connection-per-server=8", args)
            self.assertIn("--checksum=sha-256=" + rec["sha256"], args)
            self.assertIn("--auto-file-renaming=false", args)
            self.assertEqual(kwargs["env"]["HF_HUB_DISABLE_XET"], "1")
            return SimpleNamespace(returncode=0)
        with patch.object(d.subprocess, "run", side_effect=fake_download):
            self.loader.fetch(rec)
        self.assertTrue(self.loader.journal[rec["path"]]["owned"])
        self.assertEqual(self.loader.journal[rec["path"]]["status"], "verified")

    def test_no_cleanup_before_validation(self):
        with self.assertRaises(ValueError):
            self.loader.cleanup([], validation_passed=False)

    def test_verified_cleanup_is_idempotent_and_journaled(self):
        rec = record()
        path = self.root / rec["path"]
        path.parent.mkdir(parents=True)
        path.write_bytes(b"test")
        self.loader.journal[rec["path"]] = {"record": rec, "owned": True, "status": "verified", "actual": d.verify_file(path, rec)}
        self.loader.save()
        self.loader.cleanup([rec], validation_passed=True)
        self.assertFalse(path.exists())
        self.loader.cleanup([rec], validation_passed=True)
        self.assertEqual(self.loader.journal[rec["path"]]["status"], "deleted")

    def test_cleanup_detects_changed_file(self):
        rec = record()
        path = self.root / rec["path"]
        path.parent.mkdir(parents=True)
        path.write_bytes(b"test")
        self.loader.journal[rec["path"]] = {"record": rec, "owned": True, "status": "verified", "actual": d.verify_file(path, rec)}
        path.write_bytes(b"bad!")
        with self.assertRaises(ValueError):
            self.loader.cleanup([rec], validation_passed=True)
        self.assertTrue(path.exists())

    def test_resume_after_unlink_before_journal_update(self):
        rec = record()
        self.loader.journal[rec["path"]] = {"record": rec, "owned": True, "status": "delete_pending"}
        self.loader.cleanup([rec], validation_passed=True)
        self.assertEqual(self.loader.journal[rec["path"]]["status"], "deleted")

    def test_network_retries_are_bounded(self):
        with patch.object(d.subprocess, "run", return_value=SimpleNamespace(returncode=6)) as mocked, patch.object(d.time, "sleep"):
            with self.assertRaises(RuntimeError):
                self.loader.fetch(record())
            self.assertEqual(mocked.call_count, 3)

    def test_not_found_and_disk_full_fail_without_retry(self):
        for code in (3, 9, 24):
            with self.subTest(code=code), patch.object(d.subprocess, "run", return_value=SimpleNamespace(returncode=code)) as mocked:
                with self.assertRaises(RuntimeError):
                    self.loader.fetch(record())
                self.assertEqual(mocked.call_count, 1)

    def test_escape_paths_rejected(self):
        for path in ("../a", "/a", "a/../../b", "a\\b", "a//b", "C:/a"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                d.bounded_path(self.root, path)


class PipelineTests(unittest.TestCase):
    def test_explicit_episode_selection_never_expands_smoke_scope(self):
        model = {"scenes": [{"scene": "s", "episodes": [{"episode": 0}, {"episode": 7}]}]}
        selected = p.select_episodes(model, {"s": [7]})
        self.assertEqual(selected["scenes"][0]["episodes"], [{"episode": 7}])
        self.assertEqual(len(model["scenes"][0]["episodes"]), 2)
        with self.assertRaises(ValueError):
            p.select_episodes(model, {"s": [9]})

    def test_smoke_config_selects_one_episode_and_separate_paths(self):
        config, roots = p.load_config(p.HERE / "smoke_config.json")
        self.assertEqual(config["episodes"], {"001-rope": [0]})
        self.assertNotEqual(roots["source_root"], p.PROJECT / "datasets/deform360")
        self.assertNotEqual(roots["output_root"], p.PROJECT / "datasets/touch4d-v1")

    def test_mirror_config_is_ep2_direct_and_isolated(self):
        config, roots = p.load_config(p.HERE / "mirror_smoke_config.json")
        self.assertEqual(config["episodes"], {"001-rope": [2]})
        self.assertEqual(config["download"]["endpoint"], c.MIRROR_ENDPOINT)
        self.assertIsNone(config["download"]["proxy"])
        self.assertNotEqual(roots["source_root"], p.PROJECT / "datasets/deform360-smoke")
        self.assertNotEqual(roots["output_root"], p.PROJECT / "datasets/touch4d-v1-smoke")

    def test_invalid_mirror_or_credentials_rejected(self):
        for endpoint in ("http://hf-mirror.com", "https://hf-mirror.com.evil.invalid", "https://user:secret@hf-mirror.com", "https://hf-mirror.com/path"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                c.normalize_endpoint(endpoint)

    def test_batch_entry_explicitly_requests_no_gpu(self):
        self.assertIn("#SBATCH --gres=none", (p.HERE / "build.sbatch").read_text())

    def test_source_snapshot_rejects_unrecorded_or_changed_code(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / ".source_version.json"
            sha = "a" * 40
            files = {}
            for src in p.HERE.iterdir():
                if src.suffix in (".py", ".json", ".sbatch"):
                    dst = root / "scripts/rebuild_dataset" / src.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(src.read_bytes())
                    files[dst.relative_to(root).as_posix()] = p.digest(dst)
            manifest.write_text(json.dumps({"commit": sha, "files": files}))
            with patch.dict(p.os.environ, {"T4D_SOURCE_SHA": sha}):
                self.assertEqual(p.verify_source_snapshot(root), sha)
                (root / "scripts/rebuild_dataset/pipeline.py").write_text("changed")
                with self.assertRaisesRegex(ValueError, "differs"):
                    p.verify_source_snapshot(root)

    def test_offline_run_and_resume_do_not_redownload_verified_episodes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "run"
            run.mkdir()
            roots = {"source_root": root / "source", "output_root": root / "out"}
            for path in roots.values():
                path.mkdir()
            config = json.loads((p.HERE / "build_config.json").read_text())
            rec = record()
            model = {"revisions": {"raw": "a" * 40, "processed": "a" * 40},
                     "scenes": [{"scene": "001-rope", "raw_files": [],
                                 "episodes": [{"episode": 0, "download_ready": True, "files": [rec]}]}]}
            loader = unittest.mock.Mock()
            loader.missing_bytes.return_value = 4
            target = roots["output_root"] / "001-rope/episode_0"
            def command(args, log):
                if "--data-root" not in args:
                    target.mkdir(parents=True)
                    (target / "metadata.json").write_text("{}")
            with patch.object(p.Hub, "pin", return_value="a" * 40), patch.object(p, "build_catalog", return_value=model), \
                    patch.object(p, "Downloader", return_value=loader), patch.object(p, "available_bytes", return_value=10**15), \
                    patch.object(p, "run_command", side_effect=command) as commands:
                self.assertEqual(p.execute_locked(config, roots, run), 0)
                loader.fetch_many.assert_called_once_with([rec])
                self.assertEqual(p.execute_locked(config, roots, run), 0)
                loader.fetch_many.assert_called_once_with([rec])
                self.assertIn("--structure-only", commands.call_args[0][0])
            state = json.loads((run / "state.json").read_text())
            self.assertEqual(state["status"], "completed")
            self.assertEqual(len(state["attempts"]), 2)
            self.assertEqual(state["summary"], {"cleaned": 1})
            snapshot = json.loads((run / "config.json").read_text())
            snapshot["code_sha256"] = {"wrong": "version"}
            (run / "config.json").write_text(json.dumps(snapshot))
            with self.assertRaisesRegex(ValueError, "Resume rejected"):
                p.execute_locked(config, roots, run)

    def test_default_plan_has_no_network_or_file_creation(self):
        before = set(p.PROJECT.rglob("*"))
        with patch.object(p, "execute", side_effect=AssertionError("must not execute")), patch.object(c.Hub, "get", side_effect=AssertionError("no network")), patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(p.main(["--run-id", "test-plan-only"]), 0)
        self.assertEqual(before, set(p.PROJECT.rglob("*")))
        self.assertIn('"mode": "plan_only"', out.getvalue())

    def test_windows_or_non_slurm_execution_refused_before_writes(self):
        with patch.dict(p.os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                p.execute({}, {}, "x")

    def test_conversion_commands_keep_horizon_out_and_target_separate(self):
        roots = {"source_root": Path("input"), "output_root": Path("output")}
        command = p.converter_command("s", 7, roots, Path("rules"), validate=True, structure_only=True)
        self.assertIn("--episodes", command)
        self.assertIn("--scene-rules", command)
        self.assertIn("--structure-only", command)
        self.assertNotIn("--overwrite", command)
        self.assertNotIn("--depth-only", command)

    def test_config_rejects_overlapping_roots(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = json.loads((p.HERE / "build_config.json").read_text())
            config["output_root"] = config["source_root"] + "/output"
            path = root / "config.json"
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                p.load_config(path, root)

    def test_config_rejects_underestimated_output_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            config = json.loads((p.HERE / "build_config.json").read_text())
            config["space"]["output_to_splat_tar_multiplier"] = .2
            path = Path(temp) / "config.json"
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                p.load_config(path, Path(temp))

    def test_episode_full_validation_precedes_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "run"
            (run / "logs").mkdir(parents=True)
            source = root / "source"
            source.mkdir()
            roots = {"source_root": source, "output_root": root / "output"}
            target = roots["output_root"] / "001-rope/episode_0"
            state = {"episodes": {}}
            calls = []
            class FakeLoader:
                def missing_bytes(self, records):return 4
                def fetch_many(self, records):calls.append("download")
                def cleanup(self, records, validation_passed):
                    self_test.assertTrue(validation_passed)
                    self_test.assertIn("full_validate", calls)
                    calls.append("cleanup")
            self_test = self
            def command(args, log):
                if "--data-root" in args:
                    calls.append("full_validate")
                else:
                    calls.append("convert")
                    target.mkdir(parents=True)
                    (target / "metadata.json").write_text("{}")
            config = json.loads((p.HERE / "build_config.json").read_text())
            item = {"episode": 0, "download_ready": True, "files": [record()]}
            with patch.object(p, "available_bytes", return_value=10**15), patch.object(p, "run_command", side_effect=command):
                p.process_episode("001-rope", item, [], FakeLoader(), roots, run, config, state, root / "rules", {"001-rope": {0: ("right",)}})
            self.assertEqual(calls, ["download", "convert", "full_validate", "cleanup"])
            self.assertEqual(state["episodes"]["001-rope/episode_0"]["status"], "cleaned")
            self.assertTrue(state["episodes"]["001-rope/episode_0"]["full_source_validation"])

    def test_failed_validation_never_cleans(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "logs").mkdir()
            loader = unittest.mock.Mock()
            loader.missing_bytes.return_value = 0
            roots = {"source_root": root, "output_root": root / "out"}
            config = json.loads((p.HERE / "build_config.json").read_text())
            with patch.object(p, "available_bytes", return_value=10**15), patch.object(p, "run_command", side_effect=RuntimeError("invalid source")):
                with self.assertRaises(RuntimeError):
                    p.process_episode("s", {"episode": 0, "download_ready": True, "files": []}, [], loader,
                                      roots, root, config, {"episodes": {}}, root / "rules", {"s": {0: ["right"]}})
            loader.cleanup.assert_not_called()

    def test_missing_binding_still_allows_download_and_preserves_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "logs").mkdir()
            loader = unittest.mock.Mock()
            loader.missing_bytes.return_value = 0
            roots = {"source_root": root, "output_root": root / "out"}
            config = json.loads((p.HERE / "build_config.json").read_text())
            state = {"episodes": {}}
            with patch.object(p, "available_bytes", return_value=10**15), patch.object(p, "run_command") as command:
                p.process_episode("new", {"episode": 7, "download_ready": True, "files": [record()]}, [], loader,
                                  roots, root, config, state, root / "rules", {})
            loader.fetch_many.assert_called_once_with([record()])
            loader.cleanup.assert_not_called()
            command.assert_not_called()
            self.assertEqual(state["episodes"]["new/episode_7"]["status"], "needs_rules")


class SceneRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts.rebuild_dataset import rebuild
        cls.rebuild = rebuild

    def setUp(self):
        self.tables = [deepcopy(getattr(self.rebuild, key)) for key in ("SCENE_BINDINGS", "SCENE_MEDIANS", "EXTERNAL_BINDING_EVIDENCE")]

    def tearDown(self):
        for key, value in zip(("SCENE_BINDINGS", "SCENE_MEDIANS", "EXTERNAL_BINDING_EVIDENCE"), self.tables):
            getattr(self.rebuild, key).clear()
            getattr(self.rebuild, key).update(value)

    def apply(self, value):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rules.json"
            path.write_text(json.dumps(value))
            self.rebuild.load_scene_rules(path)

    def test_empty_rules_preserve_legacy_tables(self):
        self.apply({"bindings": {}, "medians": {}, "evidence": {}})
        self.assertEqual(self.rebuild.SCENE_BINDINGS, self.tables[0])
        self.assertEqual(self.rebuild.SCENE_MEDIANS, self.tables[1])

    def test_explicit_subset_bindings_and_evidence(self):
        self.apply({"bindings": {"002-rope-silk": {"7": ["left", "right"]}}, "medians": {},
                    "evidence": {"002-rope-silk": {"method": "test fixture reviewed mapping"}}})
        self.assertEqual(self.rebuild.hand_slots("002-rope-silk", 7), ["left", "right"])
        self.assertEqual(self.rebuild.binding_evidence("002-rope-silk", 7)["method"], "test fixture reviewed mapping")
        with self.assertRaises(ValueError):
            self.rebuild.hand_slots("002-rope-silk", 0)

    def test_unknown_hands_or_missing_evidence_rejected_atomically(self):
        for hands in (["left", "left"], ["unknown"], []):
            with self.subTest(hands=hands), self.assertRaises(ValueError):
                self.apply({"bindings": {"new": {"0": hands}}, "medians": {}, "evidence": {"new": {"method": "fixture"}}})
        with self.assertRaises(ValueError):
            self.apply({"bindings": {"new": {"0": ["right"]}}, "medians": {}, "evidence": {}})
        self.assertNotIn("new", self.rebuild.SCENE_BINDINGS)

    def test_multiple_baselines_do_not_silently_select_first(self):
        with tempfile.TemporaryDirectory() as temp:
            import numpy as np
            for stream in self.rebuild.STREAMS:
                directory = Path(temp) / "raw/001-rope" / ("brics-odroid_" + stream)
                directory.mkdir(parents=True)
                (directory / "median_1.npy").write_bytes(b"fixture")
                (directory / "median_2.npy").write_bytes(b"fixture")
            with self.assertRaisesRegex(ValueError, "unambiguous baseline"):
                self.rebuild.find_raw_sources(Path(temp), "001-rope", np.array([1, 2]))


if __name__ == "__main__":
    unittest.main()
