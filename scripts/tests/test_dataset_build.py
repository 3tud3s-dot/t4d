"""Offline CPU tests. No real Hub calls, Slurm jobs, or dataset conversion."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
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
    def test_numbered_selection_ranges_lists_and_invalid_inputs(self):
        self.assertEqual(c.parse_scene_ids('001-020'), list(range(1, 21)))
        self.assertEqual(c.parse_scene_ids('020,001,008'), [1, 8, 20])
        self.assertEqual(c.parse_scene_ids([20, 1]), [1, 20])
        for value in ('', '000', '020-001', '001-003,003', '../a', [], [True], [1000], [1, 1], 'all'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                c.parse_scene_ids(value)

    def test_scene_id_resolution_rejects_ambiguity_and_reports_missing(self):
        self.assertEqual(c.resolve_scene_ids([1, 3], {'001-rope', '002-other'}),
                         (['001-rope'], [{'scene_id': 3, 'reason': 'scene_id_not_published'}]))
        with self.assertRaisesRegex(ValueError, 'multiple'):
            c.resolve_scene_ids([1], {'001-rope', '001-other'})

    def catalog_fixture(self, failures=None, missing_sensor=None):
        failures = failures or set()
        class FakeHub:
            calls = []
            def tree(inner, repo, revision, path, recursive=False):
                inner.calls.append(path)
                if path in ('raw', 'processed'):
                    return [{'type': 'directory', 'path': path+'/'+name} for name in ['001-a', '002-b', '003-c']]
                if path in failures:
                    raise urllib.error.URLError('fixture temporary outage')
                if path.startswith('processed/'):
                    return [entry(path+'/episode_7/'+name) for name in c.REQUIRED]
                return [entry(path+'/brics-odroid_'+sensor+'/'+name)
                        for sensor in c.STREAMS if sensor != missing_sensor
                        for name in ('brics-odroid_'+sensor+'_1.npy', 'brics-odroid_'+sensor+'_1.txt',
                                     'median_1.npy', 'median_2.npy')]
        return FakeHub()

    def test_ids_query_only_selected_scenes_and_keep_partial_episodes_and_medians(self):
        hub=self.catalog_fixture()
        model=c.build_catalog(hub, {'raw':'a'*40,'processed':'b'*40}, scene_ids='001-002')
        self.assertEqual([s['scene'] for s in model['scenes']], ['001-a','002-b'])
        self.assertFalse(any('003-c' in x for x in hub.calls))
        self.assertTrue(all(s['episodes'][0]['download_ready'] for s in model['scenes']))
        self.assertTrue(all(s['episodes'][0]['episode']==7 for s in model['scenes']))
        self.assertEqual(len(model['scenes'][0]['streams']['tactilel_left']['medians']),2)

    def test_request_failure_is_not_unavailable_and_other_scenes_are_audited(self):
        model=c.build_catalog(self.catalog_fixture({'processed/001-a'}),
                              {'raw':'a'*40,'processed':'b'*40},scene_ids=[1,2])
        self.assertEqual(model['scenes'][0]['audit_status'],'request_failed')
        self.assertEqual(model['scenes'][1]['audit_status'],'candidate')

    def test_missing_tactile_resource_is_definitive_but_raw_request_failure_is_not(self):
        rev={'raw':'a'*40,'processed':'b'*40}
        missing=c.build_catalog(self.catalog_fixture(missing_sensor='tactiler_right'),rev,scene_ids=[1])
        self.assertEqual(missing['scenes'][0]['audit_status'],'unavailable')
        self.assertTrue(any('tactiler_right' in r for r in missing['scenes'][0]['episodes'][0]['reasons']))
        outage=c.build_catalog(self.catalog_fixture({'raw/001-a'}),rev,scene_ids=[1])
        self.assertEqual(outage['scenes'][0]['audit_status'],'request_failed')
        self.assertEqual(outage['scenes'][0]['episodes'][0]['reasons'],[])

    def test_missing_scene_ids_do_not_fall_back_to_all_scenes(self):
        hub=self.catalog_fixture()
        model=c.build_catalog(hub, {'raw':'a'*40,'processed':'b'*40},scene_ids=[999])
        self.assertEqual(model['scenes'],[])
        self.assertEqual(model['selection_errors'][0]['scene_id'],999)
        self.assertEqual(hub.calls,['raw','processed'])

    def test_successful_scene_metadata_is_cached_and_failed_requests_are_retried(self):
        rev={'raw':'a'*40,'processed':'b'*40}
        with tempfile.TemporaryDirectory() as directory:
            cache=Path(directory)
            first=self.catalog_fixture({'processed/002-b'})
            c.build_catalog(first,rev,cache_dir=cache,scene_ids=[1,2])
            second=self.catalog_fixture()
            model=c.build_catalog(second,rev,cache_dir=cache,scene_ids=[1,2])
            self.assertNotIn('processed/001-a',second.calls)
            self.assertNotIn('raw/001-a',second.calls)
            self.assertIn('processed/002-b',second.calls)
            self.assertTrue(all(s['audit_status']=='candidate' for s in model['scenes']))
            with self.assertRaisesRegex(ValueError,'revision mismatch'):
                c.build_catalog(second,{**rev,'processed':'c'*40},cache_dir=cache,scene_ids=[1])

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

    def test_sha_and_git_blob_checked(self):
        path = self.root / "a"
        path.write_bytes(b"test")
        for lfs in (False, True):
            self.assertEqual(d.verify_file(path, record(lfs=lfs))["size"], 4)
        path.write_bytes(b"bad!")
        with self.assertRaises(ValueError):
            d.verify_file(path, record())

    def test_explicit_local_rejection_discard_is_owned_verified_and_reasoned(self):
        rec=record(); path=self.root/rec['path'];path.parent.mkdir(parents=True);path.write_bytes(b'test')
        self.loader.journal[rec['path']]={'record':rec,'owned':True,'status':'verified','actual':d.verify_file(path,rec)}
        with self.assertRaises(ValueError):
            self.loader.discard([rec],reason='')
        self.assertTrue(path.exists())
        self.loader.discard([rec],reason='Unsupported robot transform dimensions')
        self.assertFalse(path.exists())
        self.assertEqual(self.loader.journal[rec['path']]['cleanup_basis']['kind'],'local_input_rejection')
        self.loader.discard([rec],reason='Unsupported robot transform dimensions')

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
    def test_formal_config_limits_first_twenty_and_rejects_conflicting_selection(self):
        config,_=p.load_config(p.HERE/'build_config.json')
        self.assertEqual(c.parse_scene_ids(config['scene_ids']),list(range(1,21)))
        self.assertNotIn('scenes',config)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'config.json'
            config['scenes']=[];path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):p.load_config(path)

    def test_unavailable_and_network_failed_entries_never_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config=json.loads((p.HERE/'build_config.json').read_text())
            for audit,status in [('unavailable','unavailable_sources'),('request_failed','network_audit_failed')]:
                loader=unittest.mock.Mock(); state={'episodes':{}}
                p.process_episode('new',{'episode':0,'files':[],'download_ready':False,'audit_status':audit,'reasons':['fixture']},
                                  [],loader,{'source_root':root,'output_root':root/'output'},root,config,state,root/'rules',{})
                loader.fetch.assert_not_called();loader.cleanup.assert_not_called();loader.discard.assert_not_called()
                self.assertEqual(state['episodes']['new/episode_0']['status'],status)

    def test_explicit_rejection_is_recorded_before_owned_cleanup_and_resumes_safely(self):
        from scripts.rebuild_dataset import rebuild
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'logs').mkdir(); loader=unittest.mock.Mock();loader.missing_bytes.return_value=0
            config=json.loads((p.HERE/'build_config.json').read_text());state={'episodes':{}}
            small=record('processed/new/episode_7/metadata.json');large=record('processed/new/episode_7/splatfacto.tar')
            entry_={'episode':7,'files':[small,large],'download_ready':True}
            def discard(records,reason):
                evidence=json.loads((root/'state.json').read_text())['episodes']['new/episode_7']
                self.assertEqual(evidence['status'],'rejected_local');self.assertEqual(evidence['reason'],reason)
            loader.discard.side_effect=discard
            with patch.object(p,'available_bytes',return_value=10**15), \
                    patch.object(p,'prepare_episode_rules',side_effect=rebuild.DataRejected('bad transform dimensions')), \
                    patch.object(p,'run_command') as convert:
                args=('new',entry_,[],loader,{'source_root':root,'output_root':root/'output'},root,config,state,root/'rules',{})
                p.process_episode(*args);p.process_episode(*args)
            loader.fetch.assert_called_once_with(small);loader.discard.assert_called_once();convert.assert_not_called()
            self.assertEqual(state['episodes']['new/episode_7']['status'],'rejected_cleaned')

    def test_programming_or_infrastructure_errors_never_trigger_rejection_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);loader=unittest.mock.Mock();loader.missing_bytes.return_value=0
            config=json.loads((p.HERE/'build_config.json').read_text());state={'episodes':{}}
            with patch.object(p,'available_bytes',return_value=10**15), \
                    patch.object(p,'prepare_episode_rules',side_effect=RuntimeError('fixture infrastructure error')):
                with self.assertRaises(RuntimeError):
                    p.process_episode('new',{'episode':0,'files':[],'download_ready':True},[],loader,
                                      {'source_root':root,'output_root':root/'output'},root,config,state,root/'rules',{})
            loader.cleanup.assert_not_called();loader.discard.assert_not_called()

    def test_network_retry_uses_immutable_manifest_generations_and_keeps_completed_scenes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'run';run.mkdir();source=root/'source';source.mkdir()
            roots={'source_root':source,'output_root':root/'out'};roots['output_root'].mkdir()
            config=json.loads((p.HERE/'build_config.json').read_text())
            records=[record(f'processed/{name}/episode_0/splatfacto.tar') for name in ['001-rope','002-rope-silk']]
            first={'revisions':{'raw':'a'*40,'processed':'a'*40},'scenes':[
                {'scene':'001-rope','raw_files':[],'audit_status':'candidate','episodes':[{'episode':0,'files':[records[0]],'download_ready':True}]},
                {'scene':'002-rope-silk','raw_files':[],'audit_status':'request_failed','episodes':[]}]}
            second=deepcopy(first); second['scenes'][1].update(audit_status='candidate',episodes=[{'episode':0,'files':[records[1]],'download_ready':True}])
            loader=unittest.mock.Mock();loader.missing_bytes.return_value=4
            def command(args,log):
                if '--data-root' not in args:
                    name=args[args.index('--scene')+1];target=roots['output_root']/name/'episode_0';target.mkdir(parents=True);(target/'metadata.json').write_text('{}')
            with patch.object(p.Hub,'pin',return_value='a'*40),patch.object(p,'build_catalog',side_effect=[first,second]) as audit, \
                    patch.object(p,'Downloader',return_value=loader),patch.object(p,'available_bytes',return_value=10**15), \
                    patch.object(p,'prepare_episode_rules',return_value=(run/'scene_rules.json',{})),patch.object(p,'run_command',side_effect=command):
                self.assertEqual(p.execute_locked(config,roots,run),2)
                original=p.digest(run/'manifest.json')
                self.assertEqual(p.execute_locked(config,roots,run),0)
                self.assertEqual(p.digest(run/'manifest.json'),original)
                self.assertEqual(p.execute_locked(config,roots,run),0)
                self.assertEqual(audit.call_count,2)
            self.assertEqual([call.args[0] for call in loader.fetch.call_args_list],records)
            state=json.loads((run/'state.json').read_text());self.assertEqual(state['completed_scenes'],['001-rope','002-rope-silk'])
            provenance=json.loads((run/'provenance.json').read_text());self.assertEqual(provenance['manifest_file'],'manifest-0002.json')
            self.assertEqual(provenance['audit_history'][0]['sha256'],original)
            (run/provenance['manifest_file']).write_text('{}')
            with patch.object(p.Hub,'pin',return_value='a'*40),self.assertRaisesRegex(ValueError,'Pinned manifest changed'):
                p.execute_locked(config,roots,run)

    def test_explicit_episode_selection_never_expands_smoke_scope(self):
        model = {"scenes": [{"scene": "s", "episodes": [{"episode": 0}, {"episode": 7}]}]}
        selected = p.select_episodes(model, {"s": [7]})
        self.assertEqual(selected["scenes"][0]["episodes"], [{"episode": 7}])
        self.assertEqual(len(model["scenes"][0]["episodes"]), 2)
        with self.assertRaises(ValueError):
            p.select_episodes(model, {"s": [9]})

    def test_smoke_config_selects_one_episode_and_separate_paths(self):
        fixture = json.loads((p.HERE / "build_config.json").read_text())
        fixture.update(source_root="datasets/test-source", output_root="datasets/test-v1",
                       scene_ids="001", episodes={"001-rope": [0]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(fixture))
            config, roots = p.load_config(path)
        self.assertEqual(config["episodes"], {"001-rope": [0]})
        self.assertNotEqual(roots["source_root"], p.PROJECT / "datasets/deform360")
        self.assertNotEqual(roots["output_root"], p.PROJECT / "datasets/touch4d-v1")

    def test_mirror_config_is_ep2_direct_and_isolated(self):
        fixture = json.loads((p.HERE / "build_config.json").read_text())
        fixture.update(source_root="datasets/test-source", output_root="datasets/test-v1",
                       scene_ids="001", episodes={"001-rope": [2]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(fixture))
            config, roots = p.load_config(path)
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
                    patch.object(p, "prepare_episode_rules", return_value=(run / "scene_rules.json", {"binding_method": "fixture"})), \
                    patch.object(p, "run_command", side_effect=command) as commands:
                self.assertEqual(p.execute_locked(config, roots, run), 0)
                loader.fetch.assert_called_once_with(rec)
                self.assertEqual(p.execute_locked(config, roots, run), 0)
                loader.fetch.assert_called_once_with(rec)
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
                def fetch(self, rec):calls.append("download")
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
            (run / "rules").write_text("{}")
            with patch.object(p, "available_bytes", return_value=10**15), patch.object(p, "run_command", side_effect=command), \
                    patch.object(p, "prepare_episode_rules", return_value=(run / "rules", {})):
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
            (root / "rules").write_text("{}")
            with patch.object(p, "available_bytes", return_value=10**15), patch.object(p, "run_command", side_effect=RuntimeError("invalid source")), \
                    patch.object(p, "prepare_episode_rules", return_value=(root / "rules", {})):
                with self.assertRaises(RuntimeError):
                    p.process_episode("s", {"episode": 0, "download_ready": True, "files": []}, [], loader,
                                      roots, root, config, {"episodes": {}}, root / "rules", {"s": {0: ["right"]}})
            loader.cleanup.assert_not_called()

    def test_missing_binding_downloads_metadata_and_defers_large_archives(self):
        from scripts.rebuild_dataset import rebuild
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "logs").mkdir()
            loader = unittest.mock.Mock()
            loader.missing_bytes.return_value = 0
            roots = {"source_root": root, "output_root": root / "out"}
            config = json.loads((p.HERE / "build_config.json").read_text())
            state = {"episodes": {}}
            small = record("processed/new/episode_7/metadata.json")
            with patch.object(p, "available_bytes", return_value=10**15), patch.object(p, "run_command") as command, \
                    patch.object(p, "prepare_episode_rules", side_effect=rebuild.RulesPending("no documented identities")):
                p.process_episode("new", {"episode": 7, "download_ready": True, "files": [small, record()]}, [], loader,
                                  roots, root, config, state, root / "rules", {})
            loader.fetch.assert_called_once_with(small)
            loader.cleanup.assert_not_called()
            command.assert_not_called()
            self.assertEqual(state["episodes"]["new/episode_7"]["status"], "needs_rules")


class AutomaticRuleTests(unittest.TestCase):
    def setUp(self):
        from scripts.rebuild_dataset import rebuild
        self.r=rebuild
        self.saved=[deepcopy(getattr(rebuild,k)) for k in ('SCENE_BINDINGS','SCENE_MEDIANS','EXTERNAL_BINDING_EVIDENCE')]
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.config={'bindings':{},'medians':{},'evidence':{}}

    def tearDown(self):
        for key,value in zip(('SCENE_BINDINGS','SCENE_MEDIANS','EXTERNAL_BINDING_EVIDENCE'),self.saved):
            target=getattr(self.r,key);target.clear();target.update(value)

    def inputs(self,meta=None,robot_extra=None,count=1,episode=7,scene='002-rope-silk',baselines=1):
        np=self.r.np;directory=self.root/'processed'/scene/f'episode_{episode}'
        (directory/'robot').mkdir(parents=True,exist_ok=True)
        (directory/'metadata.json').write_text(json.dumps(meta or {}))
        transforms=np.broadcast_to(np.eye(4),(3,count,4,4)).copy()
        openings=np.full((3,count),.08)
        actions=np.zeros((3,count,5,3));actions[:,:,0,:]=transforms[:,:,:3,3]
        actions[:,:,1:4,:]=transforms[:,:,:3,:3];actions[:,:,4,0]=openings
        robot={'T_worlds':transforms[:,0] if count==1 else transforms,
               'openings':openings[:,0] if count==1 else openings,
               'actions':actions[:,0] if count==1 else actions,'bimanual':count==2,**(robot_extra or {})}
        np.save(directory/'robot/robot.npy',robot,allow_pickle=True)
        for sensor in self.r.STREAMS:
            path=self.root/'raw'/scene/('brics-odroid_'+sensor);path.mkdir(parents=True,exist_ok=True)
            for number in range(1,baselines+1):np.save(path/f'median_{number}.npy',np.zeros((16,32)))
        return directory,robot

    def resolve(self,**kwargs):
        return self.r.resolve_episode_rules(self.root,'002-rope-silk',kwargs.pop('episode',7),self.config,**kwargs)

    def test_documented_single_hand_mapping_and_source_arrays_are_unchanged(self):
        directory,_=self.inputs({'robot_slot_hands':['left']})
        before=(directory/'robot/robot.npy').read_bytes();rules,audit=self.resolve()
        self.assertEqual(rules['bindings']['002-rope-silk']['7'],['left'])
        self.assertEqual(audit['binding_method'],'documented_source_fields')
        self.assertEqual(before,(directory/'robot/robot.npy').read_bytes())
        self.assertEqual(self.config,{'bindings':{},'medians':{},'evidence':{}})
        path=self.root/'effective.json';path.write_text(json.dumps(rules));self.r.load_scene_rules(path)
        self.assertEqual(self.r.hand_slots('002-rope-silk',7),['left'])
        evidence=self.r.binding_evidence('002-rope-silk',7)
        self.assertEqual(evidence['robot_sha256'],hashlib.sha256(before).hexdigest())
        t,o,_=self.r.load_robot(before,3,['left']);self.r.load_actions(before,3,['left'],t,o)

    def test_documented_dual_hand_slot_mapping_supports_numpy_integer_indices(self):
        np=self.r.np
        self.inputs(robot_extra={'physical_hand_to_robot_slot':{'left':np.int64(1),'right':np.int64(0)}},count=2)
        rules,_=self.resolve();self.assertEqual(rules['bindings']['002-rope-silk']['7'],['right','left'])

    def test_hand_count_episode_number_and_generic_order_never_guess_identity(self):
        self.inputs({'hand_order':['right','left']},count=2)
        with self.assertRaisesRegex(self.r.RulesPending,'No documented'):
            self.resolve()

    def test_conflicting_or_incomplete_documented_fields_remain_pending(self):
        for meta,extra in [({'robot_slot_hands':['left']},{'robot_slot_hands':['right']}),
                           ({'robot_slot_hands':['unknown']},{}),
                           ({'physical_hand_to_robot_slot':{'left':True}},{}),
                           ({'physical_hand_to_robot_slot':{'left':0,'right':0}}, {})]:
            with self.subTest(meta=meta):
                self.inputs(meta,extra)
                with self.assertRaises(self.r.RulesPending):self.resolve()

    def test_multiple_baselines_need_documented_choice_not_first_filename(self):
        self.inputs({'robot_slot_hands':['right']},baselines=2)
        with self.assertRaisesRegex(self.r.RulesPending,'Multiple baselines'):self.resolve()
        choices={s:'median_2.npy' for s in self.r.STREAMS}
        self.inputs({'robot_slot_hands':['right'],'tactile_baselines':choices},baselines=2)
        rules,audit=self.resolve();self.assertEqual(rules['medians']['002-rope-silk'],choices)
        self.assertEqual(audit['baseline_method'],'explicit')

    def test_legacy_mapping_is_preserved_and_conflicting_metadata_is_not_ignored(self):
        self.inputs(scene='001-rope',episode=0)
        rules,audit=self.r.resolve_episode_rules(self.root,'001-rope',0,self.config)
        self.assertEqual(audit['robot_slot_hands'],['right']);self.assertEqual(audit['binding_method'],'legacy_reviewed')
        self.assertEqual(rules,self.config)
        self.inputs({'robot_slot_hands':['left']},scene='001-rope',episode=0)
        with self.assertRaisesRegex(self.r.RulesPending,'conflict'):
            self.r.resolve_episode_rules(self.root,'001-rope',0,self.config)

    def test_proven_format_defect_is_distinct_from_ambiguous_identity(self):
        directory,_=self.inputs({'robot_slot_hands':['right']})
        self.r.np.save(directory/'robot/robot.npy',{'T_worlds':self.r.np.zeros((3,3))})
        with self.assertRaises(self.r.DataRejected):self.resolve()

    def test_metadata_stage_can_resolve_before_large_or_tactile_files_are_downloaded(self):
        self.inputs({'robot_slot_hands':['right']},baselines=0)
        inventory={s:['median_1.npy'] for s in self.r.STREAMS}
        rules,audit=self.resolve(available_medians=inventory)
        self.assertEqual(audit['robot_slot_hands'],['right'])
        self.assertEqual(rules['bindings']['002-rope-silk']['7'],['right'])
        with self.assertRaisesRegex(self.r.DataRejected,'No baseline'):self.resolve()

    def test_episode_evidence_files_are_immutable_and_do_not_mutate_previous_episode(self):
        self.inputs({'robot_slot_hands':['right']},episode=7)
        self.inputs({'robot_slot_hands':['left']},episode=8)
        run=self.root/'run';run.mkdir();config=run/'configured.json';config.write_text(json.dumps(self.config))
        shared=[record(f'raw/002-rope-silk/brics-odroid_{s}/median_1.npy') for s in self.r.STREAMS]
        roots={'source_root':self.root}
        first,_=p.prepare_episode_rules('002-rope-silk',7,roots,run,config,shared);original=first.read_bytes()
        second,_=p.prepare_episode_rules('002-rope-silk',8,roots,run,config,shared)
        self.assertEqual(first.read_bytes(),original);self.assertNotEqual(first,second)
        self.r.load_scene_rules(first)
        self.assertEqual(self.r.hand_slots('002-rope-silk',7),['right'])
        self.assertEqual(self.r.binding_evidence('002-rope-silk',7)['episode'],7)
        bad=json.loads(first.read_text());bad['bindings']['002-rope-silk']['7']=['left'];first.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError,'Previously resolved'):
            p.prepare_episode_rules('002-rope-silk',7,roots,run,config,shared)


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
