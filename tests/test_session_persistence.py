from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import studio_server


def test_session_persists_across_f5_download_and_server_restart_until_manual_delete(
    tmp_path: Path, monkeypatch
) -> None:
    """
    Uploading media, marking masks, rendering, downloading (cleanup=0), and restarting the
    server (simulating F5 + server reload) MUST preserve the entire session on disk until
    the user explicitly calls /cleanup ("Xóa phiên").
    """
    sessions_dir = tmp_path / "sessions"
    store1 = studio_server.SessionStore(base_dir=sessions_dir, ttl_seconds=0)
    monkeypatch.setattr(studio_server, "STORE", store1)
    client1 = TestClient(studio_server.create_app())

    sid = client1.post("/matting/api/session").json()["session_id"]

    # Upload source + background PNGs
    src = np.full((120, 120, 3), (40, 165, 45), dtype=np.uint8)
    src[25:95, 30:90] = (120, 155, 210)
    bg = np.full((120, 120, 3), (175, 170, 160), dtype=np.uint8)

    for role, arr, fn in (("source", src, "demo_src.png"), ("background", bg, "demo_bg.png")):
        raw = cv2.imencode(".png", arr)[1].tobytes()
        uid = client1.post(
            f"/matting/api/session/{sid}/upload/init",
            json={"role": role, "filename": fn, "total_size": len(raw)},
        ).json()["upload_id"]
        assert client1.put(f"/matting/api/session/{sid}/upload/{uid}/chunk?index=0", content=raw).status_code == 200
        assert client1.post(f"/matting/api/session/{sid}/upload/{uid}/complete").status_code == 200

    # Save confirmed mask on frame 0
    sess1 = store1.get(sid)
    mask0 = np.zeros((120, 120), dtype=np.uint8)
    mask0[25:95, 30:90] = 1
    studio_server._save_confirmed_mask(sess1, 0, mask0, push_undo=False)

    # Simulate completed MP4 output in session jobs
    out_mp4 = sess1.work_dir / "AGY_TachNen_Final.mp4"
    out_mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"0" * 1024)
    with sess1.lock:
        sess1.jobs["job1234567"] = {
            "job_id": "job1234567",
            "status": "done",
            "progress": 1.0,
            "message": "Hoàn tất 1 frames (0.01 MB).",
            "output_path": str(out_mp4),
            "output_mb": 0.01,
            "disk_mb": 0.1,
        }
    store1.save_meta(sess1)

    # Download with cleanup=0 MUST keep session intact!
    dl0 = client1.get(f"/matting/api/session/{sid}/download?kind=video&job_id=job1234567&cleanup=0")
    assert dl0.status_code == 200
    assert len(dl0.content) == out_mp4.stat().st_size
    assert sess1.work_dir.exists()

    # Simulate browser F5 + full server restart with empty in-memory SessionStore
    store2 = studio_server.SessionStore(base_dir=sessions_dir, ttl_seconds=0)
    monkeypatch.setattr(studio_server, "STORE", store2)
    client2 = TestClient(studio_server.create_app())

    st = client2.get(f"/matting/api/session/{sid}/state")
    assert st.status_code == 200, st.text
    j = st.json()
    assert j["session_id"] == sid
    assert j["src_meta"]["filename"] == "demo_src.png"
    assert j["src_meta"]["frame_count"] == 1
    assert j["bg_meta"]["filename"] == "demo_bg.png"
    assert j["confirmed_frames"] == [0]
    assert j["active_job"]["job_id"] == "job1234567"
    assert j["active_job"]["status"] == "done"
    assert "cleanup=0" in j["active_job"]["download_url"]

    # Re-download after restart STILL works!
    dl1 = client2.get(f"/matting/api/session/{sid}/download?kind=video&job_id=job1234567&cleanup=0")
    assert dl1.status_code == 200
    assert len(dl1.content) == out_mp4.stat().st_size

    # Preview STILL works right after F5/restart without re-uploading!
    prev = client2.post(
        f"/matting/api/session/{sid}/preview",
        json={"frame_idx": 0, "matting_engine": "classic", "use_pymatting_fg": False},
    )
    assert prev.status_code == 200, prev.text

    # ONLY explicit manual cleanup deletes the session
    cl = client2.post(f"/matting/api/session/{sid}/cleanup")
    assert cl.status_code == 200
    assert cl.json()["cleaned"] is True
    assert not (sessions_dir / sid).exists()
    assert client2.get(f"/matting/api/session/{sid}/state").status_code == 404


def test_session_not_auto_reaped_when_ttl_disabled(tmp_path: Path) -> None:
    store = studio_server.SessionStore(base_dir=tmp_path / "sessions", ttl_seconds=0)
    sess = store.create()
    sess.last_active = time.time() - 999_999
    old_ts = time.time() - 999_999
    os.utime(sess.work_dir, (old_ts, old_ts))

    swept = store.sweep_expired()
    assert swept == 0
    assert sess.work_dir.exists()


def test_frontend_preserves_session_on_f5_and_manual_delete_only() -> None:
    app_js = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    index_html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")

    # Must NEVER auto-delete session on F5 / pagehide
    assert "sendBeacon" not in app_js
    assert "pagehide" not in app_js
    # Must persist & restore session across F5 via localStorage + /state endpoint
    assert "localStorage" in app_js
    assert "/state" in app_js
    # Download button must keep session (cleanup=0) so only manual "Xóa phiên" deletes it
    assert "cleanup=0" in app_js
    assert "cleanup=1" not in app_js
    assert "Xóa phiên" in index_html
