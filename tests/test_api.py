"""端到端 API 关键流程测试。"""
import pytest

from conftest import build_wav, post_inspect


REF_TIMES = [0.5, 1.0, 1.5, 2.0]


def _ref():
    return build_wav(REF_TIMES, duration=2.6, noise_floor=0.002)


def test_health_and_empty_result(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    assert client.get("/api/result/last").status_code == 404


def test_full_normal_cycle(client):
    test = build_wav([1.1, 1.6, 2.1, 2.6], duration=3.2)  # 带起始静音
    resp = post_inspect(client, _ref(), test)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["verdict"] == "ok"
    assert body["summary"]["miss_count"] == 0
    assert body["summary"]["extra_count"] == 0
    # 结果已持久化，可再次获取
    last = client.get("/api/result/last").json()
    assert last["id"] == body["id"]


def test_full_missed_impact(client):
    test = build_wav([0.5, 1.0, 2.0], duration=2.6)  # 漏第 3 冲
    body = post_inspect(client, _ref(), test).json()
    assert body["verdict"] == "defect"
    assert body["summary"]["miss_count"] == 1
    missed = body["missed"][0]
    assert missed["ref_index"] == 2
    # 后续冲次未错位：第 4 个参考冲次仍匹配待检最后一个
    tail = [m for m in body["matches"] if m["ref_index"] == 3][0]
    assert tail["test_index"] == 2


def test_full_extra_impact_and_echo(client):
    test = build_wav(
        [0.5, 1.0, 1.25, 1.5, 2.0],
        duration=2.6,
        echoes={0.5: 0.035},  # 35ms 回声，不应算多冲
    )
    body = post_inspect(client, _ref(), test).json()
    assert body["summary"]["extra_count"] == 1
    assert body["summary"]["suppressed_echo_count"] == 1
    assert body["extras"][0]["time"] == pytest.approx(1.25, abs=0.015)


def test_tempo_minus_12_percent_accepted(client):
    t = [0.5 + (x - 0.5) * 0.88 for x in REF_TIMES]
    body = post_inspect(client, _ref(), build_wav(t, duration=2.4)).json()
    assert body["verdict"] == "ok"


def test_ambiguous_requires_manual_review(client):
    test = build_wav([0.5, 1.0, 1.5, 2.0, 2.5], duration=3.0)
    body = post_inspect(client, _ref(), test).json()
    assert body["ambiguous"] is True
    assert body["verdict"] == "ambiguous"
    assert any("人工复听" in w for w in body["warnings"])


def test_invalid_audio_does_not_overwrite_last_result(client):
    good = post_inspect(client, _ref(), build_wav(REF_TIMES, duration=2.6))
    assert good.status_code == 200
    good_id = good.json()["id"]

    # 错误采样率
    bad = build_wav(REF_TIMES, sr=8000, duration=2.6)
    resp = post_inspect(client, _ref(), bad)
    assert resp.status_code == 400
    assert "采样率" in resp.json()["detail"]

    # 参考冲次不足 4 次
    bad_ref = build_wav([0.5, 1.0, 1.5], duration=2.0)
    resp2 = post_inspect(client, bad_ref, build_wav(REF_TIMES, duration=2.6))
    assert resp2.status_code == 400
    assert "参考录音" in resp2.json()["detail"]

    # 非 WAV
    resp3 = client.post(
        "/api/inspect",
        files={
            "reference": ("ref.wav", _ref(), "audio/wav"),
            "test": ("test.txt", b"hello", "text/plain"),
        },
    )
    assert resp3.status_code == 400

    # 上一次结果仍在
    last = client.get("/api/result/last").json()
    assert last["id"] == good_id


def test_silent_test_all_missed(client):
    import io
    import wave
    import numpy as np

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.zeros(2 * 16000, dtype="<i2").tobytes())
    body = post_inspect(client, _ref(), buf.getvalue()).json()
    assert body["summary"]["test_impact_count"] == 0
    assert body["summary"]["miss_count"] == 4
    assert any("未检测到任何冲击" in w for w in body["warnings"])


def test_uploaded_audio_served_back(client):
    body = post_inspect(client, _ref(), build_wav(REF_TIMES, duration=2.6)).json()
    resp = client.get(body["test"]["url"])
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"


def test_index_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "冲压机" in resp.text


def test_masked_weak_impact_recovered_as_matched(client):
    """重击遮蔽轻冲：常规检测漏掉，补查后按匹配处理，不再报漏冲。"""
    test = build_wav(
        REF_TIMES, duration=2.6, noise_floor=0.002, gains={1.5: 0.06}
    )
    body = post_inspect(client, _ref(), test).json()
    assert body["summary"]["detected_impact_count"] == 3
    assert body["summary"]["recovered_count"] == 1
    assert body["verdict"] == "ok"
    assert body["summary"]["miss_count"] == 0
    assert body["missed"] == []
    rec = body["recovered"][0]
    assert rec["ref_index"] == 2
    assert rec["time"] == pytest.approx(1.5, abs=0.02)
    # 补回冲次进入待检冲次序列并标记为已匹配
    recovered_imp = [i for i in body["test"]["impacts"] if i["is_recovered"]]
    assert len(recovered_imp) == 1
    assert recovered_imp[0]["matched"] is True
    assert any("补回" in w for w in body["warnings"])


def test_masked_weak_impact_echo_not_counted_as_extra(client):
    """轻冲被重击遮蔽且带 35ms 回声：补回轻冲，回声仍不计多冲。"""
    test = build_wav(
        REF_TIMES, duration=2.6,
        gains={1.5: 0.06}, extra_echoes=[(1.5, 0.035, 0.08)],
    )
    body = post_inspect(client, _ref(), test).json()
    assert body["summary"]["recovered_count"] == 1
    assert body["summary"]["extra_count"] == 0
    assert body["verdict"] == "ok"


def test_recovery_tie_keeps_original_verdict_and_asks_review(client):
    """补回归属并列：该段保留原判定（漏冲仍在），并提示人工复听。"""
    # 重击 0.5/1.5/2.0 遮蔽了两个轻冲 0.95/1.05：二者都只能算 ±15% 内的
    # 同一参考冲次，补回数并列且归属不同
    test = build_wav(
        [0.5, 0.95, 1.05, 1.5, 2.0],
        duration=2.4,
        gains={0.95: 0.06, 1.05: 0.06},
    )
    body = post_inspect(client, _ref(), test).json()
    assert body["summary"]["recovered_count"] == 0
    assert body["summary"]["miss_count"] == 1
    assert body["summary"]["extra_count"] == 0  # 未采用的弱峰不计多冲
    assert body["review_required"] is True
    assert len(body["review_segments"]) == 1
    seg = body["review_segments"][0]
    assert seg["start"] == pytest.approx(0.5, abs=0.02)
    assert seg["end"] == pytest.approx(1.5, abs=0.02)
    assert any("人工复听" in w for w in body["warnings"])
