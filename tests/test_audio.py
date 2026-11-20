"""冲击检测与 WAV 校验测试。"""
import io
import wave

import numpy as np
import pytest

from app.audio import (
    AudioValidationError,
    detect_impacts,
    find_weak_peaks,
    parse_wav,
    validate_reference,
)
from conftest import SR, build_wav


def test_detect_five_impacts():
    times = [0.5, 1.0, 1.5, 2.0, 2.5]
    samples = parse_wav(build_wav(times, duration=3.0, noise_floor=0.002))
    det = detect_impacts(samples)
    assert len(det.times) == 5
    for got, want in zip(det.times, times):
        assert got == pytest.approx(want, abs=0.02)


def test_loudness_change_kept_same_count():
    """整体响度变化不应改变检测到的冲次数量。"""
    times = [0.4, 0.8, 1.2, 1.6]
    quiet = parse_wav(build_wav(times, gain=0.08, noise_floor=0.0005))
    loud = parse_wav(build_wav(times, gain=0.9, noise_floor=0.005))
    assert len(detect_impacts(quiet).times) == 4
    assert len(detect_impacts(loud).times) == 4


def test_echo_within_40ms_suppressed():
    times = [0.5, 1.0, 1.5, 2.0]
    wav = build_wav(times, echoes={0.5: 0.035, 1.5: 0.03}, duration=2.5)
    det = detect_impacts(parse_wav(wav))
    assert len(det.times) == 4
    assert len(det.suppressed_echoes) == 2
    for got, want in zip(det.times, times):
        assert got == pytest.approx(want, abs=0.015)


def test_second_impulse_at_50ms_is_detected():
    """40ms 之外的独立冲击应保留（多冲要能被发现）。"""
    times = [0.5, 0.55, 1.0, 1.5, 2.0]
    det = detect_impacts(parse_wav(build_wav(times, duration=2.5)))
    assert len(det.times) == 5


def test_masked_weak_impact_found_by_weak_scan():
    """重击抬高自适应门限时，检测漏掉的轻冲应能被弱峰补查找回。"""
    times = [0.5, 1.0, 1.5, 2.0]
    wav = build_wav(times, duration=2.6, gains={1.5: 0.08})
    samples = parse_wav(wav)
    det = detect_impacts(samples)
    assert not any(t == pytest.approx(1.5, abs=0.03) for t in det.times)  # 常规检测漏掉轻冲
    assert len(det.times) == 3

    weak = find_weak_peaks(
        samples, 0.0, det.duration,
        exclude_times=list(det.times) + list(det.suppressed_echoes),
    )
    assert any(t == pytest.approx(1.5, abs=0.02) for t in weak)


def test_weak_scan_excludes_confirmed_echoes():
    """弱峰补查沿用 40ms 去回声：已确认冲次与其回声都不再作为弱峰。"""
    times = [0.5, 1.0, 1.5, 2.0]
    wav = build_wav(
        times, duration=2.6,
        echoes={1.5: 0.035}, echo_gain=0.5,
    )
    samples = parse_wav(wav)
    det = detect_impacts(samples)
    assert len(det.suppressed_echoes) == 1
    weak = find_weak_peaks(
        samples, 0.0, det.duration,
        exclude_times=list(det.times) + list(det.suppressed_echoes),
    )
    assert weak == []


def test_weak_scan_ignores_background_ripple():
    """没有重击遮蔽时，普通背景起伏不应被当成弱峰。"""
    times = [0.5, 1.0, 1.5, 2.0]
    samples = parse_wav(
        build_wav(times, duration=2.6, gain=0.4, noise_floor=0.02)
    )
    det = detect_impacts(samples)
    assert len(det.times) == 4
    weak = find_weak_peaks(
        samples, 0.0, det.duration,
        exclude_times=list(det.times) + list(det.suppressed_echoes),
    )
    assert weak == []


def test_validate_reference_bounds():
    ok = detect_impacts(
        parse_wav(build_wav([0.2, 0.6, 1.0, 1.4], duration=1.8))
    )
    validate_reference(ok)  # 4 次、间隔 400ms，合法

    too_few = detect_impacts(
        parse_wav(build_wav([0.2, 0.6, 1.0], duration=1.5))
    )
    with pytest.raises(AudioValidationError):
        validate_reference(too_few)

    too_close = detect_impacts(
        parse_wav(build_wav([0.2, 0.3, 0.6, 1.0, 1.4], duration=1.8))
    )
    with pytest.raises(AudioValidationError):
        validate_reference(too_close)


def _rewrap(pcm: bytes, *, nch, sw, fr, nframes):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(nch)
        wf.setsampwidth(sw)
        wf.setframerate(fr)
        wf.writeframes(pcm)
    return buf.getvalue()


def test_reject_wrong_format():
    base = build_wav([0.2, 0.6, 1.0, 1.4], duration=1.8)
    with wave.open(io.BytesIO(base), "rb") as wf:
        pcm = wf.readframes(wf.getnframes())

    with pytest.raises(AudioValidationError):
        parse_wav(_rewrap(pcm, nch=2, sw=2, fr=SR, nframes=len(pcm) // 4))
    with pytest.raises(AudioValidationError):
        parse_wav(_rewrap(pcm, nch=1, sw=1, fr=SR, nframes=len(pcm) // 2))
    with pytest.raises(AudioValidationError):
        parse_wav(_rewrap(pcm, nch=1, sw=2, fr=8000, nframes=len(pcm) // 2))
    with pytest.raises(AudioValidationError):
        parse_wav(b"not a wav file")


def test_reject_duration_over_8s():
    # 9 秒静音 WAV
    pcm = np.zeros(9 * SR, dtype="<i2").tobytes()
    with pytest.raises(AudioValidationError):
        parse_wav(_rewrap(pcm, nch=1, sw=2, fr=SR, nframes=9 * SR))


def test_silence_detects_nothing():
    pcm = np.zeros(2 * SR, dtype="<i2").tobytes()
    det = detect_impacts(parse_wav(_rewrap(pcm, nch=1, sw=2, fr=SR, nframes=2 * SR)))
    assert det.times == []
