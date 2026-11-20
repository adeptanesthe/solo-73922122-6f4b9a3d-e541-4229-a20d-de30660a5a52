"""节奏匹配动态规划测试。"""
from app.matching import TIMING_TOLERANCE, align_impacts


def test_perfect_match():
    r = [0.5, 1.0, 1.5, 2.0]
    t = [1.2, 1.7, 2.2, 2.7]  # 起始静音平移，节奏完全一致
    res = align_impacts(r, t)
    assert res.miss_count == 0
    assert res.extra_count == 0
    assert res.matched == [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert res.ambiguous is False


def test_miss_does_not_shift_later_impacts():
    """一次漏冲后，后续冲次必须仍按参考序号对齐，不错位。"""
    r = [0.5, 1.0, 1.5, 2.0, 2.5]
    t = [0.5, 1.0, 2.0, 2.5]  # 第 3 冲（1.5s）漏冲
    res = align_impacts(r, t)
    assert res.miss_count == 1
    assert res.extra_count == 0
    assert res.missed_ref_indices == [2]
    # 后续冲次没有整体错位
    assert res.matched == [(0, 0), (1, 1), (3, 2), (4, 3)]


def test_double_miss():
    r = [0.5, 1.0, 1.5, 2.0, 2.5]
    t = [0.5, 1.0, 2.5]  # 连漏两冲
    res = align_impacts(r, t)
    assert res.missed_ref_indices == [2, 3]
    assert res.matched == [(0, 0), (1, 1), (4, 2)]


def test_extra_impact_reported_with_time():
    r = [0.5, 1.0, 1.5, 2.0]
    t = [0.5, 1.0, 1.25, 1.5, 2.0]  # 1.25s 多冲
    res = align_impacts(r, t)
    assert res.miss_count == 0
    assert res.extra_count == 1
    assert res.extra_test_indices == [2]


def test_leading_silence_extra_is_extra():
    r = [0.5, 1.0, 1.5, 2.0]
    t = [0.1, 0.5, 1.0, 1.5, 2.0]  # 开头多一个
    res = align_impacts(r, t)
    assert res.extra_test_indices == [0]
    assert res.matched[0] == (0, 1)


def test_tempo_change_within_15_percent_ok():
    r = [0.0, 0.5, 1.0, 1.5]
    scale = 1 + TIMING_TOLERANCE  # +15%
    t = [x * scale for x in r]
    res = align_impacts(r, t)
    assert res.miss_count == 0 and res.extra_count == 0

    t2 = [x * (1 - TIMING_TOLERANCE) for x in r]  # -15%
    res2 = align_impacts(r, t2)
    assert res2.miss_count == 0 and res2.extra_count == 0


def test_tempo_change_beyond_tolerance_forces_miss():
    r = [0.0, 0.5, 1.0, 1.5]
    t = [x * 1.30 for x in r]  # +30%，超出窗口
    res = align_impacts(r, t)
    # 无法整段匹配，最优解中必然出现漏冲/多冲
    assert res.miss_count + res.extra_count > 0


def test_ambiguous_equal_cost_alignments_flagged():
    """参考节奏均匀时，末尾多一个等间隔冲次可对应两种等价归属。"""
    r = [0.0, 0.5, 1.0, 1.5]
    t = [0.0, 0.5, 1.0, 1.5, 2.0]
    res = align_impacts(r, t)
    assert res.ambiguous is True
    assert res.optimal_way_count >= 2
    # 两种归属都是 1 次多冲，不应被当成正常
    assert res.extra_count + res.miss_count >= 1


def test_empty_test_all_missed():
    r = [0.5, 1.0, 1.5, 2.0]
    res = align_impacts(r, [])
    assert res.miss_count == 4
    assert res.matched == []
    assert res.ambiguous is False
    assert set(res.expected_missed_times) == {0, 1, 2, 3}


def test_expected_missed_time_estimation():
    r = [0.0, 0.5, 1.0, 1.5]
    t = [2.0, 2.5, 3.5]  # 参考第 3 冲漏，整体平移 2s、节奏不变
    res = align_impacts(r, t)
    assert res.missed_ref_indices == [2]
    # 漏冲预计时刻 ≈ 2.0 + 1.0 = 3.0
    assert res.expected_missed_times[2] == 3.0


# ---------------------------------------------------------------------------
# 重击遮蔽补回
# ---------------------------------------------------------------------------


def test_weak_peak_between_matches_recovered():
    """相邻两次已匹配之间的弱峰补回后，漏冲消失。"""
    r = [0.5, 1.0, 1.5, 2.0, 2.5]
    t = [0.5, 1.0, 2.0, 2.5]  # 第 1.5s 的轻冲被重击遮蔽，常规检测漏掉
    res = align_impacts(r, t, weak_times=[1.5])
    assert res.miss_count == 0
    assert res.extra_count == 0
    assert res.recovered == [(2, 1.5)]
    assert res.test_times == [0.5, 1.0, 1.5, 2.0, 2.5]
    assert res.review_segments == []
    assert res.matched == [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]


def test_recovered_weak_peak_must_keep_whole_segment_rhythm():
    """补回后从左匹配到右匹配的整段必须符合 ±15%，节奏不符则不补。"""
    r = [0.5, 1.0, 1.5, 2.0]
    t = [0.5, 2.0]  # 两次匹配间隔 1.5s = 参考 3 个间隔，允许中间漏冲
    # 1.45s 与右匹配 2.0s 相差 0.55s，相对参考 0.5s 偏差 10% 可以；
    # 1.2s 与左/右都无法同时满足窗口，不补回
    res = align_impacts(r, t, weak_times=[1.2])
    assert res.recovered == []
    assert res.missed_ref_indices == [1, 2]

    # 与左右锚点都能落在 ±15% 内的弱峰补回一个
    res_ok = align_impacts(r, t, weak_times=[1.45])
    assert res_ok.recovered == [(2, 1.45)]
    assert res_ok.missed_ref_indices == [1]


def test_recovery_only_between_matched_impacts():
    """首匹配之前 / 末匹配之后的弱峰不补回（无整段节奏约束）。"""
    r = [0.5, 1.0, 1.5, 2.0, 2.4]  # 末间隔稍短，保证前缀漏冲对齐唯一
    t = [1.5, 2.0, 2.4]  # 前两次参考冲次漏在首匹配之前
    res = align_impacts(r, t, weak_times=[0.5, 1.0])
    assert res.missed_ref_indices == [0, 1]
    assert res.recovered == []

    t2 = [0.5, 1.0, 1.5, 2.0]  # 末次参考冲次漏在末匹配之后
    r2 = [0.5, 1.0, 1.5, 2.0, 2.6]
    res2 = align_impacts(r2, t2, weak_times=[2.6])
    assert res2.missed_ref_indices == [4]
    assert res2.recovered == []


def test_recovery_tie_keeps_original_verdict_and_flags_review():
    """同一段内补回数并列但归属不同：保留原判定并提示复听，弱峰不计多冲。"""
    r = [0.5, 1.0, 1.5, 2.0]
    t = [0.5, 1.5, 2.0]  # 第 2 冲漏
    # 0.95 与 1.05 都可作为第 2 冲的补回，补回数相同、归属不同
    res = align_impacts(r, t, weak_times=[0.95, 1.05])
    assert res.recovered == []
    assert res.missed_ref_indices == [1]
    assert res.extra_count == 0  # 未采用的弱峰不计多冲
    assert res.review_segments == [(0.5, 1.5)]


def test_recovery_only_when_original_alignment_unique():
    """原匹配归属不唯一（ambiguous）时不补查弱峰。"""
    r = [0.0, 0.5, 1.0, 1.5]
    t = [0.0, 0.5, 1.0, 1.5, 2.0]  # 原对齐已歧义
    res = align_impacts(r, t, weak_times=[1.75])
    assert res.ambiguous is True
    assert res.recovered == []


def test_confirmed_extras_still_reported_with_actual_time():
    """补回弱峰不影响已确认多冲按实录时刻标出。"""
    r = [0.5, 1.0, 1.5, 2.0, 2.5]
    t = [0.5, 1.0, 1.25, 2.0, 2.5]  # 1.25s 为确认多冲，1.5s 漏冲
    res = align_impacts(r, t, weak_times=[1.5])
    assert res.recovered == [(2, 1.5)]
    assert res.extra_test_indices == [2]
    assert res.test_times[res.extra_test_indices[0]] == 1.25
    assert res.miss_count == 0


def test_recovery_never_reuses_same_weak_peak_twice():
    r = [0.5, 1.0, 1.5, 2.0, 2.5]
    t = [0.5, 1.0, 2.5]  # 中间两次漏冲
    res = align_impacts(r, t, weak_times=[1.5])
    # 只有 1.5s 一个弱峰：能补 1 次，另一个仍是漏冲
    assert len(res.recovered) == 1
    assert len(res.missed_ref_indices) == 1


def test_dense_weak_peaks_treated_as_review_not_silent():
    """弱峰过密、并列归属过多时按复听处理，不静默取一个方案。"""
    from app.matching import _segment_plans

    r = [0.5, 1.0, 1.5, 2.0]
    # 两个参考槽各约 50 个候选弱峰，可行归属组合远超上限
    c1 = [round(0.95 + i * 0.002, 3) for i in range(51)]
    c2 = [round(1.4 + i * 0.002, 3) for i in range(51)]
    assert _segment_plans(
        [1, 2], r, (0.5, 0.5), (2.0, 2.0), sorted(set(c1 + c2))
    ) is None


def test_dense_weak_peaks_keep_original_verdict_in_pipeline():
    """无法确认归属的段在整条流水线中保留原判定并提示复听。"""
    from app.matching import MatchResult, _recover_weak_matches

    ref_times = [0.5, 1.0, 1.5, 2.0, 2.5]
    test_times = [0.5, 2.0]  # 中间 ref1/ref2 漏冲
    result = MatchResult(
        matched=[(0, 0), (3, 1)],
        missed_ref_indices=[1, 2],
        miss_count=2,
        test_times=list(test_times),
    )
    c1 = [round(0.95 + i * 0.002, 3) for i in range(51)]
    c2 = [round(1.4 + i * 0.002, 3) for i in range(51)]
    _recover_weak_matches(ref_times, test_times, c1 + c2, result)
    assert result.missed_ref_indices == [1, 2]
    assert result.recovered == []
    assert result.review_segments == [(0.5, 2.0)]
    assert result.review_required is True
