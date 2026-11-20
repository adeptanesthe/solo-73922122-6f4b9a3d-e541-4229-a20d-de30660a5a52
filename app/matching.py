"""参考冲次与待检冲次的节奏匹配。

用动态规划在「参考冲次序列 R」和「待检冲次序列 T」之间找最优对齐：
- 匹配：参考冲次 i 与待检冲次 j 对应，相邻两次匹配之间的总时长相对
  参考节奏允许 ±15% 偏差（累计跨度判定，单个间隔的偏差不会被漏冲掩盖）；
- 漏冲：参考冲次没有对应的待检冲次，计 1 次漏冲；
- 多冲：待检冲次无法归属到任何参考冲次，计 1 次多冲。

关键性质：
1. 漏冲只增加漏冲计数，不会把后续冲次整体错位（对齐始终以参考序号为准）；
2. 起始静音通过「首对匹配前的待检冲次记多冲、参考冲次记漏冲」自然处理；
3. 40ms 回声已在检测阶段抑制，不进入本模块；
4. 若存在多条成本相同但归属不同的最优对齐，置 ambiguous=True，
   前端提示人工复听，而不是武断放行。

重击遮蔽补回（align_impacts 的 weak_times 参数）：
- 原匹配归属唯一时，在相邻两次已匹配冲次之间补查明显高于附近背景的弱峰；
- 补回后，从左侧匹配到右侧匹配的整段冲次须逐对满足 ±15% 节奏；
- 每段取「补回冲次最多」的方案；若最大方案存在不同归属的并列，
  该段保留原判定并标记人工复听，未采用的弱峰不计多冲；
- 已确认的多冲仍按实录时刻标出。
"""
from __future__ import annotations

from dataclasses import dataclass, field

TIMING_TOLERANCE = 0.15
MISS_COST = 1
EXTRA_COST = 1
MAX_WAY_COUNT = 2  # 只需区分「唯一最优」还是「多个等价最优」
_SEG_MAX_WAYS = 64  # 单段最多保留的并列方案数
_SEG_MAX_WORK = 40_000  # 单段 DP 状态-方案展开上限，防止密集弱峰导致组合爆炸


@dataclass
class MatchResult:
    matched: list[tuple[int, int]] = field(default_factory=list)
    missed_ref_indices: list[int] = field(default_factory=list)
    extra_test_indices: list[int] = field(default_factory=list)
    miss_count: int = 0
    extra_count: int = 0
    ambiguous: bool = False
    optimal_cost: int = 0
    optimal_way_count: int = 1
    timing_ratios: list[float] = field(default_factory=list)
    expected_missed_times: dict[int, float] = field(default_factory=dict)
    # 参与索引的待检冲次序列（补回弱峰后为扩展序列，按时间升序）
    test_times: list[float] = field(default_factory=list)
    # 补回记录：(参考冲次序号, 弱峰实录时刻)
    recovered: list[tuple[int, float]] = field(default_factory=list)
    # 补回归属并列、保留原判定并须人工复听的段：(左匹配时刻, 右匹配时刻)
    review_segments: list[tuple[float, float]] = field(default_factory=list)

    @property
    def review_required(self) -> bool:
        return bool(self.review_segments)


def _timing_ok(t_j: float, t_prev: float, r_i: float, r_prev: float) -> bool:
    """相邻两次匹配之间的累计时长是否落在 ±15% 节奏窗口内。"""
    ref_span = r_i - r_prev
    test_span = t_j - t_prev
    if ref_span <= 0:
        return False
    if test_span < 0:
        return False
    ratio = test_span / ref_span
    return abs(ratio - 1.0) <= TIMING_TOLERANCE + 1e-12


def align_impacts(
    ref_times: list[float],
    test_times: list[float],
    weak_times: list[float] | None = None,
) -> MatchResult:
    n = len(ref_times)
    m = len(test_times)
    result = MatchResult()
    result.test_times = list(test_times)

    if n == 0:
        # 参考为空（正常流程下会在 API 层拦截，这里做防御处理）
        result.extra_test_indices = list(range(m))
        result.extra_count = m
        result.optimal_cost = m * EXTRA_COST
        result.ambiguous = m > 0
        return result

    # cost[i][j]：参考前 i+1 个、待检前 j+1 个冲次，且 (i,j) 构成一对匹配
    # ways[i][j]：达到该最小成本的不同对齐路径数（封顶 MAX_WAY_COUNT）
    cost: list[list[int]] = [[0] * m for _ in range(n)]
    ways: list[list[int]] = [[0] * m for _ in range(n)]
    # back[i][j]：最优来源，元素为 (pi, pj)；锚点为 None；多来源记 "multi"
    back: list[list] = [[None] * m for _ in range(n)]

    for i in range(n):
        for j in range(m):
            anchor_cost = i * MISS_COST + j * EXTRA_COST  # 首对匹配
            best = anchor_cost
            best_ways = 1
            best_src = None
            for pi in range(i):
                for pj in range(j):
                    if ways[pi][pj] == 0:
                        continue
                    if not _timing_ok(
                        test_times[j], test_times[pj],
                        ref_times[i], ref_times[pi],
                    ):
                        continue
                    between_miss = (i - pi - 1) * MISS_COST
                    between_extra = (j - pj - 1) * EXTRA_COST
                    c = cost[pi][pj] + between_miss + between_extra
                    if c < best:
                        best = c
                        best_ways = ways[pi][pj]
                        best_src = (pi, pj)
                    elif c == best:
                        best_ways = min(
                            MAX_WAY_COUNT, best_ways + ways[pi][pj]
                        )
                        if best_src != (pi, pj):
                            best_src = "multi"
            cost[i][j] = best
            ways[i][j] = best_ways
            back[i][j] = best_src

    # 终止：匹配到 (i,j) 后，剩余参考全漏冲、剩余待检全多冲
    terminal_costs: list[tuple[int, int, int]] = []
    for i in range(n):
        for j in range(m):
            if ways[i][j] == 0:
                continue
            tail = (n - 1 - i) * MISS_COST + (m - 1 - j) * EXTRA_COST
            terminal_costs.append((cost[i][j] + tail, i, j))

    if not terminal_costs:
        # m == 0：待检一个冲次都没检测到，全部漏冲
        result.missed_ref_indices = list(range(n))
        result.miss_count = n
        result.optimal_cost = n * MISS_COST
        result.optimal_way_count = 1
        for i, r in enumerate(ref_times):
            result.expected_missed_times[i] = r
        if weak_times:
            _recover_weak_matches(ref_times, test_times, weak_times, result)
        return result

    global_best = min(tc[0] for tc in terminal_costs)
    best_terminals = [(i, j) for c, i, j in terminal_costs if c == global_best]
    total_ways = 0
    for i, j in best_terminals:
        total_ways = min(MAX_WAY_COUNT, total_ways + ways[i][j])
    # 不同最优终止点本身就是不同的冲次归属，同样构成歧义
    if len(best_terminals) > 1:
        total_ways = MAX_WAY_COUNT

    # 回溯其中一条最优路径用于展示
    end_i, end_j = best_terminals[0]
    matched: list[tuple[int, int]] = []
    i, j = end_i, end_j
    while True:
        matched.append((i, j))
        src = back[i][j]
        if src is None:
            break
        if src == "multi":  # 找到任一具体前驱即可
            found = None
            for pi in range(i):
                for pj in range(j):
                    if ways[pi][pj] == 0:
                        continue
                    if not _timing_ok(
                        test_times[j], test_times[pj],
                        ref_times[i], ref_times[pi],
                    ):
                        continue
                    between = (i - pi - 1) + (j - pj - 1)
                    if cost[pi][pj] + between * MISS_COST == cost[i][j]:
                        found = (pi, pj)
                        break
                if found:
                    break
            if found is None:  # pragma: no cover - 防御性分支
                break
            i, j = found
        else:
            i, j = src
    matched.reverse()

    matched_ref = {i for i, _ in matched}
    matched_test = {j for _, j in matched}
    missed = [i for i in range(n) if i not in matched_ref]
    extras = [j for j in range(m) if j not in matched_test]

    result.matched = matched
    result.missed_ref_indices = missed
    result.extra_test_indices = extras
    result.miss_count = len(missed)
    result.extra_count = len(extras)
    result.optimal_cost = global_best
    result.optimal_way_count = total_ways
    result.ambiguous = total_ways > 1 or len(best_terminals) > 1

    result.timing_ratios = _timing_ratios(ref_times, test_times, matched)
    result.expected_missed_times = _estimate_missed_times(
        ref_times, test_times, matched, missed
    )
    # 原匹配归属唯一时才补查弱峰
    if weak_times and not result.ambiguous:
        _recover_weak_matches(ref_times, test_times, weak_times, result)
    return result


def _timing_ratios(
    ref_times: list[float],
    test_times: list[float],
    matched: list[tuple[int, int]],
) -> list[float]:
    ratios: list[float] = []
    for (i1, j1), (i2, j2) in zip(matched, matched[1:]):
        ref_span = ref_times[i2] - ref_times[i1]
        if ref_span > 0:
            ratios.append((test_times[j2] - test_times[j1]) / ref_span)
    return ratios


# ---------------------------------------------------------------------------
# 重击遮蔽补回
# ---------------------------------------------------------------------------


def _segment_plans(
    ref_ids: list[int],
    ref_times: list[float],
    left: tuple[float, float],
    right: tuple[float, float],
    weak: list[float],
) -> list[tuple[tuple[int, int], ...]] | None:
    """求段内「补回弱峰数最多」的全部并列归属方案。

    段 = 相邻两次已匹配冲次 (left, right) 之间；ref_ids 为段内漏冲的参考
    冲次，weak 为段内待查弱峰。约束：

    - 从左匹配出发，经补回的冲次，到右匹配为止，整段相邻冲次逐对满足
      ±15% 节奏（累计跨度判定，与主对齐一致）；
    - 漏冲参考冲次与未采用的弱峰允许存在，只是不计入补回收益；
    - 目标：最大化被匹配的弱峰数。

    返回值：
    - 空列表：无法补回（最大补回数为 0 或方案都过不了右侧收口）；
    - 非空列表：达到最大补回数的归属方案，方案为 ((参考冲次序号, 弱峰下标), ...)；
    - None：弱峰过密，展开量超出上限，无法确认归属是否唯一（按并列复听处理）。
    """
    n, m = len(ref_ids), len(weak)
    if n == 0 or m == 0:
        return []

    # bucket[(rlen, clen)][(last_ref_pos, last_weak_idx)] = (补回数, {方案})
    # 末两项 -1 表示段内尚未补回任何冲次（此时以左锚点为上一个匹配）。
    start_key = (0, 0)
    bucket: dict[
        tuple[int, int], dict[tuple[int, int], tuple[int, dict[tuple, None]]]
    ] = {start_key: {(-1, -1): (0, {(): None})}}
    work = 0
    overflow = False

    def push(
        rlen: int,
        clen: int,
        last_rp: int,
        last_ci: int,
        gained: int,
        plans: dict[tuple, None],
    ) -> None:
        nonlocal work, overflow
        if overflow:
            return
        work += len(plans)
        if work > _SEG_MAX_WORK:
            overflow = True
            return
        cell = bucket.setdefault((rlen, clen), {})
        cur = cell.get((last_rp, last_ci))
        if cur is None or gained > cur[0]:
            cell[(last_rp, last_ci)] = (gained, dict(plans))
        elif gained == cur[0]:
            merged = cur[1]
            for plan in plans:
                if len(merged) >= _SEG_MAX_WAYS:
                    overflow = True
                    return
                merged.setdefault(plan, None)

    for rlen in range(n + 1):
        for clen in range(m + 1):
            for (last_rp, last_ci), (gained, plans) in list(
                bucket.get((rlen, clen), {}).items()
            ):
                # 参考漏冲（不补回）
                if rlen < n:
                    push(rlen + 1, clen, last_rp, last_ci, gained, plans)
                # 弱峰不采用
                if clen < m:
                    push(rlen, clen + 1, last_rp, last_ci, gained, plans)
                # 用弱峰补回参考冲次
                if rlen < n and clen < m:
                    r_i, t_j = ref_times[ref_ids[rlen]], weak[clen]
                    if last_ci >= 0:
                        prev_r = ref_times[ref_ids[last_rp]]
                        prev_t = weak[last_ci]
                    else:
                        prev_r, prev_t = left
                    if not _timing_ok(t_j, prev_t, r_i, prev_r):
                        continue
                    push(
                        rlen + 1, clen + 1, rlen, clen, gained + 1,
                        {plan + ((ref_ids[rlen], clen),): None for plan in plans},
                    )

    best = -1
    final: dict[tuple, None] = {}
    for (last_rp, last_ci), (gained, plans) in bucket.get((n, m), {}).items():
        # 整段收口：最后一个补回冲次（或左锚点）到右匹配仍须满足 ±15%
        if last_ci >= 0:
            prev_r = ref_times[ref_ids[last_rp]]
            prev_t = weak[last_ci]
        else:
            prev_r, prev_t = left
        if not _timing_ok(right[1], prev_t, right[0], prev_r):
            continue
        if gained > best:
            best = gained
            final = dict(plans)
        elif gained == best:
            for plan in plans:
                if len(final) >= _SEG_MAX_WAYS:
                    return None  # 并列方案过多，无法确认归属唯一
                final.setdefault(plan, None)
    if overflow:
        # 展开被截断，无法确认「最大补回数」及其归属是否唯一：
        # 保守按并列复听处理，而不是静默给出一个可能不完整的结果
        return None
    if best <= 0:
        return []
    return list(final)


def _recover_weak_matches(
    ref_times: list[float],
    test_times: list[float],
    weak_times: list[float],
    result: MatchResult,
) -> None:
    """按段补回被重击遮蔽的弱峰，结果直接写入 result。"""
    matched = result.matched
    n_ref = len(ref_times)
    if not matched:
        return

    matched_ref = {i for i, _ in matched}
    missed = sorted(set(range(n_ref)) - matched_ref)
    if not missed:
        return

    weak = sorted(set(weak_times))
    recovered: list[tuple[int, float]] = []
    review: list[tuple[float, float]] = []

    # 只在相邻两次已匹配的冲次之间补查；首匹配前 / 末匹配后没有
    # 「左匹配到右匹配」的整段节奏约束，不补回
    for (i1, j1), (i2, j2) in zip(matched, matched[1:]):
        seg_ref = [i for i in missed if i1 < i < i2]
        lo, hi = test_times[j1], test_times[j2]
        if not seg_ref:
            continue
        seg_weak = [t for t in weak if lo < t < hi]
        if not seg_weak:
            continue
        plans = _segment_plans(
            seg_ref,
            ref_times,
            (ref_times[i1], lo),
            (ref_times[i2], hi),
            seg_weak,
        )
        if plans is None:
            # 弱峰过密、无法确认归属是否唯一：保留原判定并提示复听
            review.append((lo, hi))
            continue
        if not plans:
            continue
        if len(plans) > 1:
            # 补回数并列但归属不同：保留该段原判定，提示人工复听；
            # 未采用的弱峰不计多冲
            review.append((lo, hi))
            continue
        recovered.extend(
            (ri, seg_weak[ci]) for ri, ci in plans[0]
        )

    if not recovered:
        result.review_segments = review
        return

    # 扩展待检序列并重建索引：补回冲次参与匹配，未采用的弱峰不进入序列
    ext_times = sorted(set(test_times) | {t for _, t in recovered})
    index_of = {t: k for k, t in enumerate(ext_times)}
    matched_pairs = [(i, index_of[test_times[j]]) for i, j in matched]
    matched_pairs += [(i, index_of[t]) for i, t in recovered]
    matched_pairs.sort()

    matched_ref = {i for i, _ in matched_pairs}
    matched_test = {j for _, j in matched_pairs}
    new_missed = [i for i in range(n_ref) if i not in matched_ref]
    new_extras = [j for j in range(len(ext_times)) if j not in matched_test]

    result.test_times = ext_times
    result.matched = matched_pairs
    result.missed_ref_indices = new_missed
    result.extra_test_indices = new_extras
    result.miss_count = len(new_missed)
    result.extra_count = len(new_extras)
    result.recovered = recovered
    result.review_segments = review
    result.optimal_cost = result.miss_count + result.extra_count
    result.timing_ratios = _timing_ratios(ref_times, ext_times, matched_pairs)
    result.expected_missed_times = _estimate_missed_times(
        ref_times, ext_times, matched_pairs, new_missed
    )


def _estimate_missed_times(
    ref_times: list[float],
    test_times: list[float],
    matched: list[tuple[int, int]],
    missed: list[int],
) -> dict[int, float]:
    """按匹配段估计每个漏冲参考冲次「本应出现」的时刻。"""
    expected: dict[int, float] = {}
    if not matched:
        for i in missed:
            expected[i] = ref_times[i]
        return expected

    ratios = []
    for (i1, j1), (i2, j2) in zip(matched, matched[1:]):
        ref_span = ref_times[i2] - ref_times[i1]
        if ref_span > 0:
            ratios.append((test_times[j2] - test_times[j1]) / ref_span)
    scale = sorted(ratios)[len(ratios) // 2] if ratios else 1.0

    first_i, first_j = matched[0]
    r0, t0 = ref_times[first_i], test_times[first_j]
    for i in missed:
        expected[i] = max(0.0, t0 + (ref_times[i] - r0) * scale)
    return expected
