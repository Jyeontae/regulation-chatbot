"""리랭크 후보 모으기 비교: 합친 유사도 상위 N개(max) vs 검색어별 몫(quota) — docs/2026-11-04/후보모으기_비교.md

저장된 검색어(--rewrite-from)를 재사용해 API 호출 없이 로컬에서 잰다.
모든 설정이 후보를 다듬은 검색어로 리랭크해 상위 k개를 남긴다 (지금 챗봇과 같음).
리랭커 점수는 (검색어, 청크)마다 한 번만 계산하고 쌍마다 시간을 잰다. 설정의 리랭크 시간은 그 후보 쌍 시간의 합이다.

화면에는 점수·시간과 질문 id 만 출력한다 (규정명·본문 없음).

실행:
    poetry run python scripts/compare_candidates.py --rewrite-from data/eval/results/<검색어 CSV>.csv
"""

import argparse
import time
from collections.abc import Callable
from pathlib import Path
from statistics import mean, quantiles

from eval_retrieval import EvalItem, load_queries, parse_eval_set

from regbot.config import settings
from regbot.models import get_reranker
from regbot.report import print_table
from regbot.retriever import Hit, Retriever

POOL = 20  # 검색어마다 보는 순위


def max_candidates(n: int) -> Callable[[list[Hit], list[Hit], list[Hit]], list[Hit]]:
    """지금 방식: 청크별 높은 유사도로 합친 순위의 상위 n개."""
    return lambda merged, original, rewrite: merged[:n]


def quota_candidates(m: int) -> Callable[[list[Hit], list[Hit], list[Hit]], list[Hit]]:
    """C2: 원래 질문 상위 m개 + 다듬은 검색어 상위 m개. 합친 순위 순서로 둔다 (리랭크 동점 처리용)."""

    def pick(merged: list[Hit], original: list[Hit], rewrite: list[Hit]) -> list[Hit]:
        wanted = {h.doc.id for h in original[:m]} | {h.doc.id for h in rewrite[:m]}
        return [h for h in merged if h.doc.id in wanted]

    return pick


CONFIGS = {
    "max5 (지금)": max_candidates(5),
    "max8": max_candidates(8),
    "max10": max_candidates(10),
    "quota3": quota_candidates(3),
    "quota4": quota_candidates(4),
    "quota5": quota_candidates(5),
}
BASELINE = "max5 (지금)"


class ScoreCache:
    """(검색어, 청크) 리랭커 점수와 걸린 시간. 한 쌍씩 실행한다 (retriever.rerank 와 같은 방식)."""

    def __init__(self) -> None:
        self.cache: dict[tuple[str, str], tuple[float, float]] = {}

    def get(self, query: str, hit: Hit) -> tuple[float, float]:
        key = (query, hit.doc.id)
        if key not in self.cache:
            start = time.perf_counter()
            score = float(get_reranker().predict([(query, hit.doc.page_content)], batch_size=1)[0])
            self.cache[key] = (score, time.perf_counter() - start)
        return self.cache[key]


def run(
    item: EvalItem, query: str, retriever: Retriever, scores: ScoreCache, k: int
) -> dict[str, tuple[list[str], float, int]]:
    """설정마다 (상위 k개 section_id, 리랭크 시간, 후보 수)."""
    original = retriever.search(item.question, POOL)
    rewrite = retriever.search(query, POOL) if query != item.question else original
    merged = retriever.search_all([item.question, query], POOL)
    out = {}
    for name, pick in CONFIGS.items():
        candidates = pick(merged, original, rewrite)
        scored = [(scores.get(query, h), i, h) for i, h in enumerate(candidates)]
        seconds = sum(t for (_, t), _, _ in scored)
        ranked = [h for _, _, h in sorted(scored, key=lambda x: (-x[0][0], x[1]))]
        chosen = {h.doc.id for h in ranked}
        ranked += [h for h in merged if h.doc.id not in chosen]  # 후보가 k개보다 적을 때만 쓰인다
        out[name] = ([str(h.doc.metadata["section_id"]) for h in ranked[:k]], seconds, len(candidates))
    return out


def first_rank(sids: list[str], expected: set[str]) -> int | None:
    return next((r for r, sid in enumerate(sids, start=1) if sid in expected), None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rewrite-from", type=Path, required=True)
    parser.add_argument("--eval-set", type=Path, default=settings.eval_set_path)
    parser.add_argument("-k", type=int, default=settings.retriever_k)
    args = parser.parse_args()

    _, items = parse_eval_set(args.eval_set)
    answerable = [i for i in items if i.expected]
    queries = load_queries(args.rewrite_from)
    retriever = Retriever("vector")
    get_reranker().predict([("준비", "준비")])
    scores = ScoreCache()
    print(f"[설정] 검색어 {args.rewrite_from.name} · 정답 있는 질문 {len(answerable)}개 · 상위 {args.k}개")

    per_item = {i.qid: run(i, queries[i.qid], retriever, scores, args.k) for i in answerable}

    rows, summary = [], {}
    multi = [i for i in answerable if i.qtype == "여러 조항"]
    two_plus = [i for i in answerable if len(i.expected) >= 2]
    for name in CONFIGS:
        def top(i: EvalItem) -> list[str]:
            return per_item[i.qid][name][0]

        def rec(group: list[EvalItem]) -> float:
            return mean(len(set(top(i)) & i.expected) / len(i.expected) for i in group)

        ranks = [first_rank(top(i), i.expected) for i in answerable]
        times = [per_item[i.qid][name][1] for i in answerable]
        s = {
            "multi": rec(multi),
            "two_plus": rec(two_plus),
            "hit1": mean(float(r == 1) for r in ranks),
            "hitk": mean(float(r is not None) for r in ranks),
            "mrr": mean(1 / r if r else 0.0 for r in ranks),
            "mean_t": mean(times),
            "p90_t": quantiles(times, n=10, method="inclusive")[-1],
            "cands": mean(per_item[i.qid][name][2] for i in answerable),
        }
        summary[name] = s
        rows.append([
            name, f"{s['multi']:.2f}", f"{s['two_plus']:.2f}", f"{s['hit1']:.2f}", f"{s['hitk']:.2f}",
            f"{s['mrr']:.2f}", f"{s['cands']:.1f}", f"{s['mean_t']:.1f} / {s['p90_t']:.1f}초",
        ])
    print()
    print_table(
        ["설정", "여러 조항 R@4", "정답2+ R@4", "Hit@1", f"Hit@{args.k}", "MRR", "후보 수", "리랭크 평균/p90"], rows, 1
    )

    base = summary[BASELINE]
    print("\n[규칙 판정] 지금(max5) 대비: 여러 조항 R@4 상승 / Hit@k 하락 없음 / Hit@1·MRR 하락 ≤0.05 / p90 ≤5초")
    for name, s in summary.items():
        if name == BASELINE:
            continue
        checks = {
            "주지표": s["multi"] > base["multi"] + 1e-9,
            f"Hit@{args.k}": s["hitk"] >= base["hitk"] - 1e-9,
            "Hit@1": s["hit1"] >= base["hit1"] - 0.05 - 1e-9,
            "MRR": s["mrr"] >= base["mrr"] - 0.05 - 1e-9,
            "p90": s["p90_t"] <= 5.0,
        }
        failed = [key for key, ok in checks.items() if not ok]
        print(f"  {name}: {'통과' if not failed else '탈락 (' + ', '.join(failed) + ')'}")

    print("\n[여러 조항 질문별] 찾은 정답 수 / 정답 수")
    header = ["질문"] + list(CONFIGS)
    print_table(
        header,
        [[i.qid] + [f"{len(set(per_item[i.qid][n][0]) & i.expected)}/{len(i.expected)}" for n in CONFIGS] for i in multi],
        1,
    )


if __name__ == "__main__":
    main()
