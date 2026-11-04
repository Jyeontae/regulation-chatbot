"""여러 조항 진단: 정답 조문이 2개 이상인 질문에서 정답 일부가 상위 k개(LLM 에 가는 발췌)에 못 드는 원인을 집계한다.

챗봇과 같은 검색(원래 질문 + 다듬은 검색어 합치기 → 상위 후보를 다듬은 검색어로 리랭크)을 저장된 검색어로 재현한다.
API 호출이 없다 (검색어는 --rewrite-from 의 CSV 에서 읽는다). 임베딩·리랭커는 로컬이다.

놓친 정답 조문마다 원인을 셋으로 나눈다 (docs/2026-11-03/개선_후보_조사.md 4절 단계 C):
- 순위 밀림: 리랭크 후보(상위 N개) 안에는 있었는데 리랭크 뒤 k위 밖 → 리랭커 교체·가속(C5) 또는 중복 제거(C1)
- 후보 밖: 합친 검색 결과 N+1~P위 → 몫 방식(C2)·질문 분해(C3)
- 검색 밖: 두 검색 모두 P위 안에 없음 → 참조 따라가기(C4)·질문 분해(C3)
함께 집계: 상위 k칸 중 같은 조문 조각·같은 본문 중복, 놓친 정답을 어느 검색어 목록이 더 높게 찾았는지,
놓친 정답이 상위 조문의 참조(refs)로 닿는지, 상위 k칸과 놓친 정답의 종류(본문·부칙·별표).

화면에는 숫자와 질문 id 만 출력한다 (규정명·section_id·본문 없음).

실행:
    poetry run python scripts/diagnose_multi.py --rewrite-from data/eval/results/<검색어 CSV>.csv
    poetry run python scripts/diagnose_multi.py --rewrite-from <CSV> --pool 20 --rerank-candidates 5 -k 4
"""

import argparse
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

from eval_retrieval import load_queries, parse_eval_set

from regbot.config import settings
from regbot.models import get_reranker
from regbot.report import print_table
from regbot.retriever import Hit, Retriever, rerank

# 'category/title#' 다음의 '구간N-' 까지가 같은 규정(같은 구간)을 가리키는 앞부분이다
_PREFIX = re.compile(r"^(.*?#(?:구간\d+-)?)")


def prefix_of(section_id: str) -> str:
    match = _PREFIX.match(section_id)
    return match[1] if match else section_id


def kind_of(section_id: str) -> str:
    local = section_id[len(prefix_of(section_id)) :]
    if local.startswith("부칙"):
        return "부칙"
    if local.startswith("별"):
        return "별표·별지"
    return "본문"


def body_of(hit: Hit) -> str:
    # 첫 줄은 위치 머리말이라 같은 본문이 다른 규정에 실려 있어도 다르게 보인다
    return hit.doc.page_content.split("\n", 1)[-1].strip()


@dataclass
class Missed:
    cause: str  # 순위 밀림 / 후보 밖 / 검색 밖
    kind: str
    merged_rank: int | None
    original_rank: int | None
    rewrite_rank: int | None
    via_refs: bool  # 상위 조문의 참조로 닿는가
    same_reg_in_top: bool  # 상위 k칸에 같은 규정 조문이 있는가
    indexed: bool  # 인덱스에 그 section_id 가 있는가


@dataclass
class Diagnosis:
    qid: str
    qtype: str
    expected: int
    recall_before: float  # 리랭크 전 상위 k
    recall_after: float  # 리랭크 후 상위 k (챗봇)
    recall_candidates: float  # 리랭크 후보 N개
    recall_pool: float  # 합친 검색 P개
    dup_section: int  # 상위 k칸 중 앞 칸과 같은 section_id (긴 조문 조각)
    dup_body: int  # 상위 k칸 중 앞 칸과 본문이 같은 다른 조문 (다른 규정에 실린 같은 조문)
    top_kinds: list[str] = field(default_factory=list)
    missed: list[Missed] = field(default_factory=list)


def rank_in(hits: list[Hit], section_id: str) -> int | None:
    return next((i for i, h in enumerate(hits, start=1) if h.doc.metadata["section_id"] == section_id), None)


def recall(hits: list[Hit], expected: set[str]) -> float:
    return len({h.doc.metadata["section_id"] for h in hits} & expected) / len(expected)


def ref_targets(hits: list[Hit]) -> set[str]:
    """상위 조문의 refs("제3조|별표 1")를 같은 규정의 section_id 로 바꾼 집합."""
    targets = set()
    for hit in hits:
        prefix = prefix_of(str(hit.doc.metadata["section_id"]))
        for ref in str(hit.doc.metadata.get("refs") or "").split("|"):
            if ref:
                targets.add(prefix + ref.replace(" ", ""))
    return targets


def diagnose(retriever: Retriever, indexed: set[str], qid: str, qtype: str, question: str, query: str,
             expected: set[str], k: int, n: int, pool: int, ref_top: int) -> Diagnosis:
    original_list = retriever.search(question, pool)
    rewrite_list = retriever.search(query, pool) if query != question else original_list
    merged = retriever.search_all([question, query], pool)
    before = merged[:k]
    after = rerank(query, list(merged), n)[:k]
    candidates = merged[:n]

    seen_sections, seen_bodies = set(), set()
    dup_section = dup_body = 0
    for hit in after:
        sid, body = hit.doc.metadata["section_id"], body_of(hit)
        if sid in seen_sections:
            dup_section += 1
        elif body in seen_bodies:
            dup_body += 1
        seen_sections.add(sid)
        seen_bodies.add(body)

    found = {h.doc.metadata["section_id"] for h in after}
    refs = ref_targets(after[:ref_top])
    top_prefixes = {prefix_of(str(h.doc.metadata["section_id"])) for h in after}
    missed = []
    for sid in sorted(expected - found):
        merged_rank = rank_in(merged, sid)
        if merged_rank is not None and merged_rank <= n:
            cause = "순위 밀림"
        elif merged_rank is not None:
            cause = "후보 밖"
        else:
            cause = "검색 밖"
        missed.append(
            Missed(
                cause,
                kind_of(sid),
                merged_rank,
                rank_in(original_list, sid),
                rank_in(rewrite_list, sid),
                sid in refs,
                prefix_of(sid) in top_prefixes,
                sid in indexed,
            )
        )
    return Diagnosis(
        qid, qtype, len(expected), recall(before, expected), recall(after, expected),
        recall(candidates, expected), recall(merged, expected), dup_section, dup_body,
        [kind_of(str(h.doc.metadata["section_id"])) for h in after], missed,
    )


def summarize(label: str, group: list[Diagnosis], k: int, n: int, pool: int) -> None:
    if not group:
        return
    print(f"\n[{label}] 질문 {len(group)}개, 정답 조문 {sum(d.expected for d in group)}개")
    print_table(
        ["지표", "값"],
        [
            [f"Recall@{k} 리랭크 전", f"{mean(d.recall_before for d in group):.2f}"],
            [f"Recall@{k} 리랭크 후 (챗봇)", f"{mean(d.recall_after for d in group):.2f}"],
            [f"Recall 리랭크 후보 {n}개", f"{mean(d.recall_candidates for d in group):.2f}"],
            [f"Recall 합친 검색 {pool}개", f"{mean(d.recall_pool for d in group):.2f}"],
            [f"상위 {k}칸 중 같은 조문 조각", f"{sum(d.dup_section for d in group)} / {k * len(group)}칸"],
            [f"상위 {k}칸 중 같은 본문 (다른 규정 사본)", f"{sum(d.dup_body for d in group)} / {k * len(group)}칸"],
        ],
        1,
    )
    kinds = Counter(kind for d in group for kind in d.top_kinds)
    print(f"  상위 {k}칸 종류: " + ", ".join(f"{kind} {count}" for kind, count in kinds.most_common()))

    missed = [m for d in group for m in d.missed]
    print(f"\n  놓친 정답 조문 {len(missed)}개 (상위 {k}칸 밖)")
    if not missed:
        return
    causes = Counter(m.cause for m in missed)
    print_table(
        ["원인", "개수", "처방 후보"],
        [
            ["순위 밀림 (후보 안)", str(causes["순위 밀림"]), "C1 중복 제거 · C5 리랭커"],
            [f"후보 밖 ({n + 1}~{pool}위)", str(causes["후보 밖"]), "C2 몫 방식 · C3 질문 분해"],
            [f"검색 밖 ({pool}위 밖)", str(causes["검색 밖"]), "C4 참조 따라가기 · C3 질문 분해"],
        ],
        1,
    )
    better = Counter(
        "원래 질문" if (m.original_rank or 10**6) < (m.rewrite_rank or 10**6)
        else "다듬은 검색어" if (m.rewrite_rank or 10**6) < (m.original_rank or 10**6)
        else ("같음" if m.original_rank else "둘 다 없음")
        for m in missed
    )
    print("  더 높게 찾은 검색어 목록: " + ", ".join(f"{key} {value}" for key, value in better.most_common()))
    print("  종류: " + ", ".join(f"{key} {value}" for key, value in Counter(m.kind for m in missed).most_common()))
    print(f"  상위 조문의 참조로 닿음: {sum(m.via_refs for m in missed)} / {len(missed)}")
    print(f"  같은 규정 조문이 상위 {k}칸에 있음: {sum(m.same_reg_in_top for m in missed)} / {len(missed)}")
    not_indexed = sum(not m.indexed for m in missed)
    if not_indexed:
        print(f"  [주의] 인덱스에 없는 정답 section_id: {not_indexed}개 (평가셋 정답 표기 확인 필요)")
    ranks = sorted(m.merged_rank for m in missed if m.merged_rank)
    print(f"  합친 검색 순위 (찾은 것만): {ranks or '없음'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rewrite-from", type=Path, required=True, help="다듬은 검색어를 읽을 이전 검색 평가 CSV")
    parser.add_argument("--eval-set", type=Path, default=settings.eval_set_path)
    parser.add_argument("-k", type=int, default=settings.retriever_k, help="LLM 에 가는 발췌 수 (기본: RETRIEVER_K)")
    parser.add_argument("--rerank-candidates", type=int, default=settings.rerank_candidates or 5)
    parser.add_argument("--pool", type=int, default=20, help="합친 검색에서 볼 순위 (기본: 20)")
    parser.add_argument("--ref-top", type=int, default=2, help="참조를 따라갈 상위 조문 수 (기본: 2)")
    parser.add_argument("--all", action="store_true", help="정답이 1개인 질문도 포함")
    args = parser.parse_args()

    found, items = parse_eval_set(args.eval_set)
    if not found:
        print("[중단] 평가 표를 찾지 못함")
        return
    queries = load_queries(args.rewrite_from)
    targets = [i for i in items if i.expected and (args.all or len(i.expected) >= 2)]
    missing = [i.qid for i in targets if i.qid not in queries]
    if missing:
        print(f"[중단] 검색어 없는 질문: {', '.join(missing)}")
        return

    retriever = Retriever("vector")
    indexed = {str(m["section_id"]) for m in retriever.store.get(include=["metadatas"])["metadatas"]}
    get_reranker().predict([("준비", "준비")])
    print(f"[설정] 검색어 {args.rewrite_from.name} · 합친 검색 {args.pool}개 → 리랭크 후보 {args.rerank_candidates}개"
          f"(다듬은 검색어) → 상위 {args.k}개 · 질문 {len(targets)}개")

    results = [
        diagnose(retriever, indexed, i.qid, i.qtype, i.question, queries[i.qid], i.expected,
                 args.k, args.rerank_candidates, args.pool, args.ref_top)
        for i in targets
    ]
    summarize("여러 조항 유형", [d for d in results if d.qtype == "여러 조항"], args.k, args.rerank_candidates, args.pool)
    summarize("정답 2개 이상 전체", [d for d in results if d.expected >= 2], args.k, args.rerank_candidates, args.pool)

    print("\n[질문별] 놓친 정답의 원인 (합친 검색 순위 / 원래 질문 목록 순위 / 다듬은 검색어 목록 순위)")
    for d in results:
        if d.missed:
            detail = "; ".join(
                f"{m.cause}({m.merged_rank or '-'}/{m.original_rank or '-'}/{m.rewrite_rank or '-'}"
                f"{', 참조' if m.via_refs else ''}{', 같은 규정' if m.same_reg_in_top else ''})"
                for m in d.missed
            )
            print(f"  {d.qid} [{d.qtype}] {d.expected - len(d.missed)}/{d.expected}: {detail}")


if __name__ == "__main__":
    main()
