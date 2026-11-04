"""검색 평가: 평가셋 질문으로 검색해, 정답 조문(section_id)을 몇 등에서 찾는지 Hit@k·Recall@k·MRR 로 잰다.

평가셋은 마크다운 표로 쓴다 (기본: .env 의 EVAL_SET_PATH, 템플릿: docs/2026-10-29/eval_set_template.md).
표 앞뒤에 설명 글이 있어도 되고, 코드 블록(```) 안의 표는 예시로 보고 건너뛴다.

    | id | 유형 | 질문 | 정답 section_id | 기대 답 |
    |---|---|---|---|---|
    | Q01 | 단순 사실 | 연차는 며칠 전에 신청해? | 사규/휴가규정#제5조 | 3일 전 |
    | Q02 | 여러 조항 | … | 사규/휴가규정#제5조, 사규/휴가규정#제7조 | … |
    | Q03 | 규정에 없음 | … | | |

- 정답이 여러 개면 쉼표로 구분한다. 비워 두면 규정에 없는 질문으로 보고 점수 계산에서 뺀다.
- 유형·기대 답 열은 없어도 된다. 기대 답은 답변 평가용이라 여기서는 쓰지 않는다.

기본은 LLM 을 쓰지 않아 외부 전송이 없다. --rewrite 는 질문을 LLM API 로 보내 검색어로 다듬는다 (규정 본문은 보내지 않음).
LLM 이 다듬은 검색어는 실행마다 달라지므로, 검색 방식만 바꿔 비교할 때는 --rewrite-from 으로 이전 결과의 검색어를 재사용한다.
화면에는 점수와 질문 id 만 출력한다 (질문·규정명 없음).
질문별 검색 결과는 평가셋 옆 results/ 폴더에 CSV 로 남긴다 (규정명 포함, 공유 금지).

실행:
    poetry run python scripts/eval_retrieval.py
    poetry run python scripts/eval_retrieval.py --ks 1 3 5 10 20
    poetry run python scripts/eval_retrieval.py --mode hybrid   # 검색 방식 지정 (기본: .env 의 RETRIEVER_MODE)
    poetry run python scripts/eval_retrieval.py --rewrite       # LLM 으로 질문을 검색어로 다듬어 검색 (외부 API 전송)
    poetry run python scripts/eval_retrieval.py --rewrite --merge-original   # 원래 질문과 다듬은 검색어로 각각 검색해 합침
    poetry run python scripts/eval_retrieval.py --rewrite-from data/eval/results/<이전 결과>.csv   # 검색어 재사용 (API 호출 없음)
    poetry run python scripts/eval_retrieval.py --rerank-candidates 10   # 상위 10개 후보를 로컬 리랭커로 다시 정렬
    poetry run python scripts/eval_retrieval.py --rewrite-from <이전 결과>.csv --merge-original --rerank-candidates 10 --rerank-query rewrite

리랭크하면 질문당 리랭크 시간(평균·상위 10% 경계·최대)도 출력한다. 모델 로딩과 첫 실행 준비는 시간에서 뺀다.
"""

import argparse
import csv
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, quantiles

from regbot.config import settings
from regbot.models import get_reranker, llm_name
from regbot.rag import build_rewriter, rewrite_model, rewrite_query
from regbot.report import display_path, format_cell, print_table
from regbot.retriever import CANDIDATES, SUPPORTED_MODES, Retriever, rerank

RERANK_QUERIES = ("original", "rewrite")  # 리랭커에 넣는 질문: 원래 질문 / 다듬은 검색어

NO_ANSWER = {"", "-", "없음"}  # 정답 칸에 이 값만 있으면 규정에 없는 질문


@dataclass
class EvalItem:
    qid: str
    qtype: str
    question: str
    expected: set[str]  # 정답 section_id. 비어 있으면 규정에 없는 질문


@dataclass
class Result:
    item: EvalItem
    # (section_id, 점수) 청크 순위순. LLM 에 보내는 k개와 같은 기준이라, 길어서 나뉜 조문은 같은 section_id 가 여러 번 나온다
    # 점수는 vector 면 코사인 유사도, hybrid 면 RRF 점수, 리랭크한 후보는 리랭커 점수 (후보 밖은 원래 점수)
    retrieved: list[tuple[str, float]]
    top1: float  # 벡터 검색 1등의 코사인 유사도 (검색 방식과 무관, 검색어가 여럿이면 그중 가장 높은 값)
    rerank_seconds: float  # 리랭크에 걸린 시간. 리랭크하지 않았으면 0
    query: str  # 다듬은 검색어. 다듬지 않았으면 빈 문자열

    @property
    def first_rank(self) -> int | None:
        """정답이 처음 나온 순위 (1부터). 검색 결과 안에 없으면 None."""
        return next((rank for rank, (sid, _) in enumerate(self.retrieved, start=1) if sid in self.item.expected), None)

    def hit(self, k: int) -> float:
        rank = self.first_rank
        return float(rank is not None and rank <= k)

    def recall(self, k: int) -> float:
        found = {sid for sid, _ in self.retrieved[:k]} & self.item.expected
        return len(found) / len(self.item.expected)

    @property
    def reciprocal_rank(self) -> float:
        rank = self.first_rank
        return 1 / rank if rank else 0.0


def header_columns(cells: list[str]) -> dict[str, int]:
    """머리글 칸에서 열 위치를 찾는다. 필수 열(id·질문·정답)이 없으면 빈 dict (머리글이 아님)."""
    columns: dict[str, int] = {}
    for i, cell in enumerate(cells):
        name = cell.lower()
        # '정답 section_id' 도 'id' 를 품고 있으므로 id 열은 정확히 일치할 때만 잡는다
        if name == "id":
            columns.setdefault("id", i)
        elif "질문" in name:
            columns.setdefault("question", i)
        elif "정답" in name:
            columns.setdefault("expected", i)
        elif "유형" in name:
            columns.setdefault("type", i)
    return columns if {"id", "question", "expected"} <= columns.keys() else {}


def parse_eval_set(path: Path) -> tuple[bool, list[EvalItem]]:
    """id·질문·정답 열이 있는 첫 마크다운 표를 읽는다. 코드 블록(```) 안의 표는 예시로 보고 건너뛴다.

    반환: (평가 표의 머리글을 찾았는지, 질문 목록)
    """
    columns: dict[str, int] = {}
    items: list[EvalItem] = []
    in_fence = False
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line.startswith("```"):
            if columns:  # 평가 표가 끝났다
                break
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if not line.startswith("|"):
            if columns:  # 평가 표가 끝났다
                break
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not columns:
            columns = header_columns(cells)
            continue
        if all(cell and set(cell) <= set(":-") for cell in cells):  # 구분선 |---|---|
            continue
        cells += [""] * (max(columns.values()) + 1 - len(cells))
        qid, question = cells[columns["id"]], cells[columns["question"]]
        if not qid and not question:
            continue
        expected = {sid.strip() for sid in cells[columns["expected"]].split(",")} - NO_ANSWER
        qtype = cells[columns["type"]] if "type" in columns else ""
        items.append(EvalItem(qid, qtype or "(없음)", question, expected))
    return bool(columns), items


def load_queries(path: Path) -> dict[str, str]:
    """이전 결과 CSV 의 검색어 열 (id → 검색어). 검색어가 빈 행(다듬지 않은 실행)은 뺀다."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {row["id"]: row["검색어"] for row in csv.DictReader(f) if row.get("검색어")}


def retrieve(
    retriever: Retriever, queries: list[str], k: int, rerank_query: str = "", rerank_n: int = 0
) -> tuple[list[tuple[str, float]], float, float]:
    """반환: (상위 k개 (section_id, 점수), 1등 유사도, 리랭크 시간). rerank_n 이 0 이면 리랭크하지 않는다."""
    # 1등 유사도는 'LLM 호출 전 거절' 기준 후보라 검색 방식과 무관하게 벡터 1등 값을 쓴다.
    # hybrid 에서는 벡터 1등이 상위 k 밖으로 밀릴 수 있어 후보 전체를 받은 뒤 자른다
    hits = retriever.search_all(queries, max(k, rerank_n, 2 * CANDIDATES))
    top1 = max((h.cosine for h in hits if h.vector_rank == 1 and h.cosine is not None), default=0.0)
    seconds = 0.0
    if rerank_n:
        start = time.perf_counter()
        hits = rerank(rerank_query, hits, rerank_n)
        seconds = time.perf_counter() - start
    return [(str(h.doc.metadata["section_id"]), h.score) for h in hits[:k]], top1, seconds


def print_scores(answerable: list[Result], ks: list[int], base_k: int) -> None:
    print(f"\n[전체] 정답 있는 질문 {len(answerable)}개")
    rows = [
        [str(k), f"{mean(r.hit(k) for r in answerable):.2f}", f"{mean(r.recall(k) for r in answerable):.2f}"]
        for k in ks
    ]
    print_table(["k", "Hit@k", "Recall@k"], rows, 1)
    print(f"  MRR@{max(ks)} = {mean(r.reciprocal_rank for r in answerable):.2f}")

    by_type: dict[str, list[Result]] = defaultdict(list)
    for result in answerable:
        by_type[result.item.qtype].append(result)
    print(f"\n[유형별] k = {base_k}")
    rows = [
        [
            qtype,
            format_cell(len(group)),
            f"{mean(r.hit(base_k) for r in group):.2f}",
            f"{mean(r.recall(base_k) for r in group):.2f}",
            f"{mean(r.reciprocal_rank for r in group):.2f}",
        ]
        for qtype, group in sorted(by_type.items())
    ]
    print_table(["유형", "질문", f"Hit@{base_k}", f"Recall@{base_k}", "MRR"], rows, 1)

    missed = [r.item.qid for r in answerable if r.first_rank is None]
    late = [f"{r.item.qid}({r.first_rank}위)" for r in answerable if r.first_rank and r.first_rank > base_k]
    print("\n[개선할 질문]")
    print(f"  상위 {max(ks)}개 안에 정답 없음: {', '.join(missed) or '없음'}")
    print(f"  정답이 {base_k}위 밖 (LLM 에 안 감): {', '.join(late) or '없음'}")


def print_similarity_gap(answerable: list[Result], unanswerable: list[Result]) -> None:
    """규정에 없는 질문과 있는 질문의 1등 유사도를 비교한다. 차이가 뚜렷하면 'LLM 호출 전 거절' 기준을 정할 수 있다."""
    print("\n[1등 유사도]")
    for label, group in (("정답 있는 질문", answerable), ("규정에 없는 질문", unanswerable)):
        if group:
            tops = [r.top1 for r in group]
            print(f"  {label}: 평균 {mean(tops):.3f} (최소 {min(tops):.3f} / 최대 {max(tops):.3f}, {len(group)}개)")


def print_rerank_time(results: list[Result], load_seconds: float) -> None:
    """질문당 리랭크 시간. 느린 경우는 상위 10% 경계(90번째 백분위수)로 본다."""
    times = [r.rerank_seconds for r in results]
    p90 = quantiles(times, n=10, method="inclusive")[-1] if len(times) > 1 else times[0]
    print(f"\n[리랭크 시간] 질문 {len(times)}개 (모델 로딩 {load_seconds:.1f}초는 제외)")
    print(f"  평균 {mean(times):.2f}초 / 상위 10% 경계 {p90:.2f}초 / 최소 {min(times):.2f}초 / 최대 {max(times):.2f}초")


def save_details(results: list[Result], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig: 엑셀에서 열어도 한글이 깨지지 않게 한다
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "유형", "정답 수", "정답 순위", "1등 유사도", "검색 결과 (section_id:점수, 순위순)", "검색어"])
        for r in results:
            rank = r.first_rank or ("없음" if r.item.expected else "-")
            retrieved = " ; ".join(f"{sid}:{score:.3f}" for sid, score in r.retrieved)
            writer.writerow([r.item.qid, r.item.qtype, len(r.item.expected), rank, f"{r.top1:.3f}", retrieved, r.query])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=settings.eval_set_path, help="평가셋 (기본: .env 의 EVAL_SET_PATH)")
    parser.add_argument("--ks", type=int, nargs="+", help=f"점수를 볼 k 목록 (기본: 1 3 {settings.retriever_k} 5 10)")
    parser.add_argument("--mode", choices=SUPPORTED_MODES, help="검색 방식 (기본: .env 의 RETRIEVER_MODE)")
    parser.add_argument("--rewrite", action="store_true", help="LLM 으로 질문을 검색어로 다듬어 검색 (질문이 외부 API 로 전송된다)")
    parser.add_argument("--rewrite-from", type=Path, help="이전 결과 CSV 의 검색어를 재사용 (LLM 을 부르지 않는다)")
    parser.add_argument("--merge-original", action="store_true", help="원래 질문과 다듬은 검색어로 각각 검색해 높은 점수로 합친다")
    parser.add_argument(
        "--rerank-candidates",
        type=int,
        # 실험 도구의 기본값은 v1 로 둔다 (10-30 결정). .env 의 RERANK_CANDIDATES 는 쓰지 않는다
        default=0,
        help="검색 결과 상위 N개를 로컬 리랭커로 다시 정렬 (기본: 0 = 끔)",
    )
    parser.add_argument(
        "--rerank-query",
        choices=RERANK_QUERIES,
        default="original",
        help="리랭커에 넣는 질문 (original: 원래 질문 / rewrite: 다듬은 검색어, 기본: original)",
    )
    args = parser.parse_args()
    ks = sorted(set(args.ks or (1, 3, settings.retriever_k, 5, 10)))
    if ks[0] < 1:
        parser.error("--ks 는 1 이상이어야 합니다")
    if args.rewrite and args.rewrite_from:
        parser.error("--rewrite 와 --rewrite-from 은 함께 쓰지 않는다")
    if args.merge_original and not (args.rewrite or args.rewrite_from):
        parser.error("--merge-original 은 --rewrite 또는 --rewrite-from 과 함께 쓴다")
    if args.rewrite_from and not args.rewrite_from.exists():
        parser.error(f"--rewrite-from 파일이 없음: {args.rewrite_from}")
    if args.rerank_candidates < 0:
        parser.error("--rerank-candidates 는 0 이상이어야 합니다")
    if args.rerank_candidates and args.rerank_query == "rewrite" and not (args.rewrite or args.rewrite_from):
        parser.error("--rerank-query rewrite 는 --rewrite 또는 --rewrite-from 과 함께 쓴다")
    # 유형별 표는 LLM 에 실제로 보내는 개수(RETRIEVER_K) 기준으로 본다
    base_k = min(settings.retriever_k, ks[-1])

    path: Path = args.eval_set
    if not path.exists():
        print(f"[중단] 평가셋이 없음: {display_path(path)}")
        return
    found, items = parse_eval_set(path)
    if not found:
        print(f"[중단] {display_path(path)} 에서 평가 표를 찾지 못함 → 머리글에 id·질문·정답 열이 있어야 한다 (코드 블록 안의 표는 건너뜀)")
        return
    if not items:
        print(f"[중단] 평가 표에 질문이 없음 → {display_path(path)} 의 표에 행을 추가")
        return

    try:
        retriever = Retriever(args.mode)
        rewriter = build_rewriter() if args.rewrite else None
    except ValueError as e:
        print(f"[중단] {e}")
        return
    valid_ids = {str(meta["section_id"]) for meta in retriever.store.get(include=["metadatas"])["metadatas"]}
    print(f"[인덱스] {display_path(settings.chroma_dir)} / 컬렉션 '{settings.chroma_collection}' ({len(valid_ids):,}개 조문)")
    if not valid_ids:
        print("  [중단] 인덱스가 비어 있음 → build_index.py 를 먼저 실행")
        return
    rewrite_name = llm_name(None, rewrite_model())
    saved = load_queries(args.rewrite_from) if args.rewrite_from else None
    rewriting = rewriter is not None or saved is not None
    method = retriever.mode
    if rewriter:
        method += f" + 검색어 다듬기 ({rewrite_name})"
    elif saved is not None:
        method += f" + 검색어 다듬기 (재사용: {display_path(args.rewrite_from)})"
    if args.merge_original:
        method += " + 원래 질문 합치기"
    if args.rerank_candidates:
        rerank_input = "다듬은 검색어" if args.rerank_query == "rewrite" else "원래 질문"
        method += f" + 리랭크 ({settings.reranker_model}, 후보 {args.rerank_candidates}개, 입력: {rerank_input})"
    print(f"[검색] {method}")

    duplicates = sorted(qid for qid, count in Counter(item.qid for item in items).items() if count > 1)
    blank = [item.qid for item in items if not item.question]
    unknown = [item.qid for item in items if item.expected - valid_ids]
    usable = [item for item in items if item.question and not item.expected - valid_ids]
    print(
        f"[평가셋] {display_path(path)} — 질문 {len(items)}개"
        f" (평가 {len(usable)}개: 정답 있음 {sum(1 for i in usable if i.expected)} / 규정에 없음 {sum(1 for i in usable if not i.expected)})"
    )
    if duplicates:
        print(f"  [주의] 같은 id 가 여러 번 나옴: {', '.join(duplicates)}")
    if blank:
        print(f"  [제외] 질문이 비어 있음: {', '.join(blank)}")
    if unknown:
        print(f"  [제외] 정답 section_id 가 인덱스에 없음 (오타·분할 변경 확인): {', '.join(unknown)}")
    if not usable:
        return
    if saved is not None:
        # 짝지은 비교가 되도록 한 문항이라도 검색어가 없으면 채점하지 않는다
        missing = [item.qid for item in usable if item.qid not in saved]
        if missing:
            print(f"  [중단] --rewrite-from 파일에 검색어가 없는 질문: {', '.join(missing)}")
            return

    load_seconds = 0.0
    if args.rerank_candidates:
        # 모델 로딩과 첫 실행 준비는 서버를 띄울 때 한 번만 드는 비용이라 질문당 시간에서 뺀다
        start = time.perf_counter()
        get_reranker().predict([("준비", "준비")])
        load_seconds = time.perf_counter() - start

    if rewriter:
        print(f"[전송] 질문 {len(usable)}개를 {rewrite_name} API 로 보내 검색어로 다듬습니다 (규정 본문은 보내지 않음)")
    results = []
    failed: list[str] = []
    for item in usable:
        query = item.question
        if rewriter:
            try:
                query = rewrite_query(rewriter, item.question)
            except Exception:  # 질문 하나가 실패해도 원래 질문으로 검색하고 계속한다
                failed.append(item.qid)
        elif saved is not None:
            query = saved[item.qid]
        queries = [item.question, query] if args.merge_original else [query]
        rerank_query = query if args.rerank_query == "rewrite" else item.question
        retrieved = retrieve(retriever, queries, ks[-1], rerank_query, args.rerank_candidates)
        results.append(Result(item, *retrieved, query if rewriting else ""))
    if failed:
        print(f"  [주의] 검색어 다듬기 실패 → 원래 질문으로 검색: {', '.join(failed)}")
    answerable = [r for r in results if r.item.expected]
    unanswerable = [r for r in results if not r.item.expected]
    if answerable:
        print_scores(answerable, ks, base_k)
    print_similarity_gap(answerable, unanswerable)
    if args.rerank_candidates:
        print_rerank_time(results, load_seconds)

    print("\n[지표 설명]")
    print("  Hit@k    : 상위 k개 안에 정답 조문이 하나라도 있는 질문의 비율")
    print("  Recall@k : 정답 조문 중 상위 k개 안에 든 비율의 평균 (정답이 여러 개인 질문에서 Hit@k 와 달라진다)")
    print("  MRR      : 정답이 처음 나온 순위의 역수 평균 (1위 1.0, 2위 0.5, 4위 0.25, 못 찾음 0)")
    print("  1등 유사도: 규정에 없는 질문이 확실히 낮게 나오면 그 사이 값으로 'LLM 호출 전 거절' 기준을 정할 수 있다")

    label = retriever.mode + ("-rewrite" if rewriting else "") + ("-merge" if args.merge_original else "")
    if args.rerank_candidates:
        label += f"-rerank{args.rerank_candidates}-{args.rerank_query}"
    # 보류셋 결과를 튜닝 중에 열지 않도록 파일 이름에 평가셋 이름을 넣는다
    details = path.parent / "results" / f"retrieval-{path.stem}-{label}-{datetime.now():%Y%m%d-%H%M%S}.csv"
    save_details(results, details)
    print(f"\n[저장] {display_path(details)} (질문별 검색 결과·검색어, 규정명 포함 · 공유 금지)")


if __name__ == "__main__":
    main()
