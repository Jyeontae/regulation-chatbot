"""검색 확인: 질문과 가까운 청크를 Chroma 에서 찾아 순위·위치·유사도를 출력하고, --answer 면 LLM 답변까지 생성한다.

기본 출력은 카테고리·종류·조 위치까지만 보여 준다 (규정명·본문 없음).
--answer 는 질문마다 검색된 청크 본문을 외부 LLM API 로 보낸다.
--rewrite 는 질문을 외부 LLM API 로 보내 검색어로 다듬은 뒤 검색한다 (규정 본문은 보내지 않음). 답변은 원래 질문으로 만든다.
--merge-original 을 더하면 원래 질문으로도 검색해, 청크마다 더 높은 유사도로 합친다 (출처 열: 그 유사도를 낸 쪽).
질문을 인자로 주면 한 번 검색하고, 생략하면 빈 줄을 입력할 때까지 계속 받는다.
임베딩 모델 로딩이 실행 시간의 대부분이므로 여러 질문을 확인할 때는 인자 없이 실행한다.

실행:
    poetry run python scripts/search.py "휴가 신청은 언제까지 해야 해?"
    poetry run python scripts/search.py                  # 질문을 계속 입력
    poetry run python scripts/search.py -k 10            # 상위 10개 (기본: .env 의 RETRIEVER_K)
    poetry run python scripts/search.py --mode hybrid    # 검색 방식 지정 (기본: .env 의 RETRIEVER_MODE)
    poetry run python scripts/search.py --rewrite        # LLM 으로 질문을 검색어로 다듬어 검색 (외부 API 전송)
    poetry run python scripts/search.py --rewrite --merge-original   # 원래 질문과 다듬은 검색어로 각각 검색해 합침
    poetry run python scripts/search.py --show-names     # 규정명 표시 (본인 확인용, 공유 금지)
    poetry run python scripts/search.py --show-text      # 규정명·본문 표시 (본인 확인용, 공유 금지)
    poetry run python scripts/search.py --answer         # LLM 답변 생성 (본인 확인용, 공유 금지)
    poetry run python scripts/search.py --answer --provider openai   # 답변 LLM 지정 (기본: .env 의 LLM_PROVIDER)
"""

import argparse
from collections.abc import Iterator

from langchain_core.runnables import Runnable

from regbot.config import settings
from regbot.models import SUPPORTED_PROVIDERS, llm_name
from regbot.rag import build_chain, build_rewriter, format_context, rewrite_model, rewrite_query
from regbot.report import display_path, format_cell, print_table
from regbot.retriever import SUPPORTED_MODES, Retriever

COLUMNS = ["순위", "카테고리", "종류", "위치", "유사도"]
HYBRID_COLUMNS = ["벡터", "BM25"]  # 합치기 전 각 검색에서의 순위
SOURCES = ["질문", "검색어"]  # --merge-original 에서 점수를 낸 쪽 (Retriever.search_all 의 검색어 순서)


def read_queries() -> Iterator[str]:
    """빈 줄, Ctrl+C, 입력 끝(EOF)에서 멈춘다."""
    while True:
        try:
            query = input("\n질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not query:
            return
        yield query


def search(
    retriever: Retriever,
    query: str,
    k: int,
    show_names: bool,
    show_text: bool,
    chain: Runnable | None,
    rewriter: Runnable | None,
    merge: bool,
) -> None:
    search_query = query
    if rewriter:
        try:
            search_query = rewrite_query(rewriter, query)
        except Exception as e:  # 다듬기가 실패해도 원래 질문으로 검색한다
            print(f"  [다듬기 실패] {type(e).__name__}: {e} → 원래 질문으로 검색")
        else:
            print(f"  검색어: {search_query}")
    # 유사도는 코사인 유사도(check_setup.py 와 같은 척도). hybrid 에서 벡터 후보 밖이면 "-"
    hits = retriever.search_all([query, search_query] if merge else [search_query], k)
    hybrid = retriever.mode == "hybrid"
    rows = []
    for rank, hit in enumerate(hits, start=1):
        meta = hit.doc.metadata
        cosine = "-" if hit.cosine is None else f"{hit.cosine:.3f}"
        row = [str(rank), str(meta["category"]), str(meta["type"]), "#" + hit.doc.id.rsplit("#", 1)[-1], cosine]
        if hybrid:
            row += [format_cell(hit.vector_rank), format_cell(hit.bm25_rank)]
        if merge:
            row.append(SOURCES[hit.query_index])
        rows.append(row)
    columns = COLUMNS + (HYBRID_COLUMNS if hybrid else []) + (["출처"] if merge else [])
    names = [str(hit.doc.metadata["title"]) for hit in hits] if show_names else None
    print_table(columns, rows, 4, names)

    docs = [hit.doc for hit in hits]
    if show_text:
        for rank, doc in enumerate(docs, start=1):
            print(f"\n  [{rank}]")
            for line in doc.page_content.splitlines():
                print(f"    {line}")

    if chain is None:
        return
    print("\n[답변]")
    try:
        answer = chain.invoke({"context": format_context(docs), "question": query})
    except Exception as e:  # 질문 하나가 실패해도 계속. API 오류 메시지는 요청 내용을 담지 않아 원인 확인용으로 함께 출력
        print(f"  [실패] {type(e).__name__}: {e}")
        return
    if not answer.strip():
        # 사고(thinking) 토큰만으로 출력 상한에 닿거나 모델이 응답을 거절하면 텍스트 없이 끝난다
        print("  (빈 답변: 응답에 텍스트가 없음)")
        return
    for line in answer.splitlines():
        print(f"  {line}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", help="질문. 생략하면 빈 줄을 입력할 때까지 계속 받는다")
    parser.add_argument("-k", type=int, default=settings.retriever_k, help="가져올 청크 수 (기본: .env 의 RETRIEVER_K)")
    parser.add_argument("--show-names", action="store_true", help="규정명 표시 (본인 확인용, 공유 금지)")
    parser.add_argument("--show-text", action="store_true", help="규정명과 본문 표시 (본인 확인용, 공유 금지)")
    parser.add_argument(
        "--answer", action="store_true", help="LLM 답변 생성. 검색된 청크 본문이 외부 API 로 전송된다 (본인 확인용, 공유 금지)"
    )
    parser.add_argument("--provider", choices=SUPPORTED_PROVIDERS, help="답변 LLM (기본: .env 의 LLM_PROVIDER)")
    parser.add_argument("--mode", choices=SUPPORTED_MODES, help="검색 방식 (기본: .env 의 RETRIEVER_MODE)")
    parser.add_argument("--rewrite", action="store_true", help="LLM 으로 질문을 검색어로 다듬어 검색 (질문이 외부 API 로 전송된다)")
    parser.add_argument("--merge-original", action="store_true", help="원래 질문과 다듬은 검색어로 각각 검색해 높은 점수로 합친다")
    args = parser.parse_args()
    if args.k < 1:
        parser.error("-k 는 1 이상이어야 합니다")
    if args.provider and not (args.answer or args.rewrite):
        parser.error("--provider 는 --answer 또는 --rewrite 와 함께 쓴다")
    if args.merge_original and not args.rewrite:
        parser.error("--merge-original 은 --rewrite 와 함께 쓴다")

    # 컬렉션이 없으면 open_store 가 빈 컬렉션을 만들므로 개수로 인덱싱 여부를 확인한다
    try:
        retriever = Retriever(args.mode)
    except ValueError as e:
        print(f"[중단] {e}")
        return
    print(f"[인덱스] {display_path(settings.chroma_dir)} / 컬렉션 '{settings.chroma_collection}' ({retriever.count:,}개)")
    if retriever.count == 0:
        print("  [중단] 인덱스가 비어 있음 → build_index.py 를 먼저 실행")
        return
    print(f"[검색] {retriever.mode}" + (" + 원래 질문 합치기" if args.merge_original else ""))
    print("  유사도: 코사인 유사도. 1에 가까울수록 질문과 비슷하다")
    if args.merge_original:
        print("  합치기: 원래 질문·다듬은 검색어로 각각 검색해 청크마다 더 높은 점수를 쓴다. 출처 열은 그 점수를 낸 쪽")
    if retriever.mode == "hybrid":
        print("  순위: 벡터 순위와 BM25 순위를 RRF 로 합친 순서. 벡터·BM25 열은 각 검색의 순위 (후보 밖이면 -)")
    else:
        print("  순위: 유사도가 높은 순서")

    chain = None
    if args.answer:
        # 모델 ID 누락, LangSmith 추적 같은 설정 문제를 첫 질문 전에 드러내려고 체인을 미리 만든다
        try:
            chain = build_chain(args.provider)
        except ValueError as e:
            print(f"  [중단] {e}")
            return
        print(f"[전송] 질문마다 상위 {args.k}개 청크 본문을 {llm_name(args.provider)} API 로 보냅니다")

    rewriter = None
    if args.rewrite:
        try:
            rewriter = build_rewriter(args.provider)
        except ValueError as e:
            print(f"  [중단] {e}")
            return
        print(f"[전송] 질문마다 질문을 {llm_name(args.provider, rewrite_model(args.provider))} API 로 보내 검색어로 다듬습니다")

    # 답변에는 규정명·내용이 들어가므로 --answer 면 근거 표에도 규정명을 함께 보여 준다
    show_names = args.show_names or args.show_text or args.answer
    try:
        for query in [args.query] if args.query else read_queries():
            search(retriever, query, args.k, show_names, args.show_text, chain, rewriter, args.merge_original)
    except KeyboardInterrupt:  # 답변을 기다리는 중 Ctrl+C 로 멈춰도 아래 공유 금지 안내가 나오게 한다
        print("\n  [중단] Ctrl+C")

    if args.answer or args.show_text:
        print("\n※ 규정 내용(본문·답변)이 포함된 출력입니다. 공유하지 마세요.")
    elif args.show_names:
        print("\n※ 파일 이름이 포함된 출력입니다. 공유할 때는 --show-names 없이 다시 실행하세요.")


if __name__ == "__main__":
    main()
