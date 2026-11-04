"""채팅 화면: 질문마다 Chroma 에서 청크를 검색하고, 그 청크만 근거로 LLM 답변을 스트리밍한다 (Gradio).

검색 기본값은 M4b 다 (docs/2026-10-30 실험 기록).
질문을 LLM 으로 규정 말투의 검색어로 다듬고, 원래 질문과 다듬은 검색어로 각각 검색해 청크마다 더 높은 유사도로 합친다.
답변 맨 위에 다듬은 검색어를 보여 주고, 답변은 원래 질문으로 만든다. --no-rewrite 면 원래 질문으로만 검색한다 (v1).

질문마다 질문이(다듬기), 그리고 검색된 청크 본문이(답변) 외부 LLM API 로 전송된다.
화면에 규정 내용이 나오므로 이 PC(127.0.0.1)에서만 열고, 공개 링크(share)는 만들지 않는다.
후속 질문("그럼 계약직은?")을 이해하도록 같은 대화의 직전 질문 HISTORY_TURNS 개(기본 2)를 다듬기·답변 호출에 함께 보낸다.
이전 답변과 발췌는 보내지 않는다. 대화 지우기(휴지통) 버튼으로 새 주제를 시작한다. --history 0 이면 질문마다 독립적으로 처리한다.

실행:
    poetry run python scripts/chat.py
    poetry run python scripts/chat.py --provider openai   # 답변 LLM 지정 (기본: .env 의 LLM_PROVIDER)
    poetry run python scripts/chat.py -k 6                # 검색 청크 수 (기본: .env 의 RETRIEVER_K)
    poetry run python scripts/chat.py --mode hybrid       # 검색 방식 지정 (기본: .env 의 RETRIEVER_MODE)
    poetry run python scripts/chat.py --no-rewrite        # 검색어 다듬기 끄기 (원래 질문으로만 검색)
    poetry run python scripts/chat.py --rerank-candidates 5   # 상위 5개 후보를 로컬 리랭커로 다시 정렬 (외부 전송 없음)
    poetry run python scripts/chat.py --effort medium     # 답변 LLM 의 effort (기본: .env 의 ANSWER_EFFORT, 비면 모델 기본값)
    poetry run python scripts/chat.py --history 0         # 직전 질문을 쓰지 않음 (기본: .env 의 HISTORY_TURNS)

답변 위에 단계별 시간(다듬기·검색·리랭크)을, 아래에 첫 글자와 답변 완료까지의 시간과 호출별 토큰 수를 보여 준다.
토큰 수는 입력 / 출력이고, 출력 중 생각(thinking) 토큰은 API 가 알려 줄 때만 따로 나온다.
"""

import os

# 모델은 이미 받아 둔 캐시만 쓰고 실행 중에 Hugging Face 에 접속하지 않는다.
# gradio 가 import 될 때 huggingface_hub 설정을 읽으므로 .env 가 아니라 여기(맨 앞)에서 정한다.
# 새 모델을 받아야 할 때만 셸에서 HF_HUB_OFFLINE=0 으로 실행한다
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse  # noqa: E402
import time  # noqa: E402
from collections.abc import Iterator  # noqa: E402

import gradio as gr  # noqa: E402
from langchain_core.messages.ai import UsageMetadata  # noqa: E402

from regbot.config import settings  # noqa: E402
from regbot.models import EFFORT_LEVELS, SUPPORTED_PROVIDERS, get_reranker, llm_name  # noqa: E402
from regbot.rag import (  # noqa: E402
    build_answer_model,
    build_rewriter,
    format_context,
    format_history,
    rewrite_model,
    rewrite_query_with_usage,
)
from regbot.report import display_path  # noqa: E402
from regbot.retriever import SUPPORTED_MODES, Retriever, rerank  # noqa: E402


def message_text(content) -> str:
    """Gradio 히스토리 메시지 내용 → 글자. 문자열이거나 {"type": "text", "text": ...} 목록으로 온다 (파일 등은 무시)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text")
    return ""


def previous_questions(history: list, turns: int, max_chars: int) -> list[str]:
    """같은 대화의 직전 사용자 질문 최대 turns 개 (오래된 것부터). 이전 답변은 쓰지 않는다.

    비어 있거나 길이 상한을 넘어 거절된 질문은 외부로 보낸 적이 없으므로 빼고 센다.
    """
    if turns <= 0:
        return []
    questions = []
    for message in history:
        if isinstance(message, dict) and message.get("role") == "user":
            text = message_text(message.get("content")).strip()
            if text and len(text) <= max_chars:
                questions.append(text)
    return questions[-turns:]


def usage_text(label: str, usage: UsageMetadata | None) -> str:
    """'답변 입력 2,300 / 출력 900 (생각 700 + 본문 200)'. 생각 토큰은 출력 토큰에 포함된 값이다."""
    if not usage:
        return f"{label} 토큰 정보 없음"
    output = usage["output_tokens"]
    reasoning = usage.get("output_token_details", {}).get("reasoning")
    detail = f" (생각 {reasoning:,} + 본문 {output - reasoning:,})" if reasoning is not None else ""
    return f"{label} 입력 {usage['input_tokens']:,} / 출력 {output:,}{detail}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, default=settings.retriever_k, help="가져올 청크 수 (기본: .env 의 RETRIEVER_K)")
    parser.add_argument("--provider", choices=SUPPORTED_PROVIDERS, help="답변 LLM (기본: .env 의 LLM_PROVIDER)")
    parser.add_argument("--mode", choices=SUPPORTED_MODES, help="검색 방식 (기본: .env 의 RETRIEVER_MODE)")
    parser.add_argument("--no-rewrite", action="store_true", help="검색어 다듬기를 끄고 원래 질문으로만 검색")
    parser.add_argument(
        "--rerank-candidates",
        type=int,
        default=settings.rerank_candidates,
        help="검색 결과 상위 N개를 로컬 리랭커로 다시 정렬 (0: 끔, 기본: .env 의 RERANK_CANDIDATES)",
    )
    parser.add_argument(
        "--effort",
        choices=EFFORT_LEVELS,
        default=settings.answer_effort or None,
        help="답변 LLM 의 effort (기본: .env 의 ANSWER_EFFORT, 비면 모델 기본값)",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=settings.history_turns,
        help="다듬기·답변에 함께 보낼 직전 질문 수 (0: 안 씀, 기본: .env 의 HISTORY_TURNS)",
    )
    parser.add_argument("--port", type=int, default=7860, help="서버 포트 (기본: 7860)")
    args = parser.parse_args()
    if args.k < 1:
        parser.error("-k 는 1 이상이어야 합니다")
    if args.history < 0:
        parser.error("--history 는 0 이상이어야 합니다")
    if args.rerank_candidates < 0:
        parser.error("--rerank-candidates 는 0 이상이어야 합니다")

    # 임베딩 모델 로딩·BM25 색인이 오래 걸리므로 서버를 띄우기 전에 한 번만 만든다
    try:
        retriever = Retriever(args.mode)
    except ValueError as e:
        print(f"[중단] {e}")
        return
    print(f"[인덱스] {display_path(settings.chroma_dir)} / 컬렉션 '{settings.chroma_collection}' ({retriever.count:,}개)")
    if retriever.count == 0:
        print("  [중단] 인덱스가 비어 있음 → build_index.py 를 먼저 실행")
        return
    try:
        chain = build_answer_model(args.provider, args.effort)
        rewriter = None if args.no_rewrite else build_rewriter(args.provider, timeout=settings.rewrite_timeout)
    except ValueError as e:
        print(f"  [중단] {e}")
        return
    rewrite_name = llm_name(args.provider, rewrite_model(args.provider))
    method = retriever.mode + (f" + 검색어 다듬기 ({rewrite_name}) + 원래 질문 합치기" if rewriter else "")
    if args.rerank_candidates:
        # 리랭커 모델 로딩과 첫 실행 준비도 첫 질문이 느려지지 않게 서버를 띄우기 전에 한다
        get_reranker().predict([("준비", "준비")])
        method += f" + 리랭크 (후보 {args.rerank_candidates}개)"
    print(f"[검색] {method}")
    if rewriter:
        print(f"[전송] 질문마다 질문을 {rewrite_name} API 로 보내 검색어로 다듬습니다")
    answer_name = llm_name(args.provider) + f" (effort {args.effort or '기본값'})"
    print(f"[전송] 질문마다 상위 {args.k}개 청크 본문을 {answer_name} API 로 보냅니다")
    if args.history:
        print(f"[전송] 같은 대화의 직전 질문 최대 {args.history}개를 다듬기·답변에 함께 보냅니다 (이전 답변은 보내지 않음)")

    def answer(message: str, history: list) -> Iterator[str]:
        question = message.strip()
        if not question:
            return
        # 긴 문서를 붙여 넣으면 그 전문이 다듬기·답변 두 번 외부로 나가므로 LLM 을 부르지 않고 돌려보낸다
        if len(question) > settings.max_question_chars:
            yield (
                f"질문이 {len(question):,}자로 너무 깁니다 (최대 {settings.max_question_chars:,}자). "
                "문서를 붙여 넣지 말고 궁금한 점만 짧게 물어봐 주세요. (외부로 전송하지 않았습니다)"
            )
            return
        started = time.perf_counter()
        timings: list[str] = []  # 단계별 걸린 시간 (화면 표시용)
        usages: list[str] = []  # 호출별 토큰 수 (화면 표시용)
        search_query, prefix = question, ""
        # 후속 질문을 이해하도록 같은 대화의 직전 질문을 다듬기·답변에 함께 넣는다 (이전 답변은 넣지 않음)
        earlier = previous_questions(history, args.history, settings.max_question_chars)
        history_block = format_history(earlier)
        if earlier:
            prefix = f"*이전 질문 {len(earlier)}개 참고*\n\n"
        if rewriter:
            step = time.perf_counter()
            try:
                search_query, usage = rewrite_query_with_usage(rewriter, question, history_block)
                usages.append(usage_text("다듬기", usage))
                prefix += f"*검색어: {search_query}*\n\n"
            except Exception as e:  # 다듬기가 실패하거나 REWRITE_TIMEOUT 을 넘어도 원래 질문으로 검색한다
                prefix += f"*검색어 다듬기 실패 ({type(e).__name__}) → 원래 질문으로 검색*\n\n"
            timings.append(f"다듬기 {time.perf_counter() - step:.1f}초")
            yield prefix
        # 다듬지 않았거나 실패하면 두 검색어가 같아 한 번만 검색한다
        step = time.perf_counter()
        hits = retriever.search_all([question, search_query], max(args.k, args.rerank_candidates))
        timings.append(f"검색 {time.perf_counter() - step:.1f}초")
        if args.rerank_candidates:
            # 리랭커에는 다듬은 검색어를 넣는다 (다듬기 실패·끔이면 원래 질문과 같다).
            # 근거: 두 검색어 세트 모두 다듬은 검색어가 원래 질문과 같거나 나음 (docs/2026-11-03 1절, 11-04 확정).
            # 후보 수는 .env 의 RERANK_CANDIDATES (11-04 결정 8, docs/2026-11-04/후보모으기_비교.md)
            step = time.perf_counter()
            hits = rerank(search_query, hits, args.rerank_candidates)
            timings.append(f"리랭크 {time.perf_counter() - step:.1f}초")
        docs = [hit.doc for hit in hits[: args.k]]
        prefix += f"*{' · '.join(timings)}*\n\n"
        yield prefix
        partial, first_token, full = "", None, None
        try:
            for chunk in chain.stream({"context": format_context(docs), "question": question, "history": history_block}):
                # 조각을 이어 붙이면 마지막에 토큰 사용량(usage_metadata)도 합쳐진다
                full = chunk if full is None else full + chunk
                # .text 는 생각(thinking) 블록을 빼고 본문만 돌려준다
                token = chunk.text
                if first_token is None and token:
                    first_token = time.perf_counter() - started
                partial += token
                yield prefix + partial
        except Exception as e:  # API 오류 메시지는 요청 내용을 담지 않아 원인 확인용으로 함께 보여 준다
            yield f"{prefix}{partial}\n\n[실패] {type(e).__name__}: {e}"
            return
        total = f"답변 완료 {time.perf_counter() - started:.1f}초"
        usages.append(usage_text("답변", full.usage_metadata if full is not None else None))
        footer = f"*토큰: {' · '.join(usages)}*"
        if not partial.strip():
            # 사고(thinking) 토큰만으로 출력 상한에 닿거나 모델이 응답을 거절하면 텍스트 없이 끝난다
            yield prefix + f"(빈 답변: 응답에 텍스트가 없음)\n\n*{total}*\n\n{footer}"
            return
        yield prefix + partial + f"\n\n*첫 글자 {first_token:.1f}초 · {total}*\n\n{footer}"

    demo = gr.ChatInterface(
        fn=answer,
        title="사내 규정 챗봇",
        description=(
            f"규정 발췌 {args.k}개를 근거로 답합니다 · 검색 {method} · {answer_name}"
            + (f" · 직전 질문 {args.history}개 참고 (새 주제는 대화 지우기)" if args.history else "")
        ),
        analytics_enabled=False,
        # 대화가 .gradio/flagged/ 나 브라우저에 쌓이지 않게 한다 (기본값이 환경 변수에 따라 바뀌므로 고정)
        flagging_mode="never",
        save_history=False,
    )
    # GRADIO_SERVER_NAME·GRADIO_SHARE 환경 변수가 있어도 외부에 열리지 않도록 값을 고정한다.
    # run_history: gradio 6 은 기본으로 실행 기록을 남기고, 그 화면에서 Hugging Face 저장소로 보낼 수도 있어 끈다
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=False, inbrowser=True, run_history=False)


if __name__ == "__main__":
    main()
