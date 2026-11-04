"""답변 평가: 평가셋 질문마다 챗봇과 같은 과정으로 답변을 만들고, 기대 답과 비교해 채점한다.

채점은 세 층이다.
- 규칙 (로컬): 기대 답의 숫자(금액·일수 등)가 답변에 있는가 / 규정에 없는 질문에 "찾을 수 없습니다"로 답했는가 /
  근거 줄에 인용한 발췌가 정답 조문인가
- 유사도 (로컬): 기대 답과 답변 본문을 리랭커(크로스 인코더)로 비교한 0~1 점수. 참고 지표
- LLM 판정 (외부 API): 기대 답을 정답으로 보고 맞음 / 부분 / 틀림. 질문·기대 답·답변만 보내고 발췌는 보내지 않는다

effort·프롬프트를 여러 개 주면 같은 질문·같은 발췌로 설정(조합)마다 답변을 만들어 품질·시간·토큰을 한 표로 비교한다.
시간 흔들림이 설정 사이에 고르게 퍼지도록 질문마다 설정을 번갈아 실행한다.

외부 전송: 답변 생성에 질문 + 상위 k개 발췌(답변 LLM), LLM 판정에 질문 + 기대 답 + 답변(판정 LLM).
--rewrite-from 을 주지 않으면 질문을 다듬기 LLM 으로도 보낸다.
화면에는 점수·시간·토큰과 질문 id 만 출력한다 (질문·답변·규정명 없음).
질문별 답변과 판정 이유는 평가셋 옆 results/ 폴더에 CSV 로 남긴다 (규정 내용 포함, 공유 금지).

실행:
    poetry run python scripts/eval_answers.py --rewrite-from data/eval/results/<이전 결과>.csv --limit 2   # 2문항만 시험
    poetry run python scripts/eval_answers.py --rewrite-from <이전 결과>.csv --efforts default medium low
    poetry run python scripts/eval_answers.py --rewrite-from <이전 결과>.csv --prompts current narrow   # 프롬프트 비교
    poetry run python scripts/eval_answers.py --rewrite-from <이전 결과>.csv --no-judge   # LLM 판정 끄기 (로컬 채점만)
"""

import argparse
import csv
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, quantiles

from langchain_core.prompts import ChatPromptTemplate

from eval_retrieval import EvalItem, load_queries, parse_eval_set
from regbot.config import settings
from regbot.models import EFFORT_LEVELS, get_llm, get_reranker, llm_name
from regbot.rag import (
    DEFAULT_PROMPT_VARIANT,
    NOT_FOUND,
    PROMPT_VARIANTS,
    build_answer_model,
    build_rewriter,
    format_context,
    rewrite_query,
)
from regbot.report import display_path, print_table
from regbot.retriever import Retriever, rerank

DEFAULT_EFFORT = "default"  # effort 를 지정하지 않음 (모델 기본값, Sonnet 은 high 와 같다)
VERDICT_SCORES = {"맞음": 1.0, "부분": 0.5, "틀림": 0.0}

JUDGE_SYSTEM_PROMPT = """당신은 사내 규정 챗봇 답변의 채점자입니다. [기대 답]을 정답으로 보고 [답변]이 [질문]에 맞게 답했는지 판정합니다.

- 맞음: 기대 답의 핵심 내용(금액·기한·일수·조건·대상)이 모두 들어 있고, 기대 답과 어긋나는 내용이 없다.
- 부분: 핵심 내용 중 일부만 있거나, 맞는 내용과 기대 답과 어긋나는 내용이 섞여 있다.
- 틀림: 핵심 내용이 없거나 기대 답과 다르다. 답을 찾을 수 없다고 한 경우도 틀림이다.
- 기대 답에 없는 추가 설명은 기대 답과 어긋나지 않으면 감점하지 않는다. 표현 차이(1,000,000원과 100만 원 등)도 감점하지 않는다.

첫 줄에 "판정: 맞음", "판정: 부분", "판정: 틀림" 중 하나를 쓰고, 둘째 줄에 "이유: "로 시작하는 한 문장을 씁니다."""

JUDGE_HUMAN_PROMPT = """[질문]
{question}

[기대 답]
{expected}

[답변]
{answer}"""

JUDGE_PROMPT = ChatPromptTemplate.from_messages([("system", JUDGE_SYSTEM_PROMPT), ("human", JUDGE_HUMAN_PROMPT)])

# 1,000,000 / 100만 / 3 / 0.5 같은 수. 만·억 단위는 곱해서 같은 값으로 맞춘다
NUMBER = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(억|만)?")
CITATION_LINE = re.compile(r"^\s*\**\s*근거\s*[:：]")
CITATION_NUMBER = re.compile(r"\[(\d+)\]")


@dataclass
class Answer:
    item: EvalItem
    config: str  # 설정 이름 (effort, 프롬프트가 여럿이면 effort/프롬프트)
    sections: list[str]  # LLM 에 넘긴 발췌의 section_id (번호순)
    text: str = ""
    error: str = ""
    first_text_seconds: float | None = None  # 답변 LLM 호출 시작 → 첫 본문 글자
    total_seconds: float = 0.0  # 답변 LLM 호출 시작 → 끝
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    verdict: str = ""  # 맞음 / 부분 / 틀림 / 판정 실패 / (판정 안 함이면 빈 문자열)
    verdict_reason: str = ""
    similarity: float | None = None
    scores: dict[str, float | None] = field(default_factory=dict)  # 규칙 채점 결과


def split_citation(text: str) -> tuple[str, str]:
    """(본문, 근거 줄). 근거 줄이 없으면 근거는 빈 문자열."""
    lines = text.rstrip().splitlines()
    for index in range(len(lines) - 1, -1, -1):
        if CITATION_LINE.match(lines[index]):
            return "\n".join(lines[:index]).strip(), "\n".join(lines[index:]).strip()
    return text.strip(), ""


def numbers(text: str) -> set[float]:
    values = set()
    for match in NUMBER.finditer(text):
        value = float(match.group(1).replace(",", ""))
        values.add(value * {"억": 1e8, "만": 1e4}.get(match.group(2) or "", 1))
    return values


def is_refusal(text: str) -> bool:
    """답 전체가 거절인가. 답 중간의 "○○은 찾을 수 없습니다"(일부만 못 찾음)는 거절로 보지 않는다."""
    return text.strip().strip("*").strip().rstrip(".") == NOT_FOUND.rstrip(".")


def mentions_not_found(text: str) -> bool:
    """답 일부에서 못 찾았다고 밝혔는가 (여러 조항 질문에서 필요한 조문이 발췌에 없을 때)."""
    return NOT_FOUND.rstrip(".") in text and not is_refusal(text)


def rule_scores(answer: Answer, expected_answer: str) -> dict[str, float | None]:
    """규칙 채점. 해당하지 않는 지표는 None (평균에서 뺀다)."""
    body, citation = split_citation(answer.text)
    answerable = bool(answer.item.expected)
    refused = is_refusal(body)
    scores: dict[str, float | None] = {
        # 정답 있는 질문은 거절하지 않아야, 규정에 없는 질문은 거절해야 맞다
        "거절 판단": float(refused != answerable),
        "숫자 포함": None,
        "근거 줄": None,
        "정답 인용": None,
    }
    if not answerable:
        return scores
    wanted = numbers(expected_answer)
    if wanted:
        scores["숫자 포함"] = len(wanted & numbers(body)) / len(wanted)
    scores["근거 줄"] = float(bool(citation))
    cited = {int(n) for n in CITATION_NUMBER.findall(citation)}
    cited_sections = {answer.sections[n - 1] for n in cited if 1 <= n <= len(answer.sections)}
    scores["정답 인용"] = float(bool(cited_sections & answer.item.expected))
    return scores


def parse_verdict(text: str) -> tuple[str, str]:
    verdict = re.search(r"판정\s*[:：]\s*(맞음|부분|틀림)", text)
    reason = re.search(r"이유\s*[:：]\s*(.+)", text)
    return (verdict.group(1) if verdict else "판정 실패"), (reason.group(1).strip() if reason else text.strip()[:200])


def generate(chain, docs, question: str, answer: Answer) -> None:
    """답변을 스트리밍으로 받아 본문·시간·토큰을 answer 에 채운다."""
    start = time.perf_counter()
    full = None
    try:
        for chunk in chain.stream({"context": format_context(docs), "question": question}):
            full = chunk if full is None else full + chunk
            # .text 는 생각(thinking) 블록을 빼고 본문만 돌려준다
            if chunk.text and answer.first_text_seconds is None:
                answer.first_text_seconds = time.perf_counter() - start
            answer.text += chunk.text
    except Exception as e:  # 한 문항이 실패해도 계속한다. 오류 내용은 CSV 에만 남긴다
        answer.error = f"{type(e).__name__}: {e}"
    answer.total_seconds = time.perf_counter() - start
    usage = full.usage_metadata if full is not None else None
    if usage:
        answer.input_tokens = usage["input_tokens"]
        answer.output_tokens = usage["output_tokens"]
        answer.reasoning_tokens = usage.get("output_token_details", {}).get("reasoning")


def p90(values: list[float]) -> float:
    return quantiles(values, n=10, method="inclusive")[-1] if len(values) > 1 else values[0]


def average(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return mean(present) if present else None


def cell(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def print_summary(answers: list[Answer], configs: list[str], judged: bool) -> None:
    by_config: dict[str, list[Answer]] = defaultdict(list)
    for answer in answers:
        by_config[answer.config].append(answer)

    print("\n[품질] 점수는 0~1 평균 (해당 문항만)")
    columns = ["설정", "LLM 판정", "맞음/부분/틀림", "숫자 포함", "정답 인용", "근거 줄", "거절 판단", "유사도"]
    rows = []
    for config in configs:
        group = by_config[config]
        verdicts = [a.verdict for a in group if a.item.expected and a.verdict in VERDICT_SCORES]
        counts = "/".join(str(verdicts.count(v)) for v in VERDICT_SCORES) if judged else "-"
        rows.append(
            [
                config,
                cell(average([VERDICT_SCORES[v] for v in verdicts])) if judged else "-",
                counts,
                *(cell(average([a.scores.get(name) for a in group])) for name in ("숫자 포함", "정답 인용", "근거 줄", "거절 판단")),
                cell(average([a.similarity for a in group if a.item.expected])),
            ]
        )
    print_table(columns, rows, 1)

    print("\n[시간·토큰] 답변 LLM 호출만 (검색·리랭크 제외). 시간은 평균 / 상위 10% 경계")
    columns = ["설정", "첫 글자", "답변 완료", "입력 토큰", "생각 토큰", "본문 토큰"]
    rows = []
    for config in configs:
        group = [a for a in by_config[config] if not a.error]
        firsts = [a.first_text_seconds for a in group if a.first_text_seconds is not None]
        totals = [a.total_seconds for a in group]
        reasoning = [a.reasoning_tokens for a in group]
        bodies = [
            a.output_tokens - (a.reasoning_tokens or 0) for a in group if a.output_tokens is not None
        ]
        rows.append(
            [
                config,
                f"{mean(firsts):.1f} / {p90(firsts):.1f}초" if firsts else "-",
                f"{mean(totals):.1f} / {p90(totals):.1f}초" if totals else "-",
                cell(average([a.input_tokens for a in group]), 0),
                cell(average(reasoning), 0),
                cell(average(bodies), 0),
            ]
        )
    print_table(columns, rows, 1)

    if judged:
        print("\n[유형별 LLM 판정 점수]")
        types = sorted({a.item.qtype for a in answers if a.item.expected})
        rows = []
        for qtype in types:
            row = [qtype, str(sum(1 for a in by_config[configs[0]] if a.item.qtype == qtype))]
            for config in configs:
                scores = [
                    VERDICT_SCORES[a.verdict]
                    for a in by_config[config]
                    if a.item.qtype == qtype and a.verdict in VERDICT_SCORES
                ]
                row.append(cell(average(scores)))
            rows.append(row)
        print_table(["유형", "질문", *configs], rows, 1)

    print("\n[확인할 질문]")
    for config in configs:
        group = by_config[config]
        wrong = [a.item.qid for a in group if a.verdict == "틀림"]
        partial = [a.item.qid for a in group if a.verdict == "부분"]
        refusal = [a.item.qid for a in group if a.scores.get("거절 판단") == 0.0]
        not_found = [a.item.qid for a in group if a.item.expected and mentions_not_found(split_citation(a.text)[0])]
        failed = [a.item.qid for a in group if a.error or a.verdict == "판정 실패"]
        print(f"  {config}: 틀림 {', '.join(wrong) or '없음'} / 부분 {', '.join(partial) or '없음'}"
              f" / 거절 판단 틀림 {', '.join(refusal) or '없음'} / 일부 못 찾음 {', '.join(not_found) or '없음'}"
              f" / 실패 {', '.join(failed) or '없음'}")


def save_details(answers: list[Answer], expected_answers: dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig: 엑셀에서 열어도 한글이 깨지지 않게 한다
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["id", "유형", "설정", "판정", "판정 이유", "숫자 포함", "정답 인용", "근거 줄", "거절 판단", "유사도",
             "첫 글자(초)", "답변 완료(초)", "입력 토큰", "출력 토큰", "생각 토큰", "넘긴 발췌 (section_id)",
             "기대 답", "답변", "오류"]
        )
        for a in answers:
            writer.writerow(
                [a.item.qid, a.item.qtype, a.config, a.verdict, a.verdict_reason,
                 *(cell(a.scores.get(name)) for name in ("숫자 포함", "정답 인용", "근거 줄", "거절 판단")),
                 cell(a.similarity, 3), cell(a.first_text_seconds, 2), cell(a.total_seconds, 2),
                 a.input_tokens, a.output_tokens, a.reasoning_tokens, " ; ".join(a.sections),
                 expected_answers.get(a.item.qid, ""), a.text, a.error]
            )


def expected_answer_column(path: Path) -> dict[str, str]:
    """평가 표의 '기대 답' 열 (id → 기대 답). parse_eval_set 은 이 열을 읽지 않으므로 따로 읽는다."""
    answers: dict[str, str] = {}
    columns: dict[str, int] = {}
    in_fence = False
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line.startswith("```"):
            if columns:
                break
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if not line.startswith("|"):
            if columns:
                break
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not columns:
            names = [c.lower() for c in cells]
            if "id" in names and any("기대" in c for c in cells):
                columns = {"id": names.index("id"), "answer": next(i for i, c in enumerate(cells) if "기대" in c)}
            continue
        if all(c and set(c) <= set(":-") for c in cells):
            continue
        cells += [""] * (max(columns.values()) + 1 - len(cells))
        answers[cells[columns["id"]]] = cells[columns["answer"]]
    return answers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=settings.eval_set_path, help="평가셋 (기본: .env 의 EVAL_SET_PATH)")
    parser.add_argument("--rewrite-from", type=Path, help="이전 검색 평가 CSV 의 검색어를 재사용 (없으면 다듬기 LLM 호출)")
    parser.add_argument("-k", type=int, default=settings.retriever_k, help="LLM 에 넘길 발췌 수 (기본: .env 의 RETRIEVER_K)")
    parser.add_argument(
        "--rerank-candidates", type=int, default=settings.rerank_candidates,
        help="검색 결과 상위 N개를 로컬 리랭커로 다시 정렬 (0: 끔, 기본: .env 의 RERANK_CANDIDATES)",
    )
    parser.add_argument(
        "--efforts", nargs="+", choices=(DEFAULT_EFFORT, *EFFORT_LEVELS),
        default=[settings.answer_effort or DEFAULT_EFFORT],
        help=f"비교할 답변 LLM effort ({DEFAULT_EFFORT}: 지정 안 함 = 모델 기본값, 기본: .env 의 ANSWER_EFFORT)",
    )
    parser.add_argument(
        "--prompts", nargs="+", choices=PROMPT_VARIANTS, default=[DEFAULT_PROMPT_VARIANT],
        help=f"비교할 답변 프롬프트 (발췌끼리 다를 때의 규칙만 다르다, 기본: {DEFAULT_PROMPT_VARIANT})",
    )
    parser.add_argument("--no-judge", action="store_true", help="LLM 판정을 끄고 로컬 채점만 한다 (판정용 외부 전송 없음)")
    parser.add_argument("--judge-model", help="판정 모델 ID (기본: 답변 모델과 같은 모델, effort low)")
    parser.add_argument("--limit", type=int, help="앞에서부터 N문항만 (시험 실행용)")
    args = parser.parse_args()
    if args.k < 1:
        parser.error("-k 는 1 이상이어야 합니다")
    if args.rerank_candidates < 0:
        parser.error("--rerank-candidates 는 0 이상이어야 합니다")
    if args.rewrite_from and not args.rewrite_from.exists():
        parser.error(f"--rewrite-from 파일이 없음: {args.rewrite_from}")
    efforts = list(dict.fromkeys(args.efforts))  # 같은 값을 두 번 주면 한 번만
    prompts = list(dict.fromkeys(args.prompts))
    # 설정 이름 → (effort, 프롬프트). 하나만 비교하는 쪽은 이름에서 뺀다
    configs: dict[str, tuple[str, str]] = {}
    for effort in efforts:
        for prompt in prompts:
            name = "/".join(([effort] if len(efforts) > 1 or len(prompts) == 1 else []) + ([prompt] if len(prompts) > 1 else []))
            configs[name] = (effort, prompt)

    path: Path = args.eval_set
    if not path.exists():
        print(f"[중단] 평가셋이 없음: {display_path(path)}")
        return
    found, items = parse_eval_set(path)
    expected_answers = expected_answer_column(path)
    if not found or not items:
        print(f"[중단] {display_path(path)} 에서 평가 질문을 찾지 못함")
        return
    if args.limit:
        items = items[: args.limit]
    missing_answers = [i.qid for i in items if i.expected and not expected_answers.get(i.qid)]
    if missing_answers:
        print(f"[중단] 기대 답이 비어 있는 질문: {', '.join(missing_answers)}")
        return

    try:
        retriever = Retriever()
        # DEFAULT_EFFORT 는 빈 문자열로 넘겨 .env 의 ANSWER_EFFORT 도 쓰지 않게 한다 (모델 기본값)
        chains = {
            name: build_answer_model(effort="" if effort == DEFAULT_EFFORT else effort, prompt=prompt)
            for name, (effort, prompt) in configs.items()
        }
        rewriter = None if args.rewrite_from else build_rewriter()
    except ValueError as e:
        print(f"[중단] {e}")
        return
    saved = load_queries(args.rewrite_from) if args.rewrite_from else {}
    missing_queries = [i.qid for i in items if args.rewrite_from and i.qid not in saved]
    if missing_queries:
        print(f"[중단] --rewrite-from 파일에 검색어가 없는 질문: {', '.join(missing_queries)}")
        return
    judge_model = args.judge_model or (settings.openai_model if settings.llm_provider == "openai" else settings.anthropic_model)
    judge = None if args.no_judge else JUDGE_PROMPT | get_llm(model=judge_model, effort="low")

    rewrite_note = f"재사용: {display_path(args.rewrite_from)}" if args.rewrite_from else "다듬기 LLM 호출"
    print(f"[평가셋] {display_path(path)} — 질문 {len(items)}개 (정답 있음 {sum(1 for i in items if i.expected)})")
    print(f"[검색] {retriever.mode} + 검색어 다듬기 ({rewrite_note}) + 원래 질문 합치기"
          + (f" + 리랭크 (후보 {args.rerank_candidates}개)" if args.rerank_candidates else "") + f", 발췌 {args.k}개")
    print(f"[답변] {llm_name()} · effort {', '.join(efforts)} · 프롬프트 {', '.join(prompts)}")
    print(f"[전송] 질문 {len(items)}개 × 설정 {len(configs)}개: 질문 + 발췌 {args.k}개 → {llm_name()}")
    if judge:
        print(f"[전송] LLM 판정: 질문 + 기대 답 + 답변 → {llm_name(None, judge_model)} (발췌는 보내지 않음)")
    if rewriter:
        print("[전송] 질문 → 다듬기 LLM")
    if args.rerank_candidates:
        get_reranker().predict([("준비", "준비")])

    answers: list[Answer] = []
    started = time.perf_counter()
    for index, item in enumerate(items, start=1):
        query = saved[item.qid] if args.rewrite_from else rewrite_query(rewriter, item.question)
        hits = retriever.search_all([item.question, query], max(args.k, args.rerank_candidates))
        if args.rerank_candidates:
            # 채팅 화면과 같게 다듬은 검색어로 리랭크한다 (11-04 확정)
            hits = rerank(query, hits, args.rerank_candidates)
        docs = [hit.doc for hit in hits[: args.k]]
        sections = [str(doc.metadata["section_id"]) for doc in docs]
        expected_answer = expected_answers.get(item.qid, "")
        for name in configs:
            answer = Answer(item, name, sections)
            generate(chains[name], docs, item.question, answer)
            if not answer.error:
                answer.scores = rule_scores(answer, expected_answer)
                body, _ = split_citation(answer.text)
                if item.expected and body:
                    answer.similarity = float(get_reranker().predict([(expected_answer, body)])[0])
                if judge and item.expected:
                    try:
                        reply = judge.invoke({"question": item.question, "expected": expected_answer, "answer": body})
                        answer.verdict, answer.verdict_reason = parse_verdict(reply.text)
                    except Exception as e:
                        answer.verdict, answer.verdict_reason = "판정 실패", f"{type(e).__name__}: {e}"
            answers.append(answer)
        print(f"  진행 {index}/{len(items)} ({time.perf_counter() - started:.0f}초)", flush=True)

    print_summary(answers, list(configs), judge is not None)
    print("\n[지표 설명]")
    print("  LLM 판정 : 맞음 1 / 부분 0.5 / 틀림 0 의 평균 (정답 있는 질문만)")
    print("  숫자 포함: 기대 답에 나온 수(금액·일수 등) 중 답변 본문에 있는 비율 (기대 답에 수가 있는 질문만)")
    print("  정답 인용: 근거 줄에 인용한 발췌 중 정답 조문이 있는 질문의 비율")
    print("  거절 판단: 규정에 없는 질문은 거절, 정답 있는 질문은 거절하지 않은 비율 (전체 질문)")
    print("  유사도   : 기대 답과 답변 본문의 크로스 인코더 점수 (참고용, 답이 길면 낮게 나올 수 있다)")

    label = "-".join(efforts + prompts) + (f"-rerank{args.rerank_candidates}" if args.rerank_candidates else "")
    details = path.parent / "results" / f"answers-{path.stem}-{label}-{datetime.now():%Y%m%d-%H%M%S}.csv"
    save_details(answers, expected_answers, details)
    print(f"\n[저장] {display_path(details)} (질문별 답변·판정 이유, 규정 내용 포함 · 공유 금지)")


if __name__ == "__main__":
    main()
