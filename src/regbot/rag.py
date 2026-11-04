"""답변 생성: 검색된 청크만 근거로 LLM 이 질문에 답하는 체인 (prompt | llm | parser).

LLM 은 외부 API 이므로 체인을 호출할 때마다 context 로 넘긴 청크 본문이 외부로 전송된다.
검색어 다듬기 체인은 질문만 보낸다 (규정 본문 없음).
"""

from langchain_core.documents import Document
from langchain_core.messages.ai import UsageMetadata
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langsmith.utils import tracing_is_enabled

from regbot.config import settings
from regbot.models import get_llm

NOT_FOUND = "해당 규정을 찾을 수 없습니다."

SYSTEM_PROMPT_TEMPLATE = f"""당신은 사내 규정 안내 도우미입니다. [규정 발췌]에 적힌 내용만 근거로 [질문]에 답합니다.

- 발췌에 답이 없으면 "{NOT_FOUND}"라고만 답하고, 일반 상식이나 추측으로 보충하지 않습니다.
- 금액·기한·일수·조건은 발췌에 적힌 표현 그대로 씁니다.
- {{diff_rule}}
- 답변 마지막 줄에 근거로 쓴 발췌를 "근거: [번호] 규정명 위치" 형식으로 모두 적습니다. 위치는 발췌 첫 줄 머리말에 적힌 대로 쓰고(예: 제5조, 부칙 제1조, 별표 1), 머리말에 없는 조 번호는 만들지 않습니다.
- 한국어로 간결하게 답합니다."""

# 발췌끼리 내용이 다를 때의 규칙. 같은 규정이 두 곳에 있거나(옛 버전) 본문과 별표가 어긋나면 조용히 하나만 답하지 않게 한다.
# current 는 별개 제도(지급 주체가 다른 규정)끼리도 비교해 충돌이 아닌 것을 차이로 짚고 답이 길어졌다 (11-02)
DIFF_RULES = {
    "current": "발췌끼리 내용이 다르면 어느 발췌가 어떻게 다른지 함께 알립니다.",
    "narrow": (
        "서로 다른 규정·제도의 내용은 각각 나누어 안내하고 서로 비교하지 않습니다. "
        "같은 규정의 같은 항목인데 발췌끼리 값이 다를 때만 어느 발췌가 어떻게 다른지 함께 알립니다."
    ),
}
PROMPT_VARIANTS = tuple(DIFF_RULES)
# narrow 채택 (11-02 답변 평가): 품질 같음(LLM 판정 0.82 → 0.84), 차이 안내 답변 12 → 6문항, 본문 길이는 같음
DEFAULT_PROMPT_VARIANT = "narrow"

SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.format(diff_rule=DIFF_RULES[DEFAULT_PROMPT_VARIANT])

# {history} 는 채팅 화면의 직전 질문 블록(format_history). 비어 있으면(기본) 평가 때와 같은 프롬프트다
HUMAN_PROMPT = """{history}[규정 발췌]
{context}

[질문]
{question}"""

PROMPTS = {
    variant: ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT_TEMPLATE.format(diff_rule=rule)), ("human", HUMAN_PROMPT)]
    ).partial(history="")
    for variant, rule in DIFF_RULES.items()
}
PROMPT = PROMPTS[DEFAULT_PROMPT_VARIANT]


def format_context(docs: list[Document]) -> str:
    # 청크 첫 줄의 위치 머리말(카테고리 > 규정명 > … > 조·부칙·별표)이 함께 들어가 LLM 이 출처를 인용할 수 있다
    return "\n\n".join(f"[{i}] {doc.page_content}" for i, doc in enumerate(docs, start=1))


REWRITE_SYSTEM_PROMPT = """당신은 사내 규정 검색어 작성 도우미입니다. [질문]을 사내 규정 문서에서 찾기 좋은 검색어 한 줄로 바꿉니다.

- 일상어·구어체 표현은 규정 문서에서 쓰는 명사형 용어로 바꿉니다.
- 질문에 있는 대상·조건·기간·금액 같은 내용은 모두 남기고, 질문에 없는 내용은 더하지 않습니다.
- 규정 이름이나 조 번호는 질문에 있을 때만 씁니다.
- 검색어 한 줄만 출력합니다. 설명이나 따옴표를 붙이지 않습니다."""

REWRITE_HUMAN_PROMPT = """{history}[질문]
{question}"""

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [("system", REWRITE_SYSTEM_PROMPT), ("human", REWRITE_HUMAN_PROMPT)]
).partial(history="")

# 직전 질문 블록. 지시 문장을 시스템 프롬프트가 아니라 이 블록에 넣어, 직전 질문이 없으면 평가 때와 같은 프롬프트가 되게 한다.
# 이전 답변·발췌는 넣지 않는다 (외부 전송을 질문으로만 한정, 이전 답의 오류가 이어지는 것도 막는다)
HISTORY_HEADER = (
    "[이전 질문] (오래된 것부터. [질문]이 '그럼', '그건', '계약직은?'처럼 앞 질문에 기대어 대상·조건이 빠져 있을 때만 "
    "참고해 빠진 부분을 채우고, 새 주제면 무시합니다. 여기 있는 대상·조건은 질문에 있는 내용으로 봅니다)"
)


def format_history(previous_questions: list[str]) -> str:
    """직전 질문 목록 → 프롬프트 앞에 붙일 블록. 비어 있으면 빈 문자열 (프롬프트가 바뀌지 않음)."""
    if not previous_questions:
        return ""
    lines = "\n".join(f"- {q}" for q in previous_questions)
    return f"{HISTORY_HEADER}\n{lines}\n\n"

# REWRITE_MODEL 이 비어 있을 때 쓰는 다듬기 모델. 검색어 한 줄이라 작은 모델로 충분하다 (M4·M4b 평가 모델)
REWRITE_MODEL_DEFAULTS = {"anthropic": "claude-haiku-5-5"}


def _check_tracing() -> None:
    # LangSmith 추적이 켜져 있으면 체인 입력(질문, 검색된 규정 본문)이 LangSmith 서버에도 저장된다
    if tracing_is_enabled():
        raise ValueError("LangSmith 추적이 켜져 있습니다. .env 의 LANGSMITH_TRACING / LANGCHAIN_TRACING_V2 를 끄고 실행하세요.")


def build_answer_model(
    provider: str | None = None, effort: str | None = None, prompt: str = DEFAULT_PROMPT_VARIANT
) -> Runnable:
    """입력 {"context": format_context(docs), "question": 질문} → LLM 메시지 (본문 + 토큰 사용량 usage_metadata).

    provider 생략 시 .env 의 LLM_PROVIDER. effort 생략(None) 시 .env 의 ANSWER_EFFORT 를 쓰고,
    빈 문자열이면 effort 를 지정하지 않는다 (모델 기본값, Claude 에만 적용).
    낮출수록 답하기 전에 생각하는 양이 줄어 첫 글자가 빨리 나온다.
    prompt 는 PROMPT_VARIANTS 중 하나 (발췌끼리 다를 때의 규칙만 다르다).
    """
    _check_tracing()
    effort = settings.answer_effort if effort is None else effort
    return PROMPTS[prompt] | get_llm(provider, effort=effort or None)


def build_chain(provider: str | None = None, effort: str | None = None) -> Runnable:
    """build_answer_model() 과 같은 입력 → 답변 문자열."""
    return build_answer_model(provider, effort) | StrOutputParser()


def rewrite_model(provider: str | None = None) -> str | None:
    """다듬기 모델 ID. .env 의 REWRITE_MODEL, 비어 있으면 회사별 기본값, 그것도 없으면 None (답변 모델)."""
    provider = (provider or settings.llm_provider).lower()
    return settings.rewrite_model or REWRITE_MODEL_DEFAULTS.get(provider)


def build_rewriter(provider: str | None = None, timeout: float | None = None) -> Runnable:
    """입력 {"question": 질문} → LLM 메시지 (검색어 + 토큰 사용량). 모델은 rewrite_model() 참고.

    effort 는 .env 의 REWRITE_EFFORT (기본 low, 비우면 모델 기본값).
    timeout(초)을 주면 그 시간 안에 응답이 없을 때 재시도 없이 예외를 낸다. 채팅 화면은 이때 원래 질문으로 검색한다.
    평가 스크립트는 timeout 없이 불러 검색어가 조용히 원래 질문으로 바뀌지 않게 한다.
    """
    _check_tracing()
    # 한 줄 바꾸기라 깊은 사고가 필요 없다. 지연을 줄이려고 effort 를 낮춘다
    retries = 0 if timeout is not None else None
    llm = get_llm(
        provider, rewrite_model(provider), effort=settings.rewrite_effort or None, timeout=timeout, max_retries=retries
    )
    return REWRITE_PROMPT | llm


def rewrite_query_with_usage(
    rewriter: Runnable, question: str, history: str = ""
) -> tuple[str, UsageMetadata | None]:
    """(검색어 한 줄, 토큰 사용량). 여러 줄이면 첫 줄을 쓰고, 빈 답이면 원래 질문으로 검색한다.

    history 는 format_history() 결과 (채팅 화면의 직전 질문). 비우면 질문만 보낸다.
    """
    message = rewriter.invoke({"question": question, "history": history})
    # .text 는 생각(thinking) 블록을 빼고 본문만 돌려준다
    lines = (line.strip().strip("\"'") for line in message.text.splitlines())
    return next((line for line in lines if line), question), message.usage_metadata


def rewrite_query(rewriter: Runnable, question: str) -> str:
    """검색어 한 줄 (rewrite_query_with_usage 의 검색어만)."""
    return rewrite_query_with_usage(rewriter, question)[0]
