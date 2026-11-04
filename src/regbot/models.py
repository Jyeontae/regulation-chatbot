"""임베딩·LLM 생성: .env 설정에 따라 알맞은 모델 객체를 만든다."""

from functools import lru_cache
from typing import TYPE_CHECKING

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from regbot.config import settings

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

SUPPORTED_PROVIDERS = ("openai", "anthropic")
# Claude 의 output_config.effort 값. 지정하지 않으면 Sonnet 은 high 와 같다
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
# 리랭커가 질문 + 청크에서 읽는 최대 토큰 수. 넘는 뒷부분은 잘린다 (bge-reranker-v2-m3 모델 카드 예시 값)
RERANK_MAX_LENGTH = 512


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """로컬 임베딩 모델 (기본: BAAI/bge-m3). 처음 실행 시 모델을 내려받는다."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
        # 정규화하면 내적 = 코사인 유사도
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_reranker() -> "CrossEncoder":
    """로컬 리랭커 (기본: BAAI/bge-reranker-v2-m3). 질문과 청크를 한 쌍씩 함께 읽어 관련도를 매긴다.

    처음 실행 시 모델을 내려받는다. 장치는 임베딩과 같은 EMBEDDING_DEVICE 를 쓴다.
    """
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.reranker_model, device=settings.embedding_device, max_length=RERANK_MAX_LENGTH)


def llm_name(provider: str | None = None, model: str | None = None) -> str:
    """출력용 'provider/model' 이름. provider 를 생략하면 .env 의 LLM_PROVIDER 를 사용한다."""
    provider = (provider or settings.llm_provider).lower()
    model = model or (settings.openai_model if provider == "openai" else settings.anthropic_model)
    return f"{provider}/{model}"


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    effort: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> BaseChatModel:
    """답변 생성 LLM. provider 를 생략하면 .env 의 LLM_PROVIDER, model 을 생략하면 provider 의 .env 모델을 사용한다.

    Phase 5 에서 두 모델을 비교할 때는 get_llm("openai"), get_llm("anthropic") 처럼 직접 지정한다.
    effort 는 Claude 면 output_config.effort (low~max), OpenAI 면 추론 모델의 reasoning_effort 로 넘긴다.
    gpt-5.4-mini 의 reasoning_effort 는 none(기본)·low·medium·high·xhigh 라 max 는 xhigh 로 바꾼다.
    none 은 OpenAI 에서만 뜻이 있어 Claude 에는 effort 를 지정하지 않는다(모델 기본값).
    timeout(초)·max_retries 를 생략하면 라이브러리 기본값을 쓴다 (검색어 다듬기만 짧게 지정한다).
    """
    provider = (provider or settings.llm_provider).lower()

    kwargs = {}
    if settings.llm_temperature is not None:
        kwargs["temperature"] = settings.llm_temperature
    if max_retries is not None:
        kwargs["max_retries"] = max_retries

    if provider == "openai":
        model = model or settings.openai_model
        if not model:
            raise ValueError(".env 의 OPENAI_MODEL 이 비어 있습니다. 사용할 OpenAI 모델 ID를 입력하세요.")
        from langchain_openai import ChatOpenAI

        if effort:
            kwargs["reasoning_effort"] = "xhigh" if effort == "max" else effort
        if timeout is not None:
            kwargs["request_timeout"] = timeout
        # stream_usage: 스트리밍할 때도 마지막 조각에 토큰 사용량을 받는다 (채팅 화면·답변 평가의 토큰 표시)
        # store=False: 요청·응답을 OpenAI 쪽에 저장하지 않게 명시한다 (공식 문서의 기본값 설명이 서로 달라 고정)
        return ChatOpenAI(model=model, stream_usage=True, store=False, **kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if effort and effort != "none":
            kwargs["effort"] = effort
        if timeout is not None:
            kwargs["default_request_timeout"] = timeout
        # 사고(thinking) 토큰도 max_tokens 에 포함되므로, 답변 끝의 근거 줄까지 잘리지 않게 여유를 둔다
        return ChatAnthropic(model=model or settings.anthropic_model, max_tokens=8192, **kwargs)

    raise ValueError(f"지원하지 않는 LLM_PROVIDER: {provider!r} (가능한 값: {', '.join(SUPPORTED_PROVIDERS)})")
