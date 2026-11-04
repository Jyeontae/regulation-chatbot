"""프로젝트 설정: .env 값을 읽어 한곳에서 관리한다."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _get(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    # LLM
    llm_provider: str
    openai_model: str
    anthropic_model: str
    llm_temperature: float | None
    rewrite_model: str
    rewrite_effort: str
    rewrite_timeout: float
    answer_effort: str
    # 채팅 입력
    max_question_chars: int
    history_turns: int
    # 임베딩
    embedding_model: str
    embedding_device: str
    # 경로
    docs_dir: Path
    chunks_path: Path
    chroma_dir: Path
    chroma_collection: str
    eval_set_path: Path
    # 검색
    retriever_k: int
    retriever_mode: str
    # 리랭커
    reranker_model: str
    rerank_candidates: int


def load_settings() -> Settings:
    temperature = _get("LLM_TEMPERATURE")
    return Settings(
        llm_provider=_get("LLM_PROVIDER", "anthropic").lower(),
        openai_model=_get("OPENAI_MODEL"),
        anthropic_model=_get("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
        llm_temperature=float(temperature) if temperature else None,
        rewrite_model=_get("REWRITE_MODEL"),
        # 비우면 모델 기본값. 11-02 까지는 코드에 low 로 고정돼 있었다
        rewrite_effort=_get("REWRITE_EFFORT", "low").lower(),
        rewrite_timeout=float(_get("REWRITE_TIMEOUT", "5")),
        answer_effort=_get("ANSWER_EFFORT").lower(),
        max_question_chars=int(_get("MAX_QUESTION_CHARS", "300")),
        history_turns=int(_get("HISTORY_TURNS", "2")),
        embedding_model=_get("EMBEDDING_MODEL", "BAAI/bge-m3"),
        embedding_device=_get("EMBEDDING_DEVICE", "cpu"),
        docs_dir=_path(_get("DOCS_DIR", "data/raw")),
        chunks_path=_path(_get("CHUNKS_PATH", "data/chunks/chunks.jsonl")),
        chroma_dir=_path(_get("CHROMA_DIR", "data/chroma")),
        chroma_collection=_get("CHROMA_COLLECTION", "regulations"),
        eval_set_path=_path(_get("EVAL_SET_PATH", "data/eval/eval_set.md")),
        retriever_k=int(_get("RETRIEVER_K", "4")),
        retriever_mode=_get("RETRIEVER_MODE", "vector").lower(),
        reranker_model=_get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
        rerank_candidates=int(_get("RERANK_CANDIDATES", "0")),
    )


settings = load_settings()
