"""청크 형식 정의와 chunks.jsonl 읽기·쓰기. Chroma와 BM25가 같은 청크를 읽는다."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

# Chroma metadata 에는 문자열·숫자·불리언만 넣을 수 있다 (None·리스트 불가)
MetadataValue = str | int | float | bool


@dataclass
class Chunk:
    chunk_id: str
    text: str  # 위치 머리말 + 본문. 임베딩·BM25 검색 대상이자 LLM 에 보내는 내용
    metadata: dict[str, MetadataValue]


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]
