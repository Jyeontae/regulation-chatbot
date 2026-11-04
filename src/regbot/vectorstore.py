"""벡터 저장소: Chroma(cosine) 컬렉션 열기, 청크 → Document 변환.

인덱싱과 검색이 같은 임베딩 모델·거리 함수를 쓰도록 컬렉션 설정을 여기 한곳에 둔다.
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document

from regbot.chunks import Chunk
from regbot.config import settings
from regbot.models import get_embeddings


def open_store() -> Chroma:
    """설정의 컬렉션을 연다. 없으면 만든다."""
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_dir),
        # 생성 시에만 적용되므로 거리 함수를 바꾸려면 컬렉션을 다시 만들어야 한다.
        # similarity_search_with_score 의 점수는 코사인 거리 = 1 - 코사인 유사도 (작을수록 비슷)
        collection_configuration={"hnsw": {"space": "cosine"}},
    )


def to_documents(chunks: list[Chunk]) -> list[Document]:
    return [Document(page_content=chunk.text, metadata=chunk.metadata) for chunk in chunks]
