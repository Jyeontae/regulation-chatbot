"""검색: 벡터 검색(Chroma)과 키워드 검색(BM25)을 순위로 합친다 (RRF).

- vector: Chroma 코사인 유사도 순위만 쓴다.
- hybrid: 두 검색의 상위 CANDIDATES 개씩을 RRF 로 합친다. 점수 척도가 달라 점수 대신 순위만 쓴다.

BM25 는 chunks.jsonl 을 읽어 메모리에 만들므로, 인덱스와 청크 파일이 같은 분할 결과여야 한다.
rerank() 는 검색 결과의 상위 후보를 로컬 리랭커로 다시 정렬한다 (외부 전송 없음).
"""

from dataclasses import dataclass

from kiwipiepy import Kiwi
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from regbot.chunks import load_chunks
from regbot.config import settings
from regbot.models import get_reranker
from regbot.vectorstore import open_store

SUPPORTED_MODES = ("vector", "hybrid")
CANDIDATES = 30  # 검색마다 RRF 에 넣는 후보 수
RRF_K = 60  # RRF 상수. 클수록 1등과 하위 순위의 점수 차가 줄어든다 (원 논문 값)

# 조사·어미·기호를 빼고 내용 형태소만 남긴다. NNB(의존 명사)는 "조·원·일"처럼 조문 번호·금액·기한에 쓰여 남긴다
_CONTENT_TAGS = {"NNG", "NNP", "NNB", "NR", "SN", "SL", "SH", "XR", "VV", "VA"}


@dataclass
class Hit:
    doc: Document
    score: float  # 정렬 기준. vector 는 코사인 유사도, hybrid 는 RRF 점수, rerank() 를 거친 후보는 리랭커 점수
    cosine: float | None  # 벡터 후보 밖이면 None
    vector_rank: int | None
    bm25_rank: int | None
    query_index: int = 0  # search_all 에서 이 점수를 낸 검색어의 순서


class Retriever:
    def __init__(self, mode: str | None = None) -> None:
        self.mode = (mode or settings.retriever_mode).lower()
        if self.mode not in SUPPORTED_MODES:
            raise ValueError(f"지원하지 않는 RETRIEVER_MODE: {self.mode!r} (가능한 값: {', '.join(SUPPORTED_MODES)})")
        self.store = open_store()
        self.count = len(self.store.get(include=[])["ids"])
        self._kiwi: Kiwi | None = None
        self._bm25: BM25Okapi | None = None
        self._docs: list[Document] = []
        if self.mode == "hybrid" and self.count:
            self._build_bm25()

    def _build_bm25(self) -> None:
        chunks = load_chunks(settings.chunks_path)
        # 청크 파일만 다시 만들고 인덱스를 그대로 두면 두 검색이 서로 다른 청크를 보게 된다
        if {chunk.chunk_id for chunk in chunks} != set(self.store.get(include=[])["ids"]):
            raise ValueError("chunks.jsonl 과 Chroma 인덱스의 청크가 다릅니다. build_index.py 를 다시 실행하세요.")
        self._kiwi = Kiwi()
        self._docs = [Document(id=c.chunk_id, page_content=c.text, metadata=c.metadata) for c in chunks]
        self._bm25 = BM25Okapi([self._tokens(tokens) for tokens in self._kiwi.tokenize([c.text for c in chunks])])

    @staticmethod
    def _tokens(tokens) -> list[str]:
        # 불규칙 활용 용언은 "VV-I" 처럼 태그에 접미사가 붙는다
        return [t.form.lower() for t in tokens if t.tag.split("-")[0] in _CONTENT_TAGS]

    def search(self, query: str, k: int) -> list[Hit]:
        """상위 k개. hybrid 는 후보 수가 k 와 무관하게 CANDIDATES 로 고정이라 최대 2 × CANDIDATES 개까지 나온다."""
        # relevance_scores 계열은 점수가 0~1 을 벗어나면 경고 메시지에 검색 문서 전체(규정명·본문)를 찍으므로 쓰지 않는다
        vector = [
            (doc, 1 - distance)  # 코사인 거리 → 유사도 (check_setup.py 와 같은 척도)
            for doc, distance in self.store.similarity_search_with_score(
                query, k=k if self.mode == "vector" else CANDIDATES
            )
        ]
        if self.mode == "vector":
            return [Hit(doc, cosine, cosine, rank, None) for rank, (doc, cosine) in enumerate(vector, start=1)]

        # 후보 수를 k 에 맞춰 늘리면 k 에 따라 순위가 달라지므로 고정한다
        hits: dict[str, Hit] = {}
        for rank, (doc, cosine) in enumerate(vector, start=1):
            hits[doc.id] = Hit(doc, 1 / (RRF_K + rank), cosine, rank, None)
        for rank, index in enumerate(self._bm25_top(query, CANDIDATES), start=1):
            doc = self._docs[index]
            hit = hits.setdefault(doc.id, Hit(doc, 0.0, None, None, None))
            hit.score += 1 / (RRF_K + rank)
            hit.bm25_rank = rank
        # 동점이면 벡터 순위가 높은 쪽을 앞에 둔다
        ranked = sorted(hits.values(), key=lambda h: (-h.score, h.vector_rank or len(hits) + 1))
        return ranked[:k]

    def search_all(self, queries: list[str], k: int) -> list[Hit]:
        """검색어마다 상위 k개를 찾아, 같은 청크는 가장 높은 점수 하나로 합친 상위 k개.

        모든 검색어의 점수가 같은 척도라 순위(RRF) 대신 점수를 그대로 비교한다.
        어느 목록에서도 k위 밖인 청크는 합친 결과에서도 k위 안에 들 수 없으므로 목록마다 k개만 찾으면 된다.
        """
        best: dict[str, Hit] = {}
        for index, query in enumerate(queries):
            if query in queries[:index]:  # 같은 검색어는 한 번만 검색한다
                continue
            for hit in self.search(query, k):
                hit.query_index = index
                if hit.doc.id not in best or hit.score > best[hit.doc.id].score:
                    best[hit.doc.id] = hit
        return sorted(best.values(), key=lambda h: -h.score)[:k]

    def _bm25_top(self, query: str, n: int) -> list[int]:
        scores = self._bm25.get_scores(self._tokens(self._kiwi.tokenize(query)))
        # 질문 형태소가 하나도 없는 청크(점수 0)는 순위에 넣지 않는다
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
        return [i for i in top if scores[i] > 0]


def rerank(query: str, hits: list[Hit], n: int) -> list[Hit]:
    """상위 n개 후보를 리랭커 점수로 다시 정렬하고, n위 밖은 원래 순서대로 뒤에 붙인다.

    리랭커는 질문과 청크(위치 머리말 포함)를 한 쌍씩 함께 읽어 0~1 관련도를 낸다.
    다시 매긴 후보의 score 는 리랭커 점수로 바뀌고, 코사인 유사도는 cosine 에 남는다.
    """
    head, tail = hits[:n], hits[n:]
    if not head:
        return hits
    # 한 쌍씩 돌린다. 여러 쌍을 묶으면 짧은 쌍도 가장 긴 쌍 길이로 채워 계산해 더 느렸다
    # (후보 5개 평균 2.6초 → 2.3초, docs/2026-11-02 실험 기록)
    scores = get_reranker().predict([(query, hit.doc.page_content) for hit in head], batch_size=1)
    for hit, score in zip(head, scores):
        hit.score = float(score)
    # 동점이면 원래 순위를 지킨다 (sorted 는 안정 정렬)
    return sorted(head, key=lambda h: -h.score) + tail
