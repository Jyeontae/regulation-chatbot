"""인덱싱: data/chunks/chunks.jsonl 을 임베딩해 Chroma 에 적재한다.

매번 컬렉션을 비우고 전체를 다시 적재한다 (청크 1,000개 기준 CPU 로 수 분).
화면에는 개수와 걸린 시간만 출력한다 (규정 본문·파일 이름 없음).

실행:
    poetry run python scripts/build_index.py
"""

import time
from collections import Counter

from regbot.chunks import load_chunks
from regbot.config import settings
from regbot.report import display_path
from regbot.vectorstore import open_store, to_documents

BATCH_SIZE = 64  # 임베딩·적재 단위. 진행 상황도 이 단위로 출력


def main() -> None:
    chunks = load_chunks(settings.chunks_path)
    print(f"[대상] {display_path(settings.chunks_path)} ({len(chunks):,}개)")
    if not chunks:
        return

    docs = to_documents(chunks)
    ids = [chunk.chunk_id for chunk in chunks]

    store = open_store()
    # 이전 적재분이 남으면 같은 chunkId 를 다시 넣을 수 없고, 분할에서 사라진 청크도 검색에 남는다
    store.reset_collection()

    print(f"\n[적재] {display_path(settings.chroma_dir)} / 컬렉션 '{settings.chroma_collection}'")
    start = time.perf_counter()
    for i in range(0, len(docs), BATCH_SIZE):
        store.add_documents(docs[i : i + BATCH_SIZE], ids=ids[i : i + BATCH_SIZE])
        print(f"  {min(i + BATCH_SIZE, len(docs)):,} / {len(docs):,}")
    elapsed = time.perf_counter() - start

    # 본문이 출력되지 않도록 id 만 조회한다
    saved = len(store.get(include=[])["ids"])
    categories = Counter(str(chunk.metadata["category"]) for chunk in chunks)
    print("\n[결과]")
    print(f"  저장: {saved:,}개 / 청크: {len(chunks):,}개 → {'일치' if saved == len(chunks) else '불일치, 다시 실행 필요'}")
    print(f"  카테고리: {', '.join(f'{name} {count:,}' for name, count in categories.most_common())}")
    print(f"  걸린 시간: {elapsed:,.0f}초 (청크당 {elapsed / len(chunks):.2f}초)")


if __name__ == "__main__":
    main()
