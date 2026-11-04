"""환경 구성 확인: 설정값 → 임베딩 모델 → LLM API 연결을 차례로 점검한다.

실행:
    poetry run python scripts/check_setup.py              # 전체 점검
    poetry run python scripts/check_setup.py --skip-llm   # API 호출 없이 임베딩까지만
"""

import argparse
import os

from regbot.config import settings
from regbot.models import SUPPORTED_PROVIDERS, get_embeddings, get_llm

API_KEY_NAMES = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def check_settings() -> None:
    print("[1] 설정값")
    print(f"  LLM_PROVIDER     : {settings.llm_provider}")
    print(f"  OPENAI_MODEL     : {settings.openai_model or '(비어 있음)'}")
    print(f"  ANTHROPIC_MODEL  : {settings.anthropic_model}")
    print(f"  EMBEDDING_MODEL  : {settings.embedding_model} ({settings.embedding_device})")
    print(f"  DOCS_DIR         : {settings.docs_dir}")
    print(f"  CHUNKS_PATH      : {settings.chunks_path}")
    print(f"  CHROMA_DIR       : {settings.chroma_dir}")
    for provider, key_name in API_KEY_NAMES.items():
        # 키 값 자체는 출력하지 않는다
        status = "설정됨" if os.getenv(key_name) else "없음"
        print(f"  {key_name:<17}: {status}")


def check_embeddings() -> None:
    print("\n[2] 임베딩 (처음 실행 시 모델 다운로드로 수 분 걸릴 수 있음)")
    sentences = [
        "연차휴가는 어떻게 신청하나요?",
        "휴가 쓰려면 뭘 해야 돼?",
        "법인카드 사용 한도는?",
    ]
    vectors = get_embeddings().embed_documents(sentences)
    print(f"  벡터 차원: {len(vectors[0])}")

    def similarity(a: list[float], b: list[float]) -> float:
        # normalize_embeddings=True 이므로 내적 = 코사인 유사도
        return sum(x * y for x, y in zip(a, b))

    print(f"  '{sentences[0]}' ↔ '{sentences[1]}': {similarity(vectors[0], vectors[1]):.3f}")
    print(f"  '{sentences[0]}' ↔ '{sentences[2]}': {similarity(vectors[0], vectors[2]):.3f}")
    print("  → 첫 번째 값이 두 번째 값보다 크면 정상")


def check_llm(provider: str) -> None:
    if not os.getenv(API_KEY_NAMES[provider]):
        print(f"  {provider}: 건너뜀 ({API_KEY_NAMES[provider]} 없음)")
        return
    try:
        response = get_llm(provider).invoke("'연결 확인'이라고만 답하세요.")
        print(f"  {provider}: {response.text.strip()}")
    except Exception as e:  # 설정 점검용이므로 오류를 보여주고 계속 진행
        print(f"  {provider}: 실패 - {type(e).__name__}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-llm", action="store_true", help="LLM API 호출을 건너뛴다")
    args = parser.parse_args()

    check_settings()
    check_embeddings()
    if not args.skip_llm:
        print("\n[3] LLM API 연결")
        for provider in SUPPORTED_PROVIDERS:
            check_llm(provider)


if __name__ == "__main__":
    main()
