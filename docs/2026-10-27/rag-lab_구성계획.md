# rag-lab 구성 계획 (Python LangChain vs Spring AI 비교 RAG)

> 근거: 「사내 문서 RAG 고도화 플랜」 브리프 + 2026-10-26 조사(공식 문서·Maven Central·PyPI·소스 코드·설치된 venv 확인, 영역별 반박 검증 완료)
> 상태: **계획 초안** — 7장의 결정 사항을 확정한 뒤 0단계부터 진행
> 원칙: 코드·스크립트는 Claude가 작성하고, **설치·실행·검증은 담당자가 직접** 한다. 실행 결과를 공유할 때는 집계 수치와 마스킹된 오류만 보낸다.

---

## 결정 기록

| 날짜 | 결정 | 본인 판단 근거 |
|---|---|---|
| 2026-10-27 | **Python 단독으로 구현** (브리프의 Spring AI 재구현은 제외. 필요하면 나중에 확장 과제로) | ① 검색 고도화 기능(하이브리드 검색, 한국어 형태소 분석, 리랭커, 평가 도구)이 Python 쪽에 훨씬 많다 ② 최신 기능이 Python에 먼저, 빠르게 나온다 ③ 커뮤니티 규모가 크다 |

### 결정 상세: Python 단독 구현

**검토한 선택지**
| 선택지 | 내용 |
|---|---|
| A. Python + Spring AI 이중 구현 (브리프 원안) | 같은 RAG를 두 프레임워크로 만들어 비교 |
| B. 혼합 구조 | Python이 검색·생성, Spring이 앞단(캐시·사용량 제한·관측성) |
| **C. Python 단독 (채택)** | FastAPI + LangChain 1.x에 Redis·MCP·관측성까지 구현 |

**판단 기준:** 이 프로젝트의 핵심 과제는 **한국어 규정 문서의 검색 정확도**와 **토큰 비용 절감**이다. 따라서 ① 검색 고도화 도구를 얼마나 바로 쓸 수 있는가, ② 새 기법을 얼마나 빨리 적용할 수 있는가, ③ 막혔을 때 해결 자료가 충분한가를 기준으로 삼았다.

**근거 보강 (2026-10-26 조사로 확인한 사실)**

① **검색 고도화 기능**
- 한국어 키워드 검색에 필요한 형태소 분석기 Kiwi(`kiwipiepy`)는 Python 패키지다. Java 바인딩은 확인되지 않아, Spring으로 하이브리드 검색을 하려면 한국어 토큰화를 따로 해결해야 한다.
- 학습한 고도화 기법(BM25 하이브리드와 RRF, MultiQuery, Decomposition, CrossEncoder 리랭커, 맥락 압축)과 평가 도구(krag 검색 지표, LLM-as-judge)가 모두 Python 생태계에 있고, 실습 노트북으로 이미 검증했다.
- Spring AI 2.0의 RAG 구성은 벡터 검색과 Advisor 중심이다. Python에서 튜닝한 하이브리드 검색과 리랭커를 옮기려면 Java로 재구현해야 하고, 옮기지 못하면 정확도가 떨어진다.

② **최신 기능 반영 속도와 성숙도**
- Spring AI는 2.0 GA가 2026년 6월로 비교적 최근이고, 조사 시점에 이미 2.1.0-RC1이 나와 있어 API 변화가 잦다(2.0에서 설정 키·모듈 이름 다수 변경).
- 조사 중 **문서와 실제 코드가 다른 부분**을 여러 건 확인했다.
  - 문서에 나온 시맨틱 캐시 스타터가 Maven에 존재하지 않음
  - MCP 전송 방식의 실제 기본값이 문서와 다름
  - Prometheus 지표 이름이 문서 표와 코드에서 다름
- 신생 프레임워크로 학습과 튜닝을 하면 **문제의 원인이 내 코드인지 프레임워크인지** 가르는 데 시간이 많이 든다.

③ **커뮤니티 규모**
- LLM 애플리케이션 예제, 튜토리얼, 이슈 해결 자료는 Python 쪽이 압도적으로 많다.
- 정량 근거가 필요하면 GitHub 스타 수, PyPI·Maven 다운로드 수를 직접 확인해 이 칸에 추가한다(미확인).

④ **프로젝트 조건 (추가 근거)**
- 기존 자산이 Python에 있다. LangChain 6개 섹션 학습과 실습, 동작이 확인된 설정·모델 계층(bge-m3 로컬 임베딩과 OpenAI·Claude 전환)이 그대로 이어진다.
- 1인 프로젝트에서 언어 하나에 집중하면, 두 구현을 같은 조건으로 맞추는 작업(청크 공유, 임베딩 서버 공유, 검색 일치 확인)에 들어갈 시간을 정확도 튜닝과 운영 기능에 쓸 수 있다.

**포기한 것 (트레이드오프)**
| 포기한 것 | 대응 |
|---|---|
| Java/Spring 백엔드 공고에서의 "Spring에 LLM을 붙여 봤다"는 차별점 | 캐시 키 설계, 원자적 사용량 제한, SSE, 관측성, MCP 같은 운영 역량은 Python에서도 그대로 보여줄 수 있다 |
| Spring AI가 기본 제공하는 관측 지표(토큰 사용량 등) | Python에서는 메트릭을 직접 설계·구현한다. 이 자체가 설계 경험이 된다 |
| 두 프레임워크 정량 비교 | 단계별 전후 비교(튜닝 전후 Hit@k, 캐시 전후 토큰·지연·비용)로 대체한다 |

**다시 검토할 조건**
- Java 백엔드 직무 지원이 확정되거나, 회사 Spring 시스템에 연동할 일이 생길 때 → 혼합 구조(B)로 Spring 앞단을 확장 과제로 추가
- Spring AI 2.x가 안정화되어 문서와 코드가 일치하고, 한국어 하이브리드 검색을 할 방법이 생길 때

> 아래 본문은 Spring 비교를 전제로 쓴 초안이라 Python 단독 기준으로 다시 작성할 예정이다.

## 목적과 우선순위 (2026-10-27 합의)

- **목적:** 백엔드 개발자 포트폴리오. "어느 쪽이 나은가"를 증명하는 실험이 아니라, **Spring 위에서 LLM 기능을 운영 수준으로 만든 경험**과 그 근거 수치를 남기는 것.
- **프로젝트 관계:** 기존 사내 규정 챗봇의 다음 단계(같은 프로젝트). 도메인은 가상 규정 그대로이고, 기존 코드는 `python-rag/`로 이어서 사용.
- **비중:**
  - **Spring판이 주인공:** Redis 캐시·사용량 제한, SSE, 재시도·폴백, MCP, 관측성에 시간을 가장 많이 투자.
  - **Python판은 기준선:** `/search`, `/ask`, 평가 기준선 측정까지 최소한으로.
  - **공정성 장치는 가볍게 유지:** 청크 파일 공유와 TEI 공유는 유지(설계 이야기가 됨). 검색 일치 게이트는 top-k chunkId 일치만 확인. 반복 측정은 최소화.
- **포트폴리오 재료로 기록할 것:** 각 단계의 전후 수치, 발견한 함정과 해결 과정(예: 임베딩 설정 실수로 오류 없이 엉터리 벡터가 생성되는 문제, 숫자만 다른 질문의 시맨틱 캐시 오적중, MCP 프로토콜 기본값 함정).

---

## 0. 한눈에 보기

```
docs/sample/*.md (가상 규정)
        │
        ▼
 ingest/build_index.py  ──►  index/v{n}/chunks.jsonl + manifest.json   ← 청킹은 여기서 "한 번만"
        │                              │
        │  (임베딩: TEI bge-m3, 공용)   │
        ▼                              ▼
 python-rag (FastAPI)            spring-rag (Boot 4.1 + Spring AI 2.0)
  Chroma(cosine)                  pgvector(COSINE, id-type=TEXT)
  /ask /ask/stream /search        /ask /ask/stream /search /mcp
                                  + Redis(정확·시맨틱 캐시, 이력, 사용량 제한)
        │                              │
        └──────────► eval/run_eval.py ◄┘  → results/*.json → COMPARISON.md
                     prometheus ← 두 앱 /metrics, grafana 대시보드 1개
```

**공정 비교의 핵심 3가지:** 같은 청크 파일, 같은 임베딩 서버, 같은 LLM·프롬프트. 브리프의 "같은 파라미터"만으로는 결과가 같아지지 않는다는 것을 조사에서 확인했다(1장).

---

## 1. 조사로 확인된 브리프 수정 사항

| # | 브리프 내용 | 확인된 사실 | 계획 반영 |
|---|---|---|---|
| 1 | Spring Boot 4.x + Spring AI 2.0.x | 최신 GA는 **Spring AI 2.0.2**(2026-10-23), **Boot 4.1.2**. 2.0.2는 Boot 4.1.2 기준으로 빌드됨. 2.1.0-RC1도 나와 있음 | Boot 4.1.2 + spring-ai-bom 2.0.2로 **고정**. 동적 버전(2.+) 금지 |
| 2 | Java 21 이상 | 최소 요건은 Java 17. 21+는 프로젝트 선택 사항(가상 스레드) | Java 21 또는 25(결정 필요) |
| 3 | "같은 청킹 파라미터" | Spring `TokenTextSplitter`는 cl100k 토큰 기준이고 청크 ID가 매번 랜덤. LangChain은 문자 기준 | **청킹은 Python ingest가 한 번만** 하고 두 구현은 `chunks.jsonl`을 적재만 |
| 4 | "같은 임베딩 모델" | Spring ONNX Transformers는 mean pooling + 정규화 없음(bge-m3는 CLS + L2). Ollama bge-m3는 F16 GGUF·별도 토크나이저 | **TEI 컨테이너 하나**를 두 구현이 함께 호출(3-2) |
| 5 | PDF는 `PyPDFLoader` | `langchain-community`는 2026-06-19 아카이브(유지보수 종료). pypdf는 한글 띄어쓰기 손실 + venv에 미설치 | 평가 문서는 md. PDF가 필요하면 pymupdf 직접 사용 |
| 6 | `RetrievalQA`(임시) 또는 `create_agent` | RetrievalQA는 deprecated이고 점수를 버림. create_agent는 LLM 호출이 2회 이상으로 늘어 비교가 깨짐 | **2단계 LCEL**: 점수 포함 검색 → `prompt \| llm` 1회 |
| 7 | QuestionAnswerAdvisor 계열 | 모듈명이 2.0에서 `spring-ai-vector-store-advisor`로 바뀜. 기본 프롬프트가 영어 | 한국어 프롬프트를 양쪽에 똑같이 사용 |
| 8 | 재시도·타임아웃 | OpenAI·Anthropic은 SDK 기반이라 `spring.ai.retry.*`가 적용되지 않음 → `spring.ai.{openai,anthropic}.timeout / max-retries` | 공급자별 키로 설정, 재시도 1~2회 |
| 9 | 폴백 | Spring AI 2.0.2에 내장 폴백 없음. 폴백이 일어나면 "같은 LLM" 원칙과 충돌 | 직접 구현하되 **평가 중에는 끄거나 응답에 표시** |
| 10 | `@Cacheable` + 커스텀 키 | 캐시 적중 시 메서드가 실행되지 않아 `cache`·`latencyMs`를 안에서 정할 수 없음 | 캐시 값 객체 분리, 바깥 계층에서 적중 판정 |
| 11 | 시맨틱 캐시 | Spring AI 내장 `SemanticCacheAdvisor`는 답변 텍스트만 저장(sources·usage 손실). 문서에 나온 스타터는 **존재하지 않음**(404) | **직접 구현** |
| 12 | Redis 8 또는 Redis Stack | Redis 8에 Query Engine·벡터 검색이 내장됨(Stack 통합) | `redis:8.10.3` 사용 |
| 13 | `@McpTool`, Streamable HTTP | 어노테이션 패키지는 `org.springframework.ai.mcp.annotation`. **`spring.ai.mcp.server.protocol=STREAMABLE`을 명시하지 않으면 구식 SSE가 켜짐** | protocol 명시 필수 |
| 14 | 관측성 지표 이름 | 문서 표의 이름과 코드가 다름(예: ChatClient는 `spring_ai_chat_client_seconds_*`). 비용·캐시 지표는 내장되어 있지 않음 | 실제 `/actuator/prometheus` 출력 확인 후 대시보드 작성 |
| 15 | Compose 목록 | 트레이스 백엔드가 없음 | Jaeger를 선택 프로필로 추가 |
| 16 | `topK: 4` + Hit@5 | sources가 4개면 Hit@5를 계산할 수 없음 | 검색 전용 `/search` 추가(7장) |
| 17 | `ratelimit:{userId}` | API 계약에 userId가 없음 | `X-User-Id` 헤더 추가(7장) |
| 18 | `재적재마다 indexVersion 증가` | 내용이 같아도 캐시가 무효화되어 전후 비교에 잡음이 생김 | **문서·청커·임베딩 지문이 바뀔 때만** 증가 |
| 19 | 현재 Python판을 마이그레이션 | 현재 코드는 설정·모델 계층뿐(레거시 체인 없음) | 1단계는 "정리"가 아니라 **1.x로 새로 구현** |

---

## 2. 저장소 구조

```
rag-lab/
├── docs/sample/              # 가상 규정 md (직접 작성, 실제 문서 금지)
├── ingest/
│   └── build_index.py        # 로딩·정규화·청킹 → index/v{n}/
├── index/v{n}/               # chunks.jsonl, manifest.json (git 제외 또는 커밋 선택)
├── eval/
│   ├── questions.jsonl       # 검색 평가셋 (동결)
│   ├── cache_pairs.jsonl     # 시맨틱 캐시 오적중 평가쌍
│   ├── parity_check.py       # 두 구현 검색 결과 일치 확인
│   └── run_eval.py
├── results/                  # {YYYYMMDD-HHmm}_{impl}_{label}.json
├── python-rag/               # 현재 regulation-chatbot 코드 이전 (FastAPI + LangChain 1.x)
├── spring-rag/               # Spring Boot 4.1 + Spring AI 2.0 (Gradle 또는 Maven)
├── infra/
│   ├── prometheus/prometheus.yml
│   └── grafana/{provisioning,dashboards}/
├── docker-compose.yml
├── .env.example              # 키 이름만
├── .gitattributes            # *.sh eol=lf
├── .dockerignore             # .env, data/, index/
└── COMPARISON.md
```

- **현재 상태:** 지금 폴더(regulation-chatbot)는 상위 `D:\hanil` 저장소 안에 untracked로 들어 있고, 상위 저장소에는 다른 프로젝트의 미커밋 변경이 있습니다. rag-lab은 **독립 git 저장소**로 만드는 것을 권장합니다(7장 D1).
- **폴더 이동 시:** Poetry 가상환경 경로가 바뀌므로 `poetry install`과 `check_setup.py`를 다시 실행해야 합니다.

---

## 3. 핵심 설계

### 3-1. 문서와 데이터 위치
- **실제 사내 문서:** 저장소 밖에만 둡니다. 현재 `data/raw`에는 `.gitkeep`만 있으므로 옮길 파일은 없고, `DOCS_DIR` 기본값만 `docs/sample`로 바꿉니다.
- **샘플 문서:** `docs/sample/`에 가상 규정 md **10~15개**를 둡니다(제N장/제N조/항 구조).
  - 평가 변별력을 위한 장치:
    - 비슷한 조항이 여러 문서에 걸쳐 있게
    - 일수·금액·날짜만 다른 조항
    - 부정문("~할 수 없다")
    - 답이 없는 질문용 공백 주제
  - 문서가 적으면 Hit@5가 거의 100%로 포화되므로, 전체 청크 수가 k보다 충분히 많아야 합니다(목표 150개 이상).

### 3-2. 임베딩: TEI(Text Embeddings Inference) 공용 서버 — 권장
| 항목 | 내용 |
|---|---|
| 이미지 | `ghcr.io/huggingface/text-embeddings-inference:cpu-1.9`(pull 후 digest 고정) |
| 모델 | `--model-id BAAI/bge-m3 --revision <커밋 sha>` (manifest에 기록) |
| 왜 TEI인가 | 현재 기준선(sentence-transformers fp32, CLS+L2, 8192토큰)과 같은 결과를 내고, 양쪽 모두 기본 OpenAI 호환 클라이언트로 호출 가능. Prometheus `/metrics` 제공 |
| Python 클라이언트 | `OpenAIEmbeddings(base_url=".../v1", api_key="unused", check_embedding_ctx_length=False, chunk_size=32)` |
| Spring 클라이언트 | `spring-ai-starter-model-openai`의 embedding: `base-url`, `api-key=unused`, **`metadata-mode=NONE`**, dimensions 지정 안 함 |
| 메모리 | fp32 약 3GB. Docker Desktop(WSL2) 메모리 한도 확인 필요 |

**조용히 틀어지는 함정 2가지 (반드시 막을 것)**
1. `OpenAIEmbeddings` 기본값(`check_embedding_ctx_length=True`)은 tiktoken 토큰 ID를 보냅니다. TEI는 이것을 오류 없이 받아서 **엉터리 벡터**를 돌려줍니다. 시작할 때 고정 문장의 기준 벡터와 비교하는 자가진단을 넣습니다.
2. Spring OpenAI 임베딩 기본값(`metadata-mode=EMBED`)은 메타데이터 문자열을 본문 앞에 붙여 임베딩합니다. 그러면 Python 쪽 벡터와 달라집니다.

추가 사항:
- TEI는 기본적으로 한 요청에 32개까지만 받습니다. 양쪽 배치 크기를 32 이하로 맞춥니다.
- 포트는 `127.0.0.1`에만 공개합니다(인증 없음).
- 실제 OpenAI 키가 TEI로 가지 않게 임베딩 클라이언트에는 더미 키를 씁니다.

대안인 Ollama bge-m3는 LLM까지 로컬로 돌릴 때만 이점이 있습니다. 벡터가 현재 기준선과 달라지고 기본 컨텍스트가 4096이라는 단점이 있습니다.

### 3-3. 공유 적재(ingest)와 indexVersion
- **청킹 담당:** `ingest/build_index.py`만 문서를 읽고, NFC 정규화를 하고, 조항 정규식으로 청킹합니다. 두 런타임은 다시 자르지 않습니다.
- **청크 형식(`chunks.jsonl` 한 줄):**
  ```json
  {"chunkId":"sample_a.md#0007","source":"sample_a.md","ordinal":7,"text":"…","textSha256":"…","tokenCount":312,
   "metadata":{"chapter":"제3장 휴가","article":"제7조","title":"연차휴가"}}
  ```
- **`manifest.json`:** `indexVersion`, 지문(문서 해시 목록 + 청커 이름·버전·파라미터 + 임베딩 모델·revision + TEI 이미지 digest), 생성 시각을 담습니다.
- **indexVersion 증가 규칙:** 지문이 바뀔 때만 올립니다. 내용이 같은 재적재에서는 그대로 두어 Redis 캐시가 유지됩니다.
- **원천:** manifest 한 곳. Spring은 시작할 때와 재적재할 때 manifest를 읽어 메모리에 둡니다.
- **저장소 구분:** Chroma 컬렉션 `rag_v{n}`, pgvector 테이블 `rag_chunks`에 `indexVersion` 메타데이터. 이전 버전은 필터로 삭제합니다.

### 3-4. 점수·ID·프롬프트 일치
| 항목 | Python | Spring |
|---|---|---|
| 벡터 저장소 | Chroma, **`collection_configuration={"hnsw":{"space":"cosine"}}`**(생성 후 변경 불가) | PgVectorStore, `distance-type=COSINE_DISTANCE`, `index-type=HNSW`, `dimensions=1024` |
| score | `similarity_search_with_relevance_scores` → 1 − 코사인 거리 | `Document.getScore()` = 1 − 거리 |
| chunkId | Chroma id = `sample_a.md#0007` | **`id-type=TEXT`**(테이블 생성 전에만 바꿀 수 있음) + `Document.builder().id(chunkId)` |
| source | `metadata["source"]` = 파일명 | 리더가 채워주지 않으므로 적재 시 직접 설정 |
| 프롬프트 | 한국어 템플릿 파일 1개를 양쪽이 읽음 | `QuestionAnswerAdvisor`의 `{query}`, `{question_answer_context}` 자리에 같은 문구 |
| LLM 파라미터 | 모델 ID·max_tokens·temperature 공통 설정 | 같은 값(Anthropic 기본 maxTokens가 4096이므로 명시) |

**검색 일치 게이트(간소화):** LLM 비교와 Redis 작업 전에, 평가 질문별 top-k chunkId 집합이 양쪽에서 같은지 확인합니다. 다르면 원인(저장소 차이, 설정 실수)을 기록합니다. 원인 기록 자체가 포트폴리오 재료가 됩니다.

---

## 4. 단계별 계획

> 각 단계 공통: 시작 전에 작업 계획 공유 → Claude 작성 → **담당자 실행** → 결과 수치 공유 → `results/` 기록 → 체크박스 갱신

### 0단계 · 사전 정리 (코드 이전, 결정 확정)
- [ ] 7장 결정 사항 확정
- [ ] rag-lab 저장소 생성, 현재 코드를 `python-rag/`로 이전, venv 재생성 후 `check_setup.py` 재확인
- [ ] `DOCS_DIR=docs/sample`, `.gitattributes`, `.dockerignore`, `.env.example` 정리
- [ ] 버전 고정표(5장) 확정, WSL2 메모리(`.wslconfig`) 확인
- [ ] LLM 단가를 공식 가격표에서 확인해 `.env`에 입력(비용 추정용, 코드에 하드코딩 금지)

**완료 기준:** 새 위치에서 `check_setup.py`가 통과하고 첫 커밋 완료

### 1단계 · 샘플·평가셋 → 공유 적재 → Python판
순서가 중요합니다. **평가셋을 코드보다 먼저 만들고 동결**해야 튜닝 과적합을 막을 수 있습니다.

1. **샘플과 평가셋 (동결)**
   - [ ] `docs/sample/` 가상 규정 10~15개
   - [ ] `eval/questions.jsonl` 25~30개: `{"id","question","expected_sources":[...],"expected_chunkIds":[...],"type":"fact|condition|multi|none"}`
     - 파일 단위(브리프 Hit@k)와 청크 단위를 함께 기록하고, 답이 없는 질문도 포함
2. **임베딩 서버 + ingest**
   - [ ] Compose에 TEI만 먼저 추가, 자가진단(기준 벡터 비교) 스크립트
   - [ ] `ingest/build_index.py` → `index/v1/`
3. **Python판 (`python-rag/`)**
   - [ ] 의존성 정리: `fastapi`, `uvicorn`을 직접 의존성으로 명시하고 `langchain-community`는 제거. 하이브리드 BM25를 할 경우 `rank-bm25`를 직접 사용
   - [ ] `get_embeddings()`에 `EMBEDDING_BACKEND=local|tei` 추가(비교 실행은 tei)
   - [ ] `get_vectorstore()`(Chroma cosine), 적재 명령
   - [ ] `POST /search`, `POST /ask`, `POST /ask/stream`
     - SSE는 FastAPI 0.135+ 내장 `fastapi.sse.EventSourceResponse`
     - camelCase는 pydantic `alias_generator=to_camel`
     - 토큰은 `AIMessage.usage_metadata`. 체인 끝에 `StrOutputParser`를 붙이면 usage가 사라지므로 붙이지 않음
   - [ ] LLM `timeout`, `max_retries` 명시. 폴백은 `with_fallbacks`(평가 중에는 끔)
   - [ ] 임베딩 워밍업은 lifespan에서(첫 요청 지연이 측정에 섞이지 않게)
   - [ ] 토큰·비용 추출과 5단계 공통 메트릭 이름을 **지금부터** 적용(나중에 붙이면 이전 결과와 지표가 맞지 않음)
4. **평가 스크립트**
   - [ ] `eval/run_eval.py`
     - 순차 실행, 워밍업 1회 제외, 질문당 N회 반복
     - Hit@3/Hit@5는 `/search`(topK=5), 지연·토큰은 `/ask`(topK=4)로 측정
     - 클라이언트 왕복 시간과 서버 `latencyMs`를 구분해 기록
     - 결과에 indexVersion·지문·모델 ID·git commit·기능 플래그 기록, UTF-8로 저장
     - 실패·타임아웃은 건너뛰지 말고 오류로 집계(오류율), 가능한 모델이면 temperature 0
     - 결과 파일에는 질문 ID·지표·chunkId만 남기고 답변 본문은 저장하지 않음(선택 플래그로만)
   - [ ] 기준선 측정: `results/..._python_baseline.json`

**완료 기준:** Python 기준선 결과 파일 1개, 평가셋 동결 커밋

### 2단계 · Spring AI판 기본 RAG
- [ ] 프로젝트 생성: Boot 4.1.2, spring-ai-bom 2.0.2, `spring-boot-starter-webmvc`(Boot 4에서는 `-web`이 deprecated)
- [ ] **첫날 명시:** `spring.ai.model.chat=<공급자>`, `spring.ai.model.embedding=openai`(TEI용), `spring.ai.vectorstore.type=pgvector`. 명시하지 않으면 스타터를 추가할 때 빈이 중복됩니다
- [ ] Compose: `pgvector/pgvector:0.8.7-pg17`, pgvector 설정(3-4표), `initialize-schema=true`
- [ ] 적재: `chunks.jsonl`을 읽는 커스텀 리더 → `vectorStore.add()`(`TokenTextSplitter` 사용 안 함)
- [ ] `RetrievalService` 하나를 `/search`, `/ask`, MCP가 함께 사용
- [ ] `/ask`
  - `QuestionAnswerAdvisor`를 요청마다 topK에 맞춰 생성
  - `call().chatClientResponse()`로 받아 `context().get(QuestionAnswerAdvisor.RETRIEVED_DOCUMENTS)`에서 sources, `chatResponse().getMetadata().getUsage()`에서 토큰을 읽음
- [ ] `/ask/stream`
  - `Flux<ServerSentEvent<?>>` 반환, 이벤트 순서는 `sources` → `token` → `done{usage,cache,latencyMs}`
  - `spring.threads.virtual.enabled=true`, `spring.mvc.async.request-timeout`을 LLM 타임아웃보다 길게
- [ ] 재시도·타임아웃: `spring.ai.{openai|anthropic}.timeout`, `max-retries=1~2`
- [ ] 폴백: ChatClient 두 개 + try/catch. 첫 토큰 이전 실패만 전환하고, 응답 `model`·`fallbackUsed`로 표시
- [ ] 비용 로깅: 설정의 단가표 × usage(본문·청크 원문은 로그에 남기지 않음)
- [ ] 스트리밍에서 usage가 실제로 오는지 런타임에서 확인(2.0.2 소스상 OpenAI·Anthropic 모두 포함되지만 문서 설명과 다름)
- [ ] 캐시 우회 플래그와 `cache` 필드를 **지금 미리** 구현(항상 MISS) → 3단계 전후 비교의 토대
- [ ] `eval/parity_check.py` 통과 → `results/..._spring_baseline.json`

**완료 기준:** 검색 일치 게이트 통과, Spring 기준선 결과 파일

### 3단계 · Redis (하나씩 붙이고 매번 재측정)
공통: `redis:8.10.3`(entrypoint를 덮어쓰지 말 것. 모듈 로딩이 빠짐), 기본 클라이언트 Lettuce, `maxmemory-policy volatile-lru`.
**넣지 말 것:** `spring-ai-starter-vector-store-redis`(VectorStore 빈 중복), 시맨틱 캐시 auto-config(ONNX와 외부 모델 다운로드, Jedis 강제).

| 순서 | 기능 | 구현 요점 |
|---|---|---|
| 3-1 | 정확 캐시 `rag:answer:{v}:{hash}` | 별도 빈의 `@Cacheable`(학습 목적) + `CachedAnswer` 레코드(answer, sources, usage, model, promptVersion). 바깥 계층이 적중 여부·`latencyMs`를 결정. 키 = `v + sha256(정규화 질문 + topK + 모델 + 프롬프트 버전)`. 정규화는 NFKC → 소문자 → 문장부호 제거(숫자 사이는 유지) → 공백 전부 제거. 폴백·빈 sources는 저장하지 않음. 값은 Jackson 3 `JacksonJsonRedisSerializer` |
| 3-2 | 대화 이력 `chat:history:{sessionId}` | List 기반 `ChatMemoryRepository` 직접 구현(MULTI: DEL → RPUSH → LTRIM → EXPIRE). sessionId는 서버가 발급(UUID). **이력이 있는 세션은 캐시 BYPASS** |
| 3-3 | 사용량 제한 `ratelimit:{userId}:{yyyyMMdd}` | Hash(calls, tokens) + Lua 1회 호출로 확인과 증가를 원자적으로. 날짜는 `Asia/Seoul` 기준(컨테이너는 UTC). 초과 시 429 + Retry-After. MCP 도구 호출에도 적용 |
| 3-4 | 시맨틱 캐시 `rag:semcache:{v}:{id}` | 직접 구현(Redis FT 벡터 인덱스, KNN 1, `index_version` 태그 필터, TTL 24h). 시작 임계값은 코사인 ≥ 0.95. **숫자·날짜·조번호·부정어가 있으면 건너뜀**. sources·usage까지 저장 |

- **임계값 근거:** 조사에서 bge-m3로 직접 측정해 보니 숫자만 다른 질문 쌍(예: 경조휴가 5일 vs 3일)이 0.93까지 나와 실제 패러프레이즈(0.78~0.96)와 겹쳤습니다. 임계값만으로는 막을 수 없으므로 가드가 필수입니다. 다만 14쌍 측정이라 출발점일 뿐이고, `eval/cache_pairs.jsonl`로 0.90~0.97 구간을 돌려 오적중률과 적중률을 기록합니다.
- **척도 주의:** Spring AI `RedisVectorStore`의 유사도는 (1+cos)/2 척도입니다. Python에서 잰 코사인 값을 그대로 쓰면 안 됩니다.
- **측정 3종:** cold(첫 실행), warm(같은 질문 반복 → 정확 캐시 적중률), 변형 질문 쌍(시맨틱 적중률·오적중률). 각각 `results/`에 기록합니다.

### 4단계 · MCP 서버
- [ ] 스타터 `spring-ai-starter-mcp-server-webmvc`(어노테이션 모듈 포함, 별도 추가 불필요)
- [ ] `spring.ai.mcp.server.protocol=STREAMABLE` **필수**, `streamable-http.mcp-endpoint=/mcp`, `tool-callback-converter=false`(의도하지 않은 도구 노출 방지)
- [ ] `@McpTool(name="searchDocuments", annotations=readOnlyHint…)`. 반환은 **`record SearchDocumentsResult(long indexVersion, List<Chunk> chunks)`**: List를 그대로 반환하면 MCP 스펙(outputSchema 루트는 object)을 위반합니다
- [ ] topK는 1~10으로 제한, 도구 호출에 Timer와 rate limit 적용
- [ ] Origin/Host 검증: 자동구성에는 검증이 없으므로(NOOP) `WebMvcStreamableServerTransportProvider` 빈을 직접 정의하고, `sessionIdleTimeout`도 설정
- [ ] MCP Inspector 2.12.0(Node 22.19+)
  - `npx @modelcontextprotocol/inspector@2.12.0 --cli http://localhost:8080/mcp --transport http --method tools/list`
  - Protocol Era는 기본값 legacy를 유지(Spring AI 2.0.x는 2025-11-25 스펙)
- [ ] 외부 모델 클라이언트 연결은 **Claude Code 로컬(`claude mcp add --transport http`)로만**, 색인에 `docs/sample`만 있을 때만. claude.ai 원격 커넥터는 서버를 인터넷에 공개해야 하므로 금지

### 5단계 · 관측성
- [ ] Spring: actuator + `micrometer-registry-prometheus`, `management.endpoints.web.exposure.include=health,prometheus`
- [ ] **본문 기록 차단:**
  - Spring AI content 로깅 속성 10개를 모두 `false`로 명시
  - `db.vector.query.content`(질문 원문이 span에 자동 기록됨)를 `ObservationFilter`로 제거
  - `SimpleLoggerAdvisor`는 debug 프로필에서만 사용
- [ ] 공통 메트릭 계약(양쪽 이름·라벨·버킷 동일):
  - `rag_ask_latency_seconds{cache,stream}`, `rag_cache_requests_total{result}`
  - `gen_ai_client_token_usage_total{gen_ai_token_type,gen_ai_operation_name}`: Spring은 기본 제공, Python은 직접 증가. 임베딩 토큰과 구분
  - `rag_llm_cost_usd_total{model}`, `rag_ratelimit_rejections_total{limit}`, `rag_ask_errors_total{type}`
  - 스트리밍 첫 토큰 시간 `rag_stream_first_token_seconds`
- [ ] Python: `prometheus-fastapi-instrumentator` 8.1, uvicorn 단일 워커
- [ ] Compose 전체: app 2개, postgres, redis, tei, prometheus `v3.15.1`(scrape 5s), grafana `12.4.13`(classic 파일 프로비저닝), jaeger `2.22.0`(`tracing` 프로필). 모든 포트는 `127.0.0.1:`로 공개하고 비밀번호는 `.env`에서
- [ ] "`docker compose up --build` 한 번" 조건 맞추기:
  - Spring 이미지는 빌드 스테이지를 포함한 멀티스테이지 Dockerfile(`bootBuildImage`는 별도 빌드 단계가 필요해 제외)
  - Python 이미지는 TEI를 쓰므로 torch 없이 가볍게
  - `ingest` one-shot 서비스 → 앱은 `depends_on: condition: service_completed_successfully`(빈 인덱스로 뜨지 않게)
  - TEI는 첫 기동 시 모델 2.3GB를 내려받으므로 named volume + healthcheck `start_period` 넉넉히
- [ ] OTel을 켤 경우 `spring-boot-starter-opentelemetry`가 OTLP 메트릭을 기본으로 내보내므로 `management.otlp.metrics.export.enabled=false`
- [ ] 대시보드 JSON은 **실제 `/actuator/prometheus` 출력을 확인한 뒤** 작성(문서상 지표 이름과 실제가 다름)
- [ ] 공식 p50/p95·적중률 수치는 Grafana가 아니라 `results/*.json` 기준

### 6단계 · 비교 정리 (`COMPARISON.md`)
- 브리프의 표에 다음을 추가합니다.
  - 측정 조건: indexVersion, 모델, 하드웨어, 반복 횟수
  - LOC 측정 규칙: `cloc`, 핵심 로직만, 테스트·설정 제외 여부 명시
  - 운영 편의 판단 기준
- Python판은 Redis가 없어 `cache`가 항상 MISS라는 점, 하이브리드 BM25는 Python 전용 부가 실험이라는 점을 명시합니다.
- 결론에는 수치와 판단 근거를 함께 적습니다.

---

## 5. 버전 고정표 (2026-10-26 확인)

| 구성 | 버전/태그 | 비고 |
|---|---|---|
| Spring Boot | 4.1.2 | 4.0.x는 2026-12 지원 종료 보도(미확인) |
| Spring AI | bom 2.0.2 | 2.1.0-RC1 사용 금지, GA 후 계획적으로 업그레이드 |
| Java | 21 또는 25 | 결정 필요 |
| MCP Java SDK | 2.1.0(전이) | |
| pgvector | `pgvector/pgvector:0.8.7-pg17` | HNSW 최대 2000차원(bge-m3 1024) |
| Redis | `redis:8.10.3` | |
| TEI | `text-embeddings-inference:cpu-1.9` | digest 고정 |
| Prometheus / Grafana | `v3.15.1` / `12.4.13` | Grafana 13은 dashboard v2 형식 주의 |
| Jaeger | `2.22.0` | 선택 프로필 |
| Python | 3.13, langchain-core 1.7, langchain-openai 1.7.2, langchain-anthropic 1.7.6, langchain-chroma 1.1.0(chromadb 1.5.9), fastapi 0.162, uvicorn 0.54 | `uvicorn[standard]`의 uvloop은 Windows 미지원 |
| MCP Inspector | 2.12.0 (Node 22.19+) | |

---

## 6. API 계약 보강안 (승인 필요)

```
POST /search   요청 { "query", "topK": 5 }   응답 { "indexVersion", "chunks":[{source, chunkId, score, text}] }
POST /ask      요청 { "question", "topK": 4, "sessionId"? }   헤더 X-User-Id, X-Cache-Bypass(평가용)
               응답 { answer, sources[], usage{inputTokens,outputTokens}, cache, latencyMs,
                      model, fallbackUsed, indexVersion }
               cache: MISS | HIT_EXACT | HIT_SEMANTIC | BYPASS
POST /ask/stream  SSE 이벤트: sources → token(반복) → done{usage, cache, latencyMs, model}
오류: 429 + Retry-After(사용량 초과), 504(LLM 타임아웃), 공통 오류 JSON
```

- **`/search`:** Hit@5 문제를 해결하고, MCP `searchDocuments`와 1:1로 대응해 "MCP 경로와 /ask 경로의 검색 결과가 같은지" 확인하는 데도 씁니다.

---

## 7. 결정이 필요한 사항 (권장안 포함)

| # | 결정 | 권장안 | 이유 |
|---|---|---|---|
| D1 | 저장소 위치 | `D:\hanil` 밖 독립 저장소(예: `D:\rag-lab`) | 상위 저장소의 다른 프로젝트 변경과 섞이지 않음 |
| D2 | 실제 사내 규정 범위 | **이 프로젝트에서는 제외**(docs/sample 전용) | 제외하면 외부 LLM을 유지할 수 있음. 실제 규정을 쓰려면 로컬 LLM이 필수인데 GPU가 없음 |
| D3 | 임베딩 런타임 | TEI | 현재 기준선 유지, 커스텀 코드 없음 |
| D4 | 비교용 LLM | 공급자·모델 하나로 고정(예: `gpt-5.4-mini-2026-03-17` 또는 Claude 하나), max_tokens 1024 | "같은 LLM" 원칙. 다른 모델은 부가 실험으로 |
| D5 | Python 벡터 저장소 | Chroma(cosine) 유지 | pgvector(langchain-postgres 0.0.x)는 Windows async 문제와 UUID id 제약이 있음. 저장소 차이는 검색 일치 게이트로 확인 |
| D6 | chunkId 형식 | `sample_a.md#0007`(TEXT) | 사람이 읽을 수 있고 평가셋과 바로 대조 가능 |
| D7 | 청킹·샘플 형식 | 조항 정규식(조 단위) + 긴 조문만 2차 분할, 샘플은 md만 | 규정 구조 반영. PDF 경로(PyMuPDF·띄어쓰기)는 비교 범위 밖. 필요하면 가상 md를 PDF로 변환한 샘플을 추가 |
| D8 | Hit@5 처리 | `/search` 추가 | 6장 |
| D9 | indexVersion | 지문이 바뀔 때만 증가, manifest가 원천 | 캐시 전후 비교 잡음 제거 |
| D10 | Java / 빌드 도구 | Java 25(LTS) + Gradle | 공식 Dockerfile 베이스가 25. 사내 다른 프로젝트가 Gradle 사용 |
| D11 | 시맨틱 캐시 | 직접 구현, 1차는 숫자·날짜·부정어 질문 건너뜀 | 1장 #11 |
| D12 | 대화 이력 | List 직접 구현 + 이력 있는 세션은 BYPASS | 브리프 키 설계 유지 |
| D13 | API 계약 변경 | 6장 승인 | |
| D14 | 트레이스 | Jaeger를 선택 프로필로 | 브리프 목록에 트레이스 백엔드 없음 |
| D15 | 하이브리드 BM25 | 공식 비교에서 제외, Python 부가 실험 | Spring에 같은 한국어 형태소 분석을 맞추기 어려움 |
| D16 | 답변 품질 평가 | 2차 과제(키포인트 매칭부터) | 브리프에 없음, LLM 판정은 API 비용 발생 |
| D17 | Python 실행 위치 | 개발은 호스트, 비교 측정은 컨테이너 | 같은 네트워크·OS 조건 |

---

## 8. 주요 리스크

| 리스크 | 대응 |
|---|---|
| 임베딩 클라이언트 설정 실수로 오류 없이 엉터리 벡터 생성 | 자가진단 + 검색 일치 게이트 |
| pgvector `id-type`, Chroma `space`는 생성 후 변경 불가 | 0~1단계에서 확정 후 테이블 생성 |
| 시맨틱 캐시가 다른 조항·연도·금액 질문에 엉뚱한 답을 돌려줌 | 숫자·날짜·부정어 가드, 일치한 원 질문 로그, 오적중률 측정 |
| 관측 데이터(로그·트레이스)에 문서 본문·질문 원문이 남음 | 5단계 차단 설정, OTLP는 로컬만 |
| CPU 전용 환경의 메모리·지연(TEI 3GB + JVM + PG + …) | WSL2 메모리 확인, 임베딩 시간과 LLM 시간 분리 측정 |
| 사내 프록시가 모델(2.3GB)·이미지·Maven 다운로드를 막음 | 사전 다운로드 볼륨, NO_PROXY 설정 |
| Spring AI 2.1 GA가 임박해 API 변동 가능 | 버전 고정, 업그레이드는 단계 사이에만 |
| 문서와 소스가 다른 곳이 많음(속성명·지표명·스타터) | 구현 시 2.0.2 소스·실행 결과 기준으로 확인 |
| 외부 LLM으로 top-k 청크가 나감 | docs/sample 전용(D2), MCP 외부 연결도 샘플 색인일 때만 |
| Windows 환경 문제: PowerShell 한글 인코딩, CRLF, uvloop | `PYTHONUTF8=1`, `.gitattributes`, uvicorn 기본 루프 |
