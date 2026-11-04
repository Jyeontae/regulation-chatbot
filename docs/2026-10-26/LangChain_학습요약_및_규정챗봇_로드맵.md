# LangChain 학습 요약 + 사내 규정 챗봇 로드맵

- 학습 자료: 섹션 1~6 PDF
- 실습 환경: `C:\Users\JYT\langchain_env` (노트북 001~006, langchain 0.2.16, poetry)
- 목표: 학습 내용을 바탕으로 **사내 규정 챗봇** 구축

---

## Part 1. 섹션별 핵심 요약

| 섹션 | 노트북 | 핵심 내용 | 규정 챗봇에서 쓰임 |
|---|---|---|---|
| 1. RAG·LangChain 기초 | 001_SimpleRAG | RAG 원리(검색 → 프롬프트 주입 → 생성), 6대 컴포넌트(Loader·Splitter·Embedding·VectorStore·Retriever·LLM), `create_stuff_documents_chain` + `create_retrieval_chain`, Gradio | 전체 뼈대 |
| 2. LCEL | 002_SimpleRAG_LCEL | `prompt \| llm \| parser` 파이프, invoke/stream/batch, `RunnableParallel/Passthrough/Lambda`, LCEL로 RAG 체인 재구성, stream 출력 Gradio | 체인 구성 방식의 표준 |
| 3. 데이터 처리·임베딩 | 003_Data_Processing | 로더(PDF/Web/JSON/CSV), 분할(Recursive·**정규식**·토큰·Semantic), 임베딩 비교(OpenAI `text-embedding-3-small` / HF `BAAI/bge-m3` / Ollama), 다국어 RAG(번역·언어 라우팅), 채팅 히스토리 | 규정 문서 파싱·조항 단위 분할, 임베딩 선택 |
| 4. 벡터스토어·검색 평가 | 004_Vectorstore_Retrieval | Chroma/FAISS CRUD·저장/로드, 검색(Top-K·임계값·MMR·**metadata 필터**), BM25(+kiwipiepy 한국어 토큰화), **Hybrid(EnsembleRetriever, RRF)**, 합성 QA 테스트셋, krag 지표(Hit Rate·MRR·Recall@k·Precision@k·mAP·NDCG) | 검색 엔진 + 정량 평가 |
| 5. 고급 검색 | 005_Advanced_Retrieval | 쿼리 확장(MultiQuery, Decomposition), Re-rank(CrossEncoder `bge-reranker-v2-m3`, LLM Reranker), 맥락 압축(LLMChainFilter·Extractor·EmbeddingsFilter·Pipeline) | 검색 정확도 고도화 |
| 6. 답변 생성·평가 | 006_Generation_Evaluation | LLM 공급자 비교(OpenAI·Claude·Gemini·Ollama·Groq), 휴리스틱 평가(Embedding distance·Cross-Encoder·ROUGE), LLM-as-judge(qa/context_qa/cot_qa, criteria/score_string, labeled_*) | 모델 선택 + 답변 품질 평가 |

**전체 흐름:**
문서 로드 → 분할 → 임베딩 → 벡터스토어 → (하이브리드 검색 → 재순위/압축) → 프롬프트 + LLM → 평가 → Gradio UI

### 섹션별 기억할 포인트

**섹션 1. RAG와 LangChain 기초**
- RAG = 질문 관련 정보를 검색해서 LLM 입력에 붙여 답변 → 환각 감소, 최신·도메인 지식 반영
- 활용 사례 1번이 "기업 내부 지식베이스 기반 챗봇" = 이번 목표
- 환경: conda 또는 poetry, `.env`에 `OPENAI_API_KEY` + python-dotenv

**섹션 2. LCEL**
- `chain = prompt | llm | StrOutputParser()` 형태로 컴포넌트 연결
- Output Parser: Str / Json / Pydantic (구조화 출력)
- `RunnableParallel`(병렬 실행 후 dict 반환), `RunnablePassthrough`(입력 그대로 전달), `RunnableLambda`(사용자 함수 적용)
- RAG 체인: `{"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | parser`

**섹션 3. 데이터 처리 및 임베딩**
- 분할 전략 4가지: 문자 재귀(Recursive), 정규식(1장, 2장… 같은 구조), 토큰 기준(tiktoken/HF), 의미 기준(Semantic Chunking)
- `chunk_size` 크면 맥락 많음 / 작으면 검색 정밀, `chunk_overlap`으로 정보 손실 방지
- 임베딩 선택 기준: 성능(OpenAI·좋은 HF 모델), 비용(HF·Ollama), **개인정보(로컬 HF·Ollama)**, 하드웨어
- Ollama 임베딩은 한국어 성능이 상대적으로 약함 → 한국어는 `bge-m3` 계열이 유리

**섹션 4. 벡터스토어 검색기 및 성능 평가**
- Chroma: 편의성, 문서 추가/변경/삭제, 로컬 저장·로드 / FAISS: 대규모·고속
- Semantic 검색 옵션: Top-K, 임계값, MMR(다양성), metadata 필터, 본문 필터
- BM25: 정확한 키워드에 강함, 의미·동의어는 약함
- Hybrid: `EnsembleRetriever`가 RRF(`score = Σ 1/(rank + k)`, k≈60)로 결과 통합
- 평가셋: LLM으로 QA 합성 → 사람 검수 → 반복 개선 / 지표: Hit Rate, MRR, Recall@k, Precision@k, mAP@k, NDCG@k

**섹션 5. 고급 검색 기법**
- 쿼리 확장: MultiQuery(여러 관점 쿼리 생성), Decomposition(복잡 질문 → 하위 질문 → 종합)
- Re-rank: CrossEncoder(쿼리-문서 쌍 직접 점수화, 정확), LLM Reranker(추론 필요 시, 비쌈)
- 맥락 압축: LLMChainFilter(문서 선별), LLMChainExtractor(관련 부분만 추출), EmbeddingsFilter(LLM 없이 빠름), Pipeline(순차 결합)

**섹션 6. LLM 답변 생성 및 평가**
- 공급자: OpenAI, Claude(긴 컨텍스트), Gemini(멀티모달), Ollama(로컬), Groq(초고속)
- 휴리스틱 평가: Embedding distance(빠름), Cross-Encoder(정확·느림), ROUGE(표면 일치)
- LLM-as-judge: `qa`/`context_qa`/`cot_qa`(정답 여부), `criteria`/`score_string`(정답 없이 기준 평가), `labeled_criteria`/`labeled_score_string`(정답 대비 평가)

---

## Part 2. 사내 규정 챗봇 로드맵

| Phase | 내용 | 기간(예상) | 근거 섹션 |
|---|---|---|---|
| 0 | 요구사항·보안 제약 정리 | 1주 | 3.5, 6.1 |
| 1 | 문서 수집·전처리·메타데이터 | 1~2주 | 3 |
| 2 | 인덱싱 + 기본 RAG (하이브리드) | 1주 | 2, 4 |
| 3 | 평가셋 구축·검색 평가 | 1~2주 | 4.3 |
| 4 | 검색 고도화 | 1~2주 | 5 |
| 5 | 답변 모델 선택·평가 | 1주 | 6 |
| 6 | UI·배포 | 1~2주 | 1.5, 2.4, 3.4 |
| 7 | 운영·개선 | 지속 | 4.1 |

### Phase 0. 요구사항·제약 정리
- 대상 규정 목록, 파일 형식(PDF/DOCX/**HWP** 여부), 개정 주기, 담당 부서 확인
- **보안 결정이 최우선**: 규정을 외부 API(OpenAI 등)로 보내도 되는지
  - 불가 → 임베딩은 로컬 `bge-m3`, LLM은 Ollama 또는 사내 허용 모델
- 사용자 범위·권한(전 직원 vs 부서별 열람 제한) 확인

### Phase 1. 문서 수집·전처리
- 로더: PyPDFLoader 기본, 표·서식이 많으면 PyMuPDF/Unstructured 검토. HWP는 PDF/DOCX로 변환 후 처리
- **분할: 정규식으로 `제N장 / 제N조 / ①항` 단위 분할** → 조문이 중간에 잘리지 않게. 긴 조문만 Recursive로 2차 분할
- **메타데이터 설계**: 규정명, 장·조 번호, 조 제목, 시행/개정일, 소관부서, 원본 파일·페이지 → 출처 표기·필터링에 사용

### Phase 2. 인덱싱·기본 RAG
- 임베딩 `BAAI/bge-m3`(한국어 성능, 로컬 실행) + Chroma 영구 저장(`persist_directory`)
- BM25(kiwipiepy 토크나이저) + 벡터 검색 → `EnsembleRetriever` 하이브리드
  - 규정에는 "연차", "경조금", "출장비" 같은 정확한 용어가 많아 키워드 검색 비중이 중요
- LCEL로 RAG 체인 구성, 프롬프트 원칙:
  - 검색된 조문 안에서만 답변, 없으면 "해당 규정을 찾을 수 없습니다"
  - 답변 끝에 **근거 조항(규정명 제N조) 인용**

### Phase 3. 평가셋 구축·검색 평가
- `data/qa_test.xlsx` 방식으로 규정 QA 50~100쌍: LLM 합성 + **규정 담당자 검수**
- 질문 유형 분배: 단순 사실(금액·일수), 조건 판단, 여러 조항 종합, 규정에 없는 질문
- krag로 Hit Rate / MRR / NDCG@k 측정 → 청크 크기·k·하이브리드 가중치 튜닝의 기준선

### Phase 4. 검색 고도화 (지표를 보고 선택 적용)
- Re-rank: `bge-reranker-v2-m3` CrossEncoder (가성비 최우선 후보)
- MultiQuery: 직원 구어체 질문 ↔ 규정 문어체 용어 차이 보완
- Decomposition: "출장 중 경조사가 생기면 휴가·비용은?" 같은 복합 질문
- Contextual compression: 조문이 길 때 토큰 절감
- 각 기법 적용 전후로 Phase 3 지표를 비교해 효과 있는 것만 채택

### Phase 5. 답변 생성 모델 선택·평가
- 보안 조건 안에서 후보 LLM 2~3개 비교
- LLM-as-judge: `labeled_score_string`(정답 대비 점수), `criteria`(correctness, conciseness) + "근거 조항 인용 여부" 자체 기준 추가
- 환각 방지 확인: 규정에 없는 질문에 답을 지어내지 않는지 별도 체크

### Phase 6. UI·배포
- Gradio ChatInterface + 스트리밍 + 대화 히스토리, 답변 아래 출처 조문 표시
- 사내 서버 배포, 접근 제어(사내망/SSO), 질문·답변 로그 저장
- 권한별 열람 제한이 있으면 metadata 필터로 검색 범위 제한

### Phase 7. 운영·개선
- 규정 개정 시 재인덱싱 절차: 해당 규정 문서 delete → add (Chroma 관리 기능)
- 👍/👎 피드백 + 실패한 질문을 평가셋에 추가 → 주기적 재평가
- 구 규정 vs 신 규정 질의가 필요하면 시행일 메타데이터로 필터

### 참고: 버전 관련 주의
- 학습 환경은 langchain **0.2.16**(2024년 기준). 실제 프로젝트 시작 시 최신 LangChain의 import 경로·체인 API(`create_retrieval_chain`, `langchain.evaluation` 등) 변경 여부를 먼저 확인하고, 새 프로젝트는 별도 poetry 환경으로 시작 권장
- 대화 상태·분기 로직이 복잡해지면 LangGraph 검토
