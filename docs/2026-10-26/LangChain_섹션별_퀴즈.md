# LangChain 섹션별 핵심 퀴즈 (객관식 30문항)

섹션 1~6 PDF 내용 기준, 섹션마다 5문항. 정답은 각 문항 아래 **정답 보기**를 펼쳐서 확인하세요.

---

## 섹션 1. RAG와 LangChain 기초

**Q1. RAG(Retrieval-Augmented Generation)의 기본 원리로 옳은 것은?**
1. 사내 문서로 LLM을 다시 학습(파인튜닝)한다
2. 질문과 관련된 정보를 검색해 LLM 입력에 추가한 뒤 답변을 생성한다
3. 프롬프트 길이를 줄여 응답 속도를 높인다
4. LLM이 생성한 답변을 검색 엔진으로 다시 교정한다

<details><summary>정답 보기</summary>

**정답: 2** — RAG는 관련 정보를 *검색*해서 LLM 입력에 *추가*하여 더 정확한 답을 만드는 기술입니다. 모델 재학습(1)과는 다른 접근입니다.
</details>

**Q2. 강의에서 소개한 RAG의 장점이 아닌 것은?**
1. 환각(hallucination) 감소
2. 최신 정보 반영
3. 도메인 특화 지식 활용
4. LLM의 파라미터 수 감소

<details><summary>정답 보기</summary>

**정답: 4** — RAG는 외부 지식을 붙이는 방식이라 모델 자체의 크기와는 무관합니다.
</details>

**Q3. 긴 문서를 처리 가능한 크기의 청크로 나누는 LangChain 컴포넌트는?**
1. Document Loader
2. Text Splitter
3. Retriever
4. Vector Store

<details><summary>정답 보기</summary>

**정답: 2** — Text Splitter가 문자·토큰 기반 등 다양한 전략으로 청크를 만듭니다.
</details>

**Q4. 질문에 관련된 문서 검색부터 최종 답변 생성까지 전체 RAG 파이프라인을 구축하는 함수는?**
1. `create_stuff_documents_chain`
2. `create_retrieval_chain`
3. `RunnableParallel`
4. `StrOutputParser`

<details><summary>정답 보기</summary>

**정답: 2** — `create_stuff_documents_chain`은 *검색된 문서들을 하나의 컨텍스트로 묶어 답변*하는 체인이고, 이를 검색기와 연결해 전체 파이프라인을 만드는 것이 `create_retrieval_chain`입니다.
</details>

**Q5. 강의에서 OpenAI API 키를 설정한 방법은?**
1. 코드에 키를 직접 문자열로 하드코딩
2. 프로젝트 폴더의 `.env` 파일에 `OPENAI_API_KEY`를 적고 `python-dotenv`로 로드
3. `pyproject.toml`의 `[tool.poetry.dependencies]`에 기입
4. Gradio 설정 화면에서 입력

<details><summary>정답 보기</summary>

**정답: 2** — `.env` + `python-dotenv` 방식입니다. 사내 프로젝트에서도 키를 코드에 넣지 않는 것이 원칙입니다.
</details>

---

## 섹션 2. LangChain LCEL

**Q1. LCEL에서 프롬프트, LLM, 파서를 연결할 때 쓰는 연산자는?**
1. `+`
2. `>>`
3. `|`
4. `&`

<details><summary>정답 보기</summary>

**정답: 3** — `chain = prompt | llm | StrOutputParser()` 처럼 pipe 연산자로 연결합니다.
</details>

**Q2. Gradio 챗봇에서 답변을 글자가 생성되는 대로 실시간 출력하려면 어떤 메서드를 사용하는가?**
1. `invoke`
2. `stream`
3. `batch`
4. `parse`

<details><summary>정답 보기</summary>

**정답: 2** — `invoke`는 완성된 출력, `batch`는 입력 리스트를 배치로 처리, `stream`은 실시간 스트림을 생성합니다.
</details>

**Q3. `RunnablePassthrough`의 역할은?**
1. 입력을 그대로 다음 단계로 전달한다
2. 입력을 대문자로 변환한다
3. 여러 체인을 동시에 실행한다
4. LLM 출력을 JSON으로 파싱한다

<details><summary>정답 보기</summary>

**정답: 1** — RAG 체인에서 `{"context": retriever, "question": RunnablePassthrough()}`처럼 사용자 질문을 그대로 넘길 때 씁니다. 사용자 함수 변환은 `RunnableLambda`, 동시 실행은 `RunnableParallel`입니다.
</details>

**Q4. `RunnableParallel`로 여러 체인을 실행했을 때 반환되는 결과 형태는?**
1. 마지막 체인의 문자열 결과만 반환
2. 각 체인 결과를 이어 붙인 하나의 문자열
3. 지정한 키와 각 체인 결과로 구성된 딕셔너리
4. 체인 실행 순서가 담긴 리스트

<details><summary>정답 보기</summary>

**정답: 3** — 여러 작업을 동시에 실행하고, 지정한 키에 결과를 담은 딕셔너리로 반환합니다.
</details>

**Q5. LLM의 JSON 형식 출력을 Python 딕셔너리로 변환해 주는 Output Parser는?**
1. `StrOutputParser`
2. `JsonOutputParser`
3. `PydanticOutputParser`
4. `CSVOutputParser`

<details><summary>정답 보기</summary>

**정답: 2** — `StrOutputParser`는 문자열, `PydanticOutputParser`는 Pydantic 모델로 파싱·검증합니다.
</details>

---

## 섹션 3. 데이터 처리 및 임베딩 기법

**Q1. PDF에서 텍스트를 추출해 페이지별 Document 객체로 변환하는 로더와 필요한 패키지는?**
1. `WebBaseLoader` + bs4
2. `JSONLoader` + jq
3. `PyPDFLoader` + pypdf
4. `CSVLoader` + pandas

<details><summary>정답 보기</summary>

**정답: 3** — `langchain_community`의 `PyPDFLoader`를 쓰고 `pypdf`를 설치합니다. PyMuPDF, Unstructured 같은 대안도 있습니다.
</details>

**Q2. "제1장, 제2장…" 또는 "제1조, 제2조…"처럼 구조가 정해진 문서를 정확하게 나누기에 가장 적합한 분할 방법은?**
1. 정규표현식 기반 분할
2. 고정 길이 문자 분할
3. 토큰 수 기반 분할
4. 무작위 샘플링 분할

<details><summary>정답 보기</summary>

**정답: 1** — 정규식은 특정 패턴 기준으로 나눌 수 있어 구조화된 문서에 유용합니다. 사내 규정 챗봇의 조문 단위 분할에 그대로 쓰입니다.
</details>

**Q3. Semantic Chunking(맥락 기반 분할)에 대한 설명으로 옳은 것은?**
1. 마침표 기준으로만 문장을 나눈다
2. 문장들 사이의 임베딩 차이를 기반으로 의미가 바뀌는 지점에서 나눈다
3. tiktoken으로 토큰 수를 세어 일정하게 나눈다
4. `langchain_community` 기본 기능이라 추가 설치가 필요 없다

<details><summary>정답 보기</summary>

**정답: 2** — 의미적 유사성 기반으로 분할하며, `langchain_experimental` 패키지를 설치해 사용했습니다.
</details>

**Q4. `chunk_size`와 `chunk_overlap`에 대한 설명으로 옳은 것은?**
1. `chunk_size`가 작을수록 더 많은 컨텍스트를 담는다
2. `chunk_overlap`은 청크 사이의 연속성을 유지해 정보 손실을 막는다
3. `chunk_overlap`을 크게 할수록 저장 용량이 줄어든다
4. 두 값은 검색 결과에 영향을 주지 않는다

<details><summary>정답 보기</summary>

**정답: 2** — 큰 `chunk_size`는 더 많은 컨텍스트, 작은 `chunk_size`는 더 정확한 검색에 유리하고, 오버랩은 청크 간 연속성을 유지합니다.
</details>

**Q5. 민감한 데이터(예: 사내 규정)를 다룰 때 개인정보 보호 측면에서 강의가 권장한 임베딩 선택은?**
1. OpenAI 임베딩 — 품질이 가장 높으므로
2. 로컬에서 실행 가능한 HuggingFace 또는 Ollama 임베딩
3. 어떤 모델이든 상관없음
4. 임베딩 없이 키워드 검색만 사용

<details><summary>정답 보기</summary>

**정답: 2** — OpenAI는 데이터가 외부 서버로 전송된다는 단점이 있습니다. 단, Ollama는 한국어 성능이 상대적으로 떨어질 수 있어 한국어는 `BAAI/bge-m3` 같은 HF 모델을 많이 씁니다.
</details>

---

## 섹션 4. 벡터스토어 검색기 활용 및 성능 평가

**Q1. 키워드 검색 알고리즘 BM25의 특징이 아닌 것은?**
1. TF-IDF의 한계를 보완한 랭킹 함수다
2. 문서 길이 정규화를 적용한다
3. 동의어와 관련어를 자동으로 처리한다
4. 계산 비용이 비교적 낮다

<details><summary>정답 보기</summary>

**정답: 3** — BM25는 의미적 유사성·동의어를 처리하지 못하는 것이 단점입니다. 그래서 의미 기반 검색과 합친 Hybrid Search를 씁니다.
</details>

**Q2. LangChain의 `EnsembleRetriever`가 여러 검색기 결과를 재정렬할 때 사용하는 알고리즘은?**
1. MMR (Maximal Marginal Relevance)
2. RRF (Reciprocal Rank Fusion)
3. K-means 클러스터링
4. ROUGE

<details><summary>정답 보기</summary>

**정답: 2** — `score = Σ 1 / (r + k)` (r: 각 검색기에서의 순위, k: 보통 60)로 최종 순위를 정합니다.
</details>

**Q3. FAISS에 대한 설명으로 옳지 않은 것은?**
1. Facebook AI가 만든 유사도 검색 라이브러리다
2. 대규모 벡터 세트에서 효율적인 검색이 가능하다
3. `faiss-gpu`를 설치하면 GPU 가속을 쓸 수 있다
4. 인덱싱 알고리즘이 하나로 고정되어 속도·정확도를 조절할 수 없다

<details><summary>정답 보기</summary>

**정답: 4** — FAISS는 다양한 인덱싱 알고리즘을 제공해 속도와 정확도를 조절할 수 있습니다.
</details>

**Q4. Hit Rate(적중률)의 정의로 옳은 것은?**
1. 상위 k개 결과 중 관련 문서의 비율
2. 검색 결과에 관련 문서가 하나라도 포함된 쿼리의 비율
3. 첫 번째 관련 문서 순위의 역수 평균
4. 실제 DCG를 이상적인 DCG로 나눈 값

<details><summary>정답 보기</summary>

**정답: 2** — 1은 Precision@k, 3은 MRR, 4는 NDCG@k입니다.
</details>

**Q5. 질문 3개에 대해 첫 번째 관련 문서가 각각 1위, 2위, 4위에 나왔다. MRR은?**
1. 약 0.58
2. 약 0.33
3. 약 0.75
4. 1.0

<details><summary>정답 보기</summary>

**정답: 1** — (1/1 + 1/2 + 1/4) / 3 = 1.75 / 3 ≈ 0.58
</details>

---

## 섹션 5. 고급 검색 기법

**Q1. MultiQuery Retriever의 작동 방식으로 옳은 것은?**
1. 원본 쿼리를 그대로 여러 번 검색한다
2. LLM이 원본 쿼리를 다양한 관점의 여러 관련 쿼리로 만들고, 모든 쿼리로 검색한다
3. 검색 결과를 LLM이 요약해 하나로 만든다
4. 쿼리를 다른 언어로 번역해서 검색한다

<details><summary>정답 보기</summary>

**정답: 2** — 검색의 다양성을 높이고 누락 정보를 줄입니다. 직원의 구어체 질문과 규정 문어체 용어 차이를 메우는 데 유용합니다.
</details>

**Q2. Decomposition(Least-to-Most Prompting)에 대한 설명으로 옳은 것은?**
1. 긴 문서를 작은 청크로 나누는 분할 기법이다
2. 복잡한 질문을 하위 질문으로 분해하고, 순차적으로 답한 뒤 종합해 최종 답변을 만든다
3. 검색 결과 중 관련 없는 문서를 제거한다
4. 쿼리 임베딩의 차원을 줄인다

<details><summary>정답 보기</summary>

**정답: 2** — 단계별 결과를 볼 수 있어 설명 가능성도 높아집니다.
</details>

**Q3. CrossEncoder Reranker의 작동 방식은?**
1. 쿼리와 문서를 각각 따로 임베딩해 코사인 유사도를 계산한다
2. 쿼리와 문서를 쌍으로 함께 입력해 모델이 관련성 점수를 직접 출력하고, 점수로 재정렬한다
3. BM25 점수와 벡터 점수를 평균낸다
4. LLM이 순위 조정 이유를 설명한다

<details><summary>정답 보기</summary>

**정답: 2** — 실습에서는 `BAAI/bge-reranker-v2-m3`를 사용했습니다. 1은 일반 벡터 검색(Bi-Encoder), 4는 LLM Reranker의 특징입니다.
</details>

**Q4. 문서 내용은 변경하지 않고, LLM 판단으로 관련 있는 문서만 남기는 압축기는?**
1. `LLMChainFilter`
2. `LLMChainExtractor`
3. `EmbeddingsFilter`
4. `DocumentCompressorPipeline`

<details><summary>정답 보기</summary>

**정답: 1** — `LLMChainExtractor`는 문서에서 관련 부분만 *추출·요약*하므로 내용이 바뀝니다. Pipeline은 여러 압축기를 연결하는 틀입니다.
</details>

**Q5. LLM을 사용하지 않아 빠르고 비용 효율적인 문서 압축(필터링) 방법은?**
1. `LLMChainExtractor`
2. LLM Reranker
3. `EmbeddingsFilter`
4. MultiQuery

<details><summary>정답 보기</summary>

**정답: 3** — 쿼리와 문서 임베딩의 유사도가 임계값을 넘는 문서만 남깁니다.
</details>

---

## 섹션 6. LLM 답변 생성 및 평가

**Q1. 고성능 AI 칩으로 초고속 LLM 추론을 제공하고, OpenAI API와 유사한 인터페이스를 가진 서비스는?**
1. Ollama
2. Groq
3. Google Gemini
4. Anthropic Claude

<details><summary>정답 보기</summary>

**정답: 2** — Ollama는 로컬에서 오픈소스 LLM을 실행하는 프레임워크입니다.
</details>

**Q2. Embedding Distance로 답변을 평가할 때 점수 해석으로 옳은 것은?**
1. 거리 점수가 높을수록 의미적으로 더 유사하다
2. 거리 점수가 낮을수록 의미적으로 더 유사하다
3. 거리 점수는 단어 일치 개수다
4. 거리 점수는 항상 0 또는 1이다

<details><summary>정답 보기</summary>

**정답: 2** — 예측 답변과 정답을 임베딩한 뒤 거리를 재므로, 거리가 작을수록 유사합니다.
</details>

**Q3. ROUGE-L은 무엇을 기반으로 유사성을 측정하는가?**
1. N-gram 중첩
2. 최장 공통 부분 수열(LCS)
3. 임베딩 코사인 유사도
4. LLM의 판단

<details><summary>정답 보기</summary>

**정답: 2** — ROUGE-N은 N-gram 중첩, ROUGE-W는 가중치를 준 LCS입니다.
</details>

**Q4. LangChain QA 평가기 중 최종 판단 전에 단계별 추론 과정(Chain of Thought)을 거치는 것은?**
1. `qa`
2. `context_qa`
3. `cot_qa`
4. `criteria`

<details><summary>정답 보기</summary>

**정답: 3** — `context_qa`와 비슷하지만 단계별 추론을 추가해 복잡한 질문 평가에 적합합니다.
</details>

**Q5. 정답 레이블(ground truth)이 없는 상황에서, 주어진 기준으로 답변 품질을 1~10점으로 평가하는 평가기는?**
1. `criteria`
2. `score_string`
3. `labeled_criteria`
4. `labeled_score_string`

<details><summary>정답 보기</summary>

**정답: 2** — `criteria`는 이진(Y/N) 점수, `labeled_*`는 정답 레이블이 있을 때 사용합니다.
</details>

---

## 정답 한눈에 보기

| 섹션 | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|
| 1 | 2 | 4 | 2 | 2 | 2 |
| 2 | 3 | 2 | 1 | 3 | 2 |
| 3 | 3 | 1 | 2 | 2 | 2 |
| 4 | 3 | 2 | 4 | 2 | 1 |
| 5 | 2 | 2 | 2 | 1 | 3 |
| 6 | 2 | 2 | 2 | 3 | 2 |
