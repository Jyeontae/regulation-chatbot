# 사내 규정 챗봇 (regulation-chatbot)

사내 규정 PDF를 검색해 근거 조항과 함께 답변하는 RAG 챗봇.

| 구성 | 선택 | 실행 위치 |
|---|---|---|
| 임베딩 | `BAAI/bge-m3` | 로컬 (CPU) |
| 리랭커 | `BAAI/bge-reranker-v2-m3` (후보 8개) | 로컬 (CPU) |
| 답변·검색어 다듬기 LLM | OpenAI `gpt-5.4-mini` (현재 사용) / Anthropic Claude (`.env`로 전환) | 외부 API |
| 벡터 저장소 | Chroma (cosine) | 로컬 |
| PDF 로더 | PyMuPDF | 로컬 |

규정 전체(분할·인덱싱·리랭크)는 로컬에서만 처리한다. 질문 하나에 외부 LLM API로 가는 것은 두 가지다.
1. 질문 (검색어 다듬기)
2. 질문 + 검색된 조문 발췌 몇 개(`RETRIEVER_K`, 기본 4) (답변)

채팅 화면에서는 후속 질문을 이해하도록 같은 대화의 직전 질문(`HISTORY_TURNS`, 기본 2개)도 두 호출에 함께 간다. 이전 답변과 발췌는 보내지 않는다.

설정값(effort, 리랭커 입력, 프롬프트 등)은 Claude 기준 평가 결과로 정했다. 근거는 `docs/` 날짜별 기록에 있다.

## 설치

```bash
# (선택) 가상환경을 프로젝트 폴더 안 .venv 에 만들기
poetry config virtualenvs.in-project true --local

poetry install
copy .env.example .env
```

`.env`를 열어 API 키와 모델을 입력한다.
- `OPENAI_API_KEY`, `OPENAI_MODEL` (모델 ID는 OpenAI 모델 문서에서 확인)
- `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`
- `LLM_PROVIDER` : 기본으로 쓸 LLM (`openai` 또는 `anthropic`)

현재 운영 값 (각 항목의 설명은 `.env.example` 주석 참고):

| 설정 | 값 | 근거 |
|---|---|---|
| `LLM_PROVIDER` / `OPENAI_MODEL` | `openai` / `gpt-5.4-mini-2026-03-17` | 11-02 Claude 크레딧 소진으로 전환 |
| `REWRITE_MODEL` | 비움 (OpenAI는 답변 모델과 같은 모델) | |
| `ANSWER_EFFORT` / `REWRITE_EFFORT` | `low` / `low` | `docs/2026-11-02/답변평가_effort_비교.md` |
| `RERANK_CANDIDATES` | `8` | `docs/2026-11-04/후보모으기_비교.md` |
| `REWRITE_TIMEOUT` | `5` (초) | `docs/2026-11-04/마무리_정비.md` |
| `MAX_QUESTION_CHARS` | `300` | `docs/2026-11-04/마무리_정비.md` |
| `HISTORY_TURNS` | `2` | `docs/2026-11-04/후속_질문_히스토리.md` |

## 환경 확인

```bash
poetry run python scripts/check_setup.py --skip-llm   # 임베딩까지만 (API 비용 없음)
poetry run python scripts/check_setup.py              # LLM API 연결까지
```

- 첫 실행 시 bge-m3 모델(약 2GB)을 Hugging Face에서 내려받는다. 리랭커 모델은 리랭크를 처음 켤 때 내려받는다.
- 사내망에서 다운로드가 막히면 다른 곳에서 모델 폴더를 받아 옮긴 뒤 `EMBEDDING_MODEL`·`RERANKER_MODEL`에 그 경로를 입력한다.
- 채팅 화면(`chat.py`)은 받아 둔 모델만 쓰도록 Hugging Face 오프라인으로 실행된다. 모델을 처음 받거나 바꿀 때는 셸에서 `HF_HUB_OFFLINE=0`을 설정하고 한 번 실행한다 (PowerShell: `$env:HF_HUB_OFFLINE="0"`).

## 규정 점검·분할

```bash
poetry run python scripts/inspect_docs.py   # 텍스트 추출 점검
poetry run python scripts/build_chunks.py   # 조문 단위 분할 → data/chunks/chunks.jsonl
```

- 두 스크립트 모두 화면에는 숫자만 출력한다. 파일 이름까지 보려면 `--show-names`를 붙인다 (공유 금지).
- 분할 결과는 `chunks.jsonl`을 직접 열어 확인한다. 규정 원문이 들어 있으므로 외부에 올리지 않는다.

## 벡터화 (인덱싱)

```bash
poetry run python scripts/build_index.py   # chunks.jsonl → data/chroma
```

- 실행할 때마다 컬렉션을 비우고 다시 만든다. 분할을 고쳤으면 `build_chunks.py` 다음에 다시 실행한다.
- 화면에는 개수와 걸린 시간만 출력한다.

## 검색·답변 확인

```bash
poetry run python scripts/search.py "휴가 신청은 언제까지 해야 해?"   # 한 번 검색
poetry run python scripts/search.py                                  # 빈 줄을 입력할 때까지 계속 검색
poetry run python scripts/search.py --answer                         # 검색 결과로 LLM 답변 생성 (외부 API 전송)
poetry run python scripts/search.py --answer --provider openai       # 답변 LLM 지정 (기본: LLM_PROVIDER)
poetry run python scripts/search.py --mode hybrid                    # 검색 방식 지정 (기본: RETRIEVER_MODE)
poetry run python scripts/search.py --rewrite                        # LLM 으로 질문을 검색어로 다듬어 검색 (외부 API 전송)
poetry run python scripts/search.py --rewrite --merge-original       # 원래 질문과 다듬은 검색어로 각각 검색해 합침
```

- `--rewrite`는 검색 전에 질문을 규정 말투의 검색어 한 줄로 바꾸고, 다듬은 검색어를 화면에 보여 준다. 답변(`--answer`)은 원래 질문으로 만든다. 질문만 외부로 가고 규정 본문은 가지 않는다. 모델은 `.env`의 `REWRITE_MODEL`이고, 비우면 Claude는 `claude-haiku-5-5`, OpenAI는 답변 모델을 쓴다.
- `--merge-original`을 더하면 원래 질문으로도 검색해, 청크마다 두 검색 중 높은 유사도로 합친다. 표의 출처 열은 그 유사도를 낸 쪽(질문·검색어)이다.

- 검색 방식(`RETRIEVER_MODE`)은 `vector`(기본, 벡터 검색만)와 `hybrid`(벡터 + Kiwi 형태소 BM25를 RRF로 합침) 두 가지다. `hybrid`는 `chunks.jsonl`로 BM25를 만들므로 청크와 인덱스가 같은 분할 결과여야 한다. 다르면 실행을 멈춘다.
- `hybrid` 표에는 합치기 전 각 검색의 순위(벡터·BM25 열)가 함께 나온다.

- 기본 출력은 순위·카테고리·종류·조 위치·유사도(코사인 유사도, 1에 가까울수록 비슷)다. `-k`로 개수를 바꾼다.
- `--show-names`는 규정명, `--show-text`는 규정명과 본문까지 보여 준다 (본인 확인용, 공유 금지).
- `--answer`는 질문마다 상위 k개 청크(규정명·본문)를 외부 LLM API로 보내고, 표에 규정명도 표시한다. 실제 규정 인덱스에는 정보보안 승인 후에만 쓰고, 출력은 공유하지 않는다.
- LangSmith 추적(`LANGSMITH_TRACING` 등)이 켜져 있으면 `--answer`는 실행되지 않는다. 체인 입력이 LangSmith 서버에도 저장되기 때문이다.

## 채팅 화면

```bash
poetry run python scripts/chat.py                     # http://127.0.0.1:7860 에서 열림
poetry run python scripts/chat.py --provider openai   # 답변 LLM 지정 (기본: LLM_PROVIDER)
poetry run python scripts/chat.py --mode hybrid       # 검색 방식 지정 (기본: RETRIEVER_MODE)
poetry run python scripts/chat.py --no-rewrite        # 검색어 다듬기 끄기 (원래 질문으로만 검색)
poetry run python scripts/chat.py --rerank-candidates 0   # 리랭크 끄기 (기본: RERANK_CANDIDATES)
poetry run python scripts/chat.py --effort medium     # 답변 LLM effort (기본: ANSWER_EFFORT)
poetry run python scripts/chat.py --history 0         # 직전 질문 쓰지 않기 (기본: HISTORY_TURNS)
```

질문 하나의 처리 순서:

1. **길이 확인** — `MAX_QUESTION_CHARS`(300자)를 넘으면 LLM을 부르지 않고 돌려보낸다. 긴 문서를 붙여 넣어 외부로 나가는 것을 막는다.
2. **검색어 다듬기** — 질문을 규정 말투의 검색어 한 줄로 바꾸고 답변 맨 위에 보여 준다. 같은 대화의 직전 질문(기본 2개)을 함께 넣어, "그럼 계약직은?"처럼 앞 질문에 기댄 질문을 혼자 읽혀도 되는 검색어로 바꾼다. `REWRITE_TIMEOUT`(5초) 안에 응답이 없거나 실패하면 재시도 없이 원래 질문으로 검색한다.
3. **검색** — 원래 질문과 다듬은 검색어로 각각 검색해 청크마다 더 높은 유사도로 합친다(M4b). 근거는 `docs/2026-10-30/채팅화면과_검색튜닝_실험기록.md` 5장.
4. **리랭크** — 상위 8개 후보를 로컬 리랭커가 **다듬은 검색어**와 함께 읽어 다시 정렬한다. 근거는 `docs/2026-11-03/개선_후보_조사.md` 1절, `docs/2026-11-04/마무리_정비.md` 3절, 후보 수는 `docs/2026-11-04/후보모으기_비교.md`.
5. **답변** — 상위 k개 발췌와 원래 질문(직전 질문이 있으면 함께)을 답변 LLM에 보내 스트리밍으로 보여 준다. `search.py --answer`와 같은 체인을 쓴다.

- 답변 위에 단계별 시간(다듬기·검색·리랭크), 아래에 첫 글자·완료 시간과 호출별 토큰 수가 나온다.
- 다듬기 모델은 `REWRITE_MODEL`이고, 비우면 OpenAI는 답변 모델, Claude는 `claude-haiku-5-5`를 쓴다.
- OpenAI 호출에는 `store=False`를 넣어 요청·응답이 OpenAI 쪽에 저장되지 않게 한다.
- 이 PC(127.0.0.1)에서만 열리고 공개 링크(share)는 만들지 않는다. 화면에 규정 내용이 나오므로 공유하지 않는다.
- 대화 기록을 남기지 않는다. Gradio의 flag·실행 기록(run history)·브라우저 대화 저장을 모두 끈다.
- 이전 대화는 **직전 질문만** 쓴다. 이전 답변·발췌는 다시 보내지 않는다(외부 전송을 질문으로 한정하고, 이전 답의 오류가 이어지는 것도 막는다). 직전 질문을 썼으면 답변 맨 위에 "이전 질문 N개 참고"가 나온다.
- 새 주제를 시작하려면 대화 지우기(휴지통) 버튼을 누른다. 길이 상한을 넘어 거절된 질문은 직전 질문에 넣지 않는다.
- 직전 질문이 없을 때의 프롬프트는 평가 때와 글자 하나까지 같다. 평가 스크립트는 직전 질문을 쓰지 않는다.
- `search.py`·`eval_retrieval.py`의 기본 검색은 다듬기·리랭크 없는 v1이다. `search.py --rewrite --merge-original`은 리랭크 전 단계까지 챗봇과 같고(리랭크 옵션 없음), `eval_retrieval.py`는 아래 검색 평가의 "챗봇과 같은 검색" 줄을 쓴다.

## 검색 평가

```bash
poetry run python scripts/eval_retrieval.py                  # data/eval/eval_set.md 로 평가
poetry run python scripts/eval_retrieval.py --ks 1 3 5 10 20  # 점수를 볼 k 지정
poetry run python scripts/eval_retrieval.py --mode hybrid      # 검색 방식 지정 (기본: RETRIEVER_MODE)
poetry run python scripts/eval_retrieval.py --rewrite          # 검색어 다듬기 (질문이 외부 API 로 전송, 결과 CSV 에 검색어 열)
poetry run python scripts/eval_retrieval.py --rewrite --merge-original                      # 원래 질문 검색과 합치기
poetry run python scripts/eval_retrieval.py --rewrite-from data/eval/results/<이전 결과>.csv   # 이전 실행의 검색어 재사용 (API 호출 없음)
poetry run python scripts/eval_retrieval.py --rewrite-from <이전 결과>.csv --merge-original --rerank-candidates 5 --rerank-query rewrite   # 챗봇과 같은 검색
```

- 다듬은 검색어는 실행마다 조금씩 달라진다. 검색 방식만 바꿔 비교할 때는 `--rewrite-from`으로 이전 결과 CSV의 검색어를 재사용해 같은 검색어끼리 비교한다.

- 평가셋은 마크다운 표(`| id | 유형 | 질문 | 정답 section_id | 기대 답 |`)로 쓴다. 정답이 여러 개면 쉼표로 구분하고, 규정에 없는 질문은 정답 칸을 비운다.
- 양식과 작성 가이드는 `docs/2026-10-29/eval_set_template.md`에 있다. 처음 한 번 `data/eval/eval_set.md`로 복사해서 6장의 표에 질문을 추가한다.
- 기본은 LLM을 쓰지 않아 외부 전송이 없다(`--rewrite`면 질문만 전송). 화면에는 Hit@k·Recall@k·MRR 점수와 질문 id만 나온다.
- 질문별 검색 결과는 `data/eval/results/retrieval-<평가셋>-<검색방식>-<시각>.csv`로 남는다 (규정명 포함, 공유 금지). 보류셋 결과(`retrieval-holdout_set-…`)는 튜닝 중에 열지 않는다.
- 1등 유사도는 검색 방식과 관계없이 벡터 검색 1등의 코사인 유사도다.
- `--rerank-query`의 기본값은 `original`(원래 질문)이다. 실험 비교용으로 남겨 두었고, 챗봇은 `rewrite`와 같다.

## 답변 평가

```bash
poetry run python scripts/eval_answers.py --rewrite-from data/eval/results/<이전 결과>.csv --limit 2   # 2문항만 시험
poetry run python scripts/eval_answers.py --rewrite-from <이전 결과>.csv --efforts default medium low   # effort 비교
poetry run python scripts/eval_answers.py --rewrite-from <이전 결과>.csv --prompts current narrow       # 프롬프트 비교
poetry run python scripts/eval_answers.py --rewrite-from <이전 결과>.csv --no-judge                     # 로컬 채점만
```

- 챗봇과 같은 과정(검색 → 리랭크 → 답변)으로 평가셋 답변을 만들고 세 층으로 채점한다: 규칙(숫자·거절·근거 인용, 로컬), 유사도(리랭커, 로컬), LLM 판정(맞음/부분/틀림, 외부 API).
- 외부 전송: 답변에 질문 + 발췌, 판정에 질문 + 기대 답 + 답변(발췌 없음). `--rewrite-from`이 없으면 질문을 다듬기 LLM으로도 보낸다.
- 화면에는 점수·시간·토큰과 질문 id만 나온다. 질문별 답변과 판정 이유는 `data/eval/results/`의 CSV로 남는다 (공유 금지).
- 지금까지의 설정 결정은 Claude(Sonnet 답변, Haiku 다듬기) 기준 평가로 내렸다. 실사용 LLM을 OpenAI로 바꾼 뒤 재평가는 하지 않기로 했다(11-04).

## 중복 문서 점검

```bash
poetry run python scripts/compare_docs.py               # 본문이 많이 겹치는 문서 쌍 찾기
poetry run python scripts/compare_docs.py --pair 5 29   # 두 문서를 조 단위로 비교
```

- `chunks.jsonl`을 읽어 비교하므로 `build_chunks.py`를 먼저 실행한다. 번호는 `build_chunks.py`와 같다.
- 남기지 않을 파일은 `data/raw` 밖으로 옮기고 `build_chunks.py`를 다시 실행한다. 파일을 빼면 뒤 번호가 하나씩 당겨진다.

## 폴더 구조

```
regulation-chatbot/
├─ src/regbot/
│   ├─ config.py      # .env 설정 로드
│   ├─ models.py      # get_embeddings(), get_reranker(), get_llm(), llm_name()
│   ├─ loaders.py     # PDF·DOCX 텍스트 추출
│   ├─ splitter.py    # 조문·별표·부칙 분할, 메타데이터, chunkId
│   ├─ chunks.py      # 청크 형식, chunks.jsonl 읽기·쓰기
│   ├─ vectorstore.py # Chroma(cosine) 저장소 열기, 청크 → Document
│   ├─ retriever.py   # 검색 (vector | hybrid: Kiwi BM25 + RRF), 여러 검색어 합치기, 리랭크
│   ├─ rag.py         # 답변 체인·검색어 다듬기 체인 (prompt | llm | parser)
│   └─ report.py      # 스크립트 공용 표 출력
├─ scripts/
│   ├─ check_setup.py # 환경 점검
│   ├─ inspect_docs.py# 규정 파일 추출 점검 (집계 수치만 출력)
│   ├─ build_chunks.py# data/raw → data/chunks/chunks.jsonl
│   ├─ compare_docs.py# 문서 간 중복 점검 (집계 수치만 출력)
│   ├─ build_index.py # chunks.jsonl → Chroma
│   ├─ search.py      # 검색 확인, --answer 로 LLM 답변 생성
│   ├─ chat.py        # Gradio 채팅 화면 (127.0.0.1 전용)
│   ├─ eval_retrieval.py # 검색 평가 (Hit@k·Recall@k·MRR, 점수만 출력)
│   ├─ eval_answers.py   # 답변 평가 (규칙·유사도·LLM 판정, 점수만 출력)
│   ├─ diagnose_multi.py # 여러 조항 진단 (놓친 정답의 원인 집계, API 없음, 숫자·id만 출력)
│   └─ compare_candidates.py # 리랭크 후보 모으기 비교 (max·quota, API 없음)
├─ data/
│   ├─ sample/        # 가상 규정 샘플 PDF (git 포함, 초기 개발용)
│   ├─ raw/           # 실제 규정 PDF·DOCX, 카테고리별 폴더 (git 제외)
│   ├─ excluded/      # 중복 정리로 색인에서 뺀 규정 파일 (git 제외)
│   ├─ chunks/        # 분할 결과 chunks.jsonl (git 제외, 규정 원문 포함)
│   ├─ eval/          # 평가셋 eval_set.md·평가 결과 (git 제외, 규정 내용 포함)
│   └─ chroma/        # 벡터 인덱스 (git 제외, 자동 생성)
├─ docs/              # 날짜별 작업 기록 (docs/README.md 참고)
├─ .env.example
└─ pyproject.toml
```

## 데이터 취급 원칙
- 실제 규정은 정보보안 승인(2026-10-29) 뒤 개발·평가에 쓴다. 외부 LLM에는 위 두 가지(질문, 질문 + 발췌)만 보낸다. 수신처(LLM 회사)나 보내는 항목이 바뀌면 승인 범위를 다시 확인한다.
- 실제 규정 PDF는 `data/raw/`에만 두고 저장소에 올리지 않는다.
- 규정 내용이 들어간 산출물(청크, 인덱스, 평가셋, 평가 결과 CSV)은 git에서 제외된 `data/` 아래에만 둔다.
- 스크립트 화면 출력은 숫자·id만 기본으로 하고, 규정명·본문은 `--show-names`·`--show-text`를 붙였을 때만 보여 준다 (본인 확인용).
- LangSmith 추적은 쓰지 않는다 (켜져 있으면 LLM 호출 스크립트가 멈춘다).
- `data/sample/`의 가상 규정 샘플은 초기 개발용으로 남겨 둔다.
