"""문서 중복 점검: chunks.jsonl 의 문서끼리 본문이 얼마나 겹치는지 집계 수치로 비교한다.

같은 규정이 두 카테고리에 들어 있으면 검색 결과에 같은 조문이 두 번 나와 RETRIEVER_K 자리를 낭비한다.
어느 파일을 남길지 판단할 근거를 숫자로 보여 준다. 규정 본문과 부칙 날짜는 출력하지 않는다.
번호(#01~)는 build_chunks.py 와 같은 순서로 매긴다.

실행:
    poetry run python scripts/compare_docs.py                 # 전체 문서 쌍 중 많이 겹치는 쌍 찾기
    poetry run python scripts/compare_docs.py --pair 5 29     # 두 문서를 조 단위로 자세히 비교
    poetry run python scripts/compare_docs.py --pair 5 29 --ids   # 두 문서의 chunkId 목록 (# 뒤 위치만)
    poetry run python scripts/compare_docs.py --show-names    # 파일 이름 표시 (본인 확인용, 공유 금지)
"""

import argparse
import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path

from regbot.chunks import load_chunks
from regbot.config import settings
from regbot.loaders import category_of, find_files
from regbot.report import display_path, format_cell, print_table

SHINGLE = 5  # 겹침을 셀 때 쓰는 글자 조각 길이
MIN_OVERLAP = 0.5  # 전체 비교에서 표에 올리는 겹침 기준
HIGH, MID = 0.95, 0.8  # 조 단위 대응 구간 (유사도)
TYPES = ["본문", "부칙", "별표", "별지", "별첨", "fallback"]
DATE = re.compile(r"(\d{4})\s*\.\s*(\d{1,2})\s*\.\s*(\d{1,2})")


@dataclass
class Doc:
    no: str
    category: str
    file_type: str
    title: str
    # 문서 안 위치(section_id 의 # 뒤) → 정규화한 본문. 길어서 나뉜 조각은 이어 붙인다
    sections: dict[str, str] = field(default_factory=dict)
    types: Counter = field(default_factory=Counter)
    chunk_ids: list[str] = field(default_factory=list)
    supplements: set[str] = field(default_factory=set)  # 부칙 제목 줄 (날짜 비교용, 출력하지 않음)

    @property
    def text(self) -> str:
        return "".join(self.sections.values())

    @property
    def latest_supplement(self) -> tuple[int, ...] | None:
        dates = [tuple(map(int, m)) for line in self.supplements for m in DATE.findall(line)]
        return max(dates, default=None)


def normalize(text: str) -> str:
    """머리말(첫 줄)을 빼고 글자·숫자만 남긴다. PDF 줄바꿈과 DOCX 표 구분자(|) 차이를 없앤다."""
    body = text.split("\n", 1)[1] if "\n" in text else ""
    return "".join(ch for ch in body if ch.isalnum())


def load_docs(chunks_path: Path, root: Path) -> list[Doc]:
    # 파일 목록(이름)만 읽어 번호를 매긴다. 규정 파일 내용은 열지 않는다
    targets, _ = find_files(root)
    numbers = {f"{category_of(path, root)}/{path.name}": f"#{i:02d}" for i, path in enumerate(targets, start=1)}

    docs: dict[str, Doc] = {}
    for chunk in load_chunks(chunks_path):
        meta = chunk.metadata
        source = str(meta["source"])
        if source not in docs:
            docs[source] = Doc(numbers.get(source, "#??"), str(meta["category"]), str(meta["file_type"]), str(meta["title"]))
        doc = docs[source]
        key = str(meta["section_id"]).rsplit("#", 1)[-1]
        doc.sections[key] = doc.sections.get(key, "") + normalize(chunk.text)
        doc.types[str(meta["type"])] += 1
        doc.chunk_ids.append(chunk.chunk_id)
        if meta["supplement"]:
            doc.supplements.add(str(meta["supplement"]))
    return sorted(docs.values(), key=lambda doc: doc.no)


def shingles(text: str) -> set[str]:
    return {text[i : i + SHINGLE] for i in range(len(text) - SHINGLE + 1)}


def overlap(a: set[str], b: set[str]) -> tuple[float, float]:
    """(작은 문서 기준 겹침, 자카드). 겹침이 높으면 작은 문서가 큰 문서 안에 거의 들어 있다."""
    if not a or not b:
        return 0.0, 0.0
    common = len(a & b)
    return common / min(len(a), len(b)), common / len(a | b)


def best_match(text: str, candidates: dict[str, str]) -> tuple[str, float]:
    """candidates 중 text 와 가장 비슷한 (위치, 유사도)."""
    best_key, best = "", 0.0
    for key, other in candidates.items():
        if text == other:
            return key, 1.0
        matcher = SequenceMatcher(None, text, other, autojunk=False)
        # quick_ratio 는 ratio 의 상한이라, 지금 최고점을 못 넘으면 비싼 ratio 계산을 건너뛴다
        if matcher.real_quick_ratio() <= best or matcher.quick_ratio() <= best:
            continue
        ratio = matcher.ratio()
        if ratio > best:
            best_key, best = key, ratio
    return best_key, best


def match_row(src: Doc, dst: Doc) -> list[str]:
    buckets = Counter()
    same_position = 0
    sections = {key: text for key, text in src.sections.items() if text}
    for key, text in sections.items():
        matched_key, ratio = best_match(text, dst.sections)
        if ratio == 1.0:
            buckets["exact"] += 1
        elif ratio >= HIGH:
            buckets["high"] += 1
        elif ratio >= MID:
            buckets["mid"] += 1
        else:
            buckets["none"] += 1
        if ratio >= MID and matched_key == key:
            same_position += 1
    counts = [len(sections), buckets["exact"], buckets["high"], buckets["mid"], buckets["none"], same_position]
    return [f"{src.no} → {dst.no}", *map(format_cell, counts)]


def supplement_message(a: Doc, b: Doc) -> str:
    latest_a, latest_b = a.latest_supplement, b.latest_supplement
    if latest_a is None and latest_b is None:
        return "두 문서 모두 부칙 날짜를 찾지 못함 → 파일을 열어 개정일 확인"
    if latest_b is None:
        return f"{a.no}에만 부칙 날짜가 있음"
    if latest_a is None:
        return f"{b.no}에만 부칙 날짜가 있음"
    if latest_a == latest_b:
        return "최근 부칙 날짜가 같음"
    newer = a if latest_a > latest_b else b
    return f"{newer.no}의 최근 부칙이 더 늦음 → 더 최근 개정본일 가능성"


def scan(docs: list[Doc], min_overlap: float, show_names: bool) -> None:
    sets = [shingles(doc.text) for doc in docs]
    found: list[tuple[float, list[str], str]] = []
    for (i, a), (j, b) in combinations(enumerate(docs), 2):
        contain, jaccard = overlap(sets[i], sets[j])
        if contain >= min_overlap:
            row = [
                a.no,
                b.no,
                f"{a.category} / {b.category}",
                f"{a.file_type} / {b.file_type}",
                f"{len(a.text):,} / {len(b.text):,}",
                f"{contain:.0%}",
                f"{jaccard:.0%}",
            ]
            found.append((contain, row, f"{a.title} / {b.title}"))
    found.sort(key=lambda item: item[0], reverse=True)

    pair_count = len(docs) * (len(docs) - 1) // 2
    print(f"\n[겹치는 문서 쌍] 문서 {len(docs)}개, 비교 {pair_count:,}쌍 중 겹침 {min_overlap:.0%} 이상 {len(found)}쌍")
    if not found:
        return
    columns = ["문서A", "문서B", "카테고리", "형식", "글자수", "겹침", "자카드"]
    print_table(columns, [row for _, row, _ in found], 4, [name for _, _, name in found] if show_names else None)
    print("\n[열 설명]")
    print(f"  글자수 : 머리말·공백·기호를 뺀 본문 글자 수 / 겹침·자카드: 본문을 {SHINGLE}글자 조각으로 나눠 센 비율")
    print("  겹침   : 작은 문서의 조각 중 큰 문서에도 있는 비율 (작은 문서가 큰 문서 안에 들어 있으면 100%에 가까움)")
    print("  자카드 : 두 문서 조각 전체 중 공통 조각 비율 (두 문서가 통째로 같으면 100%에 가까움)")
    print("  자세히 비교: --pair 번호A 번호B")


def compare_pair(a: Doc, b: Doc, show_names: bool) -> None:
    print("\n[문서 요약]")
    rows = [
        [doc.no, doc.category, doc.file_type, *map(format_cell, [sum(doc.types.values()), *(doc.types[t] for t in TYPES)]),
         format_cell(len(doc.sections)), format_cell(len(doc.text)), format_cell(len(doc.supplements))]
        for doc in (a, b)
    ]
    columns = ["번호", "카테고리", "형식", "청크", *TYPES, "조·별표", "글자수", "부칙제목"]
    print_table(columns, rows, 3, [a.title, b.title] if show_names else None)

    contain, jaccard = overlap(shingles(a.text), shingles(b.text))
    print(f"\n[전체 겹침] 작은 문서 기준 {contain:.0%} / 자카드 {jaccard:.0%}")

    print("\n[조 단위 대응] 왼쪽 문서의 조·별표마다 오른쪽 문서에서 가장 비슷한 것을 찾은 결과")
    columns = ["방향", "조·별표", "완전일치", f"{HIGH:.0%}이상", f"{MID:.0%}이상", "대응없음", "같은위치"]
    print_table(columns, [match_row(a, b), match_row(b, a)], 1)

    print(f"\n[부칙] {supplement_message(a, b)}")

    print("\n[열 설명]")
    print("  조·별표  : section_id 단위 개수 (길어서 나뉜 조각은 하나로 합쳐 비교)")
    print(f"  완전일치 : 머리말·공백·기호를 뺀 본문이 글자까지 같음 / {HIGH:.0%}이상·{MID:.0%}이상: 문구가 조금 다름")
    print(f"  대응없음 : 가장 비슷한 것도 {MID:.0%} 미만 (한쪽에만 있는 조)")
    print(f"  같은위치 : {MID:.0%} 이상으로 대응한 것 중 조 번호(위치)까지 같은 수")
    print("\n[판단 기준]")
    print("  양방향 모두 완전일치가 대부분 → 같은 규정. 하나만 남긴다")
    print(f"  {HIGH:.0%}이상·{MID:.0%}이상이 많음 → 개정 전후 판본일 수 있음. [부칙]과 파일을 확인해 최신본을 남긴다")
    print("  한쪽 방향만 대응없음이 많음 → 한 문서가 다른 문서를 포함 (합본 문서 등)")
    print("  남기지 않을 파일은 data/excluded/ 로 옮기고 build_chunks.py 를 다시 실행한다 (뒤 번호가 하나씩 당겨짐)")


def print_ids(docs: tuple[Doc, ...], show_names: bool) -> None:
    """chunkId 목록. 앞부분(카테고리/규정명)은 show_names 일 때만 보여 준다."""
    for doc in docs:
        print(f"\n[chunkId] {doc.no} ({len(doc.chunk_ids)}개)")
        for chunk_id in doc.chunk_ids:
            print(f"  {chunk_id if show_names else '#' + chunk_id.rsplit('#', 1)[-1]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, default=settings.chunks_path, help="청크 파일 (기본: .env 의 CHUNKS_PATH)")
    parser.add_argument("--root", type=Path, default=settings.docs_dir, help="번호를 매길 규정 폴더 (기본: .env 의 DOCS_DIR)")
    parser.add_argument("--pair", type=int, nargs=2, metavar=("A", "B"), help="자세히 비교할 두 문서 번호 (예: --pair 5 29)")
    parser.add_argument("--min-overlap", type=float, default=MIN_OVERLAP, help=f"전체 비교에서 표시할 겹침 기준 (기본 {MIN_OVERLAP})")
    parser.add_argument("--ids", action="store_true", help="--pair 의 두 문서 chunkId 목록 출력 (# 뒤 위치만)")
    parser.add_argument("--show-names", action="store_true", help="파일 이름 표시 (본인 확인용, 공유 금지)")
    args = parser.parse_args()

    docs = load_docs(args.chunks, args.root.resolve())
    print(f"[대상] {display_path(args.chunks)} (문서 {len(docs)}개)")
    if any(doc.no == "#??" for doc in docs):
        print("  [주의] data/raw 에 없는 문서가 있음 (#??) → 파일을 옮긴 뒤라면 build_chunks.py 를 다시 실행")

    if args.pair:
        by_no = {doc.no: doc for doc in docs}
        a, b = (by_no.get(f"#{n:02d}") for n in args.pair)
        if a is None or b is None or a is b:
            print("  [중단] 두 번호가 같거나 chunks.jsonl 에 없음")
            return
        if args.ids:
            print_ids((a, b), args.show_names)
        else:
            compare_pair(a, b, args.show_names)
    else:
        scan(docs, args.min_overlap, args.show_names)

    if args.show_names:
        print("\n※ 파일 이름이 포함된 출력입니다. 공유할 때는 --show-names 없이 다시 실행하세요.")


if __name__ == "__main__":
    main()
