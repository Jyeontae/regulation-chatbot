"""청크 생성: data/raw 규정 파일을 로딩·분할해 data/chunks/chunks.jsonl 로 저장한다.

화면에는 집계 수치만 출력한다 (규정 본문·파일 이름 없음). 분할 결과는 chunks.jsonl 을 직접 열어 확인한다.

실행:
    poetry run python scripts/build_chunks.py
    poetry run python scripts/build_chunks.py --show-names   # 표와 [주의]에 파일 이름 표시 (본인 확인용, 공유 금지)
"""

import argparse
import statistics
import sys
import traceback
from collections import Counter
from pathlib import Path

from regbot.chunks import Chunk, save_chunks
from regbot.config import settings
from regbot.loaders import find_files, load_file
from regbot.report import display_path, file_label, format_cell, print_table
from regbot.splitter import MAX_CHARS, SplitStats, split_document

# [주의]에 올리는 기준
PREAMBLE_WARN_CHARS = 300  # 첫 조문 이전에 버린 글자가 이보다 많으면 표지·목차가 아닌 내용일 수 있음
SUPPLEMENT_WARN_ARTICLES = 5  # 부칙 하나에 조가 이보다 많으면 다른 규정의 본문이 부칙으로 들어갔을 수 있음
MIN_KEPT_RATIO = 0.7  # 원문 대비 남은 글자 비율이 이보다 낮으면 많이 버려진 것
SHORT_CHUNK_CHARS = 50

TYPES = ["본문", "부칙", "별표", "별지", "별첨", "fallback"]
COUNT_COLUMNS = ["번호", "카테고리", "형식", "청크", *TYPES, "구간"]
# (열 이름, SplitStats 필드)
CLEAN_COLUMNS = [
    ("앞부분(자)", "preamble_chars"),
    ("머리글·바닥글", "header_footer_lines"),
    ("삭제조", "deleted_articles"),
    ("빈제목", "empty_blocks"),
    ("제목아닌조줄", "non_heading_article_lines"),
    ("역순", "out_of_order"),
    ("id중복", "duplicate_ids"),
]


def warnings_for(label: str, stats: SplitStats, chunk_count: int) -> list[str]:
    if chunk_count == 0:
        return [f"{label} 청크 0개 → 추출된 텍스트가 없음"]
    warnings = []
    if stats.fallback:
        warnings.append(f"{label} 조 제목이 2개 미만 → 길이 기준(fallback)으로 나눔")
    if stats.preamble_chars > PREAMBLE_WARN_CHARS:
        warnings.append(f"{label} 첫 조문 이전 {stats.preamble_chars:,}자를 버림 → 표지·목차가 아닌 내용이면 확인 필요")
    if stats.segments > 1:
        warnings.append(f"{label} 제1조가 {stats.segments}번 시작됨 → 여러 규정을 합친 문서인지 확인")
    if stats.out_of_order:
        warnings.append(f"{label} 번호가 앞 조보다 작은 조 제목 {stats.out_of_order}개 → 본문 속 참조를 제목으로 잡았을 수 있음")
    if stats.max_supplement_articles > SUPPLEMENT_WARN_ARTICLES:
        warnings.append(
            f"{label} 부칙 하나에 조 {stats.max_supplement_articles}개 → 다른 규정 본문이 부칙으로 들어갔을 수 있음"
        )
    if stats.duplicate_ids:
        warnings.append(f"{label} 같은 조·별표 번호가 다시 나와 chunkId {stats.duplicate_ids}개에 ~2 등을 붙임")
    # 짧은 문서는 표지·목차만 빠져도 비율이 크게 떨어지므로, 버린 글자 수가 많을 때만 알린다
    dropped = stats.total_chars - stats.kept_chars
    if dropped > PREAMBLE_WARN_CHARS and stats.kept_chars / stats.total_chars < MIN_KEPT_RATIO:
        warnings.append(f"{label} 원문 대비 남은 글자 {stats.kept_chars / stats.total_chars:.0%} → 많이 버려짐")
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=settings.docs_dir, help="규정 폴더 (기본: .env 의 DOCS_DIR)")
    parser.add_argument("--out", type=Path, default=settings.chunks_path, help="저장 파일 (기본: .env 의 CHUNKS_PATH)")
    parser.add_argument("--show-names", action="store_true", help="파일 이름 표시 (본인 확인용, 공유 금지)")
    args = parser.parse_args()
    root: Path = args.root.resolve()

    targets, skipped = find_files(root)
    print(f"[대상] {display_path(root)}")
    print(f"  처리 대상: {len(targets)}개 / 건너뜀: {len(skipped)}개")
    if not targets:
        return

    all_chunks: list[Chunk] = []
    all_stats: list[SplitStats] = []
    count_rows: list[list[str]] = []
    clean_rows: list[list[str]] = []
    names: list[str] = []
    warnings: list[str] = []
    for index, path in enumerate(targets, start=1):
        no = f"#{index:02d}"
        label = file_label(no, path, root, args.show_names)
        try:
            doc = load_file(path, root)
            chunks, stats = split_document(doc)
        except ModuleNotFoundError as e:
            warnings.append(f"{label} 패키지 없음: {e.name}")
            continue
        except Exception as e:  # 한 파일이 실패해도 계속. 메시지에 규정 내용이 섞일 수 있어 종류와 코드 위치만 출력
            frame = traceback.extract_tb(e.__traceback__)[-1]
            warnings.append(f"{label} 처리 실패: {type(e).__name__} ({Path(frame.filename).name}:{frame.lineno})")
            continue

        types = Counter(str(chunk.metadata["type"]) for chunk in chunks)
        count_rows.append(
            [no, doc.category, doc.file_type, *map(format_cell, [len(chunks), *(types[t] for t in TYPES), stats.segments])]
        )
        kept_ratio = stats.kept_chars / stats.total_chars if stats.total_chars else 0.0
        clean_rows.append([no, *(format_cell(getattr(stats, field)) for _, field in CLEAN_COLUMNS), f"{kept_ratio:.0%}"])
        names.append(doc.title)
        warnings.extend(warnings_for(label, stats, len(chunks)))
        all_chunks.extend(chunks)
        all_stats.append(stats)

    shown_names = names if args.show_names else None
    type_totals = Counter(str(chunk.metadata["type"]) for chunk in all_chunks)
    print("\n[파일별 청크 수]")
    total_row = ["합계", "", "", *map(format_cell, [len(all_chunks), *(type_totals[t] for t in TYPES)]), ""]
    print_table(COUNT_COLUMNS, [*count_rows, total_row], 3, shown_names)

    print("\n[파일별 정리 내역]")
    total_chars = sum(stats.total_chars for stats in all_stats)
    kept_chars = sum(stats.kept_chars for stats in all_stats)
    total_row = [
        "합계",
        *(format_cell(sum(getattr(stats, field) for stats in all_stats)) for _, field in CLEAN_COLUMNS),
        f"{kept_chars / total_chars:.0%}" if total_chars else "",
    ]
    print_table(["번호", *(name for name, _ in CLEAN_COLUMNS), "남은글자"], [*clean_rows, total_row], 1, shown_names)

    if all_chunks:
        # 머리말(첫 줄)을 뺀 본문 길이
        lengths = sorted(len(chunk.text.split("\n", 1)[1]) for chunk in all_chunks)
        print("\n[청크 본문 길이] (머리말 제외, 글자 수)")
        print(
            f"  최소 {lengths[0]:,} / 중앙값 {statistics.median(lengths):,.0f}"
            f" / 상위 10% {lengths[int(len(lengths) * 0.9)]:,} / 최대 {lengths[-1]:,}"
        )
        print(
            f"  {SHORT_CHUNK_CHARS}자 미만: {sum(n < SHORT_CHUNK_CHARS for n in lengths):,}개"
            f" / {MAX_CHARS:,}자 초과: {sum(n > MAX_CHARS for n in lengths):,}개"
        )

    print("\n[주의]")
    for warning in warnings or ["없음"]:
        print(f"  {warning}")

    print("\n[열 설명]")
    print("  구간         : 문서 안에서 제1조가 시작된 횟수 (2 이상이면 여러 규정을 합친 문서)")
    print("  앞부분(자)   : 첫 조문 이전이라 버린 글자 수 (표지·목차)")
    print("  머리글·바닥글: 여러 쪽의 같은 자리에 반복돼 지운 줄 수 (쪽 번호 포함)")
    print("  삭제조       : 지운 '제N조 삭제' 줄 / 빈제목: 내용이 없어 버린 제목 (조문이 이어지는 부칙 제목 줄 등)")
    print("  제목아닌조줄 : 줄 맨 앞이 '제N조'지만 조 제목으로 보지 않은 줄 (본문 속 참조 등)")
    print("  역순         : 번호가 앞 조보다 작은 조 제목 / id중복: 같은 chunkId 가 다시 나와 ~2 등을 붙인 수")
    print("  남은글자     : 원문 대비 청크에 남은 글자 비율 (공백 제외)")

    # 다른 파일끼리 chunkId 가 겹치면 Chroma 에 넣을 수 없다 (같은 폴더에 확장자만 다른 같은 이름의 파일)
    duplicates = [chunk_id for chunk_id, count in Counter(chunk.chunk_id for chunk in all_chunks).items() if count > 1]
    if duplicates:
        print(f"\n[중단] 서로 다른 파일의 chunkId {len(duplicates)}개가 겹침 → 같은 폴더에 이름이 같은 파일이 있는지 확인")
        sys.exit(1)

    save_chunks(all_chunks, args.out)
    print(f"\n[저장] {display_path(args.out)} ({len(all_chunks):,}개)")
    if args.show_names:
        print("\n※ 파일 이름이 포함된 출력입니다. 공유할 때는 --show-names 없이 다시 실행하세요.")


if __name__ == "__main__":
    main()
