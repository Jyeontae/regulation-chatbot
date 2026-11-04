"""규정 파일 추출 점검: 파일별 페이지·글자 수, 조문·별표·부칙 패턴 개수 등 집계 수치만 출력한다.

splitter 를 만들기 전에, 텍스트가 제대로 추출되는지(스캔본·띄어쓰기 손실)와
조문 표기가 어떤 형식인지 확인하는 용도. 규정 본문과 파일 이름은 출력하지 않는다.

실행:
    poetry run python scripts/inspect_docs.py
    poetry run python scripts/inspect_docs.py --show-names   # 표와 [주의]에 파일 이름 표시 (본인 확인용, 공유 금지)
"""

import argparse
import re
from collections import Counter
from pathlib import Path

from regbot.config import settings
from regbot.loaders import LoadedDoc, find_files, load_file
from regbot.report import display_path, file_label, format_cell, print_table

# 글자 수가 이보다 적은 PDF 페이지는 빈 페이지로 본다 (스캔본이면 텍스트가 거의 나오지 않는다)
EMPTY_PAGE_CHARS = 20
# 공백 비율이 이보다 낮으면 띄어쓰기 손실을 의심한다 (한국어 문서는 보통 0.15 이상)
MIN_SPACE_RATIO = 0.08

# "제 12 조"처럼 띄어 쓴 표기도 잡도록 글자 사이에 \s* 를 둔다
PATTERNS = {
    "장": re.compile(r"제\s*\d+\s*장"),
    # 줄 맨 앞의 "제12조(목적)", "제12조의2(...)" = 조문 제목
    "조제목": re.compile(r"^\s*제\s*\d+\s*조(?:\s*의\s*\d+)?\s*[(（]", re.MULTILINE),
    # 본문 속 "제12조에 따른다" 같은 참조까지 포함
    "조언급": re.compile(r"제\s*\d+\s*조"),
    "항": re.compile(r"[①-⑳]"),
    "별표": re.compile(r"별\s*표"),
    "별지": re.compile(r"별\s*지"),
    "별첨": re.compile(r"별\s*첨"),
    "부칙": re.compile(r"^\s*부\s*칙", re.MULTILINE),
}
COLUMNS = ["번호", "카테고리", "형식", "쪽/문단", "표", "글자수", "빈쪽", "공백비율", *PATTERNS]
LEFT_ALIGNED = 3  # 앞의 3개 열(번호·카테고리·형식)만 왼쪽 정렬


def inspect(doc: LoadedDoc) -> dict:
    text = doc.text
    body = text.replace("\n", "")
    is_pdf = doc.file_type == "pdf"
    stats = {
        "쪽/문단": len(doc.pages) if is_pdf else doc.paragraph_count,
        "표": doc.table_count,
        "글자수": sum(not ch.isspace() for ch in text),
        "빈쪽": sum(len("".join(page.split())) < EMPTY_PAGE_CHARS for page in doc.pages) if is_pdf else None,
        "공백비율": sum(ch.isspace() for ch in body) / len(body) if body else 0.0,
    }
    for name, pattern in PATTERNS.items():
        stats[name] = len(pattern.findall(text))
    return stats


def warnings_for(label: str, stats: dict) -> list[str]:
    if stats["글자수"] == 0:
        return [f"{label} 텍스트 없음 → 스캔본(이미지) 의심, OCR 필요"]
    warnings = []
    if stats["빈쪽"]:
        warnings.append(
            f"{label} 빈 페이지 {stats['빈쪽']}/{stats['쪽/문단']}쪽 → 표지·백지가 아니면 스캔 페이지 의심"
        )
    if stats["공백비율"] < MIN_SPACE_RATIO:
        warnings.append(f"{label} 공백 비율 {stats['공백비율']:.2f} → 띄어쓰기가 사라졌을 수 있음")
    if stats["조제목"] == 0:
        warnings.append(f"{label} 줄 맨 앞 '제N조(' 0개 → 조문 표기가 다르거나 줄바꿈이 사라짐")
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=settings.docs_dir, help="점검할 폴더 (기본: .env 의 DOCS_DIR)")
    parser.add_argument("--show-names", action="store_true", help="파일 이름 표시 (본인 확인용, 공유 금지)")
    args = parser.parse_args()
    root: Path = args.root.resolve()

    targets, skipped = find_files(root)
    skipped_by_suffix = Counter(path.suffix.lower() or "(확장자 없음)" for path in skipped)
    print(f"[점검 대상] {display_path(root)}")
    print(f"  처리 대상: {len(targets)}개 / 건너뜀: {len(skipped)}개", end="")
    print(f" ({', '.join(f'{s} {n}' for s, n in sorted(skipped_by_suffix.items()))})" if skipped else "")
    if not targets:
        return

    rows: list[list[str]] = []
    names: list[str] = []
    warnings: list[str] = []
    totals: Counter = Counter()
    categories: Counter = Counter()
    for index, path in enumerate(targets, start=1):
        no = f"#{index:02d}"
        # [주의] 항목에도 --show-names 일 때만 파일 이름을 붙인다
        label = file_label(no, path, root, args.show_names)
        try:
            doc = load_file(path, root)
        except ModuleNotFoundError:
            warnings.append(f"{label} python-docx 가 설치되지 않음 → poetry add python-docx")
            continue
        except Exception as e:  # 점검용이므로 한 파일이 실패해도 계속 진행. 메시지에 경로가 섞일 수 있어 종류만 출력
            warnings.append(f"{label} 읽기 실패: {type(e).__name__}")
            continue

        stats = inspect(doc)
        rows.append([no, doc.category, doc.file_type, *(format_cell(stats[c]) for c in COLUMNS[3:])])
        names.append(doc.title)
        warnings.extend(warnings_for(label, stats))
        categories[(doc.category, doc.file_type)] += 1
        totals.update({k: v for k, v in stats.items() if isinstance(v, int)})

    print("\n[파일별]")
    if rows:
        total_row = ["합계", "", "", *(format_cell(totals[c]) if c != "공백비율" else "" for c in COLUMNS[3:])]
        print_table(COLUMNS, [*rows, total_row], LEFT_ALIGNED, names if args.show_names else None)

    print("\n[카테고리별 파일 수]")
    for (category, file_type), count in sorted(categories.items()):
        print(f"  {category} / {file_type}: {count}")

    print("\n[주의]")
    for warning in warnings or ["없음"]:
        print(f"  {warning}")

    print("\n[열 설명]")
    print("  쪽/문단 : PDF 는 페이지 수, DOCX 는 비어 있지 않은 문단 수")
    print("  표      : DOCX 의 표 개수 (PDF 는 집계하지 않음)")
    print(f"  빈쪽    : 공백을 뺀 글자가 {EMPTY_PAGE_CHARS}자 미만인 PDF 페이지 수")
    print("  조제목  : 줄 맨 앞의 '제N조(' / 조언급: 본문 속 참조를 포함한 '제N조' 전체")
    print("  항      : ①~⑳ 기호 수 / 부칙: 줄 맨 앞의 '부칙'")
    if args.show_names:
        print("\n※ 파일 이름이 포함된 출력입니다. 공유할 때는 --show-names 없이 다시 실행하세요.")


if __name__ == "__main__":
    main()
