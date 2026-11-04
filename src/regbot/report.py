"""스크립트 공용 출력 도구: 한글이 섞인 표를 터미널에서 열을 맞춰 출력한다."""

import unicodedata
from pathlib import Path

from regbot.config import PROJECT_ROOT


def display_width(text: str) -> int:
    # 한글은 터미널에서 두 칸을 차지한다
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def format_cell(value: int | float | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return f"{value:,}"


def print_table(columns: list[str], rows: list[list[str]], left_aligned: int, names: list[str] | None = None) -> None:
    """앞의 left_aligned 개 열은 왼쪽, 나머지는 오른쪽 정렬. names 가 있으면 각 행 끝에 붙인다 (헤더·합계 행 제외)."""
    widths = [max(display_width(row[i]) for row in [columns, *rows]) for i in range(len(columns))]
    for index, row in enumerate([columns, *rows]):
        line = "  ".join(_pad(cell, width, i < left_aligned) for i, (cell, width) in enumerate(zip(row, widths)))
        # 이름은 길이가 제각각이라 정렬 없이 맨 끝에 붙인다
        if names and 0 < index <= len(names):
            line += f"  {names[index - 1]}"
        print("  " + line)


def display_path(path: Path) -> str:
    """프로젝트 안의 경로는 상대경로로 보여 준다."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def file_label(no: str, path: Path, root: Path, show_names: bool) -> str:
    """[주의] 항목 앞의 파일 표시. 파일 이름은 show_names 일 때만 붙인다."""
    return f"{no} [{path.relative_to(root).as_posix()}]" if show_names else no


def _pad(text: str, width: int, left: bool) -> str:
    gap = " " * max(width - display_width(text), 0)
    return text + gap if left else gap + text
