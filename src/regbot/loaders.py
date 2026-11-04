"""규정 파일 로딩: PDF·DOCX에서 텍스트를 추출해 공통 형식으로 돌려준다."""

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = (".pdf", ".docx")


class EncryptedPdfError(ValueError):
    """암호가 걸려 텍스트를 읽을 수 없는 PDF."""


@dataclass
class LoadedDoc:
    path: Path
    category: str  # docs_dir 바로 아래 폴더 이름 (사규, 지침 ...)
    title: str  # 확장자를 뺀 파일 이름 = 규정명
    file_type: str  # "pdf" | "docx"
    # PDF 는 페이지별 텍스트. DOCX 는 페이지 정보가 없어 문서 전체가 원소 1개
    pages: list[str]
    paragraph_count: int = 0  # DOCX 문단 수 (PDF 는 0)
    table_count: int = 0  # DOCX 표 수 (PDF 는 0)

    @property
    def text(self) -> str:
        return "\n".join(self.pages)


def find_files(root: Path) -> tuple[list[Path], list[Path]]:
    """root 아래 파일을 (처리 대상, 건너뛸 파일)로 나눈다. 숨김·임시 파일은 둘 다에서 뺀다."""
    targets: list[Path] = []
    skipped: list[Path] = []
    for path in sorted(root.rglob("*")):
        # .gitkeep 같은 숨김 파일, Word 가 열려 있을 때 생기는 ~$ 잠금 파일
        if not path.is_file() or path.name.startswith((".", "~$")):
            continue
        (targets if path.suffix.lower() in SUPPORTED_SUFFIXES else skipped).append(path)
    return targets, skipped


def category_of(path: Path, root: Path) -> str:
    parts = path.relative_to(root).parts
    return parts[0] if len(parts) > 1 else "(루트)"


def load_file(path: Path, root: Path) -> LoadedDoc:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path, root)
    if suffix == ".docx":
        return _load_docx(path, root)
    raise ValueError(f"지원하지 않는 형식: {suffix}")


def _load_pdf(path: Path, root: Path) -> LoadedDoc:
    import pymupdf

    with pymupdf.open(path) as doc:
        if doc.needs_pass:
            raise EncryptedPdfError(path.name)
        # sort=True: 위→아래, 왼쪽→오른쪽 순서로 읽는다 (머리글·바닥글이 쪽의 맨 위·아래에 오도록)
        pages = [page.get_text("text", sort=True) for page in doc]
    return LoadedDoc(path, category_of(path, root), path.stem, "pdf", pages)


def _load_docx(path: Path, root: Path) -> LoadedDoc:
    from docx import Document
    from docx.table import Table

    lines: list[str] = []
    paragraph_count = table_count = 0
    # 문단과 표를 문서에 나온 순서대로 읽는다 (별표가 표로 들어 있는 경우가 많음)
    for block in Document(str(path)).iter_inner_content():
        if isinstance(block, Table):
            table_count += 1
            lines.extend(_table_lines(block))
        elif block.text.strip():
            paragraph_count += 1
            lines.append(block.text)
    return LoadedDoc(
        path, category_of(path, root), path.stem, "docx", ["\n".join(lines)], paragraph_count, table_count
    )


def _table_lines(table) -> list[str]:
    """표 한 행을 '셀 | 셀 | 셀' 한 줄로 만든다."""
    lines = []
    for row in table.rows:
        cells: list[str] = []
        for cell in row.cells:
            text = cell.text.strip()
            # 병합된 셀은 python-docx 가 같은 값을 여러 번 돌려주므로 연속 중복을 뺀다
            if not cells or cells[-1] != text:
                cells.append(text)
        if any(cells):
            lines.append(" | ".join(cells))
    return lines
