"""규정 분할: 조문·별표·부칙 단위로 나누고 위치 머리말, 메타데이터, chunkId를 붙인다.

문서 하나를 다음 순서로 처리한다.
1. 줄 정리: 여러 쪽의 같은 자리에 반복되는 머리글·바닥글(쪽 번호 포함)과 "제N조 삭제" 줄을 지운다
2. 본문 시작 찾기: 내용이 딸린 첫 조 제목부터 본문으로 본다. 그 앞(표지·목차)은 버린다
3. 블록 나누기: 줄 맨 앞의 장·절·조·부칙·별표 제목을 기준으로 자르고, 내용 없는 제목은 버린다
4. 청크 만들기: 긴 블록은 항(①②) 단위로 다시 나누고 위치 머리말·메타데이터·chunkId 를 붙인다
조 제목이 2개 미만인 문서는 길이 기준(fallback)으로 나눈다.
"""

import math
import re
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from itertools import accumulate

from langchain_text_splitters import RecursiveCharacterTextSplitter

from regbot.chunks import Chunk, MetadataValue
from regbot.loaders import LoadedDoc

# 블록이 이보다 길면(글자 수) 항 단위로, 항 하나도 길면 길이 기준으로 다시 나눈다
MAX_CHARS = 1000
# 길이 기준으로 나눌 때 앞 조각과 겹치게 둘 글자 수
OVERLAP_CHARS = 100
# 조 제목 뒤에 이만큼(공백·숫자·기호 제외)의 내용이 있어야 본문의 시작으로 본다. 목차의 조 제목에는 쪽 번호뿐이다
MIN_CONTENT_CHARS = 10
# 머리글·바닥글 후보가 되는 줄의 최대 길이
MAX_HEADER_FOOTER_LEN = 50

# 줄은 앞뒤 공백을 지운 뒤 검사한다. "제 12 조"처럼 띄어 쓴 표기도 잡도록 글자 사이에 \s* 를 둔다
_OPEN, _CLOSE = r"[(（【\[]", r"[)）】\]]"
CHAPTER = re.compile(r"^제\s*(\d+)\s*장(?=\s|$)\s*(.{0,30})$")
DIVISION = re.compile(r"^제\s*(\d+)\s*절(?=\s|$)\s*(.{0,30})$")
# "제5조(휴가 신청) ① ...", "제5조의2(특례)". 제목 뒤에 조사가 붙은 "제3조(정의)에 따른",
# "제5조(휴가) 및 제6조" 같은 본문 속 참조는 제목으로 보지 않는다
ARTICLE = re.compile(
    rf"^제\s*(\d+)\s*조(?:\s*의\s*(\d+))?\s*{_OPEN}([^)）】\]]{{1,40}}){_CLOSE}"
    r"(?=$|\s+(?!\s|및|또는|내지)|(?!으로|부터|까지)[^\s에의를을과와은는로])"
)
ARTICLE_LIKE = re.compile(r"^제\s*\d+\s*조")
DELETED_ARTICLE = re.compile(rf"^제\s*\d+\s*조(?:\s*의\s*\d+)?\s*(?:{_OPEN}[^)）】\]]*{_CLOSE})?\s*삭\s*제")
# "부칙", "부칙 <2019. 1. 1.>", "[부칙]". "부칙에 따라", "부칙 제2조" 같은 참조는 제외
SUPPLEMENT = re.compile(
    r"^[\[【<〈〔]?\s*부\s*칙\s*[\]】>〉〕]?\s*(?:[<(〈\[【〔][^>)〉\]】〕]{0,30}[>)〉\]】〕])?"
    r"(?=\s|$)(?!\s*제\s*\d+\s*조)"
)
# "[별표 1]", "【별지 제1호 서식】", "[별첨 2]", "별표 1 출장비 기준". "별표 1에 따른다" 같은 참조는 제외
APPENDIX = re.compile(
    r"^(?:[\[【<〈〔]\s*(별\s*[표지첨])\s*(?:제\s*)?(\d*)\s*호?\s*(?:서식)?\s*[\]】>〉〕](?=\s|$|[<(〈\[])"
    r"|(별\s*[표지첨])\s*(?:제\s*)?(\d+)\s*호?(?:\s*서식)?(?=\s|$))"
)
APPENDIX_KINDS = ("별표", "별지", "별첨")
PARAGRAPH_START = re.compile(r"^[①-⑳]")
PAGE_NUMBER = re.compile(r"^(?:[-–—]\s*\d+\s*[-–—]|(?:page|p\.)\s*\d+)$", re.IGNORECASE)
REFERENCE = re.compile(r"제\s*(\d+)\s*조(?:\s*의\s*(\d+))?|별\s*([표지첨])\s*(?:제\s*)?(\d+)")

_SPACES = str.maketrans({" ": " ", "　": " ", "﻿": "", "​": ""})
_LENGTH_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=MAX_CHARS, chunk_overlap=OVERLAP_CHARS, add_start_index=True
)


@dataclass
class SplitStats:
    """문서 하나를 나누며 버리거나 고친 것. 규정 본문 없이 숫자만 담는다."""

    total_chars: int = 0  # 원문 글자 수 (공백 제외)
    kept_chars: int = 0  # 청크에 남은 글자 수 (공백·머리말 제외)
    fallback: bool = False  # 조 제목이 2개 미만이라 길이 기준으로 나눴는지
    preamble_chars: int = 0  # 첫 조문 이전이라 버린 글자 수 (표지·목차 등)
    header_footer_lines: int = 0  # 지운 머리글·바닥글·쪽 번호 줄
    deleted_articles: int = 0  # 지운 "제N조 삭제" 줄
    empty_blocks: int = 0  # 내용이 없어 버린 제목 (조문이 이어지는 부칙 제목 줄 등)
    non_heading_article_lines: int = 0  # 줄 맨 앞이 "제N조"인데 조 제목이 아닌 줄 (본문 속 참조, 다른 제목 표기)
    out_of_order: int = 0  # 번호가 앞 조보다 작은 조 제목 (새 규정의 제1조는 제외)
    segments: int = 1  # 문서 안 규정 수. 여러 규정을 합친 문서면 2 이상
    max_supplement_articles: int = 0  # 부칙 하나에 들어간 조의 최대 개수
    duplicate_ids: int = 0  # 겹쳐서 ~2 등을 붙인 chunkId 수


@dataclass
class _Line:
    page: int  # PDF 쪽 번호 (DOCX 는 0)
    text: str


@dataclass
class _Block:
    kind: str  # 본문 | 부칙 | 별표 | 별지 | 별첨
    label: str  # 위치 머리말의 마지막 칸 ("제5조(휴가 신청)", "부칙 <2019. 1. 1.>", "별표 1")
    page: int
    lines: list[str]
    head_len: int  # 첫 줄에서 제목이 차지하는 글자 수
    chapter: str = ""
    division: str = ""
    article_no: str = ""  # "5", "5의2"
    article_key: tuple[int, int] = (0, 0)
    article_title: str = ""
    supplement: str = ""
    supplement_no: int = 0
    segment: int = 1

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def body(self) -> str:
        """제목을 뺀 내용."""
        return "\n".join([self.lines[0][self.head_len :], *self.lines[1:]])

    @property
    def article_name(self) -> str:
        """조문 표기 ("제5조", "제5조의2"). 조가 아닌 블록은 빈 문자열."""
        return _article_name(*self.article_key) if self.article_no else ""


def split_document(doc: LoadedDoc) -> tuple[list[Chunk], SplitStats]:
    stats = SplitStats(total_chars=_count_chars(doc.text))
    lines = _clean_lines(doc, stats)
    start = _body_start(lines)
    if start == len(lines) or sum(bool(ARTICLE.match(line.text)) for line in lines[start:]) < 2:
        stats.fallback = True
        stats.kept_chars = sum(_count_chars(line.text) for line in lines)
        return _fallback_chunks(doc, lines), stats

    stats.preamble_chars = sum(_count_chars(line.text) for line in lines[:start])
    blocks = _parse_blocks(lines, start, stats)
    _assign_segments(blocks, stats)
    stats.kept_chars = sum(_count_chars(block.text) for block in blocks)
    return _make_chunks(doc, blocks, stats), stats


# ---------- 1. 줄 정리 ----------


def _clean_lines(doc: LoadedDoc, stats: SplitStats) -> list[_Line]:
    pages = [[text for line in page.splitlines() if (text := _normalize(line))] for page in doc.pages]
    repeated = _repeated_edge_lines(pages)
    lines: list[_Line] = []
    for page_no, page in enumerate(pages, start=1):
        edges = _edge_positions(len(page))
        for index, text in enumerate(page):
            if (edges.get(index), _shape(text)) in repeated or PAGE_NUMBER.match(text):
                stats.header_footer_lines += 1
            elif DELETED_ARTICLE.match(text):
                stats.deleted_articles += 1
            else:
                lines.append(_Line(page_no if doc.file_type == "pdf" else 0, text))
    return lines


def _normalize(line: str) -> str:
    return re.sub(r"[ \t]+", " ", line.translate(_SPACES)).strip()


def _edge_positions(count: int) -> dict[int, str]:
    """쪽의 맨 위 2줄·맨 아래 2줄의 자리. 머리글·바닥글은 이 자리에만 나온다고 본다."""
    positions = {index: f"top{index}" for index in range(min(2, count))}
    positions.update({count - 1 - index: f"bottom{index}" for index in range(min(2, count))})
    return positions


def _shape(text: str) -> str:
    """공백을 없애고 숫자를 #으로 바꾼 모양. "- 3 -"과 "- 4 -"를 같은 줄로 본다."""
    return re.sub(r"\d+", "#", re.sub(r"\s+", "", text))


def _repeated_edge_lines(pages: list[list[str]]) -> set[tuple[str, str]]:
    """절반 이상의 쪽에서 같은 자리에 반복되는 줄 = 머리글·바닥글. 3쪽 미만 문서는 판단하지 않는다."""
    if len(pages) < 3:
        return set()
    counter: Counter = Counter()
    for page in pages:
        for index, position in _edge_positions(len(page)).items():
            text = page[index]
            if len(text) <= MAX_HEADER_FOOTER_LEN and not _is_heading(text):
                counter[(position, _shape(text))] += 1
    threshold = max(3, math.ceil(len(pages) / 2))
    return {key for key, count in counter.items() if count >= threshold}


# ---------- 2. 본문 시작 찾기 ----------


def _body_start(lines: list[_Line]) -> int:
    """본문이 시작되는 줄: 내용이 딸린 첫 조 제목. 목차의 조 제목에는 쪽 번호 말고 내용이 없다."""
    for index, line in enumerate(lines):
        if not (match := ARTICLE.match(line.text)):
            continue
        content = line.text[match.end() :]
        for following in lines[index + 1 :]:
            if _meaningful_chars(content) >= MIN_CONTENT_CHARS or _is_heading(following.text):
                break
            content += following.text
        if _meaningful_chars(content) >= MIN_CONTENT_CHARS:
            return index
    return len(lines)


def _context_before(lines: list[_Line]) -> tuple[str, str]:
    """본문 첫 조가 속한 장·절 = 본문 시작 직전에 나온 마지막 장·절 제목."""
    chapter = division = ""
    for line in lines:
        if (match := CHAPTER.match(line.text)) and _is_title(match[2]):
            chapter, division = _heading(f"제{match[1]}장", match[2]), ""
        elif (match := DIVISION.match(line.text)) and _is_title(match[2]):
            division = _heading(f"제{match[1]}절", match[2])
    return chapter, division


# ---------- 3. 블록 나누기 ----------


def _parse_blocks(lines: list[_Line], start: int, stats: SplitStats) -> list[_Block]:
    chapter, division = _context_before(lines[:start])
    blocks: list[_Block] = []
    mode = "본문"  # 본문 | 부칙 | 별표 : 지금 읽고 있는 부분
    supplement, supplement_no = "", 0
    last_article = (0, 0)  # 마지막 본문 조 번호
    last_supplement_article = (0, 0)  # 지금 부칙 안의 마지막 조 번호

    for line in lines[start:]:
        text = line.text
        if (match := CHAPTER.match(text)) and _is_title(match[2]):
            mode, chapter, division = "본문", _heading(f"제{match[1]}장", match[2]), ""
            continue
        if mode == "본문" and (match := DIVISION.match(text)) and _is_title(match[2]):
            division = _heading(f"제{match[1]}절", match[2])
            continue

        if match := ARTICLE.match(text):
            key, title = (int(match[1]), int(match[2] or 0)), match[3].strip()
            # 부칙·별표 다음의 조는 다음 경우에만 본문으로 본다
            # - 앞 본문에 이어지는 번호 / 새 규정의 "제1조(목적)" / 부칙 안에서 조 번호가 다시 작아짐(새 규정 시작)
            if (
                mode == "본문"
                or key > last_article
                or (key == (1, 0) and title == "목적")
                or (mode == "부칙" and key <= last_supplement_article)
            ):
                blocks.append(_article_block("본문", line, match, chapter=chapter, division=division))
                mode, last_article = "본문", key
                continue
            if mode == "부칙":
                blocks.append(_article_block("부칙", line, match, supplement=supplement, supplement_no=supplement_no))
                last_supplement_article = key
                continue
            # 별표 안의 "제N조(...)" 줄은 표 내용이므로 아래에서 앞 블록에 붙인다
        elif ARTICLE_LIKE.match(text):
            stats.non_heading_article_lines += 1

        if match := SUPPLEMENT.match(text):
            mode, supplement_no, last_supplement_article = "부칙", supplement_no + 1, (0, 0)
            supplement = re.sub(r"\s+", " ", match[0])
            blocks.append(
                _Block("부칙", supplement, line.page, [text], match.end(), supplement=supplement, supplement_no=supplement_no)
            )
            continue
        if match := APPENDIX.match(text):
            mode, kind = "별표", re.sub(r"\s+", "", match[1] or match[3])
            blocks.append(_Block(kind, f"{kind} {match[2] or match[4] or ''}".strip(), line.page, [text], match.end()))
            continue

        blocks[-1].lines.append(text)

    kept = [block for block in blocks if _meaningful_chars(block.body) > 0]
    stats.empty_blocks = len(blocks) - len(kept)
    supplement_articles = Counter(block.supplement_no for block in kept if block.kind == "부칙" and block.article_no)
    stats.max_supplement_articles = max(supplement_articles.values(), default=0)
    return kept


def _article_name(number: int, sub: int) -> str:
    # 가지 조문은 법령 표기대로 "제5조의2". 본문 속 참조(REFERENCE)도 이 형태로 만들어 서로 비교할 수 있다
    return f"제{number}조의{sub}" if sub else f"제{number}조"


def _article_block(kind: str, line: _Line, match: re.Match, **fields) -> _Block:
    number, sub, title = int(match[1]), int(match[2] or 0), match[3].strip()
    article_no = f"{number}의{sub}" if sub else str(number)
    return _Block(
        kind,
        f"{_article_name(number, sub)}({title})",
        line.page,
        [line.text],
        match.end(),
        article_no=article_no,
        article_key=(number, sub),
        article_title=title,
        **fields,
    )


def _assign_segments(blocks: list[_Block], stats: SplitStats) -> None:
    """여러 규정을 합친 문서에서 본문 제1조가 다시 나올 때마다 구간 번호를 올린다."""
    segment, previous = 1, (0, 0)
    for block in blocks:
        if block.kind == "본문":
            if block.article_key == (1, 0) and previous > (1, 0):
                segment += 1
            elif block.article_key < previous:
                stats.out_of_order += 1
            previous = block.article_key
        block.segment = segment
    stats.segments = segment


# ---------- 4. 청크 만들기 ----------


def _make_chunks(doc: LoadedDoc, blocks: list[_Block], stats: SplitStats) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen: Counter = Counter()
    for block in blocks:
        section_id = _section_id(doc, block)
        seen[section_id] += 1
        if seen[section_id] > 1:
            stats.duplicate_ids += 1
            section_id = f"{section_id}~{seen[section_id]}"
        header = _header(doc, block)
        parts = _split_long(block)
        fields = {
            "type": block.kind,
            "label": block.label,
            "chapter": block.chapter,
            "division": block.division,
            "article_no": block.article_no,
            "article_title": block.article_title,
            "supplement": block.supplement,
            "refs": _references(block),
            "page": block.page,
            "segment": block.segment,
            "parts": len(parts),
        }
        for index, part in enumerate(parts, start=1):
            chunk_id = section_id if len(parts) == 1 else f"{section_id}-{index}"
            metadata = _metadata(doc, chunk_id, section_id, part=index, **fields)
            chunks.append(Chunk(chunk_id, f"{header}\n{part}", metadata))
    return chunks


def _section_id(doc: LoadedDoc, block: _Block) -> str:
    """조·부칙·별표 단위 id. 예: 사규/휴가규정#제5조, #제5조의2, #부칙2-제1조, #별표1, 합본 문서의 둘째 규정은 #구간2-제1조"""
    prefix = f"{doc.category}/{doc.title}#" + (f"구간{block.segment}-" if block.segment > 1 else "")
    if block.kind == "본문":
        return f"{prefix}{block.article_name}"
    if block.kind == "부칙":
        article = f"-{block.article_name}" if block.article_no else ""
        return f"{prefix}부칙{block.supplement_no}{article}"
    return prefix + block.label.replace(" ", "")


def _header(doc: LoadedDoc, block: _Block) -> str:
    """청크 맨 앞의 위치 머리말. 예: [사규 > 휴가규정 > 제2장 휴가 > 제5조(휴가 신청)]"""
    path = [doc.category, doc.title]
    if block.kind == "본문":
        path += [block.chapter, block.division]
    elif block.kind == "부칙" and block.article_no:
        path.append(block.supplement)
    path.append(block.label)
    return "[" + " > ".join(part for part in path if part) + "]"


def _references(block: _Block) -> str:
    """내용 속 다른 조·별표 참조를 "제3조|별표 1" 형태로. 다른 법령의 조문 번호도 섞일 수 있다."""
    own = block.article_name or block.label
    refs: list[str] = []
    for match in REFERENCE.finditer(block.body):
        if match[1]:
            ref = f"제{match[1]}조의{match[2]}" if match[2] else f"제{match[1]}조"
        else:
            ref = f"별{match[3]} {match[4]}"
        if ref != own and ref not in refs:
            refs.append(ref)
    return "|".join(refs)


def _metadata(doc: LoadedDoc, chunk_id: str, section_id: str, **fields: MetadataValue) -> dict[str, MetadataValue]:
    """검색 필터를 쉽게 하려고 모든 청크가 같은 키를 갖게 한다. 값이 없으면 "" 또는 0."""
    metadata: dict[str, MetadataValue] = {
        "chunk_id": chunk_id,
        "section_id": section_id,  # 길어서 나뉜 조각들이 공유하는 조·별표 단위 id (평가셋 정답과 대조)
        "category": doc.category,
        "title": doc.title,
        "source": f"{doc.category}/{doc.path.name}",
        "file_type": doc.file_type,
        "type": "",
        "label": "",
        "chapter": "",
        "division": "",
        "article_no": "",
        "article_title": "",
        "supplement": "",
        "refs": "",
        "page": 0,
        "segment": 1,
        "part": 1,
        "parts": 1,
    }
    metadata.update(fields)
    return metadata


def _split_long(block: _Block) -> list[str]:
    """MAX_CHARS 를 넘는 블록을 나눈다. 조·부칙은 항(①②) 경계에서, 별표는 줄 경계에서 자른다."""
    text = block.text
    if len(text) <= MAX_CHARS:
        return [text]
    if block.kind in APPENDIX_KINDS:
        return _repeat_table_header(block, _LENGTH_SPLITTER.split_text(text))
    paragraphs: list[list[str]] = [[]]
    for line in block.lines:
        if PARAGRAPH_START.match(line) and paragraphs[-1]:
            paragraphs.append([])
        paragraphs[-1].append(line)
    return _pack(["\n".join(lines) for lines in paragraphs])


def _pack(pieces: list[str]) -> list[str]:
    """항들을 앞에서부터 MAX_CHARS 를 넘지 않게 묶는다. 항 하나가 넘으면 길이 기준으로 자른다."""
    parts: list[str] = []
    current = ""
    for piece in pieces:
        if len(piece) > MAX_CHARS:
            parts += [current] if current else []
            parts += _LENGTH_SPLITTER.split_text(piece)
            current = ""
        elif current and len(current) + 1 + len(piece) > MAX_CHARS:
            parts.append(current)
            current = piece
        else:
            current = f"{current}\n{piece}" if current else piece
    if current:
        parts.append(current)
    return parts


def _repeat_table_header(block: _Block, parts: list[str]) -> list[str]:
    """표가 여러 조각으로 나뉘면 둘째 조각부터 표의 첫 행(열 제목)을 다시 붙인다. DOCX 표는 "셀 | 셀" 형태다."""
    header_row = next((line for line in block.lines[1:] if " | " in line), "")
    if not header_row:
        return parts
    return [parts[0], *(part if part.startswith(header_row) else f"{header_row}\n{part}" for part in parts[1:])]


def _fallback_chunks(doc: LoadedDoc, lines: list[_Line]) -> list[Chunk]:
    """조 제목이 없는 문서는 길이 기준으로 나눈다. 쪽 번호는 조각이 시작되는 줄의 쪽."""
    text = "\n".join(line.text for line in lines)
    line_starts = [0, *accumulate(len(line.text) + 1 for line in lines)]
    header = f"[{doc.category} > {doc.title}]"
    chunks: list[Chunk] = []
    for index, piece in enumerate(_LENGTH_SPLITTER.create_documents([text]) if text else [], start=1):
        line = lines[bisect_right(line_starts, max(piece.metadata["start_index"], 0)) - 1]
        chunk_id = f"{doc.category}/{doc.title}#본문{index}"
        metadata = _metadata(doc, chunk_id, chunk_id, type="fallback", page=line.page)
        chunks.append(Chunk(chunk_id, f"{header}\n{piece.page_content}", metadata))
    return chunks


# ---------- 공통 ----------


def _is_heading(text: str) -> bool:
    return any(pattern.match(text) for pattern in (CHAPTER, DIVISION, ARTICLE, SUPPLEMENT, APPENDIX))


def _is_title(rest: str) -> bool:
    """장·절 번호 뒤가 제목인지. "제3장 제5조는 ...한다."처럼 문장이면 제목이 아니다."""
    return not re.search(r"제\s*\d+\s*조|다\.?$", rest)


def _heading(prefix: str, rest: str) -> str:
    # 목차에서 온 쪽 번호·점선이 끝에 붙어 있으면 뗀다
    title = re.sub(r"[\s.·ㆍ∙…‥\d]+$", "", rest)
    return f"{prefix} {title}".strip()


def _count_chars(text: str) -> int:
    return sum(not ch.isspace() for ch in text)


def _meaningful_chars(text: str) -> int:
    """공백·숫자·기호를 뺀 글자 수. 목차 줄(제목 + 쪽 번호)의 내용은 0이 된다."""
    return len(re.sub(r"[\W\d_]", "", text))
