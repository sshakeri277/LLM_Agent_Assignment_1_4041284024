"""Deterministic answer-format layer.

The required answer format is stated inside each problem (a template such as
``Name=<integer>, ...``, a list of allowed options, "only the integer", ...).
This module extracts that format once (`extract_spec`) and turns any model
output into the exact required string (`normalize`), or reports that it
cannot (`None`).  `canonical` gives a comparison key used to decide whether
two candidate answers agree.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Optional, Tuple

# ---------------------------------------------------------------- utilities

_DIGIT_MAP = {ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")}
_DIGIT_MAP.update({ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")})
_DIGIT_MAP.update({ord("٫"): ".", ord("٬"): ","})

_ARABIC_LETTER = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")
_PLACEHOLDER = re.compile(
    r"(?:^|(?<=[\s,;:،؛(\[]))(?P<key>[^\W\d]\w*)\s*=\s*"
    r"(?P<ph><\s*(?P<type>integer|int|number|float|decimal|string|word|name|text|letter)\s*>)",
    re.IGNORECASE)
_PLACEHOLDER_TOKEN = re.compile(
    r"<\s*(?:integer|int|number|float|decimal|string|word|name|text|letter)\s*>", re.IGNORECASE)
_PREFIX = re.compile(r"^(?:final\s*answer|the\s*answer\s*is|answer|پاسخ\s*نهایی|پاسخ|جواب)\s*(?:is)?\s*[:：]?\s*",
                     re.IGNORECASE)
_QUOTES = "\"'«»“”‘’`"
_TRAILING_PUNCT = ".。!؛;,،:"
_BULLET = re.compile(r"^\s*(?:[-*•▪◦]|\(?\d{1,2}[).]|\(?[a-zA-Z][).])\s+")
_OPTION_CUE = re.compile(r"یکی\s*از|گزینه|عبارت|انتخاب|برگردان|one\s+of|options?|choose", re.IGNORECASE)
_LEN_WORDS = {"دو": 2, "سه": 3, "چهار": 4, "پنج": 5, "شش": 6, "هفت": 7, "هشت": 8, "نه": 9, "ده": 10}
_DIGITS_CUE = re.compile(r"(دو|سه|چهار|پنج|شش|هفت|هشت|نه|ده|\d+)\s*-?\s*رقمی|(\d+)\s*-\s*digit", re.IGNORECASE)
_LETTER_CUE = re.compile(r"فقط\s*حرف|حرف\s*آن|only\s+the\s+letter|single\s+letter", re.IGNORECASE)
_INTEGER_CUE = re.compile(r"عدد\s*صحیح|integer|whole\s+number", re.IGNORECASE)
_NUMBER_CUE = re.compile(r"فقط\s*(?:یک\s*)?عدد|only\s+the\s+number|just\s+the\s+number|مقدار\s*عددی", re.IGNORECASE)
_INSTR_CUE = re.compile(r"برگردان|بنویس|پاسخ|فرمت|return|answer|respond|format", re.IGNORECASE)
_RANGE = re.compile(r"\d+\s*(?:تا|to|–|-)\s*\d+", re.IGNORECASE)
_INLINE_NUM_CUE = re.compile(r"یکی\s*از\s*(?:اعداد|مقادیر|شماره)|one\s+of\s+(?:the\s+)?numbers", re.IGNORECASE)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_THOUSANDS = re.compile(r"(?<![\d.])(-?\d{1,3}(?:,\d{3})+)(?![\d])")
# Where a justification starts after the answer: "480 (since ...)", "E because ...", "۴۸۰ زیرا ...".
_EXPLANATION = re.compile(r"\s*(?:[(\[]|\bsince\b|\bbecause\b|زیرا|چون|به\s*دلیل|\s[—–]\s|[;؛])", re.IGNORECASE)
_FINAL_MARK = re.compile(r"(?:final\s*answer|پاسخ\s*نهایی)\s*\**\s*[:：]\s*\**\s*(.*)$", re.IGNORECASE)


def to_ascii_digits(s: str) -> str:
    return s.translate(_DIGIT_MAP)


def _prep(s: str) -> str:
    return to_ascii_digits(unicodedata.normalize("NFKC", s)).replace("\u200c", "")


def clean(raw: str) -> str:
    """Strip markdown, answer prefixes, quotes and trailing punctuation; ASCII digits."""
    s = _prep(raw or "").strip()
    s = re.sub(r"\\boxed\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", s)
    s = s.replace("$", "").replace("**", "").replace("__", "")
    s = re.sub(r"^#+\s*", "", s)
    previous = None
    while previous != s:
        previous = s
        s = _PREFIX.sub("", s).strip()
        s = s.strip(_QUOTES).strip()
        s = s.rstrip(_TRAILING_PUNCT).strip()
    return re.sub(r"\s+", " ", s)


def _loose(s: str) -> str:
    s = clean(s).lower()
    s = re.sub(r"\s*([=,:])\s*", r"\1", s)
    return re.sub(r"\s+", " ", s)


def canonical(s: str) -> str:
    c = re.sub(r"\s+", "", clean(s).lower())
    if re.fullmatch(r"-?\d+(?:\.\d+)?", c):
        return _decimal_text(c)
    return c


def _decimal_text(x: str) -> str:
    """Exact canonical text of a plain decimal ("007" -> "7", "12.50" -> "12.5", "-0.0" -> "0").

    Works on the digits themselves (Decimal), never through float, so no digit of a large
    integer or a long decimal is lost or rounded.
    """
    text = format(Decimal(x), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def extract_final_answer(content: str) -> Optional[str]:
    """Answer after the last 'FINAL ANSWER:' marker, else the last non-empty line."""
    if not content or not content.strip():
        return None
    lines = content.splitlines()
    for idx in range(len(lines) - 1, -1, -1):
        m = _FINAL_MARK.search(lines[idx])
        if m:
            value = m.group(1).strip()
            if value:
                return value
            for nxt in lines[idx + 1:]:
                if nxt.strip():
                    return nxt.strip()
    for line in reversed(lines):
        if line.strip():
            return line.strip()
    return None


# --------------------------------------------------------------- the spec

@dataclass(frozen=True)
class FormatSpec:
    kind: str  # template | options | digits | letter | integer | number | free
    template: Optional[str] = None
    placeholders: Tuple[Tuple[str, str], ...] = ()
    options: Tuple[str, ...] = ()
    digits: Optional[int] = None
    description: str = "Answer with only the final answer, no explanation."


def _option_like(line: str) -> Optional[str]:
    text = _BULLET.sub("", line).strip()
    if not text or len(text) > 80 or _ARABIC_LETTER.search(text) or _PLACEHOLDER_TOKEN.search(text):
        return None
    return text


def _instruction_region(lines: List[str]) -> List[str]:
    """From the last line that gives an answer instruction to the end of the problem."""
    nonempty = [i for i, line in enumerate(lines) if line.strip()]
    if not nonempty:
        return []
    cues = [i for i in nonempty if _INSTR_CUE.search(lines[i])]
    return lines[cues[-1] if cues else nonempty[-1]:]


def _trailing_options(lines: List[str]) -> Tuple[str, ...]:
    """A block of >= 2 option-like lines that ends the problem, right after a cue line."""
    block: List[str] = []
    for line in reversed(lines):
        if not line.strip():
            continue
        opt = _option_like(line)
        if opt is None:
            if len(block) >= 2 and (_OPTION_CUE.search(line) or _INSTR_CUE.search(line)):
                return tuple(reversed(block))
            return ()
        block.append(opt)
    return ()


def _inline_numeric_options(text: str) -> Tuple[str, ...]:
    for m in _INLINE_NUM_CUE.finditer(text):
        tail = text[m.end():].split("\n", 1)[0]
        tail = re.split(r"\s+را\s|\.\s|\?", tail, maxsplit=1)[0]
        if _RANGE.search(tail):
            continue  # "one of the numbers 1 to 9" is a range, not a list of choices
        nums = re.findall(r"-?\d+", tail)
        if len(nums) >= 2:
            return tuple(dict.fromkeys(nums))
    return ()


def _template_spec(lines: List[str]) -> Optional[FormatSpec]:
    for line in lines:
        found = list(_PLACEHOLDER.finditer(line))
        if found:
            template = line[found[0].start("key"):found[-1].end()].strip()
            placeholders = tuple((m.group("key"), m.group("type").lower()) for m in found)
            return FormatSpec("template", template=template, placeholders=placeholders,
                              description="Answer in exactly this format, replacing each <...> placeholder "
                                          f"with its value: {template}")
    return None


def extract_spec(problem: str) -> FormatSpec:
    """Detect the required answer format from the instruction at the end of the problem."""
    text = _prep(problem or "")
    lines = text.splitlines()
    region = _instruction_region(lines)
    region_text = "\n".join(region)

    spec = _template_spec(region) or _template_spec(lines)
    if spec:
        return spec

    options = _trailing_options(lines) or _inline_numeric_options(region_text)
    if options:
        return FormatSpec("options", options=options,
                          description="Answer with exactly one of: " + " | ".join(options))

    m = _DIGITS_CUE.search(region_text)
    if m:
        word = m.group(1) or m.group(2)
        n = int(word) if word.isdigit() else _LEN_WORDS[word]
        return FormatSpec("digits", digits=n,
                          description=f"Answer with only the {n}-digit code (digits only, no spaces).")

    if _LETTER_CUE.search(region_text):
        return FormatSpec("letter", description="Answer with only the single letter (e.g. B).")

    if _INTEGER_CUE.search(region_text):
        return FormatSpec("integer", description="Answer with only the integer (digits only, no units or words).")

    if _NUMBER_CUE.search(region_text):
        return FormatSpec("number", description="Answer with only the number (digits and an optional decimal "
                                                "point; no units or words).")

    return FormatSpec("free")


# ---------------------------------------------------------- normalization

def _as_number(s: str) -> Optional[str]:
    """The single number in `s` (repeats of the same value are fine), exactly; else None."""
    s = _THOUSANDS.sub(lambda m: m.group(1).replace(",", ""), s)
    values = {_decimal_text(x) for x in _NUMBER.findall(s)}
    return values.pop() if len(values) == 1 else None


def _as_int(s: str) -> Optional[str]:
    value = _as_number(s)
    return value if value is not None and "." not in value else None


def _normalize_template(c: str, spec: FormatSpec) -> Optional[str]:
    values = []
    for key, typ in spec.placeholders:
        m = re.search(rf"(?<!\w){re.escape(key)}\s*[=:]\s*([^,;،؛\n|]+)", c, re.IGNORECASE)
        if not m:
            return None
        raw_value = clean(m.group(1))
        if typ in ("integer", "int"):
            value = _as_int(raw_value)
        elif typ in ("number", "float", "decimal"):
            value = _as_number(raw_value)
        else:
            value = raw_value or None
        if value is None:
            return None
        values.append(value)
    template = spec.template or ""
    out, last = [], 0
    for m, value in zip(_PLACEHOLDER.finditer(template), values):
        out.append(template[last:m.start("ph")])
        out.append(value)
        last = m.end("ph")
    out.append(template[last:])
    return "".join(out)


def _normalize_option(c: str, spec: FormatSpec) -> Optional[str]:
    key = canonical(c)
    for opt in spec.options:
        if canonical(opt) == key:
            return opt
    loose = _loose(c)
    hits = [opt for opt in spec.options
            if re.search(rf"(?<![a-z0-9]){re.escape(_loose(opt))}(?![a-z0-9])", loose)]
    return hits[0] if len(hits) == 1 else None


def _answer_head(c: str, kind: str) -> str:
    """The answer without a trailing justification; for single values, the part after the last '='."""
    head = _EXPLANATION.split(c, maxsplit=1)[0]
    if kind in ("integer", "number", "digits", "letter") and "=" in head:
        head = head.rsplit("=", 1)[1]
    return clean(head)


def normalize(raw: str, spec: FormatSpec) -> Optional[str]:
    """Return `raw` in the exact required form, or None if it cannot be made valid.

    The whole answer is tried first; only if that fails is a trailing justification cut off
    ("x = 480 (since ...)" -> "480"), so an answer that is already valid is never changed.
    """
    c = clean(raw or "")
    if not c:
        return None
    value = _normalize_clean(c, spec)
    if value is None:
        head = _answer_head(c, spec.kind)
        if head and head != c:
            value = _normalize_clean(head, spec)
    return value


def _normalize_clean(c: str, spec: FormatSpec) -> Optional[str]:
    if spec.kind == "template":
        return _normalize_template(c, spec)
    if spec.kind == "options":
        return _normalize_option(c, spec)
    if spec.kind == "digits":
        runs = {r for r in re.findall(r"\d+", c) if len(r) == spec.digits}
        return runs.pop() if len(runs) == 1 else None
    if spec.kind == "letter":
        letters = {x.upper() for x in re.findall(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])", c)}
        return letters.pop() if len(letters) == 1 else None
    if spec.kind == "integer":
        return _as_int(c)
    if spec.kind == "number":
        return _as_number(c)
    return c
