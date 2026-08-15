from __future__ import annotations

import re
import unicodedata

ONES = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
SCALE_TEMPLATE = ["", "@K", "triệu", "tỷ", "@K tỷ", "triệu tỷ", "tỷ tỷ", "@K tỷ tỷ"]
ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

DEFAULT_LETTERS = {
    "a": "a", "b": "bê", "c": "xê", "d": "đê", "e": "e", "f": "ép", "g": "giê",
    "h": "hát", "i": "i", "j": "giây", "k": "ca", "l": "lờ", "m": "mờ", "n": "nờ",
    "o": "ô", "p": "pê", "q": "quy", "r": "rờ", "s": "ét", "t": "tê", "u": "u",
    "v": "vê", "w": "vê kép", "x": "ích", "y": "i", "z": "dét",
}

DEFAULT_UNITS = {
    "km/h": "ki lô mét trên giờ", "m/s": "mét trên giây", "km/s": "ki lô mét trên giây",
    "km2": "ki lô mét vuông", "cm2": "xăng ti mét vuông", "mm2": "mi li mét vuông",
    "m2": "mét vuông", "m3": "mét khối", "cm3": "xăng ti mét khối",
    "km": "ki lô mét", "cm": "xăng ti mét", "mm": "mi li mét", "dm": "đề xi mét",
    "kg": "ki lô gam", "mg": "mi li gam", "ml": "mi li lít", "kw": "ki lô oát",
    "kb": "ki lô bai", "mb": "mê ga bai", "gb": "gi ga bai", "tb": "tê ra bai",
    "khz": "ki lô héc", "mhz": "mê ga héc", "ghz": "gi ga héc", "hz": "héc",
    "ha": "héc ta", "m": "mét", "g": "gam", "l": "lít", "w": "oát", "v": "vôn",
}

DEFAULT_CURRENCY = {
    "vnđ": "đồng", "vnd": "đồng", "đ": "đồng", "usd": "đô la", "eur": "ơ rô",
    "jpy": "yên", "gbp": "bảng", "krw": "uôn", "cny": "nhân dân tệ",
}

DEFAULT_ABBREV = {
    "tp.hcm": "thành phố Hồ Chí Minh", "tphcm": "thành phố Hồ Chí Minh",
    "tp.": "thành phố", "tt.": "thị trấn", "q.": "quận", "p.": "phường",
    "ts.": "tiến sĩ", "ths.": "thạc sĩ", "gs.": "giáo sư", "pgs.": "phó giáo sư",
    "bs.": "bác sĩ", "ks.": "kỹ sư", "cn.": "cử nhân", "nxb": "nhà xuất bản",
    "v.v.": "vân vân", "vv.": "vân vân", "vd.": "ví dụ", "vd:": "ví dụ",
    "tr.": "trang", "ubnd": "ủy ban nhân dân", "hđnd": "hội đồng nhân dân",
    "tnhh": "trách nhiệm hữu hạn", "sđt": "số điện thoại", "đt.": "điện thoại",
}

DEFAULT_SYMBOLS = {
    "&": " và ", "+": " cộng ", "=": " bằng ", "@": " a còng ", "#": " số ",
    "~": " khoảng ", "<": " nhỏ hơn ", ">": " lớn hơn ", "±": " cộng trừ ",
    "×": " nhân ", "÷": " chia ", "°": " độ ", "%": " phần trăm ",
    "½": " một phần hai ", "¼": " một phần tư ", "¾": " ba phần tư ",
    "$": " đô la ", "€": " ơ rô ", "£": " bảng ", "¥": " yên ", "₫": " đồng ",
    "/": " trên ", "\\": " ", "|": " ", "*": " ", "_": " ",
}

NON_SPEECH = re.compile(
    r"[\[\(]\s*(?:nhạc|music|laugh\w*|applause|silence|inaudible|tiếng\s+[^\]\)]{0,20})[^\]\)]*[\]\)]",
    re.IGNORECASE,
)

NUMBER = r"\d+(?:[.,]\d+)*"


def roman_to_int(text: str) -> int | None:
    total, previous = 0, 0
    for char in reversed(text.upper()):
        value = ROMAN_VALUES.get(char)
        if value is None:
            return None
        if value < previous:
            total -= value
        else:
            total += value
            previous = value
    return total or None


class Normalizer:
    def __init__(self, tcfg: dict | None = None):
        t = dict(tcfg or {})
        self.le = str(t.get("le_word", "lẻ"))
        self.thousand = str(t.get("thousand_word", "nghìn"))
        self.four = str(t.get("four_after_ten", "tư"))
        self.decimal_by_digit = bool(t.get("decimal_by_digit", True))
        self.spell_acronyms = bool(t.get("spell_unknown_acronyms", True))
        self.lowercase = bool(t.get("lowercase", False))
        self.punct = str(t.get("keep_punctuation", ",.!?…"))
        self.scales = [s.replace("@K", self.thousand) for s in SCALE_TEMPLATE]

        self.letters = self._merged(DEFAULT_LETTERS, t.get("letters"))
        self.units = self._merged(DEFAULT_UNITS, t.get("units"))
        self.currency = self._merged(DEFAULT_CURRENCY, t.get("currency"))
        self.abbrev = self._merged(DEFAULT_ABBREV, t.get("abbreviations"))
        self.symbols = dict(DEFAULT_SYMBOLS)
        self.symbols.update(t.get("symbols") or {})
        self._compile()

    @staticmethod
    def _merged(base: dict, extra) -> dict:
        out = {k.lower(): v for k, v in base.items()}
        for k, v in (extra or {}).items():
            out[str(k).lower()] = str(v)
        return out

    def _compile(self) -> None:
        units_alt = "|".join(re.escape(k) for k in sorted(self.units, key=len, reverse=True))
        currency_alt = "|".join(re.escape(k) for k in sorted(self.currency, key=len, reverse=True))
        abbrev_alt = "|".join(re.escape(k) for k in sorted(self.abbrev, key=len, reverse=True))

        self.abbrev_re = re.compile(rf"(?<!\w)({abbrev_alt})(?!\w)", re.IGNORECASE)
        self.acronym_re = re.compile(r"(?<![\w.])([A-Z]{2,5})(?![\w.])")
        self.rules = [
            (re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b"), self._time),
            (re.compile(r"\b(\d{1,2})\s*[hg](\d{2})\b"), self._hour_minute),
            (re.compile(r"\b(\d{1,2})\s*h\b"), self._hour),
            (re.compile(r"\b(?:(ngày|mùng|mồng)\s+)?(\d{1,2})\s*[/-]\s*(\d{1,2})\s*[/-]\s*(\d{2,4})\b",
                        re.IGNORECASE), self._full_date),
            (re.compile(r"\b(ngày|mùng|mồng)\s+(\d{1,2})\s*[/-]\s*(\d{1,2})\b",
                        re.IGNORECASE), self._day_month),
            (re.compile(r"\b(\d{1,2})\s*/\s*(\d{4})\b"), self._month_year),
            (re.compile(r"\bthứ\s+(\d{1,2})\b", re.IGNORECASE), self._ordinal),
            (re.compile(r"\b(chương|phần|mục|tập|quyển|hồi|khóa|khoá|quý|kỳ|kì|thế\s+k[ỷỉ])\s+"
                        r"([IVXLCDM]{1,8})(?![\wÀ-ỹ])", re.IGNORECASE), self._roman),
            (re.compile(rf"({NUMBER})\s*%"), self._percent),
            (re.compile(rf"([$€£¥₫])\s*({NUMBER})"), self._currency_prefix),
            (re.compile(rf"({NUMBER})\s*°\s*([CFcf])(?!\w)"), self._temperature),
            (re.compile(rf"({NUMBER})\s*({units_alt})(?!\w)", re.IGNORECASE), self._unit),
            (re.compile(rf"({NUMBER})\s*({currency_alt})(?!\w)", re.IGNORECASE), self._currency_suffix),
            (re.compile(rf"\b({NUMBER})\s*-\s*({NUMBER})\b"), self._range),
            (re.compile(r"\b(\d{1,3})\s*/\s*(\d{1,3})\b"), self._fraction),
            (re.compile(NUMBER), self._plain_number),
        ]

    def group_of_three(self, value: int, leading: bool) -> list[str]:
        hundreds, tens, units = value // 100, (value // 10) % 10, value % 10
        parts: list[str] = []
        if hundreds > 0:
            parts += [ONES[hundreds], "trăm"]
        elif not leading and (tens > 0 or units > 0):
            parts += ["không", "trăm"]
        if tens > 1:
            parts += [ONES[tens], "mươi"]
            if units == 1:
                parts.append("mốt")
            elif units == 4:
                parts.append(self.four)
            elif units == 5:
                parts.append("lăm")
            elif units > 0:
                parts.append(ONES[units])
        elif tens == 1:
            parts.append("mười")
            if units == 5:
                parts.append("lăm")
            elif units > 0:
                parts.append(ONES[units])
        elif units > 0:
            if hundreds > 0 or not leading:
                parts.append(self.le)
            parts.append(ONES[units])
        return parts

    def integer(self, value) -> str:
        n = int(value)
        if n < 0:
            return "âm " + self.integer(-n)
        if n == 0:
            return ONES[0]
        if n >= 10 ** (3 * len(self.scales)):
            return self.digit_by_digit(str(n))
        groups: list[int] = []
        while n > 0:
            groups.append(n % 1000)
            n //= 1000
        top = len(groups) - 1
        parts: list[str] = []
        for i in range(top, -1, -1):
            if groups[i] == 0:
                continue
            parts += self.group_of_three(groups[i], i == top)
            if self.scales[i]:
                parts.append(self.scales[i])
        return " ".join(parts)

    def digit_by_digit(self, text: str) -> str:
        return " ".join(ONES[int(c)] for c in str(text) if c.isdigit())

    def decimal(self, whole: str, fraction: str) -> str:
        head = self.integer(int(whole or 0))
        tail = self.digit_by_digit(fraction) if self.decimal_by_digit else self.integer(int(fraction))
        return f"{head} phẩy {tail}"

    def number_token(self, token: str) -> str:
        text = str(token).strip()
        negative = text.startswith("-")
        if negative:
            text = text[1:]
        if re.fullmatch(r"\d{1,3}(?:\.\d{3})+,\d+", text):
            whole, fraction = text.rsplit(",", 1)
            body = self.decimal(whole.replace(".", ""), fraction)
        elif re.fullmatch(r"\d{1,3}(?:,\d{3})+\.\d+", text):
            whole, fraction = text.rsplit(".", 1)
            body = self.decimal(whole.replace(",", ""), fraction)
        elif re.fullmatch(r"\d{1,3}(?:\.\d{3})+", text):
            body = self.integer(text.replace(".", ""))
        elif re.fullmatch(r"\d{1,3}(?:,\d{3})+", text):
            body = self.integer(text.replace(",", ""))
        elif re.fullmatch(r"\d+[.,]\d+", text):
            whole, fraction = re.split(r"[.,]", text, maxsplit=1)
            body = self.decimal(whole, fraction)
        elif re.fullmatch(r"\d+", text):
            if (len(text) > 1 and text[0] == "0") or len(text) >= 10:
                body = self.digit_by_digit(text)
            else:
                body = self.integer(text)
        else:
            body = self.digit_by_digit(text)
        return ("âm " + body) if negative else body

    def month_word(self, month: int) -> str:
        if month == 4:
            return self.four
        return self.integer(month)

    def _time(self, m: re.Match) -> str:
        hour, minute = int(m.group(1)), int(m.group(2))
        second = int(m.group(3)) if m.group(3) else None
        if hour > 23 or minute > 59 or (second is not None and second > 59):
            return m.group(0)
        out = f" {self.integer(hour)} giờ"
        if minute:
            out += f" {self.integer(minute)} phút"
        if second:
            out += f" {self.integer(second)} giây"
        return out + " "

    def _hour_minute(self, m: re.Match) -> str:
        hour, minute = int(m.group(1)), int(m.group(2))
        if hour > 23 or minute > 59:
            return m.group(0)
        return f" {self.integer(hour)} giờ {self.integer(minute)} phút "

    def _hour(self, m: re.Match) -> str:
        hour = int(m.group(1))
        if hour > 23:
            return m.group(0)
        return f" {self.integer(hour)} giờ "

    def _full_date(self, m: re.Match) -> str:
        prefix = m.group(1) or "ngày"
        day, month, year = int(m.group(2)), int(m.group(3)), m.group(4)
        if not (1 <= day <= 31 and 1 <= month <= 12):
            return m.group(0)
        value = int(year)
        if len(year) == 2:
            value += 2000 if value < 50 else 1900
        return (f" {prefix} {self.integer(day)} tháng {self.month_word(month)} "
                f"năm {self.integer(value)} ")

    def _day_month(self, m: re.Match) -> str:
        day, month = int(m.group(2)), int(m.group(3))
        if not (1 <= day <= 31 and 1 <= month <= 12):
            return m.group(0)
        return f" {m.group(1)} {self.integer(day)} tháng {self.month_word(month)} "

    def _month_year(self, m: re.Match) -> str:
        month, year = int(m.group(1)), int(m.group(2))
        if not 1 <= month <= 12:
            return m.group(0)
        return f" tháng {self.month_word(month)} năm {self.integer(year)} "

    def _ordinal(self, m: re.Match) -> str:
        value = int(m.group(1))
        if value == 1:
            word = "nhất"
        elif value == 4:
            word = self.four
        else:
            word = self.integer(value)
        return f" thứ {word} "

    def _roman(self, m: re.Match) -> str:
        value = roman_to_int(m.group(2))
        if value is None or value > 200:
            return m.group(0)
        return f" {m.group(1)} {self.integer(value)} "

    def _percent(self, m: re.Match) -> str:
        return f" {self.number_token(m.group(1))} phần trăm "

    def _currency_prefix(self, m: re.Match) -> str:
        name = self.symbols.get(m.group(1), "").strip() or "đơn vị"
        return f" {self.number_token(m.group(2))} {name} "

    def _currency_suffix(self, m: re.Match) -> str:
        name = self.currency.get(m.group(2).lower(), m.group(2))
        return f" {self.number_token(m.group(1))} {name} "

    def _temperature(self, m: re.Match) -> str:
        scale = "xê" if m.group(2).lower() == "c" else "ép"
        return f" {self.number_token(m.group(1))} độ {scale} "

    def _unit(self, m: re.Match) -> str:
        name = self.units.get(m.group(2).lower(), m.group(2))
        return f" {self.number_token(m.group(1))} {name} "

    def _range(self, m: re.Match) -> str:
        return f" {self.number_token(m.group(1))} đến {self.number_token(m.group(2))} "

    def _fraction(self, m: re.Match) -> str:
        num, den = int(m.group(1)), int(m.group(2))
        if den == 0:
            return m.group(0)
        joiner = "phần" if num < den and den <= 10 else "trên"
        return f" {self.integer(num)} {joiner} {self.integer(den)} "

    def _plain_number(self, m: re.Match) -> str:
        return f" {self.number_token(m.group(0))} "

    def _expand_abbrev(self, m: re.Match) -> str:
        return f" {self.abbrev.get(m.group(1).lower(), m.group(1))} "

    def _spell_acronym(self, m: re.Match) -> str:
        token = m.group(1)
        spelled = [self.letters.get(c.lower(), c) for c in token]
        return " " + " ".join(spelled) + " "

    def _pre_clean(self, text: str) -> str:
        t = unicodedata.normalize("NFC", str(text))
        for ch in (" ", " ", " "):
            t = t.replace(ch, " ")
        for ch in ("​", "‌", "‍", "﻿", "­"):
            t = t.replace(ch, "")
        t = re.sub(r"[“”„‟«»]", '"', t)
        t = re.sub(r"[‘’‚‛`´]", "'", t)
        t = re.sub(r"[–—―‒]", "-", t)
        t = re.sub(r"[♪♫🎵🎶]", " ", t)
        t = NON_SPEECH.sub(" ", t)
        t = re.sub(r"\.{3,}|…{2,}", "…", t)
        t = (t.replace("km²", "km2").replace("m²", "m2").replace("cm²", "cm2")
             .replace("m³", "m3").replace("cm³", "cm3"))
        t = re.sub(r"(?<=[^\W\d_])-(?=[^\W\d_])", " ", t)
        return re.sub(r"\s+", " ", t).strip()

    def _post_clean(self, text: str) -> str:
        allowed = set(self.punct) | {" "}
        chars = []
        for ch in text:
            if ch in allowed:
                chars.append(ch)
            elif ch.isalpha() and "LATIN" in unicodedata.name(ch, ""):
                chars.append(ch)
            else:
                chars.append(" ")
        t = "".join(chars)
        t = re.sub(r"\s+([,.!?…;:])", r"\1", t)
        t = re.sub(r"([,.!?…])(?:\s*[,.!?…])+", r"\1", t)
        t = re.sub(r"([,.!?…])(?=[^\s])", r"\1 ", t)
        t = re.sub(r"\s+", " ", t).strip(" ,;:-")
        if self.lowercase:
            t = t.lower()
        if t and t[-1] not in ".!?…":
            t += "."
        return t

    def __call__(self, text: str) -> str:
        t = self._pre_clean(text)
        if not t:
            return ""
        t = self.abbrev_re.sub(self._expand_abbrev, t)
        for pattern, handler in self.rules:
            t = pattern.sub(handler, t)
        if self.spell_acronyms:
            t = self.acronym_re.sub(self._spell_acronym, t)
        for symbol, replacement in self.symbols.items():
            if symbol in t:
                t = t.replace(symbol, replacement)
        if any(c.isdigit() for c in t):
            t = re.sub(r"\d+", lambda m: f" {self.digit_by_digit(m.group(0))} ", t)
        return self._post_clean(t)


def build(tcfg: dict | None = None) -> Normalizer:
    return Normalizer(tcfg)
