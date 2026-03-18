from typing import Dict, Any, List, Optional, Tuple
import re

try:
    from presidio_analyzer import RecognizerResult, EntityRecognizer
except Exception:
    try:
        from presidio_analyzer.recognizer_registry import RecognizerResult, EntityRecognizer
    except Exception:
        RecognizerResult = None
        EntityRecognizer = object


SUPPORTED_ENTITIES = [
    "PERSON",
    "LOCATION",
    "ORGANIZATION",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "ID",
    "DATE_TIME",
    "AGE",
    "URL",
]

# GLiNER zero-shot labels -> Presidio entity type
# GLiNER takes free-text label descriptions; map them back to Presidio names.
LABEL_MAP: Dict[str, str] = {
    "person": "PERSON",
    "person name": "PERSON",
    "姓名": "PERSON",
    "人名": "PERSON",
    "患者姓名": "PERSON",
    "location": "LOCATION",
    "address": "LOCATION",
    "地点": "LOCATION",
    "地址": "LOCATION",
    "住址": "LOCATION",
    "organization": "ORGANIZATION",
    "company": "ORGANIZATION",
    "机构": "ORGANIZATION",
    "组织": "ORGANIZATION",
    "公司": "ORGANIZATION",
    "phone number": "PHONE_NUMBER",
    "手机号": "PHONE_NUMBER",
    "手机号码": "PHONE_NUMBER",
    "电话号码": "PHONE_NUMBER",
    "email": "EMAIL_ADDRESS",
    "email address": "EMAIL_ADDRESS",
    "邮箱": "EMAIL_ADDRESS",
    "电子邮箱": "EMAIL_ADDRESS",
    "id": "ID",
    "id number": "ID",
    "identity card": "ID",
    "身份证": "ID",
    "身份证号": "ID",
    "date": "DATE_TIME",
    "time": "DATE_TIME",
    "日期": "DATE_TIME",
    "时间": "DATE_TIME",
    "age": "AGE",
    "年龄": "AGE",
    "url": "URL",
    "website": "URL",
    "网址": "URL",
    "网站": "URL",
}

# Labels passed to GLiNER.predict_entities().
# Keep the list short — too many labels degrades GLiNER accuracy.
GLINER_LABELS = [
    "person name",
    "location",
    "organization",
    "phone number",
    "email address",
    "id number",
    "age",
    "url",
]


def _normalize_label(label: str) -> str:
    return LABEL_MAP.get(label.lower(), label.upper())


def _refine_person_span(text: str, start: int, end: int) -> Tuple[int, int]:
    """Trim noisy PERSON spans like '白雅宁是一位...' to a likely Chinese name."""
    if start < 0 or end > len(text) or start >= end:
        return start, end

    span = text[start:end]

    # Prefer a leading Chinese name token (2-4 chars)
    leading = re.match(r"^([\u4e00-\u9fff]{2,4})", span)
    if leading:
        name = leading.group(1)
        return start, start + len(name)

    # Or a common pattern ending with title, e.g. 韩雪梅医生
    titled = re.search(r"([\u4e00-\u9fff]{2,4})(医生|女士|先生)", span)
    if titled:
        name = titled.group(1)
        name_pos = span.find(name)
        if name_pos >= 0:
            abs_start = start + name_pos
            return abs_start, abs_start + len(name)

    return start, end


class GlinerNlpEngine:
    """
    Presidio NlpEngine adapter for GLiNER.
    GLiNER is a zero-shot NER model; we pass explicit label descriptions and
    it returns character-level spans directly.
    """

    def __init__(self, gliner_model, threshold: float = 0.5):
        self.model = gliner_model
        self.threshold = threshold
        self._loaded = gliner_model is not None

    # --- Presidio NlpEngine interface ---

    def load(self) -> None:
        self._loaded = self.model is not None

    def is_loaded(self) -> bool:
        return self._loaded

    def get_supported_entities(self) -> List[str]:
        return SUPPORTED_ENTITIES

    def get_supported_languages(self) -> List[str]:
        return ["zh", "en"]

    def process_text(self, text: str, language: str):
        return self.analyze(text=text, language=language)

    def process_batch(
        self,
        texts: List[str],
        language: str,
        batch_size: Optional[int] = None,
        n_process: Optional[int] = None,
        **kwargs,
    ):
        for text in texts:
            yield self.process_text(text=text, language=language)

    def is_stopword(self, word: str, language: str) -> bool:
        return False

    def is_punct(self, word: str, language: str) -> bool:
        return False

    def analyze(
        self,
        text: str,
        entities: Optional[List[str]] = None,
        logging_enable: bool = False,
        language: str = "zh",
    ) -> Dict[str, Any]:
        """
        Call GLiNER to detect entities in text.

        GLiNER API:
            model.predict_entities(text, labels, threshold=0.5)
            -> [{"start": int, "end": int, "text": str, "label": str, "score": float}, ...]
        """
        # Character-level tokens (Presidio doesn't strictly need them for span-based NER)
        tokens = [text[i] for i in range(len(text))]

        raw = self.model.predict_entities(text, GLINER_LABELS, threshold=self.threshold)

        parsed_entities: List[Dict[str, Any]] = []
        requested = set(entities or SUPPORTED_ENTITIES)
        for ent in raw:
            label = _normalize_label(ent.get("label", ""))
            if label not in requested:
                continue
            start = int(ent["start"])
            end = int(ent["end"])
            if label == "PERSON":
                start, end = _refine_person_span(text, start, end)
                if start >= end:
                    continue
            parsed_entities.append({
                "start": start,
                "end": end,
                "type": label,
                "score": float(ent.get("score", 1.0)),
            })

        # Deduplicate overlapping duplicates keeping the highest score.
        dedup: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
        for ent in parsed_entities:
            key = (ent["start"], ent["end"], ent["type"])
            if key not in dedup or ent["score"] > dedup[key]["score"]:
                dedup[key] = ent

        return {"tokens": tokens, "entities": list(dedup.values())}


class GlinerRecognizer(EntityRecognizer):
    def __init__(
        self,
        nlp_engine: GlinerNlpEngine,
        supported_entities: Optional[List[str]] = None,
        name: str = "GlinerRecognizer",
        supported_language: str = "zh",
    ):
        entities = supported_entities or SUPPORTED_ENTITIES
        try:
            super().__init__(
                supported_entities=entities,
                supported_language=supported_language,
                name=name,
            )
        except Exception:
            self.supported_entities = entities
            self.supported_language = supported_language
            self.name = name

        self.nlp_engine = nlp_engine

    def analyze(
        self,
        text: str,
        entities: Optional[List[str]] = None,
        nlp_artifacts: Optional[Dict[str, Any]] = None,
    ):
        if RecognizerResult is None:
            return []

        if isinstance(nlp_artifacts, dict) and "entities" in nlp_artifacts:
            artifacts = nlp_artifacts
        else:
            artifacts = self.nlp_engine.analyze(text=text, language=self.supported_language)

        requested = set(entities or self.supported_entities)
        out = []
        for ent in artifacts.get("entities", []):
            etype = ent.get("type")
            start = ent.get("start")
            end = ent.get("end")
            score = float(ent.get("score", 0.85))
            if etype not in requested or start is None or end is None:
                continue
            out.append(RecognizerResult(entity_type=etype, start=start, end=end, score=score))
        return out

    def load(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Rule-based recognizer for Chinese PII patterns
# ---------------------------------------------------------------------------

CHINESE_PATTERNS: List[tuple] = [
    # Name (2-3 chars) immediately before a professional title
    # Using {2,3} prevents greedily absorbing preceding function words like 在/正/于.
    # e.g. "目前正在韩雪梅医生" → tries all positions; only "韩雪梅" (3 chars) + "医生" succeeds.
    (re.compile(r"([\u4e00-\u9fff]{2,3})(?=医生|护士|大夫|主任|院长|主席|律师|老师|经理|女士|先生|博士|教授)"), "PERSON", 0.9),
    # Name at absolute text start or after sentence-ending punctuation, followed by 是
    # e.g. "白雅宁是一位…" → captures "白雅宁"
    # Narrow lookahead to only 是 (avoids "现居住于"/"最近的" false positives)
    (re.compile(r"(?:^|(?<=[。！？]))([\u4e00-\u9fff]{2,4})(?=是)"), "PERSON", 0.85),
    # Possessive: name + 的 + PII noun (白雅宁的信用评分, 白雅宁的身份证)
    (re.compile(r"([\u4e00-\u9fff]{2,4})(?=的(?:信用|年收|工资|账户|手机|邮箱|身份|病历|体重|年龄|姓名|地址))"), "PERSON", 0.85),
    # Chinese mobile: 1[3-9]\d{9}
    (re.compile(r"(?<![\d])1[3-9]\d{9}(?![\d])"), "PHONE_NUMBER", 0.95),
    # Chinese ID card: 18 digits, last may be X
    (re.compile(r"(?<![\d])\d{17}[\dXx](?![\d])"), "ID", 0.95),
    # Email (handles ASCII and common patterns)
    (re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"), "EMAIL_ADDRESS", 0.95),
]


class ChinesePatternRecognizer(EntityRecognizer):
    """Regex-based recognizer for Chinese PII: person names by title/context, phone, ID, email."""

    def __init__(
        self,
        supported_entities: Optional[List[str]] = None,
        name: str = "ChinesePatternRecognizer",
        supported_language: str = "zh",
    ):
        entities = supported_entities or SUPPORTED_ENTITIES
        try:
            super().__init__(
                supported_entities=entities,
                supported_language=supported_language,
                name=name,
            )
        except Exception:
            self.supported_entities = entities
            self.supported_language = supported_language
            self.name = name

    def analyze(
        self,
        text: str,
        entities: Optional[List[str]] = None,
        nlp_artifacts=None,
    ) -> List:
        if RecognizerResult is None:
            return []
        requested = set(entities or self.supported_entities)
        results = []
        seen: set = set()  # (start, end, type) to avoid duplicates
        for pattern, entity_type, score in CHINESE_PATTERNS:
            if entity_type not in requested:
                continue
            for m in pattern.finditer(text):
                # For name patterns that use capture group 1, take just the name
                if m.lastindex and m.lastindex >= 1:
                    start, end = m.start(1), m.end(1)
                else:
                    start, end = m.start(), m.end()
                key = (start, end, entity_type)
                if key in seen:
                    continue
                seen.add(key)
                results.append(
                    RecognizerResult(entity_type=entity_type, start=start, end=end, score=score)
                )
        return results

    def load(self) -> None:
        pass
