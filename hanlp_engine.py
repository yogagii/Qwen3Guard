from typing import List, Dict, Any, Optional, Tuple

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
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "ID",
    "IP_ADDRESS",
    "MAC_ADDRESS",
    "LOCATION",
    "DATE_TIME",
    "AGE",
    "ORGANIZATION",
    "URL",
    "IBAN_CODE",
    "CRYPTO",
]


def _normalize_label(label: str) -> str:
    m = {
        "PER": "PERSON",
        "PERSON": "PERSON",
        "NR": "PERSON",
        "LOC": "LOCATION",
        "LOCATION": "LOCATION",
        "GPE": "LOCATION",
        "NS": "LOCATION",
        "ORG": "ORGANIZATION",
        "ORGANIZATION": "ORGANIZATION",
        "NT": "ORGANIZATION",
        "PHONE": "PHONE_NUMBER",
        "TEL": "PHONE_NUMBER",
        "MOBILE": "PHONE_NUMBER",
        "EMAIL": "EMAIL_ADDRESS",
        "MAIL": "EMAIL_ADDRESS",
        "WWW": "URL",
        "URL": "URL",
        "AGE": "AGE",
        "DATE": "DATE_TIME",
        "TIME": "DATE_TIME",
        "DATETIME": "DATE_TIME",
    }
    return m.get((label or "").upper(), (label or "").upper())


def _build_token_offsets(text: str, tokens: List[str]) -> List[Tuple[int, int]]:
    """Map each token to its character start/end offsets in the source text."""
    offsets: List[Tuple[int, int]] = []
    cursor = 0

    for token in tokens:
        if not token:
            offsets.append((cursor, cursor))
            continue

        start = text.find(token, cursor)
        if start < 0:
            # Fallback for occasional tokenization mismatches. Keep searching
            # globally so we can still recover a useful span when possible.
            start = text.find(token)
        if start < 0:
            raise ValueError(
                f"Token {token!r} not found in text starting from position {cursor}."
            )

        end = start + len(token)
        offsets.append((start, end))
        cursor = end

    return offsets


def _resolve_candidate_span(
    text: str,
    ent_text: Optional[str],
    start: Any,
    end: Any,
    token_offsets: List[Tuple[int, int]],
    cursor: int,
) -> Tuple[Optional[int], Optional[int], int]:
    """
    Resolve HanLP entity span to character offsets.

    HanLP pipeline NER commonly returns token spans: (entity_text, label, token_start, token_end).
    Some models may instead return char spans. We detect both.
    """

    if isinstance(start, int) and isinstance(end, int):
        # Char-span case
        if 0 <= start < end <= len(text):
            return start, end, end

        # Token-span case
        if 0 <= start < end <= len(token_offsets):
            char_start = token_offsets[start][0]
            char_end = token_offsets[end - 1][1]
            return char_start, char_end, char_end

    # Last-resort fallback: sequential string search
    if isinstance(ent_text, str) and ent_text:
        idx = text.find(ent_text, cursor)
        if idx < 0:
            idx = text.find(ent_text)
        if idx >= 0:
            end_idx = idx + len(ent_text)
            return idx, end_idx, end_idx

    return None, None, cursor


class HanLPNlpEngine:
    def __init__(self, hanlp_model: Any):
        if hanlp_model is None:
            raise ValueError("Please pass a loaded hanlp_model to HanLPNlpEngine.")
        self.model = hanlp_model
        self._loaded = True

    def load(self) -> None:
        self._loaded = self.model is not None

    def is_loaded(self) -> bool:
        return self._loaded

    def get_supported_entities(self) -> List[str]:
        return SUPPORTED_ENTITIES

    def get_supported_languages(self) -> List[str]:
        return ["zh"]

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
        # Ignore batch_size/n_process for this simple adapter, keep API compatibility
        for text in texts:
            yield self.process_text(text=text, language=language, **kwargs)

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
        try:
            raw = self.model(text)
        except Exception:
            raw = self.model([text])

        # Fallback for some standalone NER models expecting token list
        if raw in (None, [], {}):
            try:
                raw = self.model(list(text))
            except Exception:
                pass

        parsed_entities: List[Dict[str, Any]] = []
        candidates = []
        tokens: List[str] = []

        if isinstance(raw, dict):
            # pipeline output usually contains "ner"
            if "tok" in raw and isinstance(raw["tok"], list):
                tokens = raw["tok"]
            if "ner" in raw:
                candidates = raw["ner"]
            elif "entities" in raw:
                candidates = raw["entities"]
        elif isinstance(raw, list):
            candidates = raw

        token_offsets = _build_token_offsets(text, tokens) if tokens else []

        cursor = 0
        for e in candidates:
            ent_text = None
            label = None
            start = None
            end = None
            score = 0.85

            if isinstance(e, dict):
                ent_text = e.get("text") or e.get("entity")
                label = e.get("type") or e.get("label")
                start = e.get("start")
                end = e.get("end")
                score = float(e.get("score", score))
            elif isinstance(e, (list, tuple)):
                # common HanLP tuple: (text, label, start, end)
                if len(e) >= 4 and isinstance(e[0], str) and isinstance(e[1], str):
                    ent_text, label, start, end = e[0], e[1], e[2], e[3]
                    if len(e) >= 5:
                        try:
                            score = float(e[4])
                        except Exception:
                            pass

            if label is None:
                continue
            label = _normalize_label(label)
            if entities and label not in entities:
                continue

            start, end, cursor = _resolve_candidate_span(
                text=text,
                ent_text=ent_text,
                start=start,
                end=end,
                token_offsets=token_offsets,
                cursor=cursor,
            )
            if start is None or end is None:
                continue

            if isinstance(ent_text, str) and ent_text and text[start:end] != ent_text:
                # If the computed span doesn't reproduce the entity text, prefer
                # a direct search fallback. This catches token/span mismatches.
                alt_start, alt_end, cursor = _resolve_candidate_span(
                    text=text,
                    ent_text=ent_text,
                    start=None,
                    end=None,
                    token_offsets=[],
                    cursor=cursor,
                )
                if alt_start is not None and alt_end is not None:
                    start, end = alt_start, alt_end

            parsed_entities.append(
                {"start": start, "end": end, "type": label, "score": float(score)}
            )

        return {"tokens": tokens, "entities": parsed_entities}


class HanLPRecognizer(EntityRecognizer):
    def __init__(
        self,
        nlp_engine: HanLPNlpEngine,
        supported_entities: Optional[List[str]] = None,
        name: str = "HanLPRecognizer",
        score_threshold: float = 0.5,
    ):
        entities = supported_entities or SUPPORTED_ENTITIES
        try:
            super().__init__(supported_entities=entities, supported_language="zh", name=name)
        except Exception:
            self.supported_entities = entities
            self.supported_language = "zh"
            self.name = name

        self.nlp_engine = nlp_engine
        self.score_threshold = score_threshold

    def analyze(
        self,
        text: str,
        entities: Optional[List[str]] = None,
        nlp_artifacts: Optional[Dict[str, Any]] = None,
    ):
        if RecognizerResult is None:
            return []

        # Only trust nlp_artifacts if it's our expected structure
        if isinstance(nlp_artifacts, dict) and "entities" in nlp_artifacts:
            artifacts = nlp_artifacts
        else:
            artifacts = self.nlp_engine.analyze(text=text, language="zh")

        requested = set(entities or self.supported_entities)
        out = []
        for ent in artifacts.get("entities", []):
            etype = None
            start = None
            end = None
            score = 0.85

            if isinstance(ent, dict):
                etype = ent.get("type")
                start = ent.get("start")
                end = ent.get("end")
                score = float(ent.get("score", score))
            elif isinstance(ent, (list, tuple)) and len(ent) >= 4:
                # tolerate tuple format: (text, label, start, end[, score])
                etype = _normalize_label(str(ent[1]))
                start = ent[2]
                end = ent[3]
                if len(ent) >= 5:
                    try:
                        score = float(ent[4])
                    except Exception:
                        pass
            else:
                # skip unknown artifact shape (e.g., plain string)
                continue

            if (
                etype in requested
                and isinstance(start, int)
                and isinstance(end, int)
                and 0 <= start < end <= len(text)
                and score >= self.score_threshold
            ):
                out.append(
                    RecognizerResult(
                        entity_type=etype,
                        start=start,
                        end=end,
                        score=score,
                    )
                )
        return out
