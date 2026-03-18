import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import spacy
from gliner import GLiNER
from gliner.data_processing.tokenizer import SpaCyTokenSplitter
from presidio_analyzer.nlp_engine import NlpArtifacts, NlpEngine

try:
    from presidio_analyzer import EntityRecognizer, RecognizerResult
except Exception:
    try:
        from presidio_analyzer.recognizer_registry import EntityRecognizer, RecognizerResult
    except Exception:
        EntityRecognizer = object
        RecognizerResult = None


SUPPORTED_ENTITIES = [
    "PERSON",
    "LOCATION",
    "ORGANIZATION",
    "AGE",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "ID",
    "GENDER",
    "PROFESSION",
    "DIAGNOSIS",
    "MEDICATION",
    "TRANSACTION",
    "INCOME",
    "CREDIT_SCORE",
]

GLINER_LABEL_MAPPING = {
    "person": "PERSON",
    "location": "LOCATION",
    "organization": "ORGANIZATION",
    "age": "AGE",
    "email address": "EMAIL_ADDRESS",
    "phone number": "PHONE_NUMBER",
    "personal identification number": "ID",
    "gender": "GENDER",
    "profession": "PROFESSION",
    "medical condition": "DIAGNOSIS",
    "medication": "MEDICATION",
    "transaction": "TRANSACTION",
    "income": "INCOME",
    "credit score": "CREDIT_SCORE",
}


class GLiNERNlpEngine(NlpEngine):
    def __init__(
        self,
        model_name: str,
        labels: List[str],
        label_mapping: Optional[Dict[str, str]] = None,
        threshold: float = 0.3,
        language: str = "zh",
    ):
        self.model_name = model_name
        self.labels = labels
        self.label_mapping = label_mapping or GLINER_LABEL_MAPPING
        self.threshold = threshold
        self.language = language
        self.model = None
        self.nlp = None
        self.words_splitter = SpaCyTokenSplitter(lang=language)

    def load(self) -> None:
        load_start = time.time()
        print(f"Loading GLiNER model from {self.model_name} on CPU...", flush=True)
        self.model = GLiNER.from_pretrained(self.model_name, map_location="cpu")
        # The downloaded config defaults to stanza for multilingual tokenization,
        # which is both slow here and broken on this machine for zh-hans.
        self.model.data_processor.words_splitter = self.words_splitter
        if hasattr(self.model, "config"):
            self.model.config.words_splitter_type = "spacy"
        self.nlp = spacy.blank(self.language)
        print(f"GLiNER model loaded in {time.time() - load_start:.1f}s", flush=True)

    def is_loaded(self) -> bool:
        return self.model is not None and self.nlp is not None

    def get_supported_entities(self) -> List[str]:
        return sorted(set(self.label_mapping.values()))

    def get_supported_languages(self) -> List[str]:
        return [self.language]

    def process_text(self, text: str, language: str) -> NlpArtifacts:
        if not self.is_loaded():
            self.load()
        if language != self.language:
            raise ValueError(f"Unsupported language {language!r}, expected {self.language!r}")

        predict_start = time.time()
        print("Running GLiNER inference...", flush=True)
        predictions = self.model.predict_entities(
            text,
            self.labels,
            threshold=self.threshold,
        )
        print(f"GLiNER inference finished in {time.time() - predict_start:.1f}s", flush=True)

        doc = self.nlp(text)
        spans = []
        parsed_entities = []
        scores = []

        for pred in predictions:
            raw_label = pred.get("label")
            mapped_label = self.label_mapping.get(raw_label)
            start = pred.get("start")
            end = pred.get("end")
            score = float(pred.get("score", self.threshold))

            if not mapped_label or start is None or end is None or start >= end:
                continue

            span = doc.char_span(start, end, label=mapped_label, alignment_mode="expand")
            if span is None:
                continue

            spans.append(span)
            scores.append(score)
            parsed_entities.append(
                {
                    "text": text[start:end],
                    "type": mapped_label,
                    "start": start,
                    "end": end,
                    "score": score,
                }
            )

        artifacts = NlpArtifacts(
            entities=spans,
            tokens=doc,
            tokens_indices=[token.idx for token in doc],
            lemmas=[token.lemma_ if token.lemma_ else token.text for token in doc],
            nlp_engine=self,
            language=language,
            scores=scores,
        )
        artifacts.gliner_entities = parsed_entities
        return artifacts

    def process_batch(
        self,
        texts: Iterable[str],
        language: str,
        batch_size: int = 1,
        n_process: int = 1,
        **kwargs,
    ) -> Iterator[Tuple[str, NlpArtifacts]]:
        for text in texts:
            yield text, self.process_text(text=text, language=language)

    def is_stopword(self, word: str, language: str) -> bool:
        if not self.is_loaded():
            self.load()
        return self.nlp.vocab[word].is_stop

    def is_punct(self, word: str, language: str) -> bool:
        if not self.is_loaded():
            self.load()
        return self.nlp.vocab[word].is_punct


class GLiNERRecognizer(EntityRecognizer):
    def __init__(
        self,
        nlp_engine: GLiNERNlpEngine,
        supported_entities: Optional[List[str]] = None,
        name: str = "GLiNERRecognizer",
        score_threshold: float = 0.3,
    ):
        entities = supported_entities or nlp_engine.get_supported_entities()
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
        nlp_artifacts: Optional[NlpArtifacts] = None,
    ):
        if RecognizerResult is None:
            return []

        if nlp_artifacts is None or not hasattr(nlp_artifacts, "gliner_entities"):
            nlp_artifacts = self.nlp_engine.process_text(text=text, language="zh")

        requested = set(entities or self.supported_entities)
        results = []
        for ent in nlp_artifacts.gliner_entities:
            if ent["type"] not in requested or ent["score"] < self.score_threshold:
                continue
            results.append(
                RecognizerResult(
                    entity_type=ent["type"],
                    start=ent["start"],
                    end=ent["end"],
                    score=ent["score"],
                )
            )
        return results
