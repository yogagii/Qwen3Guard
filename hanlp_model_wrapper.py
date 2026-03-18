from typing import Dict, List, Optional

from presidio_evaluator import InputSample
from presidio_evaluator.models import BaseModel
from presidio_evaluator.span_to_tag import span_to_tag

from hanlp_engine import HanLPNlpEngine, HanLPRecognizer


class HanLPModelWrapper(BaseModel):
    """
    Evaluator-facing wrapper for HanLP + Presidio recognizer.

    This intentionally avoids PresidioAnalyzerWrapper/BatchAnalyzerEngine because
    the custom HanLP analyzer path does not get translated into token tags
    correctly there. Instead, we directly call HanLPRecognizer.analyze() and
    convert the returned spans to evaluator tags.
    """

    def __init__(
        self,
        recognizer: HanLPRecognizer,
        nlp_engine: HanLPNlpEngine,
        entities_to_keep: List[str] = None,
        labeling_scheme: str = "IO",
        entity_mapping: Optional[Dict[str, str]] = None,
        verbose: bool = False,
        language: str = "zh",
    ):
        super().__init__(
            entities_to_keep=entities_to_keep,
            verbose=verbose,
            labeling_scheme=labeling_scheme,
            entity_mapping=entity_mapping,
        )
        self.recognizer = recognizer
        self.nlp_engine = nlp_engine
        self.language = language
        self.name = "HanLP Model Wrapper"

        if not self.nlp_engine.is_loaded():
            self.nlp_engine.load()

    def _predict_results(self, sample: InputSample):
        return self.recognizer.analyze(
            text=sample.full_text,
            entities=self.entities,
            nlp_artifacts=None,
        )

    def predict(self, sample: InputSample, **kwargs) -> List[str]:
        results = self._predict_results(sample)

        starts = []
        ends = []
        tags = []
        scores = []

        for res in results:
            starts.append(0 if res.start is None else res.start)
            ends.append(res.end)
            tags.append(res.entity_type)
            scores.append(res.score)

        response_tags = span_to_tag(
            scheme=self.labeling_scheme,
            text=sample.full_text,
            starts=starts,
            ends=ends,
            tags=tags,
            tokens=sample.tokens,
            scores=scores,
        )
        return response_tags

    def batch_predict(self, dataset: List[InputSample], **kwargs) -> List[List[str]]:
        return [self.predict(sample, **kwargs) for sample in dataset]

    def to_log(self) -> Dict:
        data = super().to_log()
        data.update(
            {
                "language": self.language,
                "recognizer": getattr(self.recognizer, "name", "HanLPRecognizer"),
            }
        )
        return data
