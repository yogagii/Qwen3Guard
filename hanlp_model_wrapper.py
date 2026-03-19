from typing import Dict, List, Optional

from presidio_evaluator import InputSample
from presidio_evaluator.models import BaseModel
from presidio_evaluator.span_to_tag import span_to_tag

class HanLPModelWrapper(BaseModel):
    """
    Evaluator-facing wrapper for a full Presidio AnalyzerEngine.

    This intentionally avoids PresidioAnalyzerWrapper/BatchAnalyzerEngine because
    the custom HanLP analyzer path did not get translated into token tags
    correctly there. Instead, we call analyzer_engine.analyze() directly and
    convert the returned RecognizerResults into evaluator tags.
    """

    def __init__(
        self,
        analyzer_engine,
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
        self.analyzer_engine = analyzer_engine
        self.language = language
        self.name = "HanLP Model Wrapper"

    def _predict_results(self, sample: InputSample):
        return self.analyzer_engine.analyze(
            text=sample.full_text,
            language=self.language,
            entities=self.entities,
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
                "analyzer": self.analyzer_engine.__class__.__name__,
            }
        )
        return data
