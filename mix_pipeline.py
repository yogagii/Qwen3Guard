import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Optional, Sequence

import hanlp
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_evaluator import InputSample
from presidio_evaluator.evaluation import Plotter, SpanEvaluator
from presidio_evaluator.experiment_tracking import get_experiment_tracker
from presidio_evaluator.models import BaseModel, PresidioAnalyzerWrapper
from presidio_evaluator.span_to_tag import span_to_tag

from hanlp_engine import HanLPNlpEngine, HanLPRecognizer
from zh_pattern_recognizers import register_zh_pattern_recognizers


def merge_results(*result_lists):
    merged = []
    seen = set()
    for results in result_lists:
        for result in results:
            key = (result.entity_type, result.start, result.end)
            if key in seen:
                continue
            seen.add(key)
            merged.append(result)
    return sorted(merged, key=lambda r: (r.start, r.end, r.entity_type))


@dataclass
class HybridAnalyzerConfig:
    project_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "data")
    dataset_name: str = "mixed_cn_en_500.json"
    token_model_version: str = "zh_core_web_sm"
    cuda_visible_devices: str = ""

    primary_backend: str = "hanlp"
    primary_language: str = "zh"
    secondary_backend: str = "spacy"
    secondary_language: str = "en"
    evaluation_language: str = "zh"

    zh_tokenizer_model: str = "COARSE_ELECTRA_SMALL_ZH"
    zh_ner_model: str = "MSRA_NER_ELECTRA_SMALL_ZH"
    en_spacy_model: str = "en_core_web_sm"

    gliner_model_name: str = "urchade/gliner_multi-v2.1"
    gliner_threshold: float = 0.5
    gliner_labels: Sequence[str] = field(
        default_factory=lambda: [
            "person name",
            "location",
            "organization",
            "phone number",
            "email address",
            "id number",
            "age",
            "url",
        ]
    )

    labeling_scheme: str = "IO"
    iou_threshold: float = 0.7
    plot_output_dir: str = "plotter_output"
    plot_beta: int = 2
    allow_missing_mappings: bool = True
    extra_entity_mappings: Dict[str, str] = field(default_factory=lambda: {"GENDER": "GENDER"})

    @property
    def dataset_path(self) -> Path:
        return self.data_dir / self.dataset_name

    @property
    def plot_output_path(self) -> Path:
        return self.project_dir / self.plot_output_dir

    def resolve_hanlp_model(self, category: str, model_name: str):
        namespace = getattr(hanlp.pretrained, category)
        return getattr(namespace, model_name)


@dataclass
class EvaluationArtifacts:
    dataset: List[InputSample]
    evaluator: SpanEvaluator
    evaluation_results: List
    results: object


class DatasetInspector:
    @staticmethod
    def get_entity_counts(dataset: List[InputSample]) -> Counter:
        entity_counter = Counter()
        for sample in dataset:
            entity_counter.update(sample.tags)
        return entity_counter

    @classmethod
    def summarize(cls, dataset: List[InputSample]) -> Dict[str, object]:
        entity_counts = cls.get_entity_counts(dataset)
        token_lengths = [len(sample.tokens) for sample in dataset]
        text_lengths = [len(sample.full_text) for sample in dataset]
        return {
            "sample_count": len(dataset),
            "entity_counts": entity_counts,
            "min_tokens": min(token_lengths),
            "max_tokens": max(token_lengths),
            "min_text_length": min(text_lengths),
            "max_text_length": max(text_lengths),
        }

    @staticmethod
    def print_summary(summary: Dict[str, object]) -> None:
        print(summary["sample_count"])
        print("Count per entity:")
        pprint(summary["entity_counts"].most_common(), compact=True)
        print(
            "\nMin and max number of tokens in dataset: "
            f"Min: {summary['min_tokens']}, Max: {summary['max_tokens']}"
        )
        print(
            "Min and max sentence length in dataset: "
            f"Min: {summary['min_text_length']}, Max: {summary['max_text_length']}"
        )

    @staticmethod
    def print_sample_tokens(sample: InputSample) -> None:
        for i, (token, tag) in enumerate(zip(sample.tokens, sample.tags)):
            print(f"{i:03d}\t{token}\t{tag}")


class AnalyzerFactory:
    def __init__(self, config: HybridAnalyzerConfig):
        self.config = config

    def build_primary_analyzer(self) -> AnalyzerEngine:
        if self.config.primary_backend == "hanlp":
            return self._build_hanlp_analyzer()
        if self.config.primary_backend == "gliner":
            return self._build_gliner_analyzer()
        raise ValueError(f"Unsupported primary backend: {self.config.primary_backend}")

    def build_secondary_analyzer(self) -> AnalyzerEngine:
        if self.config.secondary_backend != "spacy":
            raise ValueError(f"Unsupported secondary backend: {self.config.secondary_backend}")
        return self._build_spacy_analyzer()

    def _build_hanlp_analyzer(self) -> AnalyzerEngine:
        tok = hanlp.load(
            self.config.resolve_hanlp_model("tok", self.config.zh_tokenizer_model)
        )
        ner = hanlp.load(
            self.config.resolve_hanlp_model("ner", self.config.zh_ner_model)
        )
        hanlp_model = hanlp.pipeline().append(tok, output_key="tok").append(
            ner,
            input_key="tok",
            output_key="ner",
        )

        nlp_engine = HanLPNlpEngine(hanlp_model)
        recognizer = HanLPRecognizer(nlp_engine)

        registry = RecognizerRegistry(supported_languages=[self.config.primary_language])
        registry.add_recognizer(recognizer)
        register_zh_pattern_recognizers(registry)

        return AnalyzerEngine(
            nlp_engine=nlp_engine,
            registry=registry,
            supported_languages=[self.config.primary_language],
        )

    def _build_gliner_analyzer(self) -> AnalyzerEngine:
        from gliner_engine import (
            GLINER_LABEL_MAPPING,
            GLiNERNlpEngine,
            GLiNERRecognizer,
        )

        print('loading config ', self.config.gliner_model_name, GLINER_LABEL_MAPPING, list(self.config.gliner_labels))
        nlp_engine = GLiNERNlpEngine(
            model_name=self.config.gliner_model_name,
            label_mapping=GLINER_LABEL_MAPPING,
            labels=list(self.config.gliner_labels),
            threshold=self.config.gliner_threshold
        )

        print('loading recognizer')
        recognizer = GLiNERRecognizer(
            nlp_engine=nlp_engine,
            score_threshold=self.config.gliner_threshold,
        )
        print('end recognizer')

        registry = RecognizerRegistry(supported_languages=[self.config.primary_language])
        registry.add_recognizer(recognizer)
        register_zh_pattern_recognizers(registry)

        return AnalyzerEngine(
            nlp_engine=nlp_engine,
            registry=registry,
            supported_languages=[self.config.primary_language],
        )

    def _build_spacy_analyzer(self) -> AnalyzerEngine:
        spacy_engine = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [
                    {
                        "lang_code": self.config.secondary_language,
                        "model_name": self.config.en_spacy_model,
                    }
                ],
            }
        ).create_engine()

        registry = RecognizerRegistry(supported_languages=[self.config.secondary_language])
        registry.load_predefined_recognizers(nlp_engine=spacy_engine)

        return AnalyzerEngine(
            nlp_engine=spacy_engine,
            registry=registry,
            supported_languages=[self.config.secondary_language],
        )


class HybridAnalyzerModelWrapper(BaseModel):
    def __init__(
        self,
        primary_analyzer: AnalyzerEngine,
        secondary_analyzer: AnalyzerEngine,
        primary_language: str,
        secondary_language: str,
        primary_backend: str,
        secondary_backend: str,
        entities_to_keep: Optional[List[str]] = None,
        labeling_scheme: str = "IO",
        verbose: bool = False,
    ):
        super().__init__(
            entities_to_keep=entities_to_keep,
            verbose=verbose,
            labeling_scheme=labeling_scheme,
        )
        self.primary_analyzer = primary_analyzer
        self.secondary_analyzer = secondary_analyzer
        self.primary_language = primary_language
        self.secondary_language = secondary_language
        self.primary_backend = primary_backend
        self.secondary_backend = secondary_backend
        self.name = f"{primary_backend} + {secondary_backend} Model Wrapper"

    def _predict_results(self, sample: InputSample):
        primary_results = self.primary_analyzer.analyze(
            text=sample.full_text,
            language=self.primary_language,
            entities=self.entities,
        )
        secondary_results = self.secondary_analyzer.analyze(
            text=sample.full_text,
            language=self.secondary_language,
            entities=self.entities,
        )
        return merge_results(primary_results, secondary_results)

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
        return span_to_tag(
            scheme=self.labeling_scheme,
            text=sample.full_text,
            starts=starts,
            ends=ends,
            tags=tags,
            tokens=sample.tokens,
            scores=scores,
        )

    def batch_predict(self, dataset: List[InputSample], **kwargs) -> List[List[str]]:
        return [self.predict(sample, **kwargs) for sample in dataset]

    def to_log(self) -> Dict[str, object]:
        data = super().to_log()
        data.update(
            {
                "primary_backend": self.primary_backend,
                "secondary_backend": self.secondary_backend,
                "primary_language": self.primary_language,
                "secondary_language": self.secondary_language,
            }
        )
        return data


class HybridAnalyzerExperiment:
    def __init__(self, config: Optional[HybridAnalyzerConfig] = None):
        self.config = config or HybridAnalyzerConfig()
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config.cuda_visible_devices
        self.dataset: Optional[List[InputSample]] = None
        self.entities_mapping: Optional[Dict[str, str]] = None
        self.primary_analyzer: Optional[AnalyzerEngine] = None
        self.secondary_analyzer: Optional[AnalyzerEngine] = None
        self.anonymizer = AnonymizerEngine()

    def load_dataset(self) -> List[InputSample]:
        self.dataset = InputSample.read_dataset_json(
            self.config.dataset_path,
            token_model_version=self.config.token_model_version,
        )
        return self.dataset

    def summarize_dataset(self) -> Dict[str, object]:
        dataset = self._require_dataset()
        summary = DatasetInspector.summarize(dataset)
        DatasetInspector.print_summary(summary)
        return summary

    def align_dataset_entities(self) -> List[InputSample]:
        dataset = self._require_dataset()
        mapping = dict(PresidioAnalyzerWrapper.presidio_entities_map)
        mapping.update(self.config.extra_entity_mappings)
        self.entities_mapping = mapping

        self.dataset = SpanEvaluator.align_entity_types(
            dataset,
            entities_mapping=mapping,
            allow_missing_mappings=self.config.allow_missing_mappings,
        )

        print("Entities mapping:")
        pprint(mapping)
        print("\nCount per entity after alignment:")
        pprint(DatasetInspector.get_entity_counts(self.dataset).most_common(), compact=True)
        return self.dataset

    def build_analyzers(self):
        factory = AnalyzerFactory(self.config)
        self.primary_analyzer = factory.build_primary_analyzer()
        self.secondary_analyzer = factory.build_secondary_analyzer()
        return self.primary_analyzer, self.secondary_analyzer

    def describe_analyzers(self) -> None:
        primary_analyzer, secondary_analyzer = self._require_analyzers()
        print(f"Primary backend: {self.config.primary_backend}")
        pprint(primary_analyzer.get_supported_entities(self.config.primary_language), compact=True)
        print("\nLoaded recognizers for primary analyzer:")
        pprint(
            [
                rec.name
                for rec in primary_analyzer.registry.get_recognizers(
                    self.config.primary_language,
                    all_fields=True,
                )
            ],
            compact=True,
        )
        print(f"\nSecondary backend: {self.config.secondary_backend}")
        pprint(secondary_analyzer.get_supported_entities(self.config.secondary_language), compact=True)
        print("\nLoaded recognizers for secondary analyzer:")
        pprint(
            [
                rec.name
                for rec in secondary_analyzer.registry.get_recognizers(
                    self.config.secondary_language,
                    all_fields=True,
                )
            ],
            compact=True,
        )

    def inspect_sample(self, sample_index: int = 0) -> InputSample:
        dataset = self._require_dataset()
        sample = dataset[sample_index]
        DatasetInspector.print_sample_tokens(sample)
        return sample

    def debug_sample(self, sample_index: int = 0) -> Dict[str, object]:
        dataset = self._require_dataset()
        sample = dataset[sample_index]
        debug_info = self.debug_text(sample.full_text)
        debug_info["sample"] = sample
        return debug_info

    def debug_text(self, text: str, return_decision_process: bool = True) -> Dict[str, object]:
        primary_analyzer, secondary_analyzer = self._require_analyzers()

        primary_results = primary_analyzer.analyze(
            text=text,
            language=self.config.primary_language,
            return_decision_process=return_decision_process,
        )
        secondary_results = secondary_analyzer.analyze(
            text=text,
            language=self.config.secondary_language,
            return_decision_process=return_decision_process,
        )
        combined_results = merge_results(primary_results, secondary_results)
        anonymized_result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=combined_results,
        )

        print(text)
        print(f"\n{self.config.primary_backend}_results:")
        print(primary_results)
        print(f"\n{self.config.secondary_backend}_results:")
        print(secondary_results)
        print("\ncombined_results:")
        print(combined_results)
        print("\nanonymized_text:")
        print(anonymized_result)

        return {
            "text": text,
            "primary_results": primary_results,
            "secondary_results": secondary_results,
            "combined_results": combined_results,
            "anonymized_text": anonymized_result,
        }

    def anonymize_text(self, text: str):
        primary_analyzer, secondary_analyzer = self._require_analyzers()
        primary_results = primary_analyzer.analyze(
            text=text,
            language=self.config.primary_language,
        )
        secondary_results = secondary_analyzer.analyze(
            text=text,
            language=self.config.secondary_language,
        )
        combined_results = merge_results(primary_results, secondary_results)
        return self.anonymizer.anonymize(text=text, analyzer_results=combined_results)

    @staticmethod
    def print_results(results, text: str) -> None:
        for result in results:
            print(
                f"Entity: {result.entity_type}, "
                f"Text: {text[result.start:result.end]}, "
                f"Span: ({result.start}, {result.end}), "
                f"Score: {result.score}"
            )

    def create_model(self) -> HybridAnalyzerModelWrapper:
        primary_analyzer, secondary_analyzer = self._require_analyzers()
        return HybridAnalyzerModelWrapper(
            primary_analyzer=primary_analyzer,
            secondary_analyzer=secondary_analyzer,
            primary_language=self.config.primary_language,
            secondary_language=self.config.secondary_language,
            primary_backend=self.config.primary_backend,
            secondary_backend=self.config.secondary_backend,
            labeling_scheme=self.config.labeling_scheme,
        )

    def run_evaluation(self) -> EvaluationArtifacts:
        dataset = self._require_dataset()
        model = self.create_model()
        evaluator = SpanEvaluator(model=model, iou_threshold=self.config.iou_threshold)

        experiment = get_experiment_tracker()
        params = {
            "dataset_name": self.config.dataset_name,
            "model_name": evaluator.model.name,
        }
        params.update(evaluator.model.to_log())
        experiment.log_parameters(params)
        experiment.log_dataset_hash(dataset)
        if self.entities_mapping is not None:
            experiment.log_parameter(
                "entity_mappings",
                json.dumps(self.entities_mapping, ensure_ascii=False),
            )

        evaluation_results = evaluator.evaluate_all(
            dataset,
            language=self.config.evaluation_language,
        )
        results = evaluator.calculate_score(evaluation_results)

        experiment.log_metrics(results.to_log())
        entities, confmatrix = results.to_confusion_matrix()
        experiment.log_confusion_matrix(matrix=confmatrix, labels=entities)
        experiment.end()

        return EvaluationArtifacts(
            dataset=dataset,
            evaluator=evaluator,
            evaluation_results=evaluation_results,
            results=results,
        )

    @staticmethod
    def get_predicted_label_counts(evaluation_results: List) -> Counter:
        return Counter(
            tag
            for row in evaluation_results
            for tag in row.predicted_tags
            if tag != "O"
        )

    def plot_results(self, results, model_name: Optional[str] = None) -> Path:
        plotter = Plotter(
            results=results,
            model_name=model_name or self._build_model_name(),
            save_as="svg",
            beta=self.config.plot_beta,
        )
        output_folder = self.config.plot_output_path
        plotter.plot_scores(output_folder=output_folder)
        return output_folder

    def _build_model_name(self) -> str:
        return f"{self.config.primary_backend} + {self.config.secondary_backend} Model Wrapper"

    def _require_dataset(self) -> List[InputSample]:
        if self.dataset is None:
            raise ValueError("Dataset is not loaded. Call load_dataset() first.")
        return self.dataset

    def _require_analyzers(self):
        if self.primary_analyzer is None or self.secondary_analyzer is None:
            raise ValueError("Analyzers are not built. Call build_analyzers() first.")
        return self.primary_analyzer, self.secondary_analyzer


# Backward-compatible aliases for the current notebook names.
HanLPMixConfig = HybridAnalyzerConfig
HanLPMixExperiment = HybridAnalyzerExperiment
