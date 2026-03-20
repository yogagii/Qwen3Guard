import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from hanlp_engine import HanLPNlpEngine, HanLPRecognizer
import hanlp

# --- Chinese analyzer (HanLP) ---
tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
ner = hanlp.load(hanlp.pretrained.ner.MSRA_NER_ELECTRA_SMALL_ZH)
hanlp_model = hanlp.pipeline().append(tok, output_key="tok").append(
    ner, input_key="tok", output_key="ner"
)

nlp_engine = HanLPNlpEngine(hanlp_model)
recognizer = HanLPRecognizer(nlp_engine)

zh_registry = RecognizerRegistry(supported_languages=["zh"])
zh_registry.add_recognizer(recognizer)

zh_analyzer = AnalyzerEngine(
    nlp_engine=nlp_engine,
    registry=zh_registry,
    supported_languages=["zh"],
)

# --- English analyzer (spaCy) ---
spacy_engine = NlpEngineProvider(nlp_configuration={
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
}).create_engine()

en_registry = RecognizerRegistry(supported_languages=["en"])
en_registry.load_predefined_recognizers(nlp_engine=spacy_engine)

en_analyzer = AnalyzerEngine(
    nlp_engine=spacy_engine,
    registry=en_registry,
    supported_languages=["en"],
)

anonymizer = AnonymizerEngine()

cn_text = "陶立轩是一位62岁的男性软件开发工程师。他居住在浙江省杭州市西湖区文三路826号，持有身份证号330106196012139416。"
en_text = "My IBAN is GB59IFUE40226315499137"

mix_text = cn_text + en_text
print(mix_text)

results_chinese = zh_analyzer.analyze(text=mix_text, language="zh")
print("results_chinese:")
print(results_chinese)

anonymized_cn_text = anonymizer.anonymize(text=mix_text, analyzer_results=results_chinese)
print("anonymized_cn_text:")
print(anonymized_cn_text)

results_english = en_analyzer.analyze(text=mix_text, language="en")
print("results_english:")
print(results_english)

anonymized_en_text = anonymizer.anonymize(text=mix_text, analyzer_results=results_english)
print("anonymized_en_text:")
print(anonymized_en_text)

results = results_chinese + results_english
print("combined results:")
print(results)

anonymized_text = anonymizer.anonymize(text=mix_text, analyzer_results=results)
print("anonymized_text:")           
print(anonymized_text)
