import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine
from hanlp_engine import HanLPNlpEngine, HanLPRecognizer
import hanlp

# Build pipeline: tok -> ner (more reliable than loading NER model alone)
tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
ner = hanlp.load(hanlp.pretrained.ner.MSRA_NER_ELECTRA_SMALL_ZH)
hanlp_model = hanlp.pipeline().append(tok, output_key="tok").append(
    ner, input_key="tok", output_key="ner"
)

nlp_engine = HanLPNlpEngine(hanlp_model)
recognizer = HanLPRecognizer(nlp_engine)

registry = RecognizerRegistry(supported_languages=["zh"])
registry.add_recognizer(recognizer)

from presidio_analyzer import PatternRecognizer, Pattern

email_pattern = Pattern(
    name="email",
    regex=r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    score=0.85,
)


email_recognizer = PatternRecognizer(
    supported_entity="EMAIL_ADDRESS",
    patterns=[email_pattern],
    name="EmailRecognizer",
    supported_language="zh",
)

registry.add_recognizer(email_recognizer)

phone_pattern = Pattern(
    name="cn_phone",
    regex=r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)",
    score=0.9,
)

phone_recognizer = PatternRecognizer(
    supported_entity="PHONE_NUMBER",
    patterns=[phone_pattern],
    name="PhoneRecognizer",
    supported_language="zh",
)

registry.add_recognizer(phone_recognizer)

id_pattern = Pattern(
    name="cn_id",
    regex=r"(?<!\d)\d{17}[\dXx](?!\d)",
    score=0.95,
)

id_recognizer = PatternRecognizer(
    supported_entity="ID",
    patterns=[id_pattern],
    name="IdRecognizer",
    supported_language="zh",
)

registry.add_recognizer(id_recognizer)

analyzer = AnalyzerEngine(
    nlp_engine=nlp_engine,
    registry=registry,
    supported_languages=["zh"],
)

anonymizer = AnonymizerEngine()

cn_text = "陶立轩是一位62岁的男性软件开发工程师。他居住在浙江省杭州市西湖区文三路826号，持有身份证号330106196012139416。他的电子邮箱是taolixuan@163.com，联系电话是13857123456。他被诊断为肥胖症，症状包括体重超标、呼吸短促和关节不适。他的主治医生张敏华为他开具了布洛芬缓释胶囊。陶立轩的信用评分为608分，年收入为785,798.06元。最近有一笔来自印度法证服务的转账记录。"

print(cn_text)
results_chinese = analyzer.analyze(text=cn_text, language="zh")
print("results_chinese:")
print(results_chinese)

anonymized_cn_text = anonymizer.anonymize(text=cn_text, analyzer_results=results_chinese)
print("anonymized_cn_text:")
print(anonymized_cn_text)

# raw = hanlp_model(cn_text)
# print(raw)
# print(raw["tok"])
# print(raw["ner"])
