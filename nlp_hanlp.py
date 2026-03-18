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

analyzer = AnalyzerEngine(
    nlp_engine=nlp_engine,
    registry=registry,
    supported_languages=["zh"],
)

anonymizer = AnonymizerEngine()

cn_text = "白雅宁是一位43岁的女性口腔卫生师，现居住于黑龙江省哈尔滨市南岗区中山路123号，可通过邮箱baiyaning@163.com或手机13945671234联系。她的身份证号码为230103198008273629。近期她出现不明肿块、持续疲劳和体重下降等症状，经诊断为癌症。目前正在韩雪梅医生的指导下使用青霉素进行治疗。白雅宁的信用评分为850分，年收入为56万元人民币。最近的交易记录包括一笔央行内部资金划转。"

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
