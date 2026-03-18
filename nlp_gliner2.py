from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine
from gliner_engine2 import GlinerNlpEngine, GlinerRecognizer, ChinesePatternRecognizer
from gliner import GLiNER

# urchade/gliner_multi-v2.1 是支持中文在内的多语言零样本 NER 模型
# gliner_model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")
gliner_model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")

nlp_engine = GlinerNlpEngine(gliner_model, threshold=0.1)
recognizer = GlinerRecognizer(nlp_engine, supported_language="zh")

registry = RecognizerRegistry(supported_languages=["zh"])
registry.add_recognizer(recognizer)
# registry.add_recognizer(ChinesePatternRecognizer())

analyzer = AnalyzerEngine(
    nlp_engine=nlp_engine,
    registry=registry,
    supported_languages=["zh"],
)

anonymizer = AnonymizerEngine()

cn_text = "白雅宁是一位43岁的女性口腔卫生师，现居住于黑龙江省哈尔滨市南岗区中山路123号，可通过邮箱baiyaning@163.com或手机13945671234联系。她的身份证号码为230103198008273629。近期她出现不明肿块、持续疲劳和体重下降等症状，经诊断为癌症。目前正在韩雪梅医生的指导下使用青霉素进行治疗。白雅宁的信用评分为850分，年收入为56万元人民币。最近的交易记录包括一笔央行内部资金划转。"

print(cn_text)
results = analyzer.analyze(text=cn_text, language="zh")
print("results:")
print(results)

anonymized = anonymizer.anonymize(text=cn_text, analyzer_results=results)
print("anonymized:")
print(anonymized)