import os

# 必须在 transformers/hanlp/presidio 初始化前设置
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HF_HUB_DISABLE_XET"] = "1"   # 避免走 cas-bridge.xethub.hf.co
# os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer.nlp_engine import TransformersNlpEngine, NerModelConfiguration
# Use a pipeline as a high-level helper
#from transformers import pipeline

# pipe = pipeline("token-classification", model="StanfordAIMI/stanford-deidentifier-base")
# # Load model directly
# from transformers import AutoTokenizer, AutoModel

# tokenizer = AutoTokenizer.from_pretrained("StanfordAIMI/stanford-deidentifier-base")
# model = AutoModel.from_pretrained("StanfordAIMI/stanford-deidentifier-base")

model_config = [
    {"lang_code": "en",
     "model_name": {
         "spacy": "en_core_web_sm", # for tokenization, lemmatization
         "transformers": "StanfordAIMI/stanford-deidentifier-base"
    }
}]

# Entity mappings between the model's and Presidio's
mapping = dict(
    PER="PERSON",
    LOC="LOCATION",
    ORG="ORGANIZATION",
    AGE="AGE",
    ID="ID",
    EMAIL="EMAIL",
    DATE="DATE_TIME",
    PHONE="PHONE_NUMBER",
    PERSON="PERSON",
    LOCATION="LOCATION",
    GPE="LOCATION",
    ORGANIZATION="ORGANIZATION",
    NORP="NRP",
    PATIENT="PERSON",
    STAFF="PERSON",
    HOSP="LOCATION",
    PATORG="ORGANIZATION",
    TIME="DATE_TIME",
    HCW="PERSON",
    HOSPITAL="LOCATION",
    FACILITY="LOCATION",
    VENDOR="ORGANIZATION",
)

labels_to_ignore = ["O"]

ner_model_configuration = NerModelConfiguration(
    model_to_presidio_entity_mapping=mapping,
    alignment_mode="expand", # "strict", "contract", "expand"
    aggregation_strategy="max", # "simple", "first", "average", "max"
    labels_to_ignore = labels_to_ignore)

transformers_nlp_engine = TransformersNlpEngine(
    models=model_config,
    ner_model_configuration=ner_model_configuration)

# Transformer-based analyzer
analyzer = AnalyzerEngine(
    nlp_engine=transformers_nlp_engine, 
    supported_languages=["en"]
)

anonymizer = AnonymizerEngine()

cn_text = "白雅宁是一位43岁的女性口腔卫生师，现居住于黑龙江省哈尔滨市南岗区中山路123号，可通过邮箱baiyaning@163.com或手机13945671234联系。她的身份证号码为230103198008273629。近期她出现不明肿块、持续疲劳和体重下降等症状，经诊断为癌症。目前正在韩雪梅医生的指导下使用青霉素进行治疗。白雅宁的信用评分为850分，年收入为56万元人民币。最近的交易记录包括一笔央行内部资金划转。"

results_chinese = analyzer.analyze(text=cn_text, language="en", return_decision_process=True)
print("results_chinese:")
print(results_chinese)

anonymized_cn_text = anonymizer.anonymize(text=cn_text, analyzer_results=results_chinese)
print("anonymized_cn_text:")
print(anonymized_cn_text)

en_text="The address of Persint is 6750 Koskikatu 25 Apt. 864\nArtilleros\n, CO\n Uruguay 64677"
results_english = analyzer.analyze(text=en_text, language="en")
print(results_english)
anonymized_en_text = anonymizer.anonymize(text=en_text,analyzer_results=results_english)
print(anonymized_en_text)