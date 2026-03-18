from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

# Create configuration containing engine name and models
configuration = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "zh", "model_name": "zh_core_web_sm"},
                {"lang_code": "en", "model_name": "en_core_web_lg"}],
}

# Create NLP engine based on configuration
provider = NlpEngineProvider(nlp_configuration=configuration)
nlp_engine_with_chinese = provider.create_engine()

# Pass the created NLP engine and supported_languages to the AnalyzerEngine
analyzer = AnalyzerEngine(
    nlp_engine=nlp_engine_with_chinese, 
    supported_languages=["en", "zh"]
)
anonymizer = AnonymizerEngine()

cn_text="白雅宁是一位43岁的女性口腔卫生师，现居住于黑龙江省哈尔滨市南岗区中山路123号，可通过邮箱baiyaning@163.com或手机13945671234联系。她的身份证号码为230103198008273629。近期她出现不明肿块、持续疲劳和体重下降等症状，经诊断为癌症。目前正在韩雪梅医生的指导下使用青霉素进行治疗。白雅宁的信用评分为850分，年收入为56万元人民币。最近的交易记录包括一笔央行内部资金划转。"
# Analyze in different languages
results_chinese = analyzer.analyze(text=cn_text, language="zh")
print(results_chinese)
anonymized_cn_text = anonymizer.anonymize(text=cn_text,analyzer_results=results_chinese)
print(anonymized_cn_text)

en_text="The address of Persint is 6750 Koskikatu 25 Apt. 864\nArtilleros\n, CO\n Uruguay 64677"
results_english = analyzer.analyze(text=en_text, language="en")
print(results_english)
anonymized_en_text = anonymizer.anonymize(text=en_text,analyzer_results=results_english)
print(anonymized_en_text)