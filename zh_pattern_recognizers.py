from presidio_analyzer import Pattern, PatternRecognizer


def register_zh_pattern_recognizers(registry):
    """Register regex-based recognizers for Chinese text."""
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

    gender_pattern = Pattern(
        name="cn_gender",
        regex=(
            r"(男性|女性|先生|女士|小姐)"
            r"|(?:(?<=性别)|(?<=性别为)|(?<=性别是)|(?<=\d岁)|(?<=\d岁的))(男|女)"
        ),
        score=0.8,
    )

    gender_recognizer = PatternRecognizer(
        supported_entity="GENDER",
        patterns=[gender_pattern],
        name="GenderRecognizer",
        supported_language="zh",
    )

    registry.add_recognizer(gender_recognizer)