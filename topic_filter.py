from cross_encoding import classify_yes_no


def is_about_virology_and_or_epidemics(text: str) -> tuple[bool, float]:
    positive = 'about virology or epidemiology'
    negative = 'not about virology and not about epidemiology'

    topic_result = classify_yes_no(
        text=text,
        positive=positive,
        negative=negative
    )
    return topic_result
