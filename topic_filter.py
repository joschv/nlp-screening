from cross_encoding import classify_yes_no


def is_about_topics(text: str, topics: list[str] = ('virology', 'epidemiology')) -> tuple[bool, float]:
    positive = 'about ' + ' or about '.join(topics)
    negative = 'not about ' + ' and not about '.join(topics)

    topic_result = classify_yes_no(
        text=text,
        positive=positive,
        negative=negative
    )
    return topic_result
