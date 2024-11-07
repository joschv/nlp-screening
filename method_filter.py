from cross_encoding import classify_yes_no


def includes_deep_learning_tech(text: str) -> tuple[bool, float]:
    if text == '':
        return False, 1.0

    positive = 'uses deep learning'
    negative = 'does not use deep learning'

    topic_result = classify_yes_no(
        text=text,
        positive=positive,
        negative=negative
    )
    return topic_result
