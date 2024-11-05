from cross_encoding import classify


def is_about_virology_and_or_epidemics(text: str) -> bool:
    positive = 'about virology or epidemiology'
    negative = 'not about virology and not about epidemiology'
    topic_result = classify(
        text=text,
        candidate_labels=[positive, negative]
    )
    topic_result.pop('sequence')
    print(topic_result)

    labels = topic_result.get('labels')
    if labels[0] == negative:
        return False
    return True
