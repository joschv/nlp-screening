from cross_encoding import classify


def includes_deep_learning_tech(text: str) -> bool:
    positive = 'uses deep learning'
    negative = 'does not use deep learning'

    classification_result = classify(
        text=text,
        candidate_labels=[positive, negative]
    )
    classification_result.pop('sequence')
    print(classification_result)

    labels = classification_result.get('labels')
    if labels[0] == negative:
        return False
    return True
