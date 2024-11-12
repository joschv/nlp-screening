import torch

from dataset_definitions import USES_DEEP_LEARNING, USES_DEEP_LEARNING_CONFIDENCE
from confidence_score import confidence_score, normalize_overscore
from cross_encoding import classify_yes_no, zero_shot_classifier


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


def method_filter(method: str, dict_key: str):
    """
    Filter method to classify whether studies implemented given methods using a NLI zero-shot-classifier.
    :param method: Method which a study should use to pass the filter.
    """
    POSITIVE_METHOD = 'the method of the study includes the usage of {}'
    NEGATIVE_METHOD = 'the method of the study does not include the usage of {}'
    positive = POSITIVE_METHOD.format(method)
    negative = NEGATIVE_METHOD.format(method)
    candidate_labels = [positive, negative]

    def ff(batch):
        abstracts = batch['Abstract']
        titles = batch['Title']

        # if no abstract available then use title
        texts = [a if a is not None and len(a) > 0 else t for a, t in zip(abstracts, titles)]

        with torch.no_grad():
            classification_results = zero_shot_classifier(texts, candidate_labels)

        results, confidence_scores = [], []
        for cr in classification_results:
            results += [cr['labels'][0] == positive]
            confidence_scores += [confidence_score(
                normalized_overscore=normalize_overscore(
                    score=cr['scores'][0],
                    n=len(candidate_labels)
                )
            )]

        return {
            f'uses_{dict_key}': results,
            f'uses_{dict_key}_confidence': confidence_scores
        }
    return ff
