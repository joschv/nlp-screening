import torch

from batch_helpers import get_texts
from confidence_score import confidence_score, normalize_overscore
from cross_encoding import zero_shot_classifier


def method_filter(method: str, method_key: str):
    """
    Filter method to classify whether studies implemented given methods using a NLI zero-shot-classifier.
    :param method: Method which a study should use to pass the filter.
    :param method_key: How the method is inserted as dict key.
    """
    positive_method = 'the method of the study includes the usage of {}'
    negative_method = 'the method of the study does not include the usage of {}'
    positive = positive_method.format(method)
    negative = negative_method.format(method)
    candidate_labels = [positive, negative]

    def pp(batch):
        texts = get_texts(batch=batch)

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
            f'uses_{method_key}': results,
            f'uses_{method_key}_confidence': confidence_scores
        }
    return pp
