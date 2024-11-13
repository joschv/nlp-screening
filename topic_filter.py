import torch

from dataset_definitions import IS_ON_TOPIC, IS_ON_TOPIC_CONFIDENCE
from confidence_score import confidence_score, normalize_overscore
from cross_encoding import classify_yes_no, zero_shot_classifier


def is_about_topics(text: str, topics: list[str] = ('virology', 'epidemiology')) -> tuple[bool, float]:
    positive = 'about ' + ' or about '.join(topics)
    negative = 'not about ' + ' and not about '.join(topics)

    topic_result = classify_yes_no(
        text=text,
        positive=positive,
        negative=negative
    )
    return topic_result


def topic_filter(topics: list[str]):
    """
    Filter method to classify whether studies are about on of the given topics using a NLI zero-shot-classifier.
    :param topics: List of topics about which a study should be to pass the filter.
    """
    positive = 'about ' + ' or about '.join(topics)
    negative = 'not about ' + ' and not about '.join(topics)
    candidate_labels = [positive, negative]

    def pp(batch):
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
            IS_ON_TOPIC: results,
            IS_ON_TOPIC_CONFIDENCE: confidence_scores
        }
    return pp


def topic_journal_heuristics(title_kws: list[str], journal_kws: list[str]):
    """ Filter method to assume studies that are published in journals/books that include certain keywords
        to be about the topic. Keyword based search should be reliable in this case. """

    def pp(r):
        title = r['Title'].lower()
        journal = r['Journal/Book'].lower()

        for kw in title_kws:
            if kw in title:
                return {
                    IS_ON_TOPIC: True,
                    IS_ON_TOPIC_CONFIDENCE: 0.95
                }

        for kw in journal_kws:
            if kw in journal:
                return {
                    IS_ON_TOPIC: True,
                    IS_ON_TOPIC_CONFIDENCE: 0.95
                }

        return {
            IS_ON_TOPIC: False,
            IS_ON_TOPIC_CONFIDENCE: 0.0
        }

    return pp
