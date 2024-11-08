import torch
from datasets import load_dataset, concatenate_datasets

from confidence_score import confidence_score, normalize_overscore
from cross_encoding import zero_shot_classifier

DEBUG = True
BATCH_SIZE = 8

# import dataset
dataset = load_dataset('csv', data_files='collection_with_abstracts.csv')['train']

# for debugging
if DEBUG:
    dataset = dataset.shuffle()
    dataset = dataset.take(50)
    print(dataset)


# FILTER BY TOPIC

IS_ON_TOPIC = 'is_about_epidemiology_or_virology'
IS_ON_TOPIC_CONFIDENCE = 'is_about_epidemiology_or_virology_confidence'

def topic_journal_heuristics(r):
    """ Filter method to assume studies that are published in journals/books that include certain keywords
        to be about the topic. Keyword based search should be reliable in this case. """
    journal = r['Journal/Book'].lower()

    keywords = ['virus', 'virology', 'epidemiology', 'epidemics']
    for kw in keywords:
        if kw in journal:
            print(journal)
            return {
                IS_ON_TOPIC: True,
                IS_ON_TOPIC_CONFIDENCE: 0.95
            }
    return {
        IS_ON_TOPIC: False,
        IS_ON_TOPIC_CONFIDENCE: 0.0
    }

dataset = dataset.map(
    topic_journal_heuristics,
    batched=False
)

# define topic filter batch method
def topic_filter(topics: list[str]):
    positive = 'about ' + ' or about '.join(topics)
    negative = 'not about ' + ' and not about '.join(topics)
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
            IS_ON_TOPIC: results,
            IS_ON_TOPIC_CONFIDENCE: confidence_scores
        }
    return ff

# configure topic filter
topics = ['epidemiology', 'epidemics', 'pandemics', 'covid', 'virus', 'virology']
topic_filter_fn = topic_filter(topics=topics)

# split into studies previously deemed ontopic
bypass_subset = dataset.filter(lambda r: r[IS_ON_TOPIC] is True)
filter_subset = dataset.filter(lambda r: r[IS_ON_TOPIC] is False)


# run topic filter method on unclassified data
filter_subset = filter_subset.map(
    topic_filter_fn,
    batched=True,
    batch_size=BATCH_SIZE,
)

# combine into one dataset
dataset = concatenate_datasets([bypass_subset, filter_subset])

TOPIC_FILTER_CONFIDENCE_THRESHOLD = 0.15

# keep studies that were not confidently ruled out for further human exploration as a byproduct
possibly_ontopic = dataset.filter(lambda r: r[IS_ON_TOPIC_CONFIDENCE] < TOPIC_FILTER_CONFIDENCE_THRESHOLD)
possibly_ontopic.to_csv('possibly_on_topic.csv')

# remove studies that did not pass the topic filter to reduce further computation costs
dataset = dataset.filter(lambda r: r[IS_ON_TOPIC] is True and r[IS_ON_TOPIC_CONFIDENCE] > TOPIC_FILTER_CONFIDENCE_THRESHOLD)


print(dataset)
dataset.to_csv('out_b.csv', batch_size=BATCH_SIZE)
