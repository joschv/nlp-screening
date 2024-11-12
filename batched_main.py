import nltk
from datasets import load_dataset, concatenate_datasets

from dataset_definitions import IS_ON_TOPIC, IS_ON_TOPIC_CONFIDENCE, USES_DEEP_LEARNING, USES_DEEP_LEARNING_CONFIDENCE
from method_filter import method_filter
from semantic_search import ss_eqa
from topic_filter import topic_filter, topic_journal_heuristics

nltk.download('punkt_tab')

DEBUG = True
BATCH_SIZE = 16

# import dataset
dataset = load_dataset('csv', data_files='data/collection_with_abstracts.csv')['train']

# for debugging
if DEBUG:
    dataset = dataset.shuffle()
    dataset = dataset.take(50)
    print('Debug subset:', dataset)


cur_number_of_studies = dataset.num_rows
ORIGINAL_NUMBER_OF_STUDIES = cur_number_of_studies

print(f'Dataset contains information about a total of {ORIGINAL_NUMBER_OF_STUDIES} studies.')

# FILTER BY TOPIC

# apply topic heuristics method
topic_journal_heuristics_fn = topic_journal_heuristics(
    title_kws=['covid'],
    journal_kws=['virus', 'virology', 'epidemiology', 'epidemics']
)
dataset = dataset.map(
    topic_journal_heuristics_fn,
    batched=False
)

# export topic heuristics results for feedback loop
dataset.to_csv('topic_heuristics.csv')

# define topic filter batch method
topic_filter_fn = topic_filter(topics=['epidemiology', 'epidemics', 'pandemics', 'covid', 'virus', 'virology'])

# split studies by previously deemed ontopic or not
bypass_subset = dataset.filter(lambda r: r[IS_ON_TOPIC] is True)
filter_subset = dataset.filter(lambda r: r[IS_ON_TOPIC] is False)

print(f'Share of studies deemed on-topic by topic heuristics: {bypass_subset.num_rows/ORIGINAL_NUMBER_OF_STUDIES:.2%} (bypassing topic filter)')

# run topic filter method on unclassified (by topic heuristics) data
filter_subset = filter_subset.map(
    topic_filter_fn,
    batched=True,
    batch_size=BATCH_SIZE,
)

# combine into one dataset
dataset = concatenate_datasets([bypass_subset, filter_subset])

# keep studies that were not confidently ruled out for further human exploration as a byproduct
TOPIC_FILTER_CONFIDENCE_THRESHOLD = 0.05
possibly_ontopic = dataset.filter(
    lambda r: r[IS_ON_TOPIC_CONFIDENCE] < TOPIC_FILTER_CONFIDENCE_THRESHOLD
)
possibly_ontopic.to_csv('possibly_on_topic.csv')

print(f'Share of studies deemed edge cases by topic filter: {possibly_ontopic.num_rows/ORIGINAL_NUMBER_OF_STUDIES:.2%}')

# remove studies that did not pass the topic filter to reduce further computation costs
dataset = dataset.filter(
    lambda r: r[IS_ON_TOPIC] is True and r[IS_ON_TOPIC_CONFIDENCE] > TOPIC_FILTER_CONFIDENCE_THRESHOLD
)

print('Share of studies deemed on-topic studies by topic filtering:', dataset, f'{dataset.num_rows/ORIGINAL_NUMBER_OF_STUDIES:.2%}')


# FILTER BY METHOD

QUERY_BLUEPRINT = 'the method of the study does include the usage of {}?'

# semantic search / extractive QA
deep_learning_ss_fn = ss_eqa(
    query=QUERY_BLUEPRINT.format('deep learning'),
    dict_key='uses_deep_learning_context'
)

dataset = dataset.map(
    deep_learning_ss_fn,
    batched=True,
    batch_size=BATCH_SIZE
)

# remove the studies that do not have any phrases that are relevant to the method query
dataset = dataset.filter(
    lambda r: r['uses_deep_learning_context'] != ''
)
print('Subset after removal of studies that have no relevant phrases to deep learning method:', dataset, f'{dataset.num_rows/ORIGINAL_NUMBER_OF_STUDIES:.2%}')

# define method filter batch method
DEEP_LEARNING_METHOD = 'deep-learning, multi-layer neural nets or similar machine-learning ' \
    'and artificial-intelligence technologies'
COMP_VISION_METHOD = 'image-classification, image-segmentation or other computer-vision technologies'
NLP_METHOD = 'text-mining'

# configure method filter
deep_learning_filter_fn = method_filter(
    method=DEEP_LEARNING_METHOD,
    dict_key='deep_learning'
)

# run method filter method on dataset
dataset = dataset.map(
    deep_learning_filter_fn,
    batched=True,
    batch_size=BATCH_SIZE,
)

# remove studies that did not pass the method filter to reduce further computation costs
METHOD_FILTER_CONFIDENCE_THRESHOLD = 0.15
dataset = dataset.filter(lambda r: r[USES_DEEP_LEARNING] is True and
                                   r[USES_DEEP_LEARNING_CONFIDENCE] > METHOD_FILTER_CONFIDENCE_THRESHOLD)

print('Subset after removal of studies that are deemed to not use deep learning:', dataset, f'{dataset.num_rows/ORIGINAL_NUMBER_OF_STUDIES:.2%}')

# CLASSIFY DEEP LEARNING METHOD

# todo: classify dl method


print('Final dataset:', dataset, f'{dataset.num_rows/ORIGINAL_NUMBER_OF_STUDIES:.2%}')
dataset.to_csv('out.csv')