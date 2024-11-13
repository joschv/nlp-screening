import logging
import sys
from logging import StreamHandler
from os import mkdir, startfile
from pathlib import Path

import nltk
from datasets import load_dataset, concatenate_datasets, Dataset

from cross_encoding import classify_deep_learning_method
from dataset_definitions import IS_ON_TOPIC, IS_ON_TOPIC_CONFIDENCE, USES_DEEP_LEARNING, USES_DEEP_LEARNING_CONFIDENCE, \
    USES_DEEP_LEARNING_CONTEXT, IS_ON_TOPIC_CONTEXT, DEEP_LEARNING_METHOD_LABEL_CONFIDENCE
from method_filter import method_filter
from semantic_search import extractive_question_answering
from topic_filter import topic_filter, topic_journal_heuristics

# RUN CONFIG
DEBUG = True
BATCH_SIZE = 16
out_path = f'./out'
# DATASET MODIFICATION OPTIONS
REMOVE_OFFTOPIC = True
EXPORT_TOPIC_EDGE_CASES = True
REMOVE_NON_DEEP_LEARNING = True
EXPORT_DEEP_LEARNING_EDGE_CASES = True
WITH_CONFIDENCES = False
WITH_CONTEXTS = False
# TOPIC FILTER OPTIONS
TOPIC_FILTER_CONFIDENCE_THRESHOLD = 0.075
TOPIC_HEURISTICS_TITLE_KEYWORDS = ['covid']
TOPIC_HEURISTICS_JOURNAL_KEYWORDS = ['virus', 'virology', 'epidemiology', 'epidemics']
TOPIC_QUERY_BLUEPRINT = 'is the study about {}?'
TOPICS_LABEL = 'virology, epidemiology, covid, covid-19'
TOPIC_FILTER_LABELS = ['epidemiology', 'epidemics', 'pandemics', 'covid', 'virus', 'virology']
# METHOD FILTER OPTIONS
USES_DL_CONFIDENCE_THRESHOLD = 0.075
METHOD_QUERY_BLUEPRINT = 'the method of the study does include the usage of {}?'
METHOD_DEEP_LEARNING_QUERY = 'deep learning or ai models'
DEEP_LEARNING_METHOD = 'deep-learning, multi-layer neural-nets or similar machine-learning-models'
COMP_VISION_METHOD = 'computer-vision or image-classification or image-segmentation'
TEXT_MINING_METHOD = 'text-mining'

def run_screening():
    """
    Run NLP-screening on the given dataset.
    First heuristics are utilized to quickly qualify studies as on-topic (by keywords in Journal & Title)
    and thereby reduce computation costs.
    Then Extractive Question Answering is used to determine whether there are text passages that are relevant to the
    specified topics. If there are none, the study is disqualified as off-topic, otherwise Zero-Shot-Classification
    is used to classify as either on- or off-topic. All studies deemed off-topic are excluded (if desired).
    The same procedure of Extractive QA and then Zero-Shot-Classification is used to classify deep learning usage.
    Finally, the used deep learning method is classified using the Zero-Shot-Classification pipeline
    of a NLI CrossEncoder model.
    """
    # init
    logging.basicConfig(
        filename='basic.log',
        encoding='utf-8',
        level=logging.INFO,
        filemode='w',
        format='<%(asctime)s> [%(levelname)s]: %(message)s',
    )
    logging.getLogger().addHandler(StreamHandler(sys.stdout))  # add sys out to logging outputs

    nltk.download('punkt_tab')

    if not Path(out_path).exists():
        print(f'Creating output dir.. ({Path(out_path).resolve()}')
        mkdir(out_path)

    # import dataset
    dataset = load_dataset('csv', data_files='data/collection_with_abstracts.csv')['train']

    # for debugging purposes take a smaller subset
    if DEBUG:
        dataset = dataset.shuffle(seed=42)
        dataset = dataset.take(250)
        print('Debug subset:', dataset)

    total_num_of_studies = dataset.num_rows
    print(f'Dataset initially contains information about a total of {total_num_of_studies} studies.')

    def info_msg(msg: str, **kwargs):
        ds: Dataset = kwargs.get('dataset', dataset)
        logging.info(f'{msg}, {ds.num_rows / total_num_of_studies:.2%} of total studies. {ds if DEBUG else ""}')

    # FILTER BY TOPIC

    # apply topic heuristics method
    topic_journal_heuristics_fn = topic_journal_heuristics(
        title_kws=TOPIC_HEURISTICS_TITLE_KEYWORDS,
        journal_kws=TOPIC_HEURISTICS_JOURNAL_KEYWORDS
    )
    dataset = dataset.map(
        topic_journal_heuristics_fn,
        batched=False
    )
    # export topic heuristics results for feedback loop
    print(f'Export dataset after applying topic heuristics for feedback loop.')
    dataset.to_csv(f'{out_path}/topic_heuristics.csv')

    # semantic search/extractive QA
    topic_qa_fn = extractive_question_answering(
        query=TOPIC_QUERY_BLUEPRINT.format(TOPICS_LABEL),
        dict_key=IS_ON_TOPIC_CONTEXT,
    )
    dataset = dataset.map(
        topic_qa_fn,
        batched=True,
        batch_size=BATCH_SIZE
    )
    if REMOVE_OFFTOPIC:
        # remove the studies that do not have any phrases that are relevant to the method query
        dataset = dataset.filter(
            lambda r: r[IS_ON_TOPIC_CONTEXT] != ''
        )
        info_msg('After removal of studies that have no relevant phrases to topics')

    # classify on/off-topic
    # define topic filter batch classification method
    topic_filter_fn = topic_filter(topics=TOPIC_FILTER_LABELS)

    # split studies by previously deemed ontopic or not
    bypass_subset = dataset.filter(lambda r: r[IS_ON_TOPIC] is True)
    filter_subset = dataset.filter(lambda r: r[IS_ON_TOPIC] is False)
    info_msg('Share of studies deemed on-topic by topic heuristics  (bypassing topic filter)', dataset=bypass_subset)

    # run topic filter method on unclassified (by topic heuristics) data
    filter_subset = filter_subset.map(
        topic_filter_fn,
        batched=True,
        batch_size=BATCH_SIZE,
    )

    # merge previously split dataset
    dataset = concatenate_datasets([bypass_subset, filter_subset])

    if EXPORT_TOPIC_EDGE_CASES:
        # keep studies that were not confidently ruled out for further human exploration as a byproduct
        topic_edge_cases_file = f'{out_path}/on_topic_edge_cases.csv'

        possibly_ontopic = dataset.filter(
            lambda r: r[IS_ON_TOPIC] is True and r[IS_ON_TOPIC_CONFIDENCE] < TOPIC_FILTER_CONFIDENCE_THRESHOLD
        )
        possibly_ontopic.to_csv(topic_edge_cases_file)
        info_msg(f'Share of studies deemed edge cases by topic filter '
                 f'(review at "{Path(topic_edge_cases_file).resolve()})"', dataset=possibly_ontopic)

    if REMOVE_OFFTOPIC:
        # remove studies that did not pass the topic filter to reduce further computation costs
        dataset = dataset.filter(
            lambda r: r[IS_ON_TOPIC] is True and r[IS_ON_TOPIC_CONFIDENCE] > TOPIC_FILTER_CONFIDENCE_THRESHOLD
        )
        info_msg('Share of studies deemed on-topic by topic filtering')

    # FILTER BY METHOD

    # semantic search / extractive QA
    deep_learning_ss_fn = extractive_question_answering(
        query=METHOD_QUERY_BLUEPRINT.format(METHOD_DEEP_LEARNING_QUERY),
        dict_key=USES_DEEP_LEARNING_CONTEXT
    )

    dataset = dataset.map(
        deep_learning_ss_fn,
        batched=True,
        batch_size=BATCH_SIZE
    )

    # remove the studies that do not have any phrases that are relevant to the method query
    dataset = dataset.filter(
        lambda r: r[USES_DEEP_LEARNING_CONTEXT] != ''
    )
    info_msg('After removal of those without relevant phrases to deep learning method')

    # configure method filter
    deep_learning_filter_fn = method_filter(
        method=DEEP_LEARNING_METHOD,
        method_key='deep_learning'
    )

    # run method filter method on dataset
    dataset = dataset.map(
        deep_learning_filter_fn,
        batched=True,
        batch_size=BATCH_SIZE,
    )

    if EXPORT_DEEP_LEARNING_EDGE_CASES:
        # keep studies that were not confidently ruled out for further human exploration as a byproduct
        uses_dl_edge_cases_file = f'{out_path}/deep_learning_usage_edge_cases.csv'

        possibly_uses_deep_learning = dataset.filter(
            lambda r: r[USES_DEEP_LEARNING] is True and r[USES_DEEP_LEARNING_CONFIDENCE] < USES_DL_CONFIDENCE_THRESHOLD
        )
        possibly_uses_deep_learning.to_csv(uses_dl_edge_cases_file)
        info_msg(f'Share of studies deemed edge cases by topic filter '
                 f'(review at "{Path(uses_dl_edge_cases_file).resolve()})"', dataset=possibly_uses_deep_learning)

    if REMOVE_NON_DEEP_LEARNING:
        # remove studies that did not pass the method filter to reduce further computation costs
        dataset = dataset.filter(lambda r: r[USES_DEEP_LEARNING] is True and
                                           r[USES_DEEP_LEARNING_CONFIDENCE] > USES_DL_CONFIDENCE_THRESHOLD)

        info_msg('Share of studies after deep learning usage filter')


    # CLASSIFY DEEP LEARNING METHOD

    # natural language inference cross encoder: assign method label
    classify_deep_learning_method_fn = classify_deep_learning_method(
        computer_vision_label=COMP_VISION_METHOD,
        text_mining_label=TEXT_MINING_METHOD,
        other_label='no computer vision and no text mining',
    )
    dataset = dataset.map(
        classify_deep_learning_method_fn,
        batched=True,
        batch_size=BATCH_SIZE
    )

    info_msg('Final dataset')
    columns_to_remove = []
    if not WITH_CONFIDENCES:
        columns_to_remove += [IS_ON_TOPIC_CONFIDENCE, USES_DEEP_LEARNING_CONFIDENCE,
                              DEEP_LEARNING_METHOD_LABEL_CONFIDENCE]
    if not WITH_CONTEXTS:
        columns_to_remove += [IS_ON_TOPIC_CONTEXT, USES_DEEP_LEARNING_CONTEXT]
    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)

    dataset.to_csv(f'{out_path}/out.csv')

    # open output folder
    startfile(Path(out_path))

if __name__ == '__main__':
    run_screening()
