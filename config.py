# RUN CONFIG
DEBUG = False  # debug mode for testing with smaller dataset and more console output
DEBUG_DATASET_SIZE = 50  # if debug is True, a subset of the specified size is taken from the dataset for sped up tests
BATCH_SIZE = 16  # size of the batches that are calculated together (see https://huggingface.co/docs/datasets/about_map_batch)
out_path = f'./out'
# DATASET MODIFICATION OPTIONS
REMOVE_OFFTOPIC = True  # discard studies that are deemed to be off-topic (as soon as possible) speeds up computation
EXPORT_TOPIC_EDGE_CASES = True  # exports edge cases which are less clearly classifiable (for manual review)
REMOVE_NON_DEEP_LEARNING = True  # discard studies that are deemed to not uses deep learning (asap) speeds up computation
EXPORT_DEEP_LEARNING_EDGE_CASES = True  # exports edge cases which are less clearly classifiable (for manual review)
WITH_CONFIDENCES = True  # include confidence scores (a representation of how confident a choice was made over others) in dataset exports
WITH_CONTEXTS = True  # include context phrases scores in dataset exports (to support quicker manual review)
# TOPIC FILTER OPTIONS
TOPIC_FILTER_CONFIDENCE_THRESHOLD = 0.075  # set between 0 and 1, higher means more studies are filtered out only leaving clearly classifiable ones
TOPIC_HEURISTICS_TITLE_KEYWORDS = ['covid']  # a list of keywords which, if contained in a study title, lets a study bypass the topic filter
TOPIC_HEURISTICS_JOURNAL_KEYWORDS = ['virus', 'virology', 'epidemiology', 'epidemics']  # a list of keywords which, if contained in a study's journal/book name, lets a study bypass the topic filter
TOPIC_QUERY_BLUEPRINT = 'is the study about {}?'  # blueprint for extractive question answering query about topics
TOPICS_LABEL = 'virology, epidemiology, covid, covid-19'  # the topic labels for the extractive QA query
TOPIC_FILTER_LABELS = ['epidemiology', 'epidemics', 'pandemics', 'covid', 'virus', 'virology']  # the topic labels for topic classification
# METHOD FILTER OPTIONS
USES_DL_CONFIDENCE_THRESHOLD = 0.075  # set between 0 and 1, higher means more studies are filtered out only leaving clearly classifiable ones
METHOD_QUERY_BLUEPRINT = 'the method of the study does include the usage of {}?'  # blueprint for extractive question answering query about methods used
METHOD_DEEP_LEARNING_QUERY = 'deep learning or ai models'  # the topic labels for the extractive QA query
DEEP_LEARNING_METHOD = 'deep-learning, multi-layer neural-nets or similar machine-learning-models'  # a method labels for method classification
COMP_VISION_METHOD = 'computer-vision or image-classification or image-segmentation'  # a method labels for method classification
TEXT_MINING_METHOD = 'text-mining'  # a method labels for method classification
