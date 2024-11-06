from cross_encoding import classify, predict_tech
from csv_io import import_collection
from method_filter import includes_deep_learning_tech
from semantic_search import semantic_filter
from topic_filter import is_about_virology_and_or_epidemics

c = import_collection("collection_with_abstracts.csv")

for r in c[0:4]:
    print(r)

    title = r.get('Title', '')
    abstract = r.get('Abstract', '').strip()

    if len(abstract) == 0:
        text = title
    else:
        if 'METHOD:' in abstract:
            # could take paragraph about method?
            # until \n or RESULT(S): or CONCLUSION:
            pass
        text = abstract

    text = text.replace('\n', ' ')

    # act according to confidence in decision?

    # filter by topic
    zsc_topic_score = is_about_virology_and_or_epidemics(text)
    zsc_method_score = includes_deep_learning_tech(text)


    queries = [
        'did the method of the study include the usage of deep-learning or similar machine-learning and ai technologies?',
        'did the method of the study include the usage of image-classification, image-segmentation or other computer-vision technologies?',
        'did the method of the study include the usage of text-mining or other natural-language-processing technologies?',
    ]

    phrases = text.split('. ')  # todo: improve sentence segmentation (spacy?)

    relevant_phrases_per_query = []

    for q in queries:
        relevant_phrases = semantic_filter(
            query=q,
            texts=phrases,
            top_k=4,
            min_relevance=0.2
        )
        relevant_phrases_per_query += [relevant_phrases]

    print(relevant_phrases_per_query)

    # todo: compare classification result using relevant phrases to overall phrases
    zsc_method_score_2 = includes_deep_learning_tech('. '.join(relevant_phrases_per_query[0]))

    classification_result = classify(
        text=text,
        candidate_labels=['computer vision or image segmentation or image classification', 'text mining or natural language processing', 'no computer vision and no text mining']
    )
    classification_result.pop('sequence')
    print(classification_result)

    print('\n')
