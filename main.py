import csv

from confidence_score import confidence_score, ConfidenceMethod, normalize_overscore
from cross_encoding import classify
from csv_io import import_collection
from method_filter import includes_deep_learning_tech
from semantic_search import semantic_filter
from topic_filter import is_about_topics

c = import_collection('data/collection_with_abstracts.csv')
total_entries = len(c)
print(f'Starting analysis of collection of {total_entries} entries..')

OUTPUT_FILE = 'out.csv'
STOP_AFTER = 12
current_progress = 0

with open(OUTPUT_FILE, 'w+', newline='', encoding='utf-8') as f:
    wrote_header = False

    while len(c) > 0:
        current_progress += 1
        print(f'Analysing entry #{current_progress}')
        r = c.pop(0)
        print(r)

        # PREPARE DATA
        title = r.get('Title', '')
        abstract = r.get('Abstract', '').strip()

        # possible heuristics approach:
        # publications before 2000 are unlikely to use deep learning
        # as computation limitations were gatekeeping deep learning success
        publication_year = r.get('Publication Year')
        if len(publication_year) == 4:
            publication_year = int(publication_year)
            if publication_year < 2000:
                print('too old for deep learning?')

        if len(abstract) == 0:
            text = title
        else:
            if 'METHOD:' in abstract:
                # some papers have a dedicated paragraph about method
                # could take paragraph about method? until \n or RESULT(S): or CONCLUSION:
                pass
            text = abstract

        text = text.replace('\n', ' ')
        phrases = text.split('. ')  # todo: improve sentence segmentation (spacy?)


        # FILTER OUT
        TOPICS = ['virology and viral infections', 'epidemics', 'covid']
        FILTER_CONFIDENCE_THRESHOLD = 0.2

        # filter by topic

        # heuristics approach: if COVID in title, then it is about epidemics?
        if 'covid' in title.lower():
            zsc_topic_score = True, 0.99
        else:
            zsc_topic_score = is_about_topics(text, topics=TOPICS)

            if zsc_topic_score[0] is False and zsc_topic_score[1] > FILTER_CONFIDENCE_THRESHOLD:
                # filter out studies that are confidently not about the topic
                continue


        # filter by method
        zsc_method_score = includes_deep_learning_tech(text)

        if zsc_method_score[0] is False and zsc_method_score[1] > FILTER_CONFIDENCE_THRESHOLD:
            # filter out studies that are confidently not about the topic
            pass  # todo: continue to discard


        # SEMANTIC SEARCH FOR MOST RELEVANT PHRASES
        METHODS = (
            'deep-learning, multi-layer neural nets or similar machine-learning and artificial-intelligence technologies',
            'image-classification, image-segmentation or other computer-vision technologies',
            'text-mining'
        )

        queries = [f'did the method of the study include the usage of {m}?' for m in METHODS]

        relevant_phrases_per_query = []
        relevancy_scores = []
        relevant_texts = []

        # run semantic searches
        for q in queries:
            relevant_phrases = semantic_filter(
                query=q,
                texts=phrases,
                top_k=4,
                min_relevance=0.2
            )
            combined_relevancy_score = 0.
            combined_relevant_text = ""

            # collect results across relevant phrases
            for (relevancy, phrase) in relevant_phrases:
                combined_relevancy_score += relevancy
                combined_relevant_text += f'{phrase}. '

            relevancy_scores += [combined_relevancy_score]
            relevant_texts += [combined_relevant_text]
            relevant_phrases_per_query += [relevant_phrases]


        results = {
            'queries': queries,
            'relevant_phrases': relevant_phrases_per_query,
            'relevancy_sums': relevancy_scores,
            'relevant_texts': relevant_texts
        }
        print("results", results)

        # todo: compare classification result using relevant phrases to overall phrases
        zsc_method_score_2 = includes_deep_learning_tech(results['relevant_texts'][0])
        print("compare zsc method scores", zsc_method_score, zsc_method_score_2)

        METHOD_CLASSIFICATION_CANDIDATES = [
            'computer vision or image segmentation or image classification',
            'text mining',
            'no computer vision and no text mining'
        ]
        METHOD_LABELS = [
            'computer vision',
            'text mining',
            'other'
        ]

        classification_result = classify(
            text=text,
            candidate_labels=METHOD_CLASSIFICATION_CANDIDATES
        )
        classification_result.pop('sequence')

        # special case: computer vision and text mining scores very similar and higher than negative => both
        if (classification_result['labels'][:2] == METHOD_CLASSIFICATION_CANDIDATES[:2] and
                classification_result['scores'][0] * 0.95 < classification_result['scores'][1]):
                # very similar
                deep_learning_method = 'both'
                dlm_confidence = confidence_score(
                    normalized_overscore=(sum(classification_result['scores'][:2])-(2/3))/(1/3),
                    confidence_method=ConfidenceMethod.SIN
                )
        else:
            deep_learning_method = METHOD_LABELS[METHOD_CLASSIFICATION_CANDIDATES.index(classification_result['labels'][0])]
            dlm_confidence = confidence_score(
                normalized_overscore=normalize_overscore(classification_result['scores'][0], 3),
                confidence_method=ConfidenceMethod.LOG
            )
        print(classification_result)
        print(f'method classification: {deep_learning_method}, {dlm_confidence:.0%}')

        print('\n')

        r.pop('Abstract')
        out = r | {
            'about_virology_or_epidemics': zsc_topic_score[0],
            'about_virology_or_epidemics_confidence': f'{zsc_topic_score[1]:.0%}',
            'about_deep_learning': zsc_method_score[0],
            'about_deep_learning_confidence': f'{zsc_method_score[1]:.0%}',
            'deep_learning_method': deep_learning_method,
            'deep_learning_method_confidence': f'{dlm_confidence:.0%}',
            'about_deep_learning_context': relevant_texts[0],
        }

        if not wrote_header:
            writer = csv.DictWriter(f, fieldnames=out.keys())
            writer.writeheader()
            wrote_header = True

        writer.writerow(out)

        if current_progress >= STOP_AFTER:
            break
