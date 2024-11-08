from sentence_transformers import CrossEncoder
from transformers import pipeline

from confidence_score import confidence_score, normalize_overscore

cross_encoder_model = CrossEncoder('cross-encoder/nli-MiniLM2-L6-H768')
zero_shot_classifier = pipeline("zero-shot-classification", model='cross-encoder/nli-MiniLM2-L6-H768')

def classify(text: str, candidate_labels: list[str] = ("computer vision", "text mining", "deep learning", "no deep learning")) -> dict:
    return zero_shot_classifier(text, candidate_labels)

def classify_batched(texts: list[str], candidate_labels: list[str]):
    return zero_shot_classifier(texts, candidate_labels)

def classify_yes_no(text: str, positive: str, negative: str) -> tuple[bool, float]:
    # classify between two candidates, giving a confidence score for the decision
    topic_result = classify(
        text=text,
        candidate_labels=[positive, negative]
    )
    topic_result.pop('sequence')
    print(topic_result)

    labels = topic_result.get('labels')
    scores = topic_result.get('scores')

    result = (labels[0] == positive)

    score = scores[0]
    print(scores, score, positive, negative, result)

    confidence = confidence_score(normalize_overscore(score, 2))

    print(f'y/n classification: {result}, {confidence:.2%}')
    return result, confidence

def classify_yes_no_batched(texts_batch: list[str], positive: str, negative: str):
    topic_results = classify_batched(
        texts=texts_batch,
        candidate_labels=[positive, negative]
    )
    print(topic_results)

def predict(text: str, assumption: str):
    # does not seem to be quite effective without fine-tuning onto machine learning vocab
    scores = cross_encoder_model.predict([(assumption, text)])

    # convert scores to labels
    label_mapping = ['contradiction', 'entailment', 'neutral']
    labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
    # print(labels)
    # print(scores)
    prediction = labels[0]

    print(f'>> Predict:\n{prediction} < {assumption} & {text}')
    return prediction

def predict_tech(text: str, tech: str):
    return predict(text, f'In this study the method includes the usage of {tech}.')
