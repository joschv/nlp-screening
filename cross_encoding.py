import math

from sentence_transformers import CrossEncoder
from transformers import pipeline


model = CrossEncoder('cross-encoder/nli-MiniLM2-L6-H768')
classifier = pipeline("zero-shot-classification", model='cross-encoder/nli-MiniLM2-L6-H768')

def classify(text: str, candidate_labels: list[str] = ("computer vision", "text mining", "deep learning", "no deep learning")) -> dict:
    res = classifier(text, candidate_labels)
    return res

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

    result = labels[0] == positive
    confidence = math.sin(max(min((abs(scores[0]) - 0.5), 0.4), 0.1) * math.pi)

    print(f'y/n classification: {result}, {confidence}')
    return result, confidence

def predict(text: str, assumption: str):
    # does not seem to be quite effective without fine-tuning onto machine learning vocab
    scores = model.predict([(assumption, text)])

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
