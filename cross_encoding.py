from sentence_transformers import CrossEncoder
from transformers import pipeline

from confidence_score import confidence_score, normalize_overscore

model_checkpoint = 'cross-encoder/nli-MiniLM2-L6-H768'  # very small natural language inference model
cross_encoder_model = CrossEncoder(model_checkpoint)
zero_shot_classifier = pipeline("zero-shot-classification", model=model_checkpoint)

def classify(text: str, candidate_labels: list[str]) -> dict:
    """ Zero-shot-classify the text for the candidates.
        :returns: A dictionary containing all information regarding the classification including labels, scores etc.
        """
    return zero_shot_classifier(text, candidate_labels)

def classify_yes_no(text: str, positive: str, negative: str) -> tuple[bool, float]:
    """ Classify between two candidates, giving a confidence score for the decision.
        :returns: True if positive candidate has higher score than negative candidate,
            and a confidence score representing the confidence in choosing that label over the other.
    """
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

def predict(text: str, assumption: str) -> str:
    """ Predict (entailment, neutral, contradiction) for assumption based on text. """
    # does not seem to be quite effective without fine-tuning onto machine learning vocab
    # or requires more literal entailment
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
