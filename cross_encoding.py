import torch
from sentence_transformers import CrossEncoder
from transformers import pipeline

from batch_helpers import get_texts
from confidence_score import confidence_score, normalize_overscore
from dataset_definitions import DEEP_LEARNING_METHOD_LABEL_CONFIDENCE, DEEP_LEARNING_METHOD_LABEL

model_checkpoint = 'cross-encoder/nli-MiniLM2-L6-H768'  # very small natural language inference model
cross_encoder_model = CrossEncoder(model_checkpoint)
zero_shot_classifier = pipeline("zero-shot-classification", model=model_checkpoint)

def classify(text: str, candidate_labels: list[str]) -> dict:
    """ Zero-shot-classify the text for the candidates.
        :returns: A dictionary containing all information regarding the classification including labels, scores etc.
        """
    with torch.no_grad():
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


def assign_deep_learning_method_label(scores: dict) -> tuple[str, float]:
    """ Assigns the label for the determined deep learning method used based on the given scores. """
    cv_score = scores.get('computer vision', 0.0)
    tm_score = scores.get('text mining', 0.0)
    other_score = scores.get('other', 0.0)
    single_threshold = 0.4
    duo_threshold = 0.75

    # special case: computer vision and text mining scores very similar
    if abs(cv_score - tm_score) < 0.2 and (cv_score + tm_score) > duo_threshold:
        return 'both', confidence_score(
            normalized_overscore=(cv_score + tm_score - (2/3)) / (1/3)
        )
    # not very similar => return the highest specific method above threshold
    elif cv_score > tm_score and cv_score > single_threshold:
        return 'computer vision', confidence_score(
            normalized_overscore=normalize_overscore(cv_score, 3)
        )
    elif tm_score > cv_score and tm_score > single_threshold:
        return 'text mining', confidence_score(
            normalized_overscore=normalize_overscore(tm_score, 3)
        )
    else:
        return 'other', confidence_score(
            normalized_overscore=normalize_overscore(max(other_score, 1/3), 3)
        )


def classify_deep_learning_method(computer_vision_label: str, text_mining_label: str, other_label: str):
    """ Use Zero-Shot-Classification with the NLI CrossEncoder model to derive which deep learning method was used. """
    candidates_out = [
        'computer vision',
        'text mining',
        'other'
    ]

    candidates_in = [
        computer_vision_label,
        text_mining_label,
        other_label
    ]

    assert len(candidates_in) == len(candidates_out)

    blueprint = 'the study used {}'

    candidates = [blueprint.format(ci) for ci in candidates_in]

    def pp(batch):
        texts = get_texts(batch=batch)
        with torch.no_grad():
            classification_results = zero_shot_classifier(texts, candidates)

        labels = []
        confidences = []

        for class_res in classification_results:
            class_res.pop('sequence')

            scores = {}
            for i in range(len(candidates_in)):
                scores |= {candidates_out[i]: class_res['scores'][class_res['labels'].index(candidates[i])]}

            dlm_label_result = assign_deep_learning_method_label(scores)
            labels += [dlm_label_result[0]]
            confidences += [dlm_label_result[1]]

        return {
            DEEP_LEARNING_METHOD_LABEL: labels,
            DEEP_LEARNING_METHOD_LABEL_CONFIDENCE: confidences,
        }
    return pp


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
