from sentence_transformers import CrossEncoder
from transformers import pipeline


model = CrossEncoder('cross-encoder/nli-MiniLM2-L6-H768')
classifier = pipeline("zero-shot-classification", model='cross-encoder/nli-MiniLM2-L6-H768')

def classify(text: str, candidate_labels: list[str] = ("computer vision", "text mining", "deep learning", "no deep learning")) -> dict:
    res = classifier(text, candidate_labels)
    return res

def predict(text: str, tech: str):
    assumption = f'{tech}'
    scores = model.predict([(assumption, text)])

    # convert scores to labels
    label_mapping = ['contradiction', 'entailment', 'neutral']
    labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
    # print(labels)
    # print(scores)
    prediction = labels[0]

    print(f'>> Predict:\n{labels[0]} < {assumption} & {text}')

    return prediction
