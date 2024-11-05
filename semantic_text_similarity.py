from sentence_transformers import CrossEncoder

# for other models for the task go to:
# https://www.sbert.net/docs/cross_encoder/pretrained_models.html#stsbenchmark

model = CrossEncoder("cross-encoder/stsb-roberta-base")


def semantic_text_similarity(t1: str, t2: str) -> float:
    score = model.predict((t1, t2))
    return float(score)


def semantic_text_similarity_batch(batch: list[tuple[str, str]]) -> list[float]:
    scores = model.predict(batch)
    return [float(score) for score in scores]


if __name__ == '__main__':
    s1 = semantic_text_similarity("It's a wonderful day outside.", "It's so sunny today!")
    print(s1)
    s2 = semantic_text_similarity_batch([("It's a wonderful day outside.", "It's so sunny today!"), ("It's a wonderful day inside.", "It's so sunny today!")])
    print(s2)