import nltk
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

from batch_helpers import get_texts

# load model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
model = AutoModel.from_pretrained("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

# mean pooling - take average of all tokens
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# encode text
def encode(texts):
    # tokenize sentences
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    # compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input, return_dict=True)

    # perform pooling
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings


def semantic_search(query: str, texts: list[str]) -> list[tuple[float, str]]:
    # encode query and docs
    query_emb = encode(query)
    text_emb = encode(texts)

    # compute dot score between query and all document embeddings
    scores = torch.mm(query_emb, text_emb.transpose(0, 1))[0].cpu().tolist()

    # combine docs & scores
    text_score_pairs = list(zip(texts, scores))

    # sort by decreasing score
    text_score_pairs = sorted(text_score_pairs, key=lambda x: x[1], reverse=True)

    # output passages & scores
    # for doc, score in doc_score_pairs:
        # print(score, doc)
    return [(float(score), text) for text, score in text_score_pairs]


def semantic_filter(query: str, texts: list[str], top_k: int = 3, min_relevance: float = 0.) -> list[tuple[float, str]]:
    # print(f'>> {query}')
    semantic_scores = semantic_search(query, texts)

    top_scores = filter_top_k(
        scores=semantic_scores,
        k=top_k
    )

    top_scores = filter_by_score(
        semantic_scores=top_scores,
        threshold=min_relevance
    )

    # print(f'  {top_scores}')
    return top_scores

def filter_top_k(scores: list[tuple[float, str]], k: int = 3) -> list[tuple[float, str]]:
    return scores[:min(len(scores), k)]

def filter_by_score(semantic_scores: list[tuple[float, str]], threshold: float) -> list[tuple[float, str]]:
    # find index of last score >= threshold (requires sorted scores)
    # (could implement binary search here later)
    last_above_threshold = 0
    for (score, _) in semantic_scores:
        if score >= threshold:
            last_above_threshold += 1
        else:
            break
    return semantic_scores[:last_above_threshold]


def extractive_question_answering(query: str, dict_key: str, top_k: int = 3, min_relevance: float = 0.2):
    """ Given a query, for each example of the batch, extracts relevant phrases sorted by relevancy (desc). """
    def pp(batch):
        texts = get_texts(batch=batch)

        texts_phrases = [nltk.tokenize.sent_tokenize(t, language='english') for t in texts]

        texts_combined_relevant_phrases = []
        for phrases in texts_phrases:
            semantic_scores = semantic_filter(
                query=query,
                texts=phrases,
                top_k=top_k,
                min_relevance=min_relevance
            )
            combined_phrases = ' '.join([phrase for (score, phrase) in semantic_scores])
            texts_combined_relevant_phrases += [combined_phrases]

        return {
            dict_key: texts_combined_relevant_phrases
        }

    return pp


def text_relevancy_scores(queries: list[str], labels: list[str], **kwargs):
    """ Given a list of queries and labels, score how relevant the text is to the query. """
    assert len(queries) > 0 and len(queries) == len(labels)

    def pp(batch):
        texts = get_texts(batch=batch)
        texts_phrases = [nltk.tokenize.sent_tokenize(t, language='english') for t in texts]

        ret_dict = {}

        for q, l in zip(queries, labels):
            accumulated_scores = []
            highest_scores = []
            contexts = []

            for phrases in texts_phrases:
                semantic_scores = semantic_filter(
                    query=q,
                    texts=phrases,
                    top_k=kwargs.get('top_k', 4),
                    min_relevance=kwargs.get('min_relevance', 0.25)
                )
                scores = [score for (score, phrase) in semantic_scores]
                phrases = [phrase for (score, phrase) in semantic_scores]

                accumulated_scores += [sum(scores) if len(scores) > 0 else 0.]
                highest_scores += [max(scores) if len(scores) > 0 else 0.]
                contexts += [' '.join(phrases)]


            ret_dict |= {
                f'{l}_accumulated_relevancy_score': accumulated_scores,
                f'{l}_highest_relevancy_score': highest_scores,
                f'{l}_context': contexts,
            }

        return ret_dict
    return pp


def test_filter_by_score():
    assert filter_by_score([(0.31, ''), (0.30, ''), (0.29, ''), (-0.3, '')], 0.5) == []
    assert filter_by_score([(0.31, ''), (0.30, ''), (0.29, ''), (-0.3, '')], 0.3) == [(0.31, ''), (0.30, '')]
    assert filter_by_score([(0.31, ''), (0.30, ''), (0.29, ''), (-0.3, '')], 0.2) == [(0.31, ''), (0.30, ''), (0.29, '')]
    assert filter_by_score([(0.31, ''), (0.30, ''), (0.29, ''), (-0.3, '')], -0.3) == [(0.31, ''), (0.30, ''), (0.29, ''), (-0.3, '')]


if __name__ == '__main__':
    # sentences we want sentence embeddings for
    query = "Did the method of the study include the usage of deep learning, text mining or computer vision?"
    texts = ["Around 9 Million people live in London", "London is known for its financial district", "Our purpose is to develop a computer vision approach to quantify intra-arterial thickness on digital pathology images of kidney biopsies as a computational biomarker of arteriosclerosis."]

    print(semantic_search(query, texts))

    test_filter_by_score()
