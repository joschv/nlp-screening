# nlp-screening
Filtering papers by their abstract information about usage of deep learning techniques.

## Installation instructions

### Spacy

- download BERT model by running the following command (in venv):
  - python -m spacy download en_core_web_sm

## Implementation choices explanation

- Entity extraction
  - problem: for example "deep learning" is by default not necessarily recognized as one entity ("deep"=adjective, "learning"=noun)
- (Zero Shot) Classification
- Question answering
  - 
- Semantic similarity

### Why not keyworld-based?

- prone to deception, the following examples could be falsely assumed to use deep learning techniques:
    - "as such we chose to not use a computer vision approach" 
    - "compared to deep learning techniques this method requires more labor but is more accurate"
    