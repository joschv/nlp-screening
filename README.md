# NLP Screening Challenge

Filtering papers by their abstract (and title, journal) information about virology, epidemics and usage of deep learning techniques.

## Installation instructions

Developed using a Python 3.11 environment.

### Install requirements using requirements.txt

`` pip install -r requirements.txt ``

Models are downloaded automatically from huggingface when the application requires them for the first time.

## Run instructions

0. Set source file path in main.py (only if a different source file should be used).
1. Configure settings at the top of the config.py file.
2. Run main.py
3. After execution, review output files (in "out" subfolder - it is opened on completion).
4. If adjustments are needed continue with Step 1 (for feedback loop, set DEBUG=True)

## Which NLP techniques have i used?

- Extractive Question Answering
  - using a small [Sentence Transformer model](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1) designed for semantic search
  - to extract relevant phrases (of abstracts) for a certain query (e.g. 'is the study about virology?')
  - no relevant phrases indicates it is not about virology
- Zero Shot Classification
  - using a small [Natural Language Inference Cross Encoder model](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768)
  - to classify whether a study is about a topic or uses a method

And supporting NLP techniques such as tokenization, sentence tokenization (nltk). 
Fine-tuning a BERT model for different GLUE tasks [was experimented with](https://github.com/joschv/nlp-screening/blob/finetuning/finetune_deberta.py) but was not gone for as the solution due to time constraints.


## Why not keyword-based?

- prone to deception, the following examples could be falsely assumed to use deep learning techniques:
  - "as such we chose to not use a computer vision approach" 
  - "compared to deep learning techniques this method requires more labor but is more accurate"
- too reliable on syntax compared to semantics
  - easily misses paraphrases that do not use a certain keyword
  - which also means listing all sensible keywords is quite a task and with a growing list false positives become more likely
  - embeddings of synonyms should be similar and therefore lead to similar classification results

## Resulting dataset statistics

The results using the default configuration (but without discarding disqualified studies immediately for complete info):

| [NLP screening results](https://docs.google.com/spreadsheets/d/18oMYjod0NT1aDrDnBfFb1OEUKeoBRm0VRzLrpEClVwE/edit?gid=0#gid=0) |     |               | **Uses deep learning** |               |
|:-----------------------------------------------------------------------------------------------------------------------------:|:---:|:-------------:|:----------------------:|:-------------:|
|                                                                                                                               |     |      yes      |           no           |      sum      |
|                                                                                                                               | yes | 5932 (51,81%) |      700 (6,11%)       | 6632 (57,92%) |
|                                              **About Epidemiology or Virology**                                               | no  | 3424 (29,90%) |     1394 (12,17%)      | 4818 (42,08%) |
|                                                                                                                               | sum | 9356 (81,71%) |     2094  (18,29%)     | 11450 (100%)  |

- 5932 of 11450 (52%) were classified as both on topic (Epidemiology/Virology) and using the specified method (Deep Learning).
- Of the 5932 passing studies:
  - 4168 (70%) were classified as using computer vision
  - 196 (3%) as using text mining
  - 967 (16%) as using both
  - 601 (10%) as using neither computer vision nor text mining (but other deep learning methods)


### Extra notes
#### Implementation choices explanation

- Entity extraction
  - problem: for example "deep learning" is by default not necessarily recognized as one entity ("deep"=adjective, "learning"=noun)
  - does not yield any direct results regarding the questions
- (Zero Shot) Classification
  - adequate for assigning labels like "about virology", "uses deep learning"
  - but for best results, a fine-tuned model is probably the way to go (experimented with, but not implemented due to time constraints)
- Extractive question answering
  - used to find phrases that are relevant to a question and possibly improve classification
  - not a result in itself as NLI is still required
- Data handling and batching
  - using the huggingface dataset library and batching to speed up the calculation and potentially reduce memory usage which should greatly benefit especially smaller machines as noted in the task but may increase installation burden
