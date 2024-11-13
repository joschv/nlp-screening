
def get_texts(batch):
    """ Takes abstract - or title if abstract is missing - for each example of batch. """
    titles = [t.lower() for t in batch['Title']]
    abstracts = [a.lower() if a is not None else '' for a in batch['Abstract']]
    texts = [a if a is not None and len(a) > 0 else t for a, t in zip(abstracts, titles)]
    return texts
