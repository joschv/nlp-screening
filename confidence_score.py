import math
from enum import Enum


class ConfidenceMethod(Enum):
    LIN = 0
    EXP = 1
    SIN = 2
    LOG = 3
    # variance?


def confidence_score(normalized_overscore: float, confidence_method: ConfidenceMethod = ConfidenceMethod.SIN) -> float:
    """ Calculates confidence score for normalized overscore
    (score above proportionate score (0.5 for 2 candidates etc.)."""
    if confidence_method == ConfidenceMethod.EXP:
        return ((normalized_overscore + 1) ** 2.0 - 1) / 3
    elif confidence_method == ConfidenceMethod.SIN:
        return math.sin(normalized_overscore * 0.5 * math.pi)
    elif confidence_method == ConfidenceMethod.LOG:
        if normalized_overscore == 0.0:
            return 0.0
        return max(0.0, math.log10(normalized_overscore * 100) / 2)
    else:
        # linear
        return normalized_overscore


def normalize_overscore(score: float, n: int) -> float:
    """ Calculate normalized overscore from the score given the number of scored labels. """
    return (score - (1/n)) / ((n-1)/n)


def test():
    for m in ConfidenceMethod:
        print(m)
        for i in range(11):
            cur = i/10
            cur_confidence = confidence_score(normalized_overscore=i/10, confidence_method=m)
            print(cur, cur_confidence)

            assert 0.0 <= cur_confidence <= 1.0


if __name__ == '__main__':
    test()
