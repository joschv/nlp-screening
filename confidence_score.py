import math
from enum import Enum


class ConfidenceMethod(Enum):
    EXP = 1
    SIN = 2


def confidence_score(normalized_overscore: float, confidence_method: ConfidenceMethod = ConfidenceMethod.SIN) -> float:
    # calculates a confidence score for normalized overscore (score above proportionate score (0.5 for 2 cancidates etc)
    if confidence_method == ConfidenceMethod.EXP:
        return ((normalized_overscore + 1) ** 2.0 - 1) / 3
    elif confidence_method == ConfidenceMethod.SIN:
        return math.sin(normalized_overscore * 0.5 * math.pi)
    else:
        return normalized_overscore


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
