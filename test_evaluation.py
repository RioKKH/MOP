#!/usr/bin/env python

from evaluation import evaluate

# from individual import Individual
from NSGA2 import Individual
# from NSGA2 import evaluate


def test_evaluate():
    # 線要素が0の場合、x1 = 0なので、f1 = 0, g = 1, f2 = 1
    vector = [0.0] * 30
    ind = Individual(vector)
    evaluate(ind)

    assert abs(ind.objectives[0] - 0.0) < 1e-6, "f1 が 0.0 であるべき"
    assert abs(ind.objectives[1] - 1.0) < 1e-6, "f2 が 1.0 であるべき"
    print("test_evaluate passed.")


if __name__ == "__main__":
    test_evaluate()
