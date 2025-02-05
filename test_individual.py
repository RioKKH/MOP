#!/usr/bin/env python

from individual import Individual


def test_individual():
    vector = [0.1] * 30
    ind = Individual(vector)
    assert ind.vector.shape[0] == 30, "vector の次元は 30であるべき"
    print("test_individual passed.")


if __name__ == "__main__":
    test_individual()
