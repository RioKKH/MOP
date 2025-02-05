#!/usr/bin/env python

from nsga2 import nsga2


def test_nsga2():
    final_population = nsga2(population_size=50, generations=10)
    assert len(final_population) == 50, "最終集団サイズは50であるべき"
    for ind in final_population:
        assert len(ind.objectives) == 2, "各個体は2つの目的値を持つべき"
    print("test_nsga2 passed.")


if __name__ == "__main__":
    test_nsga2()
