#!/usr/bin/env python

import random
from individual import Individual
from visualization import NSGA2Visualizer


def test_visualization():
    # ダミーの個体群を作成
    population = []
    for _ in range(20):
        ind = Individual([random.random() for _ in range(30)])
        ind.rank = random.randint(1, 3)
        ind.objectives = [random.random(), random.random()]
        population.append(ind)
    viz = NSGA2Visualizer(x_range=(0, 1), y_range=(0, 1))
    viz.plot_population(population, generation=0)
    print("Visualization test: プロットを確認してください。")


if __name__ == "__main__":
    test_visualization()
