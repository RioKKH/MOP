#!/usr/bin/env python

from nsga2 import nsga2
from visualization import NSGA2Visualizer


def main():
    # 目的関数の値のレンジはZDT1ではf1∊ [0, 1],
    # f2はg*(1-sqrt(x1/g))の値域なので適時調整すること
    visualizer = NSGA2Visualizer(x_range=(0, 6), y_range=(0, 6))
    final_population = nsga2(
        population_size=100, generations=500, visualizer=visualizer
    )
    visualizer.plot_population(final_population, generation=30)

    print("\n最終世代の個体(解、目的値、ランク、クラウディング距離):")
    for ind in final_population:
        print(
            f"解: {ind.vector}, 目的値: {ind.objectives}, ランク: {ind.rank}, クラウディング距離: {ind.crowding_distance}"
        )


if __name__ == "__main__":
    main()
