#!/usr/bin/env python

import numpy as np
import random
import matplotlib.pyplot as plt

# from benchmark_problems import evaluate


"""
MOEA/D(Multi-Objective Evolutionaly Algorithm based on Decomposition)は、多目的最適化問題
を多数のスカラー最適化問題に分解して解く手法。 各スカラー最適化問題は「サブ問題」として
扱い、予め用意した重みベクトル(λ)を用いて目的関数群を1つの集約関数(例えばTchebycheff方式)
に変換する。

アルゴリズムの大まかな流れは以下の通り。

1. 重みベクトルの生成と近傍の決定
問題の目的数に応じた重みベクトルを生成し、各重みベクトルに対してユークリッド距離等により
近傍(Neighborhood)を決定する。(例えば、2目的問題では、各重みは[w, 1-w]の形となり、各重み
に対して最も近いT個の重みを近傍とする。)

2. 初期化
各サブ問題に対して解(個体)を初期化し、各個体の目的関数値を評価します。
また、
"""


# 個体(解)を表すクラス
class Individual:
    def __init__(self, vector):
        self.vector = np.array(vector)  # ここでは１次元の解を扱う
        self.objectives = []  # 各目的関数の値


# 目的関数の評価
def evaluate(ind):
    # def evaluate_individual(ind):
    # ind.objectives = evaluate(ind, problem="DTLZ1", M=3, k=5)
    x = ind.vector[0]
    f1 = x
    f2 = 1 - np.sqrt(x)
    ind.objectives = [f1, f2]


# 重みベクトルの生成 (2目的の場合)
def generate_weight_vectors(pop_size):
    """
    重みベクトル {λ1, λ2, ..., λN} を生成する。
    例: λ_i = [w_i, 1 - w_i] with w_i = i / (N-1)
    """
    weight_vectors = []
    for i in range(pop_size):
        w = i / (pop_size - 1)
        weight_vectors.append(np.array([w, 1 - w]))
    return weight_vectors


# 各重みベクトルの近傍を決定する
def get_neighborhood(weight_vectors, T):
    """
    各重みベクトルに対して、ユークリッド距離により近傍B(i)を決定する。
    近傍サイズをTとする
    """
    neighborhoods = []
    N = len(weight_vectors)
    for i in range(N):
        distances = [
            np.linalg.norm(weight_vectors[i] - weight_vectors[j]) for j in range(N)
        ]
        sorted_indices = np.argsort(distances)
        neighborhoods.append(sorted_indices[:T])
    return neighborhoods


# 交叉(算術交叉)
def crossover(parent1, parent2, crossover_rate=0.9):
    child1 = Individual(parent1.vector.copy())
    child2 = Individual(parent2.vector.copy())
    if random.random() < crossover_rate:
        alpha = random.random()
        child1.vector = alpha * parent1.vector + (1 - alpha) * parent2.vector
        child2.vector = alpha * parent2.vector + (1 - alpha) * parent1.vector
    return child1, child2


# 突然変位(ガウスノイズ)
def mutate(ind, mutation_rate=0.1, sigma=0.1):
    for i in range(len(ind.vector)):
        if random.random() < mutation_rate:
            ind.vector[i] += random.gauss(0, sigma)
            ind.vector[i] = max(0, min(1, ind.vector[i]))


# 集約関数 (Tchebycheff法)
def tchebycheff(ind, weight, z):
    """各目的について、weight_i * |f_i - z_i|の最大値を返す"""
    values = [weight[i] * abs(ind.objectives[i] - z[i]) for i in range(len(weight))]
    return max(values)


# 理想点の更新
def update_ideal(z, ind):
    for i in range(len(z)):
        if ind.objectives[i] < z[i]:
            z[i] = ind.objectives[i]


# 可視化の為のクラス
class MOEADVisualizer:
    def __init__(self, x_range=(0, 1), y_range=(0, 1), weight_vectors=None):
        plt.ion()  # インタラクティブモードON (描画を更新)
        self.fig, self.ax = plt.subplots()
        self.x_range = x_range
        self.y_range = y_range

        # weight_vectorsは、各サブ門d内に対応する重みベクトルのリスト
        # 例: [w, 1-w], ...]
        self.weight_vectors = weight_vectors

    def plot_population(self, population, generation):
        self.ax.clear()
        # 各個体の目的関数値を抽出
        f1_vals = [ind.objectives[0] for ind in population]
        f2_vals = [ind.objectives[1] for ind in population]

        # 重みベクトルが与えられていれば、その第一成分で色分けする
        if self.weight_vectors is not None and len(self.weight_vectors) == len(
            population
        ):
            # 各サブ問題に対応する重みwを取得(例ではweight_vectors[i] = [w, 1-w])
            colors = [w[0] for w in self.weight_vectors]
            scatter = self.ax.scatter(
                f1_vals, f2_vals, c=colors, cmap="viridis", markers="o"
            )
            cbar = self.fig.colorbar(scatter, ax=self.ax)
            cbar.set_label("Weight w (for Objective 1")
        else:
            self.ax.scatter(f1_vals, f2_vals, c="blue", marker="o")

        self.ax.set_xlabel("Objective 1")
        self.ax.set_ylabel("Objective 2")
        self.ax.set_title(f"Generation {generation}")
        self.ax.set_xlim(self.x_range)
        self.ax.set_ylim(self.y_range)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True)
        plt.draw()
        plt.pause(0.5)  # 0.5秒間表示it__


# MOEA/Dのメインアルゴリズム
def moead(population_size=100, generations=50, T=20, visualizer=None):
    # 重みベクトルの生成と近傍の決定
    weight_vectors = generate_weight_vectors(population_size)
    neighborhoods = get_neighborhood(weight_vectors, T)

    # 初期集団の生成と評価
    # population = [Individual([random.random()]) for _ in range(population_size)]
    population = [Individual([random.random()]) for _ in range(population_size)]
    for ind in population:
        evaluate(ind)
        # evaluate_individual(ind)

    # 初期集団をプロット
    if visualizer is not None:
        visualizer.plot_population(population, generation=0)

    # 理想点Zの初期化(各目的の最小値)
    z = [min(ind.objectives[i] for ind in population) for i in range(2)]

    # 進化ループ
    for gen in range(generations):
        for i in range(population_size):
            # サブ問題iの近傍からランダムに2つの解を選択
            nb_indices = neighborhoods[i]
            parent_indices = random.sample(list(nb_indices), 2)
            parent1 = population[parent_indices[0]]
            parent2 = population[parent_indices[1]]

            # 交叉・突然変異二より子個体を生成 (ここでは１つの子を採用)
            child, _ = crossover(parent1, parent2)
            mutate(child)
            evaluate(child)
            # evaluate_individual(child)

            # 理想点zの更新
            update_ideal(z, child)

            # サブ問題iの近傍の各解と比較し、Tchebycheff集約値が改善する場合置換
            for j in nb_indices:
                f_child = tchebycheff(child, weight_vectors[j], z)
                f_current = tchebycheff(population[j], weight_vectors[j], z)
                if f_child < f_current:
                    population[j] = child

        # 現世代集団をプロット
        if visualizer is not None:
            visualizer.plot_population(population, generation=(gen + 1))

        print(f"Generation {gen + 1} completed.")
    return population


# メイン処理
if __name__ == "__main__":
    visualizer = MOEADVisualizer()
    final_population = moead(
        population_size=50, generations=30, T=10, visualizer=visualizer
    )
    print("\n最終世代の個体(解ベクトルと目的関数値):")
    for ind in final_population:
        print(f"Solution: {ind.vector}, Objectives: {ind.objectives}")
