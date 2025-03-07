#!/usr/bin/env python

import numpy as np
import random
import matplotlib.pyplot as plt


# 個体を表すクラス
class Individual:
    def __init__(self, vector):
        self.vector = np.array(vector)  # 解の表現(ここでは１次元)
        self.objectives = []  # 各目的関数の値
        self.rank = None  # 非裂開ソーティングでのランク(フロント番号)
        self.crowding_distance = 0.0  # クラウディング距離


# 目的関数の評価
def evaluate(ind):
    # # 例として、1変数x ∊ [0, 1]の解に対し、
    # # 目的関数 f1 = x, f2 = 1 - sqrt(x) (どちらも最小化)とする
    # x = ind.vector[0]
    # f1 = x
    # f2 = 1 - np.sqrt(x)
    # ind.objectives = [f1, f2]
    """
    ZDT1 ベンチマーク問題の評価関数
    ind.vectorは30次元のリスト/配列と想定
    結果は ind.objectives に[f1, v2]をセットする
    """
    n = len(ind.vector)
    f1 = ind.vector[0]
    g = 1 + 9 / (n - 1) * sum(ind.vector[1:])
    f2 = g * (1 - np.sqrt(ind.vector[0] / g))
    ind.objectives = [f1, f2]


# 個体pが個体qを支配しているかどうか
def dominates(p, q):
    and_condition = all(p_i <= q_i for p_i, q_i in zip(p.objectives, q.objectives))
    or_condition = any(p_i < q_i for p_i, q_i in zip(p.objectives, q.objectives))
    return and_condition and or_condition


# 非劣解ソーティング (fast non-dominated sort)
def fast_non_dominated_sort(population):
    fronts = [[]]
    for p in population:
        p.domination_count = 0
        p.dominated_solutions = []
        for q in population:
            if dominates(p, q):
                p.dominated_solutions.append(q)
            elif dominates(q, p):
                p.domination_count += 1
        if p.domination_count == 0:
            p.rank = 1
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in p.dominated_solutions:
                q.domination_count -= 1
                if q.domination_count == 0:
                    q.rank = i + 2
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    fronts.pop()  # 最後の空リストを削除
    return fronts


# クラウディング距離の計算
def calculate_crowding_distance(front):
    num_individuals = len(front)
    if num_individuals == 0:
        return
    for ind in front:
        ind.crowding_distance = 0
    num_objectives = len(front[0].objectives)
    for m in range(num_objectives):
        # 目的mに対してソート
        front.sort(key=lambda ind: ind.objectives[m])
        # 最小値と最大値は無限大の距離を与える
        front[0].crowding_distance = float("inf")
        front[-1].crowding_distance = float("inf")
        m_values = [ind.objectives[m] for ind in front]
        m_min = min(m_values)
        m_max = max(m_values)
        if m_max - m_min == 0:
            continue
        # 内部の個体に対して距離を加算
        for i in range(1, num_individuals - 1):
            front[i].crowding_distance += (
                front[i + 1].objectives[m] - front[i - 1].objectives[m]
            ) / (m_max - m_min)


# バイナリトーナメント選択(ランクが低いもの、同ランクならクラウディング距離が大きいものを選ぶ)
def tournament_selection(population):
    i, j = random.sample(range(len(population)), 2)
    if population[i].rank < population[j].rank:
        return population[i]
    elif population[i].rank > population[j].rank:
        return population[j]
    else:
        # 同じランクならクラウディング距離で比較(大きい方が選ばれやすい)
        return (
            population[i]
            if population[i].crowding_distance > population[j].crowding_distance
            else population[j]
        )


# 交叉(ここでは簡単な算術交叉を採用)
def crossover(parent1, parent2, crossover_rate=0.9):
    child1 = Individual(parent1.vector.copy())
    child2 = Individual(parent2.vector.copy())
    if random.random() < crossover_rate:
        alpha = random.random()
        child1.vector = alpha * parent1.vector + (1 - alpha) * parent2.vector
        child2.vector = alpha * parent2.vector + (1 - alpha) * parent1.vector
    return child1, child2


# 突然変異(各遺伝子に対してガウスノイズを加える)
def mutate(ind, mutation_rate=0.1, sigma=0.1):
    for i in range(len(ind.vector)):
        if random.random() < mutation_rate:
            ind.vector[i] += random.gauss(0, sigma)
            # 解の範囲[0, 1] に制限
            ind.vector[i] = max(0, min(1, ind.vector[i]))


# 可視化の為のクラス
class NSGA2Visualizer:
    def __init__(self, x_range=(0, 1), y_range=(0, 1)):
        # インタラクティブモードON (描画を更新)
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.x_range = x_range
        self.y_range = y_range

    def plot_population(self, population, generation, max_rank=3):
        self.ax.clear()
        # ランクごとの色リスト: max_rankまでの色を指定。足りなければデフォルト色を利用
        rank_colors = ["blue", "green", "red"]

        # ランクごとにフィルタしてプロットする
        for rank in range(1, max_rank + 1):
            rank_population = [ind for ind in population if ind.rank == rank]
            if rank_population:
                # 各個体の目的関数値を抽出
                f1_vals = [ind.objectives[0] for ind in population]
                f2_vals = [ind.objectives[1] for ind in population]
                color = (
                    rank_colors[rank - 1] if rank - 1 < len(rank_colors) else "black"
                )
                self.ax.scatter(
                    f1_vals, f2_vals, c=color, marker="o", alpha=0.5, label=f"{rank}"
                )
        # max_rank寄り大きいランクの個体は別色でプロット
        others = [ind for ind in population if ind.rank > max_rank]
        if others:
            f1_vals = [ind.objectives[0] for ind in population]
            f2_vals = [ind.objectives[1] for ind in population]
            self.ax.scatter(
                f1_vals,
                f2_vals,
                c="gray",
                marker="o",
                alpha=0.5,
                lable=f"Rank > {max_rank}",
            )

        self.ax.set_xlabel("Objective 1")
        self.ax.set_ylabel("Objective 2")
        self.ax.set_title(f"Generation {generation}")
        self.ax.set_xlim(self.x_range)
        self.ax.set_ylim(self.y_range)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True)
        plt.draw()
        plt.pause(0.5)  # 0.5秒間表示


# NSGA-IIのメインアルゴリズム
def nsga2(population_size=100, generations=50, visualizer=None):
    # 初期集団の生成と評価
    population = [Individual([random.random()]) for _ in range(population_size)]
    for ind in population:
        evaluate(ind)

    # 初期集団に対して非劣解ソーティングとクラウディング距離の計算を行う
    fronts = fast_non_dominated_sort(population)
    for front in fronts:
        calculate_crowding_distance(front)

    # 初期集団をプロット
    if visualizer is not None:
        visualizer.plot_population(population, generation=0)

    # 世代ループ
    for gen in range(generations):
        offspring = []
        # 親個体から子個体を生成(集団サイズ分)
        while len(offspring) < population_size:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child1, child2 = crossover(parent1, parent2)
            mutate(child1)
            mutate(child2)
            evaluate(child1)
            evaluate(child2)
            offspring.extend([child1, child2])

        # 集団サイズを合わせる
        offspring = offspring[:population_size]

        # 親集団と子孫集団を結合してRを作成
        combined_population = population + offspring
        fronts = fast_non_dominated_sort(combined_population)

        # 新たな集団を選択(エリート保存)
        new_population = []
        for front in fronts:
            calculate_crowding_distance(front)
            if len(new_population) + len(front) <= population_size:
                new_population.extend(front)
            else:
                # 集団サイズが上限に到達するまで、クラウディング距離が大きい個体から選択
                front.sort(key=lambda ind: ind.crowding_distance, reverse=True)
                remaining = population_size - len(new_population)
                new_population.extend(front[:remaining])
                break
        population = new_population

        # 現世代集団をプロット
        if visualizer is not None:
            visualizer.plot_population(population, generation=(gen + 1))

        print(f"Generation {gen + 1} completed.")

    return population


if __name__ == "__main__":
    visualizer = NSGA2Visualizer()
    final_population = nsga2(population_size=50, generations=30, visualizer=visualizer)
    print("\n最終世代の個体 (解ベクトルと目的関数値) : ")
    for ind in final_population:
        print(
            f"解: {ind.vector}, 目的値: {ind.objectives}, ランク: {ind.rank}, クラウディング距離: {ind.crowding_distance}"
        )
