#!/usr/bin/env python

import matplotlib.pyplot as plt


class NSGA2Visualizer:
    def __init__(self, x_range=(0, 1), y_range=(0, 1)):
        # インタラクティブモードON (描画を更新)
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.x_range = x_range
        self.y_range = y_range

    def plot_population(self, population, generation, max_rank=5):
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
                    f1_vals,
                    f2_vals,
                    c=color,
                    marker="o",
                    s=5,
                    alpha=0.5,
                    label=f"{rank}",
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
                s=5,
                alpha=0.5,
                label=f"Rank > {max_rank}",
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
