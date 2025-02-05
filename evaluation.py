#!/usr/bin/env python

import numpy as np


def evaluate(ind):
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
