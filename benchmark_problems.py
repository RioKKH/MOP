#!/usr/bin/env python

import numpy as np
import math


def evaluate_dtlz1(ind, M=2, k=5):
    """
    DTLZ1の評価関数 (任意のM目的版)

    入力:
        - ind.vector: 決定変数の配列。長さ n = M + k - 1であることが前提
        - M: 目的数
        - k: g 関数に用いるパラメータ(通常5~7)
    """
    x = np.array(ind.vector)
    # n = len(x)
    # インデックスは0始まりなので、gはx[M-1]からx[n-1]を使う
    g = 100 * (
        k + np.sum((x[M - 1 :] - 0.5) ** 2 - np.cos(20 * math.pi * (x[M - 1 :] - 0.5)))
    )
    f = []
    # i = 1,...,M (pythonでは1~Mとして扱い、インデックス変換する)
    for i in range(1, M + 1):
        prod = 1.0
        # 積の対象: j=1,...,M-i -> pythonではインデックス0,...,M-i-1
        for j in range(0, M - i):
            prod *= x[j]
        if i == 1:
            f_i = 0.5 * (1 + g) * prod
        else:
            # 対応する因子はx_{M-i+1} -> pythonでは x[M-i]
            f_i = 0.5 * (1 + g) * prod * (1 - x[M - i])
        f.append(f_i)
    return f


def evaluate(ind, problem="DTLZ1", **kwargs):
    """
    汎用評価関数
    """
    prob = problem.upper()
    if prob == "DTLZ1":
        return evaluate_dtlz1(ind, **kwargs)
    else:
        raise ValueError(f"Undefined problem: {problem}")
