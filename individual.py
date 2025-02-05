#!/usr/bin/env python

import numpy as np


class Individual:
    def __init__(self, vector):
        self.vector = np.array(vector)
        self.objectives = []  # evalute() 呼び出し後に設定される
        self.rank = None
        self.crowding_distance = 0.0
