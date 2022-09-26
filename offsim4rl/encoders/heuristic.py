import argparse
import copy
from datetime import datetime
import logging
import os
import numpy as np
import pandas as pd


class CartpoleBoxEncoder():
    '''Heuristic Encoder Based on http://incompleteideas.net/sutton/book/code/pole.c'''

    N_BOXES = 162  # Number of disjoint boxes of state space.
    ONE_DEGREE = 0.0174532  # 2pi/360
    SIX_DEGREES = 0.1047192
    TWELVE_DEGREES = 0.2094384
    FIFTY_DEGREES = 0.87266

    @staticmethod
    def get_box(x, x_dot, theta, theta_dot):
        box = 0

        if x < -2.4 or x > 2.4 or theta < - CartpoleBoxEncoder.TWELVE_DEGREES or theta > CartpoleBoxEncoder.TWELVE_DEGREES:
            return -1  # to signal failure

        if x < - 0.8:
            box = 0
        elif x < 0.8:
            box = 1
        else:
            box = 2

        if x_dot < - 0.5:
            pass
        elif x_dot < 0.5:
            box += 3
        else:
            box += 6

        if theta < - CartpoleBoxEncoder.SIX_DEGREES:
            pass
        elif theta < -CartpoleBoxEncoder.ONE_DEGREE:
            box += 9
        elif theta < 0:
            box += 18
        elif theta < CartpoleBoxEncoder.ONE_DEGREE:
            box += 27
        elif theta < CartpoleBoxEncoder.SIX_DEGREES:
            box += 36
        else:
            box += 45

        if theta_dot < - CartpoleBoxEncoder.FIFTY_DEGREES:
            pass
        elif theta_dot < CartpoleBoxEncoder.FIFTY_DEGREES:
            box += 54
        else:
            box += 108

        return box

    def __init__(self):
        pass

    def encode(self, observations):
        emb = []
        for obs in observations:
            x, x_dot, theta, theta_dot = obs
            emb.append(CartpoleBoxEncoder.get_box(x, x_dot, theta, theta_dot))

        return np.array(emb)
