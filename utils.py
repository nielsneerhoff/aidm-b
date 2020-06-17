import numpy as np

MAX_ITERATIONS = 1000000
MAX_EPISODES = 5000
NO_RUNS = 5
GAMMA = 0.95
DELTA = 0.01

DELTA_T = 0.20
DELTA_R = 0.05

def BETA(reward_range, nS, nA, m):
    """
    Returns beta with convergence guarantee (see lemma 7).

    """

    return (
        (reward_range[1] - reward_range[0]) / (1 - GAMMA)) * np.sqrt(np.log(2 * nS * nA * m / DELTA) / 2)