import numpy as np

MAX_ITERATIONS = 1000000
MAX_EPISODES = 5000
GAMMA = 0.95
DELTA = 0.01

DELTA_T = 0.01
DELTA_R = 0.01

def beta(nS, nA, m, delta, gamma):
    """
    Returns a beta based on nS, nA and m and delta (confidence measures). For
    the latter two, takes defaults if not passed.

    Interpretation: with returned beta, Q(s, a) will converge to opt with
    probability 1 - ( delta / 2 ).

    See lemma 7 of paper.

    """

    return np.sqrt(np.log( 2 * nS * nA * m / delta) / 2) * 1 / (1 - gamma)

def delta_r(nS, nA, delta, m):
    """
    Returns delta_r such that all CIs are admissable (see lemma 5 of paper).

    """

    return delta / (2 * nS, nA, m)

def delta_t(nS, nA, delta, m):
    """
    Returns delta_t such that all CIs are admissable (see lemma 5 of paper).

    """
    return delta / (2 * nS, nA, m)