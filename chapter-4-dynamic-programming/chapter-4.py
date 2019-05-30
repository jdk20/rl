import numpy as np
import matplotlib.pyplot as plt

from environments import GridWorld


def iterative_policy_evaluation(pi, gamma=0.9, theta=0.1):
    """
    Iterative Policy Evaluation (Prediction), pg. 97, this is the in-place version
    Compute the state value function v_pi for any arbitrary policy pi

    p (float): Probability
    r (float): Reward
    gamma (float): Discount factor
    theta (float): Accuracy threshold
    """

    gw = GridWorld([4, 4], -1, np.array([[0, 0], [3, 3]]))

    # Initialize V(s) for all s, except V(s=terminal) = 0
    num_states = 12
    V = np.zeros(num_states)

    # while delta < theta (threshold)
    delta = 0

    # Loop over all states
    for s in range(0, num_states):
        v = V[s]
        V[s] = 99
        delta = np.max([delta, np.abs(v - V[s])])

