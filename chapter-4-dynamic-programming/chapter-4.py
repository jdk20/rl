import numpy as np
import matplotlib.pyplot as plt

from environments import GridWorld


"""
====================================
AGENTS
====================================
"""


def iterative_policy_evaluation(g, pi, gamma=0.9, theta=0.1, verbose=False):
    """
    Iterative Policy Evaluation (Prediction), pg. 97, this is the non-inplace version
    Compute the state value function v_pi for any arbitrary policy pi

    p (float): Probability
    r (float): Reward
    gamma (float): Discount factor
    theta (float): Accuracy threshold
    """

    for k in range(0, 11):
        g.print_values(k)

        # Non-inplace, only update values after every state sweep
        V = np.zeros((g.height, g.width))

        # Loop over all states
        for i in range(0, g.height):
            for j in range(0, g.width):
                # Check if terminal state
                if not g.gridworld[i, j].is_terminal:
                    # Over possible actions
                    temp_1 = 0
                    #for _, a in enumerate(g.gridworld[i, j].a):
                    # Over possible states
                    temp_2 = 0
                    for _, sn in enumerate(g.gridworld[i, j].sn):
                        # p(s',r|s,a) = g.gridworld[i, j].p[a]
                        # r = sn.r
                        # V(s') = sn.v
                        temp_2 += (g.gridworld[i, j].p['north'] * (sn.r + (gamma*sn.v)))

                    # pi(a|s) = pi[a]
                    temp_1 += 0.25*temp_2

                    # Update estimated V(s)
                    # g.gridworld[i, j].v = temp_1
                    V[i, j] = temp_1

        # Non-inplace, update values after each state sweep k
        for i in range(0, g.height):
            for j in range(0, g.width):
                g.gridworld[i, j].v = V[i, j]


"""
====================================
EXERCISES AND FIGURES
====================================
"""

"""
Iterative policy evaluation algorithm (pg. 98) used to estimate V for an arbitrary policy. This examples uses a random 
action policy. Results are shown in Fig. 4.1 on pg. 99.
"""
height = 4
width = 4
reward = -1
terminal_states = np.array([[0, 0], [3, 3]])

# Assume all actions are equally probably and all states have the same actions
pi = {'north': 0.25, 'south': 0.25, 'east': 0.25, 'west': 0.25}
g = GridWorld([height, width], reward, terminal=terminal_states)

iterative_policy_evaluation(g, pi, gamma=1.0, theta=0.1, verbose=True)
