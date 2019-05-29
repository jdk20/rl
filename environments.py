import numpy as np


class Bandit:
    """
    Enviroment with only one state and k actions. Often described as a slot machine (bandit) with k arms (levers).
    inputs: (k, sigma_q, sigma_r)
    output: (r)

    Attributes:
        k (int): Number of actions that can be taken for this environment (analogous to the number of arms).
        sigma_q (float): Standard deviation for the random generation of the true action value.
        sigma_r (float): Standard deviation for the random generation of rewards around the true action value.
        nonstationary (bool): Nonstationary bandits have their true action values randomly change after every step
    """
    def __init__(self, k, sigma_q=1, sigma_r=1, nonstationary=False):
        super().__init__()
        self.k = k                                  # number of actions
        self.sigma_r = sigma_r                      # standard deviation of rewards
        self.nonstationary = nonstationary          # nonstationary allows q to randomly change after each step

        self.q = sigma_q*np.random.randn(self.k)    # true action values for all k actions

    def step(self, a):
        """
        Take action and return the reward.
        :param a: (int) Action to be completed, aka the arm to select.
        :return: Reward for that action.
        """

        if -1 < a < self.k:  # Check if the selected action is in-bounds
            r = self.sigma_r*self.q[a] + np.random.randn()
            if self.nonstationary:
                self.q += 0.01*np.random.randn(self.k)
        else:
            print('Action has to be from [0, ' + str(self.k) + ']')
            r = 0

        return r

