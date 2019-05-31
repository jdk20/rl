import numpy as np


class Bandit:
    """
    Chapter 2
    Environment with only one state and k actions. Often described as a slot machine (bandit) with k arms (levers).
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

        Attributes:
            a (int): Action to be completed, aka the arm to select.
            r (float): Reward for that action.
        """

        if -1 < a < self.k:  # Check if the selected action is in-bounds
            r = self.sigma_r*self.q[a] + np.random.randn()
            if self.nonstationary:
                self.q += 0.01*np.random.randn(self.k)
        else:
            print('Action has to be from [0, ' + str(self.k) + ']')
            r = 0

        return r


class GridWorld:
    """
    Chapter 4: Dynamic Programming

    Rectangle Grid World creation, pass indexes containing terminal states of unreachable states. Assume four cardinal
    directions (north, south, east, west) and all transitions have the same reward.

    Attributes:
        size (list): Height and width of the gridworld, giving height*width cells (each cell containing one state)
        r (float): Reward for transitions around the gridworld (assume all transitions are R for now)
                    Improvement: dictionary[s][a] = list[r]
        a (list): Actions applied to all cells in the gridworld (assume all cells have the same actions)
                 Improvement: dictionary[s] = list[a]
        unreachable (matrix): x,y- coordinates of unreachable states that should be eliminated from the gridworld
    """
    def __init__(self, size, r, terminal=np.array([0, 0]), unreachable=np.array([False])):
        super().__init__()
        self.height = size[0]
        self.width = size[1]
        self.r = r

        # Create gridworld with empty GridCell()
        self.gridworld = np.empty((self.height, self.width), dtype=object)
        for i in range(0, self.height):
            for j in range(0, self.width):
                self.gridworld[i, j] = GridCell()

        # Set unreachable GridCell locations to False
        if unreachable.any():
            for i in range(0, unreachable.shape[0]):
                self.gridworld[unreachable[i, 0], unreachable[i, 1]] = False

        # Set any terminal states
        Splus = GridCell()
        Splus.is_terminal = True
        for i in range(0, terminal.shape[0]):
            self.gridworld[terminal[i, 0], terminal[i, 1]] = Splus  # Pointer to a single S+ object

        # Reiterate over gridworld and set appropriate GridCell parameters
        for i in range(0, self.height):
            for j in range(0, self.width):
                # Check if reachable
                if self.gridworld[i, j]:
                    # Use string names to denote actions for interpretability
                    self.gridworld[i, j].a = ['north', 'south', 'east', 'west']
                    self.gridworld[i, j].r = self.r  # Assume reward is constant for all actions/states

                    # Set action->new_state transitions and action-rewards
                    for _, a in enumerate(self.gridworld[i, j].a):
                        if a is 'north':
                            if (i-1) >= 0 and self.gridworld[i-1, j]:
                                self.gridworld[i, j].sn.append(self.gridworld[i-1, j])  # Pointer to accessible states
                                self.gridworld[i, j].p[a] = 1.0  # Keep it deterministic
                            else:
                                # Transition off the grid possible but leads to remaining in the same state
                                self.gridworld[i, j].sn.append(self.gridworld[i, j])
                                self.gridworld[i, j].p[a] = 1.0
                        elif a is 'south':
                            if (i+1) < self.height and self.gridworld[i+1, j]:
                                self.gridworld[i, j].sn.append(self.gridworld[i+1, j])
                                self.gridworld[i, j].p[a] = 1.0
                            else:
                                self.gridworld[i, j].sn.append(self.gridworld[i, j])
                                self.gridworld[i, j].p[a] = 1.0
                        elif a is 'east':
                            if (j+1) < self.width and self.gridworld[i, j+1]:
                                self.gridworld[i, j].sn.append(self.gridworld[i, j+1])
                                self.gridworld[i, j].p[a] = 1.0
                            else:
                                self.gridworld[i, j].sn.append(self.gridworld[i, j])
                                self.gridworld[i, j].p[a] = 1.0
                        elif a is 'west':
                            if (j-1) >= 0 and self.gridworld[i, j-1]:
                                self.gridworld[i, j].sn.append(self.gridworld[i, j-1])
                                self.gridworld[i, j].p[a] = 1.0
                            else:
                                self.gridworld[i, j].sn.append(self.gridworld[i, j])
                                self.gridworld[i, j].p[a] = 1.0

    def print_values(self, k):
        print(k)
        for i in range(0, self.height):
            v = '| '
            for j in range(0, self.width):
                v += str(np.round(self.gridworld[i, j].v, 1)) + ' | '
            print(v)
        print('\n')


class GridCell:
    """
    Chapter 4: Dynamic Programming

    Each GridCell instance represents a unique state, and contains information on what rewards, actions,
    and other states are accessible (probability>0).

    Attributes:
        sn (list): List possible states (sn=s') that can reached from this state
        a (array): List of possible actions from this state
        r (float): Reward for all possible actions from this state
        p (dict): 4D matrix containing probabilities for p(sn | a) (no dependence on r or s)
        v (float): Estimated value of the state
        is_terminal (bool): Denoting if this cell is the terminal state S+
    """
    def __init__(self):
        self.sn = []
        self.a = 0
        self.r = 0.0
        self.p = {}
        self.v = 0.0
        self.is_terminal = False
