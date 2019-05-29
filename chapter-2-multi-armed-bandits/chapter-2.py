import numpy as np
import matplotlib.pyplot as plt

from environments import Bandit


def simple_bandit(k, ax, eps=0.0, a=0.0, Q1=0.0, c=0.0, nonstationary=False, n_runs=2000, timesteps=1000):
    """
    A simple bandit algorithm (pg. 54)
    Estimates action-values (Q) using incremental sample averages Q = (1/N)*sum_{i=n}^{N}{Ri} == Q + (1/N)(R - Q)
    Balances exploration and exploitation using eps-greedy (eps=0 always greedy, eps=1 always exploring)
    Stationary: rewards do not change over time
    Nonstationary: rewards do change over time (want to give more weight to recent rewards)
    Upper confidence bound (ucb): Select greedy action taking into account the uncertainty around Q

    Q (array): Estimated action values
    Q1 (float): Initial estimate of Q
    N (array): Number of times each action has been performed
    A (int): Action to be taken
    R (float): Reward
    a (float): alpha, constant step size parameter
    eps (float): epsilon, probability to randomly explore
    """
    # Create a dictionary to store information about rewards received and actions taken
    metrics = {}
    metrics['reward'] = np.zeros((n_runs, timesteps))
    metrics['action'] = np.zeros((n_runs, timesteps))
    metrics['optimal_action'] = np.zeros((n_runs, timesteps))

    for n in range(0, n_runs):
        # Initialize, for a=1 to k:
        Q = Q1*np.ones(k)                   # Estimated action values for each action k
        N = np.zeros(k)                     # Step size parameter for each action
        bandit = Bandit(k, nonstationary=nonstationary)

        # Loop forever:
        for t in range(0, timesteps):
            pr = np.random.rand()           # Uniform draw

            if pr <= eps:                   # Take random action
                A = np.random.randint(k)
            else:                           # Take greedy action with or without upper confidence bound
                if c > 0:  # UCB, greater c indicates more precise CI, add 1 to t to avoid division by 0
                    A = np.argmax(Q + (c*np.sqrt(np.log(t+1)/(N+1e-6))))
                else:
                    A = np.argmax(Q)

            # Interact with the environment
            R = bandit.step(A)
            N[A] += 1

            # Incremental implementation of sample averages (Section 2.4)
            # for steps i = 1, 2, ..., n, n+1, n+2, ..., N
            # Q_{n+1} = (1/n) * sum_{i=1}^{n} {R_{i}}
            # Q_{n+1} = (1/n) * (R_{n} + sum_{i=1}^{n-1} {R_{i}})
            #
            # with Q_{n} = (1/(n-1)) * sum_{i=1}^{n-1} {R_{i}}
            # with Q_{n} * (n-1) = sum_{i=1}^{n-1} {R_{i}}
            #
            # substitute into Q_{n+1} = (1/n) * (R_{n} + Q_{n} * (n-1))
            # Q_{n+1} = (1/n) * (R_{n} + n*Q_{n} - Q_{n})
            # Q_{n+1} = R_{n}/n + Q_{n} - Q_{n}/n
            # Q_{n+1} = Q_{n} + R_{n}/n - Q_{n}/n
            # Q_{n+1} = Q_{n} + (1/n)*(R_{n} - Q_{n})

            # Or in other terms:
            # NewEstimate = OldEstimate + StepSize * (Target - OldEstimate)
            # NewEstimate: Q_{n+1}
            # OldEstimate: Q_{n}
            # StepSize: 1/n (this value changes from step to step, ex: step 3 = 1/3, step 1000 = 1/1000)
            # Target: R_{n}
            # "Error": (Target - OldEstimate) == R_{n} - Q_{n}

            # Can use step size parameter 1/n or constant step size a
            if a == 0:                      # Use 1/n
                Q[A] = Q[A] + (1/N[A])*(R - Q[A])
            else:                           # Use constant step size parameter for discounting previous rewards
                Q[A] = Q[A] + a*(R - Q[A])

            # Store information, not part of the algorithm
            metrics['reward'][n, t] = R
            metrics['action'][n, t] = A
            if A == np.argmax(bandit.q):    # Compare to the action with the optimal value
                metrics['optimal_action'][n, t] = 1
            else:
                metrics['optimal_action'][n, t] = 0

    # Plotting, also not part of the algorithm
    tt = np.arange(0, timesteps, 1)

    ax[0].plot(tt, np.mean(metrics['reward'], axis=0), '-', label='eps=' + str(eps))
    ax[0].set_ylabel('Average reward')
    ax[0].legend()

    ax[1].plot(tt, np.mean(metrics['optimal_action'], axis=0), '-', label='eps=' + str(eps))
    ax[1].set_xlabel('Timesteps')
    ax[1].set_ylabel('% Optimal action')
    ax[1].legend()

    return ax


def gradient_bandit(k, ax, a=0.1, nonstationary=False, kill_baseline = False, n_runs=2000, timesteps=1000):
    """
    The gradient bandit algorithm (pg. 59)

    H (array): Preference of selecting an action
    pi (array): Probability of selecting an action (constructed by applying softmax() to H)
    N (array): Number of times each action has been performed
    A (int): Action to be taken
    R (float): Reward
    Rmean (float): Reward baseline, necessary for rewards not distributed from N(0,1)
    a (float): alpha, constant step size parameter
    """
    # Create a dictionary to store information about rewards received and actions taken
    metrics = {}
    metrics['reward'] = np.zeros((n_runs, timesteps))
    metrics['action'] = np.zeros((n_runs, timesteps))
    metrics['optimal_action'] = np.zeros((n_runs, timesteps))

    for n in range(0, n_runs):
        # Initialize, for a=1 to k:
        N = np.zeros(k)                     # Step size parameter for each action
        H = np.zeros(k)                     # Preference for each action, converted to probability pi using softmax
        Rmean = 0.0                         # Baseline (average) for all rewards received
        bandit = Bandit(k, nonstationary=nonstationary)

        if kill_baseline:  # This is to recreate Fig. 2.5
            bandit.q += 4

        # Loop forever:
        for t in range(0, timesteps):
            # Calculate pi_t{A}, the probability of taking action A at timestep t using softmax
            pi = np.exp(H)/np.sum(np.exp(H))

            # Select a random action using the probabilities in pi (this function seems slow for some reason)
            A = np.random.choice(k, 1, p=pi)

            # Interact with the environment
            R = bandit.step(A)
            N[A] += 1

            if kill_baseline:  # This is to recreate Fig. 2.5
                Rmean = 0
            else:
                if nonstationary:  # Average reward computed incrementally with step size (for nonstationary)
                    Rmean = Rmean + a * (R - Rmean)
                else:  # Average reward computed incrementally
                    Rmean = Rmean + (1/(t+1))*(R - Rmean)

            # Update probability of taken action A
            H[A] = H[A] + a*(R - Rmean)*(1 - pi[A])

            # Update probabilities for all other actions
            for b in range(1, k):
                if b != A:
                    H[b] = H[b] - a*(R - Rmean)*pi[b]

            # Store information, not part of the algorithm
            metrics['reward'][n, t] = R
            metrics['action'][n, t] = A
            if A == np.argmax(bandit.q):    # Compare to the action with the optimal value
                metrics['optimal_action'][n, t] = 1
            else:
                metrics['optimal_action'][n, t] = 0

    # Plotting, also not part of the algorithm
    tt = np.arange(0, timesteps, 1)

    ax[0].plot(tt, np.mean(metrics['reward'], axis=0), '-', label='a=' + str(a))
    ax[0].set_ylabel('Average reward')
    ax[0].legend()

    ax[1].plot(tt, np.mean(metrics['optimal_action'], axis=0), '-', label='a=' + str(a))
    ax[1].set_xlabel('Timesteps')
    ax[1].set_ylabel('% Optimal action')
    ax[1].legend()

    return ax


"""
Fig. 2.2: A simple bandit algorithm (pg. 54) for recreating Fig. 2.2 on pg. 51
"""
if False:
    k = 10                              # Number of actions (arms)

    f, ax = plt.subplots(2, 1)
    ax = simple_bandit(k, ax, eps=0.0)
    ax = simple_bandit(k, ax, eps=0.01)
    ax = simple_bandit(k, ax, eps=0.1)
    plt.show()


"""
Exercise 2.5: A simple bandit algorithm (pg. 54) for Excercise 2.5 on pg. 56 (nonstationary bandit problem)
Here the true action value randomly changes after each step (Real life applications are mostly nonstationary).
"""
if False:
    k = 10                              # Number of actions (arms)
    eps = 0.1                           # Probability of taking a greedy (exploiting) action

    n_runs = 100
    timesteps = 10000                    # Number of timesteps for one run

    f, ax = plt.subplots(2, 1)
    ax = simple_bandit(k, ax, eps=eps, n_runs=n_runs, timesteps=timesteps, nonstationary=True)
    ax = simple_bandit(k, ax, eps=eps, a=0.1, n_runs=n_runs, timesteps=timesteps, nonstationary=True)
    plt.show()


"""
Fig. 2.3: A simple bandit algorithm (pg. 54) for recreating Fig. 2.3 on pg. 56 (Optimistic initial values)
Encourage early exploration by setting the initial estimate Q to a very positive value (ex. 5 instead of 0).
Not well suited for nonstationary problems where future exploration is also desired.

Exercise 2.6: Mysterious spikes are related to the use of the constant step size parameter
"""
if False:
    k = 10                              # Number of actions (arms)

    n_runs = 2000

    f, ax = plt.subplots(2, 1)
    ax = simple_bandit(k, ax, eps=0, a=0.1, Q1=5, n_runs=n_runs, nonstationary=False)
    ax = simple_bandit(k, ax, eps=0.1, a=0.1, Q1=0, n_runs=n_runs, nonstationary=False)
    plt.show()


"""
Fig. 2.4: A simple bandit algorithm (pg. 54) for recreating Fig. 2.4 on pg. 58 (Upper confidence bound)
Not practical for nonstationary problems
"""
if False:
    k = 10                              # Number of actions (arms)

    n_runs = 2000

    f, ax = plt.subplots(2, 1)
    ax = simple_bandit(k, ax, c=2, a=0, n_runs=n_runs, nonstationary=False)
    ax = simple_bandit(k, ax, eps=0.1, a=0, n_runs=n_runs, nonstationary=False)
    plt.show()


"""
Fig. 2.5: Gradient algorithm (pg. 59) for recreating Fig. 2.5 on pg. 60 (Gradient bandit)
"""
if True:
    k = 10                              # Number of actions (arms)

    n_runs = 100

    f, ax = plt.subplots(2, 1)
    ax = gradient_bandit(k, ax, a=0.1, n_runs=n_runs, nonstationary=False)
    ax = gradient_bandit(k, ax, a=0.1, n_runs=n_runs, kill_baseline=True, nonstationary=False)
    plt.show()







