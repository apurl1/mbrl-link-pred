import random
import numpy as np
import matplotlib.pyplot as plt
import gridworld

class ValueIteration():
    def __init__(self, env, pi, theta, df):
        # values
        self.V = np.zeros(25)
        self.V_means = []
        # action probabilities
        self.pi = pi
        self.env = env
        self.theta = theta
        self.df = df
        self.steps = 0
    
    def get_policy(self):
        for s in range(25):
            pis = np.zeros(4)
            for a in range(4):
                # s1 is next state
                for s1 in range(25):
                    r = self.env.r(s, a)
                    p = self.env.p(s1, s, a)
                    pis[a] += p * (r + self.df * self.V[s1])
            action = np.argmax(pis)
            self.pi[s, action] = 1
        return self.pi, self.V

    def get_steps(self):
        return self.steps
    
    def get_means(self):
        return self.V_means

    def iterate(self):
        delta = 1
        while delta >= self.theta:
            delta = 0
            self.steps += 1
            for s in range(25):
                vi = self.V[s]
                # calculate new probabilities
                pis = np.zeros(4)
                for a in range(4):
                    # s1 is next state
                    for s1 in range(25):
                        r = self.env.r(s, a)
                        p = self.env.p(s1, s, a)
                        pis[a] += p * (r + self.df * self.V[s1])
                # value of best action
                v = np.max(pis)
                self.V[s] = v
                delta = max(delta, np.abs(vi - v))
            self.V_means.append(np.mean(self.V))

def main():
    # Create environment
    env = gridworld.GridWorld(hard_version=False)

    # Initialize simulation
    s = env.reset()

    # Create log to store data from simulation
    log = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
    }

    pi = np.ones((25, 4)) / 4
    val_iter = ValueIteration(env, pi, 0.001, 0.95)

    # go through value iteration to find optimal policy
    val_iter.iterate()
    pi, v = val_iter.get_policy()

    # Simulate until episode is done
    done = False
    while not done:
        a = np.argmax(pi[s])
        (s, r, done) = env.step(a)

        log['t'].append(log['t'][-1] + 1)
        log['s'].append(s)
        log['a'].append(a)
        log['r'].append(r)

    # plot trajectory
    plt.plot(log['t'], log['s'])
    plt.plot(log['t'][:-1], log['a'])
    plt.plot(log['t'][:-1], log['r'])
    plt.legend(['s', 'a', 'r'])
    plt.title('Value Iteration Trajectory')
    plt.savefig('val_iter_gridworld.png')

    # plot learning curve
    plt.figure()
    plt.plot(np.arange(val_iter.get_steps()), val_iter.get_means())
    plt.title('Value Iteration Learning Curve')
    plt.savefig('val_iter_means.png')

    # visualize policy
    plt.figure()
    plt.pcolor(pi)
    plt.title('Value Iteration Policy')
    plt.savefig('val_iter_policy.png')

if __name__ == '__main__':
    main()
