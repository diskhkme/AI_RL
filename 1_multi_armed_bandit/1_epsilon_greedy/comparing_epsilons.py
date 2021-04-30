import matplotlib.pyplot as plt
import numpy as np

NUM_TRIALS = 10000
EPS = 0.1
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]

class BanditArm: # Gaussian distribution reward
    def __init__(self, m):
        # p: the win rate
        self.m = m
        self.m_estimate = 0
        self.N = 0

    def pull(self):
        # draw a 1 with probability p
        return np.random.normal() + self.m

    def update(self, x):
        self.N += 1
        self.m_estimate = (self.m_estimate*(self.N-1) + x)/self.N

def experiment(m1, m2, m3, eps, N):
    bandits = [BanditArm(m1), BanditArm(m2), BanditArm(m3)]

    means = np.array([m1,m2,m3])
    true_best = np.argmax(means)
    count_suboptimal = 0

    data = np.empty(N)

    for i in range(N):

        p = np.random.random()

        if p < eps:
            j = np.random.randint(len(bandits))
        else:
            j = np.argmax([b.m_estimate for b in bandits])

        # pull the arm for the bandit with the largest sample
        x = bandits[j].pull()
        #update the distribution for the bandit whose arm we just pulled
        bandits[j].update(x)

        if j != true_best:
            count_suboptimal += 1

        #update rewards log
        data[i] = x



    # plot the results
    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)

    plt.plot(cumulative_average)
    plt.plot(np.ones(N) * m1)
    plt.plot(np.ones(N) * m2)
    plt.plot(np.ones(N) * m3)
    plt.xscale('log')
    plt.show()

    for b in bandits:
        print(b.m_estimate)

    print(f"percent suboptimal for epsilon = {eps}: {float(count_suboptimal)/N}")

    return cumulative_average

if __name__ == "__main__":
    m1, m2, m3 = 1.5, 2.5, 3.5
    c_1 = experiment(m1,m2,m3, 0.1, 100000)
    c_05 = experiment(m1, m2, m3, 0.05, 100000)
    c_01 = experiment(m1, m2, m3, 0.01, 100000)

    # log scale plot
    plt.plot(c_1, label='eps = 0.1')
    plt.plot(c_05, label='eps = 0.05')
    plt.plot(c_01, label='eps = 0.01')
    plt.legend()
    plt.xscale('log')
    plt.show()

