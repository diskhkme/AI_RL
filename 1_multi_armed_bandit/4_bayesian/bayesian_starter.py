import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

NUM_TRIALS = 2000
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]

class Bandit:
    def __init__(self, p):
        # p: the win rate
        self.p = p
        self.a = 1
        self.b = 1
        self.N = 0 # for information only

    def pull(self):
        # draw a 1 with probability p
        return np.random.random() < self.p

    def sample(self):
        # TODO - draw a sample from Beta(a, b)
        return np.random.beta(self.a, self.b)

    def update(self, x): 
        self.N += 1
        self.a += x
        self.b += 1-x

def plot(bandits, trial):
    x = np.linspace(0,1,200)
    for b in bandits:
        y = beta.pdf(x, b.a, b.b)
        plt.plot(x,y,label=f"real p: {b.p:.4f}, win rate = {b.a-1}/{b.N}")
    plt.title(f"Bandit distributions after {trial} trials")
    plt.legend()
    plt.show()

def run_experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]

    sample_points = [5,10,20,50,100,200,500,1000,1500,1999]
    rewards = np.zeros(NUM_TRIALS)

    for i in range(NUM_TRIALS):
        j = np.argmax([b.sample() for b in bandits])

        if i in sample_points:
            plot(bandits, i)

        x = bandits[j].pull()
        rewards[i] = x
        bandits[j].update(x)

    print("total reward earned:", rewards.sum())

if __name__ == "__main__":
    run_experiment()