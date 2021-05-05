import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

NUM_TRIALS = 2000
BANDIT_MEANS = [1,2,3]

class Bandit:
    def __init__(self, true_mean):
        self.true_mean = true_mean
        # parameters for mu - prior is N(0,1)
        self.predicted_mean = 0
        self.lambda_ = 1
        self.sum_x = 0 # for convenience
        self.tau = 1
        self.N = 0

    def pull(self):
        return np.random.randn() / np.sqrt(self.tau) + self.true_mean

    def sample(self):
        return np.random.randn() / np.sqrt(self.lambda_) + self.predicted_mean

    def update(self, x):
        self.lambda_ += self.tau
        self.sum_x += x
        self.predicted_mean = self.tau * self.sum_x/self.lambda_
        self.N += 1

def plot(bandits, trial):
    x = np.linspace(-3,6,200)
    for b in bandits:
        y = norm.pdf(x, b.predicted_mean, np.sqrt(1./b.lambda_))
        plt.plot(x,y,label=f"real mean: {b.true_mean:.4f}, num_plays = {b.N}")
    plt.title(f"Bandit distributions after {trial} trials")
    plt.legend()
    plt.show()

def run_experiment():
    bandits = [Bandit(p) for p in BANDIT_MEANS]

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