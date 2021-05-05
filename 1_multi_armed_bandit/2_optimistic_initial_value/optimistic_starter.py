import matplotlib.pyplot as plt
import numpy as np

NUM_TRIALS = 10000
# EPS = 0.1
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]

class Bandit:
    def __init__(self, p):
        # p: the win rate
        self.p = p
        self.p_estimate = 10
        self.N = 1 # 초기값을 1로 설정한 것에 주의

    def pull(self):
        # draw a 1 with probability p
        return np.random.random() < self.p

    def update(self, x):
        self.N += 1
        self.p_estimate = (self.p_estimate*(self.N-1) + x)/self.N

def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]

    rewards = np.zeros(NUM_TRIALS)

    for i in range(NUM_TRIALS):

        # use epsilon-greedy to select the next bandit
        j = np.argmax([b.p_estimate for b in bandits])

        # pull the arm for the bandit with the largest sample
        x = bandits[j].pull()

        #update rewards log
        rewards[i] = x

        #update the distribution for the bandit whose arm we just pulled
        bandits[j].update(x)

    # print mean estimated for each bandit
    for b in bandits:
        print(f"mean estimate: {b.p_estimate}")

    # print total reward
    print(f"total reward earned: {rewards.sum()}")
    print(f"overall win rate: {rewards.sum() / NUM_TRIALS}")
    print(f"num times selected each bandit : {[b.N for b in bandits]}")

    # plot the results
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
    plt.ylim([0,1])
    plt.plot(win_rates)
    plt.plot(np.ones(NUM_TRIALS)*np.max(BANDIT_PROBABILITIES))
    plt.show()

if __name__ == "__main__":
    experiment()