import numpy as np
import matplotlib.pyplot as plt
from environment.grid_world import standard_grid, negative_grid, print_values, print_policy, ACTION_SPACE

GAMMA = 0.9

def play_game(grid, policy, max_steps=20):
    start_states = list(grid.actions.keys())
    start_idx = np.random.choice(len(start_states))
    grid.set_state(start_states[start_idx])

    s = grid.current_state()

    states = [s]
    rewards = [0]

    a = np.random.choice(ACTION_SPACE)
    actions = [a]

    # 내 구현과 동일, 좀 더 짧은 코드
    for _ in range(max_steps):
        r = grid.move(a)
        s = grid.current_state()

        states.append(s)
        rewards.append(r)

        if grid.game_over():
            break
        else:
            a = policy[s]
            actions.append(a)

    return states, actions, rewards

def max_dict(d):
    max_val = max(d.values())
    max_keys = [key for key,val in d.items() if val == max_val]

    return np.random.choice(max_keys), max_val

if __name__ == "__main__":
    grid = standard_grid()

    print("rewards:")
    print_values(grid.rewards, grid)

    # Random initial policy
    policy = {}
    for s in grid.actions.keys(): # 동일. 짧은 코드
        policy[s] = np.random.choice(ACTION_SPACE)

    Q = {}
    sample_counts = {} # return을 모두 저장하지 않고, average를 업데이트만 하기 위해 수정
    states = grid.all_states()
    for s in states:
        if s in grid.actions: # if not terminal state
            Q[s] = {}
            sample_counts[s] = {}
            for a in ACTION_SPACE:
                Q[s][a] = 0
                sample_counts[s][a] = 0


    deltas = [] # for convergence check?
    for it in range(10000):
        if it % 1000 == 0:
            print(it)

        biggest_change = 0

        states, actions, rewards = play_game(grid,policy)
        state_actions = list(zip(states,actions)) # state-action pair for lookup

        G = 0
        T = len(states)

        for t in range(T-2,-1,-1):
            s = states[t]
            a = actions[t]

            G = rewards[t+1] + GAMMA * G

            if (s,a) not in state_actions[:t]:
                old_q = Q[s][a]
                sample_counts[s][a] += 1
                lr = 1/sample_counts[s][a]
                Q[s][a] = old_q + lr * (G-old_q)

                policy[s] = max_dict(Q[s])[0]

                biggest_change = max(biggest_change, np.abs(old_q - Q[s][a]))

        deltas.append(biggest_change)

    plt.plot(deltas)
    plt.show()

    print(f"Result policy")
    print_policy(policy, grid)

    # find V
    V = {}
    for s, Qs in Q.items():
        V[s] = max_dict(Q[s])[1]

    print(f"Result value")
    print_values(V, grid)



