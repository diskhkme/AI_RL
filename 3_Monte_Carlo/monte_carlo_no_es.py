import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from environment.grid_world import standard_grid, negative_grid, print_values, print_policy, ACTION_SPACE

GAMMA = 0.9

def epsilon_greedy(policy, s, eps=0.1):
    p = np.random.random()
    if p < (1-eps):
        return policy[s]
    else:
        return np.random.choice(ACTION_SPACE)

def play_game(grid, policy, max_steps=20):

    # 이제는 정해진 start position(state)가 있는 상태의 문제를 가정함.
    s = grid.reset()

    a = epsilon_greedy(policy,s)

    states = [s]
    rewards = [0]
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
            a = epsilon_greedy(policy,s) # action을 선택할 때 epsilon-greedy를 따름
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

    # Random initial policy, with probability
    policy = {}
    for s in grid.actions.keys(): # 동일. 짧은 코드
        policy[s] = np.random.choice(ACTION_SPACE)

    Q = {}
    sample_counts = {} # return을 모두 저장하지 않고, average를 업데이트만 하기 위해 수정
    state_sample_count = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions: # if not terminal state
            Q[s] = {}
            sample_counts[s] = {}
            state_sample_count[s] = 0
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
                state_sample_count[s] += 1

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

    print("state_sample_count:")
    state_sample_count_arr = np.zeros((grid.rows,grid.cols))
    for i in range(grid.rows):
        for j in range(grid.cols):
            if (i,j) in state_sample_count:
                state_sample_count_arr[i,j] = state_sample_count[(i,j)]

    df = pd.DataFrame(state_sample_count_arr)
    print(df)



