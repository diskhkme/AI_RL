import numpy as np
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

    steps = 0
    while not grid.game_over():
        r = grid.move(a)
        next_s = grid.current_state()

        states.append(next_s)
        rewards.append(r)

        if grid.game_over():
            break

        a = policy[next_s]
        actions.append(a)

        steps += 1
        if steps >= max_steps:
            break

        s = next_s

    return states, actions, rewards


if __name__ == "__main__":
    grid = standard_grid()

    print("rewards:")
    print_values(grid.rewards, grid)

    # Random initial policy
    policy = {(2, 0): np.random.choice(ACTION_SPACE),
              (1, 0): np.random.choice(ACTION_SPACE),
              (0, 0): np.random.choice(ACTION_SPACE),
              (0, 1): np.random.choice(ACTION_SPACE),
              (0, 2): np.random.choice(ACTION_SPACE),
              (1, 2): np.random.choice(ACTION_SPACE),
              (2, 1): np.random.choice(ACTION_SPACE),
              (2, 2): np.random.choice(ACTION_SPACE),
              (2, 3): np.random.choice(ACTION_SPACE), }

    Q = {}
    returns = {}
    states = grid.all_states()
    for s in states:
        for a in ACTION_SPACE:
            Q[(s, a)] = 0
            if not grid.is_terminal(s):
                returns[(s,a)] = []

    for i in range(10000):
        states, actions, rewards = play_game(grid,policy)
        G = 0
        T = len(states)
        for t in range(T-2,-1,-1):
            s = states[t]
            r = rewards[t+1]
            a = actions[t]
            G = r + GAMMA * G

            if s not in states[:t]:
                returns[(s,a)].append(G)
                Q[(s,a)] = np.mean(returns[(s,a)])
                state_set = [x for x,y in Q.items() if s in x]
                value_set = [y for x,y in Q.items() if s in x]
                best_ind = np.argmax(value_set)
                policy[s] = state_set[best_ind][1]

    print(f"Result policy")
    print_policy(policy, grid)

    # find V
    V = {}
    for s in grid.all_states():
        value_set = [y for x, y in Q.items() if s in x]
        V[s] = max(value_set)

    print(f"Result value")
    print_values(V, grid)



