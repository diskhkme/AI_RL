import numpy as np
from environment.grid_world import standard_grid, negative_grid, print_values, print_policy

GAMMA = 0.9

def play_game(grid, policy, max_steps=20):
    start_states = list(grid.actions.keys())
    start_idx = np.random.choice(len(start_states))
    grid.set_state(start_states[start_idx])

    s = grid.current_state()

    states = [s]
    rewards = [0]

    steps = 0
    while not grid.game_over():
        a = policy[s]
        r = grid.move(a)
        next_s = grid.current_state()

        states.append(next_s)
        rewards.append(r)

        steps += 1
        if steps >= max_steps:
            break

        s = next_s

    return states, rewards


if __name__ == "__main__":
    grid = standard_grid()

    print("rewards:")
    print_values(grid.rewards, grid)

    # Given policy
    policy = {(2, 0): "U",
              (1, 0): "U",
              (0, 0): "R",
              (0, 1): "R",
              (0, 2): "R",
              (1, 2): "U",
              (2, 1): "R",
              (2, 2): "U",
              (2, 3): "L", }

    V = {}
    returns = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            returns[s] = []
        else:
            V[s] = 0

    for i in range(100):
        states, rewards = play_game(grid,policy)
        G = 0
        T = len(states)
        for t in range(T-2,-1,-1):
            s = states[t]
            r = rewards[t+1]
            G = r + GAMMA * G

            if s not in states[:t]:
                returns[s].append(G)
                V[s] = np.mean(returns[s])


    print(f"Result policy")
    print_policy(policy, grid)

    print(f"Result value")
    print_values(V, grid)



