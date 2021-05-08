import numpy as np
from environment.grid_world import standard_grid, ACTION_SPACE

SMALL_ENOUGH = 1e-3

def print_values(V,g):
    for i in range(g.rows):
        print("------------------------")
        for j in range(g.cols):
            v = V.get((i,j),0) # <-- set에서 데이터를 가져올 때, 없으면 0을 반환
            if v >= 0:
                print(f" {v:.2f} |", end="")
            else:
                print(f"{v:.2f} |", end="")
        print("")

def print_policy(P,g):
    for i in range(g.rows):
        print("------------------------")
        for j in range(g.cols):
            a = P.get((i,j), ' ')
            print(f"  {a}  |", end="")
        print("")

if __name__ == "__main__":
    transition_probs = {}
    rewards = {}

    grid = standard_grid()

    for i in range(grid.rows):
        for j in range(grid.cols):
            s = (i,j)
            if not grid.is_terminal(s):
                for a in ACTION_SPACE:
                    s2 = grid.get_next_state(s,a)
                    transition_probs[(s,a,s2)] = 1 # sparse representation, 없는 경우 0
                    if s2 in grid.rewards:
                        rewards[(s,a,s2)] = grid.rewards[s2] # 계산의 편의성을 위해 rewards tuple을 생성

    policy = {(2, 0): "U",
              (1, 0): "U",
              (0, 0): "R",
              (0, 1): "R",
              (0, 2): "R",
              (1, 2): "U",
              (2, 1): "R",
              (2, 2): "U",
              (2, 3): "L", }
    print_policy(policy,grid)

    V = {}
    for s in grid.all_states():
        V[s] = 0

    gamma = 0.9

    it = 0
    while True:
        biggest_change = 0
        for s in grid.all_states():
            if not grid.is_terminal(s):
                old_v = V[s]
                new_v = 0
                for a in ACTION_SPACE:
                    for s2 in grid.all_states():
                        action_prob = 1 if policy.get(s) == a else 0
                        r = rewards.get((s,a,s2),0)
                        new_v += action_prob * transition_probs.get((s,a,s2),0) * (r+gamma*V[s2])

                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(old_v-V[s]))

        print(f"iter: {it}, biggest change: {biggest_change}")
        print_values(V,grid)

        if biggest_change < SMALL_ENOUGH:
            break

        it += 1
