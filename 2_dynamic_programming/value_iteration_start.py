import numpy as np
from grid_world import windy_grid, windy_grid_penalized, ACTION_SPACE
from iterative_policy_evaluation_deterministic_start import print_values, print_policy

SMALL_ENOUGH = 1e-3
GAMMA = 0.9


def get_transition_probs_and_rewards(grid):
    transition_probs = {}
    rewards = {}

    for (s,a), v in grid.probs.items():
        for s2, p in v.items():
            transition_probs[(s,a,s2)] = p
            rewards[(s,a,s2)] = grid.rewards.get(s2,0)

    return transition_probs, rewards

if __name__ == "__main__":
    # grid = windy_grid_penalized(-0.2)
    grid = windy_grid()

    transition_probs, rewards = get_transition_probs_and_rewards(grid)

    # Random initialize policy
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ACTION_SPACE)

    # print initial policy
    print_policy(policy, grid)

    V = {}
    for s in grid.all_states():
        V[s] = 0

    it = 0
    while True:
        delta = 0

        # Single iteration value update
        for s in grid.all_states():
            if not grid.is_terminal(s):
                old_v = V[s]
                new_v = float('-inf')
                for a in ACTION_SPACE: # s에 대해 가능한 action에 대한 max value들을 계산하여 업데이트
                    v = 0
                    for s2 in grid.all_states():
                        r = rewards.get((s, a, s2), 0)
                        v += transition_probs.get((s, a, s2), 0) * (r + GAMMA * V[s2])  # Q를 계산하여 V 갱신

                    if v > new_v:
                        new_v = v

                V[s] = new_v
                # 유의미한 업데이트가 없을경우 중단
                delta = max(delta,np.abs(old_v-V[s]))
        it += 1
        if delta < SMALL_ENOUGH:
            break

    # Single iteration policy update

    for s in grid.all_states():
        new_a = None
        best_value = float('-inf')
        if not grid.is_terminal(s):
            for a in ACTION_SPACE:
                v = 0
                for s2 in grid.all_states():
                    r = rewards.get((s, a, s2), 0)
                    v += transition_probs.get((s, a, s2), 0) * (r + GAMMA * V[s2])  # 위에서 계산한 V값을 기준으로 value를 다시 계산

                if v >= best_value:  # 가장 높은 value를 도출한 a를 새로운 action으로 채택
                    best_value = v
                    new_a = a

            policy[s] = new_a

    print(f"Result policy")
    print_policy(policy, grid)

    print(f"Result value")
    print_values(V, grid)



