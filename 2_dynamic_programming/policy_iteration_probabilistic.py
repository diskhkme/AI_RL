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


def evaluate_deterministic_policy(grid, policy):
    """
    주어진 grid와 policy에 대해 value값을 계산. policy는 여전히 deterministic
    """
    V = {}
    for s in grid.all_states():
        V[s] = 0

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
                        new_v += action_prob * transition_probs.get((s,a,s2),0) * (r+GAMMA*V[s2])

                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(old_v-V[s]))

        if biggest_change < SMALL_ENOUGH:
            break

        it += 1

    # print_values(V, grid)

    return V


if __name__ == "__main__":
    grid = windy_grid_penalized(-0.2) # (1,2)의 D가 U로 바뀐것에 주의. 돌아서 가는것의 Penalty가 Lose condition에 빠지는 penalty보다 커지기 때문!
                                        # 극단적으로, penalty를 -2정도로 주면 바로 Lose condition으로 들어가는 경우를 볼 수 있음.
    # grid = windy_grid()

    transition_probs, rewards = get_transition_probs_and_rewards(grid)

    # Random initialize policy
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ACTION_SPACE)

    # print initial policy
    print_policy(policy, grid)

    last_V = {}
    it = 0
    while True:
        V = evaluate_deterministic_policy(grid, policy)

        is_policy_converged = True
        for s in grid.actions.keys():
            old_a = policy[s]
            new_a = None
            best_value = float('-inf')

            for a in ACTION_SPACE:
                v = 0
                for s2 in grid.all_states():
                    r = rewards.get((s, a, s2), 0)
                    v += transition_probs.get((s, a, s2), 0) * (r + GAMMA * V[s2])  # Q를 계산하여 V 갱신

                if v > best_value:  # V가 개선된 경우, a를 새로운 action으로 채택
                    best_value = v
                    new_a = a

            policy[s] = new_a
            if new_a != old_a:
                is_policy_converged = False

        if is_policy_converged:
            break

        it += 1

    print(f"Result policy")
    print_policy(policy, grid)

    print(f"Result value")
    print_values(V, grid)



