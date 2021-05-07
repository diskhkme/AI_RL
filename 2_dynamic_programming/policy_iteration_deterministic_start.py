import numpy as np
from grid_world import standard_grid, ACTION_SPACE
from iterative_policy_evaluation_deterministic_start import print_values, print_policy

SMALL_ENOUGH = 1e-3
GAMMA = 0.9

def get_transition_probs_and_rewards(grid):
    transition_probs = {}
    rewards = {}

    for i in range(grid.rows):
        for j in range(grid.cols):
            s = (i,j)
            if not grid.is_terminal(s):
                for a in ACTION_SPACE:
                    s2 = grid.get_next_state(s,a)
                    transition_probs[(s,a,s2)] = 1 # sparse representation, 없는 경우 0
                    if s2 in grid.rewards:
                        rewards[(s,a,s2)] = grid.rewards[s2] # 계산의 편의성을 위해 rewards tuple을 생성

    return transition_probs, rewards

def evaluate_deterministic_policy(grid, policy):
    """
    주어진 grid와 policy에 대해 value값을 계산
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
                        r = rewards.get((s, a, s2), 0)
                        new_v += action_prob * transition_probs.get((s, a, s2), 0) * (r + GAMMA * V[s2])

                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))

        if biggest_change < SMALL_ENOUGH:
            break

        it += 1

    # print_values(V, grid)

    return V

if __name__ == "__main__":
    grid = standard_grid()

    transition_probs, rewards = get_transition_probs_and_rewards(grid)

    # Random initialize policy
    policy = {}
    for s in grid.actions.keys():
        rand_int = np.random.randint(0,len(grid.actions[s]))
        policy[s] = grid.actions[s][rand_int]

    # print initial policy
    print_policy(policy, grid)

    last_V = {}
    it = 0
    while True:
        V = evaluate_deterministic_policy(grid, policy)
        print(f"Iteration: {it}")
        print_values(V,grid)
        policy_update_flag = True
        for s in grid.actions.keys():
            old_v = V[s]

            for a in grid.actions[s]:
                s2 = grid.get_next_state(s,a)
                r = rewards.get((s, a, s2), 0)
                q_a = transition_probs.get((s, a, s2), 0) * (r + GAMMA * V[s2])
                if q_a > old_v:
                    # update할 policy 결정됨
                    policy[s] = a
                    policy_update_flag = False
                    break

        # policy가 더이상 업데이트 되지 않으면 종료
        if policy_update_flag == True:
            break

        if last_V == V:
            break

        last_V = V

    print(f"Result policy")
    print_policy(policy,grid)

    print(f"Result value")
    print_values(V, grid)



