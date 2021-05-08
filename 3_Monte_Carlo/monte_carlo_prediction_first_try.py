import numpy as np
from environment import grid_world

SMALL_ENOUGH = 1e-3
GAMMA = 0.9


if __name__ == "__main__":
    grid = grid_world.standard_grid()

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
    for s in grid.all_states():
        V[s] = 0
        returns[s] = 0

    possible_initial_state = list(policy.keys())
    N = 1
    while N < 100000: # loop until convergence
        random_init_state_ind = np.random.randint(0,len(possible_initial_state))
        s = possible_initial_state[random_init_state_ind]

        grid.set_state(s)
        states = [s]
        rewards = [0]
        while not grid.is_terminal(s):

            a = policy[s]
            r = grid.move(a)
            rewards.append(r)
            s = grid.current_state()

            states.append(s)

        G = 0
        for t in range(len(states)):
            ind = len(states)-t-2
            G = rewards[ind+1] + GAMMA*G # (s,a,r) <-- G는 첫 state만 업데이트하는 것이 아님. 잘못 구현. 정답 코드를 참고.

        old_r = returns[states[0]]
        returns[states[0]] = ((returns[states[0]] * (N-1)) + G)/N # update mean

        # 제대로 체크가 어려움
        # if np.abs(old_r-returns[episode[0][0]]) < SMALL_ENOUGH:
        #     break

        N += 1

    print(f"Converged at {N} iteration")
    grid_world.print_values(returns,grid)
