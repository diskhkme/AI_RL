from environment.grid_world import standard_grid

# 1. Define helper functions to print value and policy
def print_policy(policy, g):
    all_states = g.all_states()
    max_row = 0
    max_col = 0
    for state in all_states:
        max_row = max(max_row, state[0])
        max_col = max(max_col, state[1])

    for i in range(max_row+1):
        separator = ""
        str = ""
        for j in range(max_col+1):
            separator += "----"
            if (i,j) in policy:
                str += f" {policy[(i,j)]} |"
            else:
                str += "   |"

        print(separator)
        print(str)

def print_value(value, g):
    all_states = g.all_states()
    max_row = 0
    max_col = 0
    for state in all_states:
        max_row = max(max_row, state[0])
        max_col = max(max_col, state[1])

    for i in range(max_row+1):
        separator = ""
        str = ""
        for j in range(max_col+1):
            separator += "----"
            if (i,j) in policy:
                str += f" {value[(i,j)]:.2f} |"
            else:
                str += "   |"

        print(separator)
        print(str)

def build_state_transition_dict(policy, g):
    all_states = g.all_states()
    max_row = 0
    max_col = 0
    for state in all_states:
        max_row = max(max_row, state[0])
        max_col = max(max_col, state[1])

    state_tr = {}
    for i in range(max_row):
        for j in range(max_col):
            if policy.get((i,j)) == None:
                continue

            if policy[(i,j)] == "R":
                state_tr[(i,j,i,j+1)] = 1
            elif policy[(i,j)] == "L":
                state_tr[(i, j, i,j - 1)] = 1
            elif policy[(i,j)] == "U":
                state_tr[(i, j,i+1, j)] = 1
            elif policy[(i,j)] == "D":
                state_tr[(i, j, i-1,j)] = 1

    return state_tr

def get_zero_values_dict(g):
    all_states = g.all_states()
    max_row = 0
    max_col = 0
    for state in all_states:
        max_row = max(max_row, state[0])
        max_col = max(max_col, state[1])

    values = {}
    for i in range(max_row):
        for j in range(max_col):
            values[(i,j)] = 0

    return values

g = standard_grid()

policy = {(2,0): "U",
          (1,0): "U",
          (0,0): "R",
          (0,1): "R",
          (0,2): "R",
          (1,2): "U",
          (2,1): "R",
          (2,2): "U",
          (2,3): "L",}

print_policy(policy,g)

state_tr = build_state_transition_dict(policy, g)

MAX_EPS = 0.01
GAMMA = 0.9
values = get_zero_values_dict(g)

while True:
    all_states = g.all_states()
    max_row = 0
    max_col = 0
    for state in all_states:
        max_row = max(max_row, state[0])
        max_col = max(max_col, state[1])

    for i in range(max_row):
        for j in range(max_col):
            possible_actions_in_state = [x for x in g.actions[(i,j)]]
            for a in possible_actions_in_state:
                s_prime = g.get_next_state((i,j),a)
                state_tr_key = (i,j,s_prime[0],s_prime[1])
                if g.rewards[s_prime] is not None:
                    reward = g.rewards[s_prime]
                else:
                    reward = 0

# 코드가 지저분하네...


