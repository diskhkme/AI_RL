import numpy as np

ACTION_SPACE = ("L","R","U","D")

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

class Grid:
    def __init__(self, rows, cols, start):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.i = start[0]
        self.j = start[1]

    def reset(self):
        self.i = self.start[0]
        self.j = self.start[1]
        return (self.i, self.j)

    def set(self, rewards, actions):
        '''
        initialize grid world's rewards and actions

        Arguments:
            rewards (dict): (row,col):reward
            actions (dict): (row,col):actions list
        Returns:
            None
        '''
        self.rewards = rewards
        self.actions = actions

    def set_state(self, s):
        self.i = s[0]
        self.j = s[1]

    def current_state(self):
        return (self.i, self.j)

    def game_over(self):
        return (self.i, self.j) not in self.actions

    def all_states(self):
        """
        Returns possible all states
        """
        return set(self.actions.keys()) | set(self.rewards.keys())

    def is_terminal(self, s):
        return s not in self.actions

    def get_next_state(self, s, a):
        i, j = s[0], s[1]
        if a in self.actions[(i,j)]:
            if a == 'U':
                i -= 1
            elif a == 'D':
                i += 1
            elif a == 'L':
                j -= 1
            elif a == 'R':
                j += 1

        return i,j

    def move(self, action):
        if action in self.actions[(self.i, self.j)]:
            if action == 'U':
                self.i -= 1
            elif action == 'D':
                self.i += 1
            elif action == 'L':
                self.j -= 1
            elif action == 'R':
                self.j += 1

        return self.rewards.get((self.i, self.j), 0) # reward에 없는경우 0을 return

    def undo_move(self, action):
        if action in self.actions[(self.i, self.j)]:
            if action == 'U':
                self.i += 1
            elif action == 'D':
                self.i -= 1
            elif action == 'L':
                self.j += 1
            elif action == 'R':
                self.j -= 1

        assert(self.current_state() in self.all_states())

class WindyGrid:
    """
    State transition probability is "NOT" deterministic
    """
    def __init__(self, rows, cols, start):
        self.rows = rows
        self.cols = cols
        self.i = start[0]
        self.j = start[1]

    def set(self, rewards, actions, probs):
        '''
        initialize grid world's rewards and actions

        Arguments:
            rewards (dict): (row,col):reward
            actions (dict): (row,col):actions list
            probs (dict): (row,col),'action': (row2,col2),prob2 , (row3,col3),prob3, ... (sum of prob2~N) must be 1
        Returns:
            None
        '''
        self.rewards = rewards
        self.actions = actions
        self.probs = probs

    def set_state(self, s):
        self.i = s[0]
        self.j = s[1]

    def current_state(self):
        return (self.i, self.j)

    def game_over(self):
        return (self.i, self.j) not in self.actions

    def all_states(self):
        """
        Returns possible all states
        """
        return set(self.actions.keys()) | set(self.rewards.keys())

    def is_terminal(self, s):
        return s not in self.actions

#    def get_next_state(self, s, a): # next state가 deterministic이 아니기 때문에, get_next_state는 없음
#    def undo_move(self, action): # undo도 이전 상태가 불확정이기 때문에 없음
    def move(self, action):
        s = (self.i, self.j)
        a = action

        next_state_probs = self.probs[(s,a)]
        next_states = list(next_state_probs.keys())
        next_probs = list(next_state_probs.values())
        s2 = np.random.choice(next_states, p=next_probs)

        self.i, self.j = s2
        return self.rewards.get(s2,0)

def windy_grid_penalized(step_cost = -0.1):
    """
    probabilistic env에서 움직임에 대한 ponalty(negative reward를 추가)
    """

    g = WindyGrid(3,4,(2,0))
    rewards = {(0, 3): 1,
               (1, 3): -1,
               (0, 0): step_cost,
               (0, 1): step_cost,
               (0, 2): step_cost,
               (1, 0): step_cost,
               (1, 1): step_cost,
               (1, 2): step_cost,
               (2, 0): step_cost,
               (2, 1): step_cost,
               (2, 2): step_cost,
               (2, 3): step_cost,}

    actions = {(0, 0): ("D", "R"),
               (0, 1): ("L", "R"),
               (0, 2): ("L", "R", "D"),
               (1, 0): ("U", "D"),
               (1, 2): ("U", "D", "R"),
               (2, 0): ("U", "R"),
               (2, 1): ("L", "R"),
               (2, 2): ("L", "R", "U"),
               (2, 3): ("L", "U"), }

    probs = {
        ((2, 0), 'U'): {(1, 0): 1.0},
        ((2, 0), 'D'): {(2, 0): 1.0},
        ((2, 0), 'L'): {(2, 0): 1.0},
        ((2, 0), 'R'): {(2, 1): 1.0},
        ((1, 0), 'U'): {(0, 0): 1.0},
        ((1, 0), 'D'): {(2, 0): 1.0},
        ((1, 0), 'L'): {(1, 0): 1.0},
        ((1, 0), 'R'): {(1, 0): 1.0},
        ((0, 0), 'U'): {(0, 0): 1.0},
        ((0, 0), 'D'): {(1, 0): 1.0},
        ((0, 0), 'L'): {(0, 0): 1.0},
        ((0, 0), 'R'): {(0, 1): 1.0},
        ((0, 1), 'U'): {(0, 1): 1.0},
        ((0, 1), 'D'): {(0, 1): 1.0},
        ((0, 1), 'L'): {(0, 0): 1.0},
        ((0, 1), 'R'): {(0, 2): 1.0},
        ((0, 2), 'U'): {(0, 2): 1.0},
        ((0, 2), 'D'): {(1, 2): 1.0},
        ((0, 2), 'L'): {(0, 1): 1.0},
        ((0, 2), 'R'): {(0, 3): 1.0},
        ((2, 1), 'U'): {(2, 1): 1.0},
        ((2, 1), 'D'): {(2, 1): 1.0},
        ((2, 1), 'L'): {(2, 0): 1.0},
        ((2, 1), 'R'): {(2, 2): 1.0},
        ((2, 2), 'U'): {(1, 2): 1.0},
        ((2, 2), 'D'): {(2, 2): 1.0},
        ((2, 2), 'L'): {(2, 1): 1.0},
        ((2, 2), 'R'): {(2, 3): 1.0},
        ((2, 3), 'U'): {(1, 3): 1.0},
        ((2, 3), 'D'): {(2, 3): 1.0},
        ((2, 3), 'L'): {(2, 2): 1.0},
        ((2, 3), 'R'): {(2, 3): 1.0},
        ((1, 2), 'U'): {(0, 2): 0.5, (1, 3): 0.5},
        ((1, 2), 'D'): {(2, 2): 1.0},
        ((1, 2), 'L'): {(1, 2): 1.0},
        ((1, 2), 'R'): {(1, 3): 1.0},
    }

    g.set(rewards,actions,probs)
    return g

def windy_grid():
    g = WindyGrid(3,4,(2,0))
    rewards = {(0, 3): 1,
               (1, 3): -1}
    actions = {(0, 0): ("D", "R"),
               (0, 1): ("L", "R"),
               (0, 2): ("L", "R", "D"),
               (1, 0): ("U", "D"),
               (1, 2): ("U", "D", "R"),
               (2, 0): ("U", "R"),
               (2, 1): ("L", "R"),
               (2, 2): ("L", "R", "U"),
               (2, 3): ("L", "U"), }

    probs = {
        ((2, 0), 'U'): {(1, 0): 1.0},
        ((2, 0), 'D'): {(2, 0): 1.0},
        ((2, 0), 'L'): {(2, 0): 1.0},
        ((2, 0), 'R'): {(2, 1): 1.0},
        ((1, 0), 'U'): {(0, 0): 1.0},
        ((1, 0), 'D'): {(2, 0): 1.0},
        ((1, 0), 'L'): {(1, 0): 1.0},
        ((1, 0), 'R'): {(1, 0): 1.0},
        ((0, 0), 'U'): {(0, 0): 1.0},
        ((0, 0), 'D'): {(1, 0): 1.0},
        ((0, 0), 'L'): {(0, 0): 1.0},
        ((0, 0), 'R'): {(0, 1): 1.0},
        ((0, 1), 'U'): {(0, 1): 1.0},
        ((0, 1), 'D'): {(0, 1): 1.0},
        ((0, 1), 'L'): {(0, 0): 1.0},
        ((0, 1), 'R'): {(0, 2): 1.0},
        ((0, 2), 'U'): {(0, 2): 1.0},
        ((0, 2), 'D'): {(1, 2): 1.0},
        ((0, 2), 'L'): {(0, 1): 1.0},
        ((0, 2), 'R'): {(0, 3): 1.0},
        ((2, 1), 'U'): {(2, 1): 1.0},
        ((2, 1), 'D'): {(2, 1): 1.0},
        ((2, 1), 'L'): {(2, 0): 1.0},
        ((2, 1), 'R'): {(2, 2): 1.0},
        ((2, 2), 'U'): {(1, 2): 1.0},
        ((2, 2), 'D'): {(2, 2): 1.0},
        ((2, 2), 'L'): {(2, 1): 1.0},
        ((2, 2), 'R'): {(2, 3): 1.0},
        ((2, 3), 'U'): {(1, 3): 1.0},
        ((2, 3), 'D'): {(2, 3): 1.0},
        ((2, 3), 'L'): {(2, 2): 1.0},
        ((2, 3), 'R'): {(2, 3): 1.0},
        ((1, 2), 'U'): {(0, 2): 0.5, (1, 3): 0.5},
        ((1, 2), 'D'): {(2, 2): 1.0},
        ((1, 2), 'L'): {(1, 2): 1.0},
        ((1, 2), 'R'): {(1, 3): 1.0},
    }

    g.set(rewards,actions,probs)
    return g


def standard_grid():
    g = Grid(3, 4, (2, 0))
    rewards = {(0, 3): 1,
               (1, 3): -1}
    actions = {(0, 0): ("D", "R"),
               (0, 1): ("L", "R"),
               (0, 2): ("L", "R", "D"),
               (1, 0): ("U", "D"),
               (1, 2): ("U", "D", "R"),
               (2, 0): ("U", "R"),
               (2, 1): ("L", "R"),
               (2, 2): ("L", "R", "U"),
               (2, 3): ("L", "U"), }
    g.set(rewards, actions)
    return g

def negative_grid(step_cost = -0.1):
    g = Grid(3, 4, (2, 0))
    rewards = {(0, 3): 1,
               (1, 3): -1,
               (0, 0): step_cost,
               (0, 1): step_cost,
               (0, 2): step_cost,
               (1, 0): step_cost,
               (1, 1): step_cost,
               (1, 2): step_cost,
               (2, 0): step_cost,
               (2, 1): step_cost,
               (2, 2): step_cost,
               (2, 3): step_cost,}
    actions = {(0, 0): ("D", "R"),
               (0, 1): ("L", "R"),
               (0, 2): ("L", "R", "D"),
               (1, 0): ("U", "D"),
               (1, 2): ("U", "D", "R"),
               (2, 0): ("U", "R"),
               (2, 1): ("L", "R"),
               (2, 2): ("L", "R", "U"),
               (2, 3): ("L", "U"), }
    g.set(rewards, actions)
    return g