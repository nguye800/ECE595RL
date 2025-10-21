import random
import matplotlib.pyplot as plt

grid = [[0,0,0,0,1],
        [0,None,-1,0,0],
        [0,0,0,0,0],
        [0,0,None,None,0],
        [0,0,0,0,0]]
transitions = [["right", "right", "right", "right", "up"],
                ["left", "up", "left", "left", "up"],
                ["up", "up", "right", "right", "right"],
                ["up", "down", "down", "down", "up"],
                ["up", "right", "right", "up", "up"]]
discount_factor = 0.95
ACTIONS = ['up','down','left','right']

# get value function of each state in a trajectory
def getValue(trajectory): 
    values = [[ [0, 0] for _ in range(5)] for _ in range(5)] # initialize all states to 0
    seen = set() # set of all seen states for first visit

    for i, step in enumerate(trajectory):
        state, reward, action = step
        if state in seen:
            continue
        else:
            r,c = state
            # since all rewards are 0 expect terminal state reward
            # G = gamma^(terminal-i) * reward at terminal
            values[r][c][0] += discount_factor**(len(trajectory) - 1 - i) * trajectory[-1][1]
            values[r][c][1] += 1
    return values


def generateTrajectory(startrow, startcol): # keep track of all actions and states throughout trajectory
    trajectory = []
    r = startrow
    c = startcol
    reward = 0

    while reward == 0:    
        # update reward
        reward = grid[r][c]    
        # Random movement selection with weighted probabilities
        upProb = 0.85 if transitions[r][c] == "up" else 0.05
        downProb = 0.85 if transitions[r][c] == "down" else 0.05
        leftProb = 0.85 if transitions[r][c] == "left" else 0.05
        rightProb = 0.85 if transitions[r][c] == "right" else 0.05
        directions = ['left', 'right', 'up', 'down']
        probabilities = [leftProb, rightProb, upProb, downProb]

        if reward != 0:
            direction = 'terminate'
        else:
            direction = random.choices(directions, weights=probabilities)[0]

        # append to trajectory: state, reward, action
        if reward == None:
            trajectory.append(((r,c), 0, direction))
        else:
            trajectory.append(((r,c), reward, direction))
        
        # Update position based on random direction
        if direction == 'left' and c > 0 and grid[r][c-1] is not None:
            c -= 1
        elif direction == 'right' and c < 4 and grid[r][c+1] is not None:
            c += 1
        elif direction == 'up' and r > 0 and grid[r-1][c] is not None:
            r -= 1
        elif direction == 'down' and r < 4 and grid[r+1][c] is not None:
            r += 1
        # If movement is invalid, stay in current position

    return trajectory

# expected value of taking action a from (r,c)
def step(rr, cc, move):
    nr, nc = rr, cc
    if move == 'up':    nr -= 1
    if move == 'down':  nr += 1
    if move == 'left':  nc -= 1
    if move == 'right': nc += 1
    # blocked or off-grid -> stay
    if not (0 <= nr < 5 and 0 <= nc < 5) or grid[nr][nc] is None:
        nr, nc = rr, cc
    return nr, nc

def expected_next_value(V, r, c, a):
    probs = {m:0.05 for m in ACTIONS}
    probs[a] = 0.85
    ev = 0.0
    for move, p in probs.items():
        nr, nc = step(r, c, move)
        ev += p * V[nr][nc]
    return ev

def next_value(prev, r, c, a):
    # absorbing transitions for special cells
    if grid[r][c] in (1, -1) or grid[r][c] is None:
        return prev[r][c]  # stay

    return expected_next_value(prev, r, c, a)

def value_iteration(T=149):
    V = [[0.0]*5 for _ in range(5)]
    for _ in range(T):
        newV = [[0.0]*5 for _ in range(5)]
        for r in range(5):
            for c in range(5):
                if grid[r][c] is None:
                    continue
                reward = grid[r][c] or 0
                if grid[r][c] in (1, -1):       # absorbing reward cells
                    newV[r][c] = reward + discount_factor * V[r][c]
                else:
                    best = max(next_value(V, r, c, a) for a in ACTIONS)
                    newV[r][c] = reward + discount_factor * best
        V = newV
    # greedy policy extraction
    policy = [['']*5 for _ in range(5)]
    for r in range(5):
        for c in range(5):
            if grid[r][c] is None:
                policy[r][c] = None
            elif grid[r][c] in (1,-1):
                policy[r][c] = 'stay'
            else:
                q = {a: next_value(V, r, c, a) for a in ACTIONS}
                policy[r][c] = max(q, key=q.get)
    return V, policy

def eval_policy(pi, T):
    """Iterative policy evaluation for T sweeps from V0=0."""
    V = [[0.0]*5 for _ in range(5)]
    for _ in range(T):
        newV = [[0.0]*5 for _ in range(5)]
        for r in range(5):
            for c in range(5):
                if grid[r][c] is None:
                    continue
                reward = grid[r][c] or 0.0
                # policy is a distribution over actions in normal cells
                if isinstance(pi[r][c], dict):
                    ev = 0.0
                    for a, pa in pi[r][c].items():
                        ev += pa * expected_next_value(V, r, c, a)
                    newV[r][c] = reward + discount_factor * ev
                else:
                    # deterministic label treat as prob 1 on that action
                    a = pi[r][c]
                    ev = expected_next_value(V, r, c, a)
                    newV[r][c] = reward + discount_factor * ev
        V = newV
    return V

def improve_policy(V):
    """Greedy improvement: argmax_a E[r + gamma V(s')]."""
    new_pi = [[None]*5 for _ in range(5)]
    policy_stable = True
    for r in range(5):
        for c in range(5):
            if grid[r][c] is None:
                continue
            reward = grid[r][c] or 0.0
            # compute Q(s,a) = r(s) + gamma * E[V(s') | a]
            q_vals = {a: reward + discount_factor * expected_next_value(V, r, c, a) for a in ACTIONS}
            best_a = max(q_vals, key=q_vals.get)
            # if the old policy had multiple actions (stochastic), we still switch to deterministic best
            new_pi[r][c] = best_a
    return new_pi

def init_uniform_policy():
    """π0(a|s)=0.25 for each action in non-wall cells."""
    pi = [[None]*5 for _ in range(5)]
    for r in range(5):
        for c in range(5):
            if grid[r][c] is None:
                continue
            pi[r][c] = {a: 0.25 for a in ACTIONS}
    return pi

def policies_equal(pi1, pi2):
    for r in range(5):
        for c in range(5):
            if grid[r][c] is None:
                continue
            # treat dict vs str comparison carefully
            a1 = pi1[r][c]
            a2 = pi2[r][c]
            if isinstance(a1, dict) or isinstance(a2, dict):
                return False  # once we improve, we switch to deterministic
            if a1 != a2:
                return False
    return True

def init_uniform_policy():
    """π0(a|s)=0.25 for each action in non-wall cells."""
    pi = [[None]*5 for _ in range(5)]
    for r in range(5):
        for c in range(5):
            if grid[r][c] is None:
                continue
            pi[r][c] = {a: 0.25 for a in ACTIONS}
    return pi


if __name__ == '__main__':
    # print("part 1")
    valueAggregate = [[[0, 0] for _ in range(5)] for _ in range(5)]
    for r in range(5):
        for c in range(5):
            for sample in range(3):
                trajectory = generateTrajectory(r,c)
                values = getValue(trajectory)
                # loop through all states in the trajectory to update total
                for trajr in range(5):
                    for trajc in range(5):
                        g, n = values[trajr][trajc]
                        if n != 0:
                            valueAggregate[trajr][trajc][0] += g
                            valueAggregate[trajr][trajc][1] += n
    # aggregate all values
    valueFunction = [[0]*5 for _ in range(5)]
    for r in range(5):
        for c in range(5):
            g,n = valueAggregate[r][c]
            valueFunction[r][c] = g / n if n > 0 else 0
    print(valueFunction)
