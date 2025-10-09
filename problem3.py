import random
import matplotlib.pyplot as plt

grid = [[0,0,0,0,1],
        [0,None,-1,0,0],
        [0,0,0,0,0],
        [0,0,None,None,0],
        [0,0,0,0,0]]
transitions = [["right", "right", "right", "right", "right"],
                ["left", "up", "left", "left", "up"],
                ["up", "up", "right", "right", "right"],
                ["up", "down", "down", "down", "up"],
                ["up", "right", "right", "up", "up"]]
discount_factor = 0.95
threshold = 0.00000001

def getValue(prevValue):
    newValue = [[0]*5 for i in range(5)]
    converged = True
    for r in range(5):
        for c in range(5):
            if grid[r][c] == None:
                continue
            value = grid[r][c] if grid[r][c] is not None else 0
            transition = transitions[r][c]
            expected = 0
            # top
            percentage = 0.85 if transition == "up" else 0.05
            topval = prevValue[r-1][c] if r > 0 and grid[r-1][c] != None else prevValue[r][c]
            expected += percentage * topval
            # bottom
            percentage = 0.85 if transition == "down" else 0.05
            botval = prevValue[r+1][c] if r < 4 and grid[r+1][c] != None else prevValue[r][c]
            expected += percentage * botval
            # left
            percentage = 0.85 if transition == "left" else 0.05
            leftval = prevValue[r][c-1] if c > 0 and grid[r][c-1] != None else prevValue[r][c]
            expected += percentage * leftval
            # right
            percentage = 0.85 if transition == "right" else 0.05
            rightval = prevValue[r][c+1] if c < 4 and grid[r][c+1] != None else prevValue[r][c]
            expected += percentage * rightval

            newValue[r][c] = value + discount_factor*expected
            if abs(newValue[r][c] - prevValue[r][c]) > threshold:
                converged = False
    return newValue, converged

def generateTrajectory(timesteps):
    total = 0
    r = 4
    c = 0
    
    for i in range(timesteps):
        # Get immediate reward and apply discount
        value = grid[r][c] if grid[r][c] is not None else 0
        total += value * (discount_factor ** i)
        
        # Random movement selection with weighted probabilities
        upProb = 0.85 if transitions[r][c] == "up" else 0.05
        downProb = 0.85 if transitions[r][c] == "down" else 0.05
        leftProb = 0.85 if transitions[r][c] == "left" else 0.05
        rightProb = 0.85 if transitions[r][c] == "right" else 0.05
        directions = ['left', 'right', 'up', 'down']
        probabilities = [leftProb, rightProb, upProb, downProb]
        direction = random.choices(directions, weights=probabilities)[0]
        
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
        
    return total

ACTIONS = ['up','down','left','right']

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
    # print("part a")
    # value = [[0]*5 for i in range(5)]
    # convergence = False
    # while not convergence:
    #     value, convergence = getValue(value)
    #     count += 1
    # print("value function for all states")
    # print(value)
    # print("final converged value for start")
    # print(value[4][0])

    # print("part b")
    # total = 0
    # num_samples = 100
    # for i in range(num_samples):
    #     total += generateTrajectory(50)
    # print("mean: ", total/num_samples)

    # print("part c")
    # value = [[0]*5 for i in range(5)]
    # iterations = 150
    # for i in range(iterations):
    #     value, convergence = getValue(value)
    # print("value function for all states")
    # print(value)
    # print("final value for start")
    # print(value[4][0])

    # print("part e")
    # # get convergence value
    # value = [[0]*5 for i in range(5)]
    # convergence = False
    # errors = []
    # while not convergence:
    #     value, convergence = getValue(value)

    # true_value = value

    # value = [[0]*5 for i in range(5)]
    # convergence = False
    # errors = []
    # count = 0
    # while not convergence:
    #     value, convergence = getValue(value)
    #     diff = max(abs(value[r][c] - true_value[r][c])
    #            for r in range(5) for c in range(5)
    #            if grid[r][c] is not None)
    #     errors.append(diff)
    #     count += 1

    # plt.figure()
    # plt.plot(range(count), errors, marker='o')
    # plt.xlabel('Iteration t')
    # plt.ylabel('Error')
    # plt.title('Convergence of Approximate Policy Evaluation')
    # # plt.yscale('log')  # optional: log-scale helps see exponential decay
    # plt.grid(True)
    # plt.show()

    # print("part 2")
    # value, policy = value_iteration()
    # print("learned policy")
    # print(policy)
    # print("value at all states")
    # print(value)

    print("part 3")
    # --- Policy Iteration driver ---
    pi = init_uniform_policy()
    max_outer_iters = 50 
    for k in range(max_outer_iters):
        V = eval_policy(pi, T=50)  # partial eval to speed up iterations
        new_pi = improve_policy(V)
        # If first time moving from stochastic to deterministic, just continue
        if isinstance(pi[0][0], dict):
            pi = new_pi
            continue
        # Check for stability after switched to deterministic
        if policies_equal(pi, new_pi):
            pi = new_pi
            print("policy converged at: ", k, "iterations")
            break
        pi = new_pi

    # Final guaranteed-accuracy evaluation:
    V_final = eval_policy(pi, T=149) 

    # Pretty-print learned policy and values
    for r in range(5):
        row = []
        for c in range(5):
            if grid[r][c] is None:
                row.append('#')
            else:
                row.append(pi[r][c])  # 'up'/'down'/'left'/'right'
        print(row)
    for row in V_final:
        print(row)
