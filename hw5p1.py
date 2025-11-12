import random
import matplotlib.pyplot as plt
import numpy as np
import math

# Transition matrix: rows: 1,2,3 cols: 1,2,3
transition_g = [[0.1,0.8,0.1],
                [0.1,0.1,0.8],
                [0.8,0.1,0.1]]

transition_h = [[0.8,0.1,0.1],
                [0.1,0.8,0.1],
                [0.1,0.1,0.8]]

reward = [[0, 0],
          [0, 0],
          [0, 1]]

# policies rows: 1,2,3 cols: g,h
target = [[0.9, 0.1],
          [0.9, 0.1],
          [0.1, 0.9]]

behavior = [[0.85, 0.15],
            [0.88, 0.12],
            [0.1, 0.9]]

def get_value(policy, value, discount=0.95):
    new_value = [0]*3
    for i in range(3):
        for action in range(2):
            temp_val = 0.0
            P = transition_g if action == 0 else transition_h
            for state in range(3):
                temp_val += P[i][state] * (reward[i][action] + discount*value[state])
            temp_val *= policy[i][action]
            new_value[i] += temp_val
    return new_value

def step(state, action):
    P = transition_g if action == 0 else transition_h
    next_state = random.choices([0,1,2], weights=P[state])[0]
    r = reward[state][action]          # R(s,a)
    return r, next_state

def simulate_trajectory_from_s1(policy, H):
    """Follow 'policy' for exactly H steps starting from state 0 (s=1)."""
    s = 0
    states, actions, rewards = [], [], []
    for t in range(H):
        a = random.choices([0,1], weights=policy[s])[0]
        r, s_next = step(s, a)
        states.append(s); actions.append(a); rewards.append(r)
        s = s_next
    return states, actions, rewards

def discounted_return(rewards, gamma):
    G = 0.0
    p = 1.0
    for r in rewards:
        G += p * r
        p *= gamma
    return G

def trajectory_ratio(states, actions, pi_t, pi_b):
    rho = 1.0
    for s, a in zip(states, actions):
        # assume both policies assign nonzero prob to both actions (true here)
        rho *= pi_t[s][a] / pi_b[s][a]
    return rho

if __name__ == "__main__":
    print("Part 1")
    iterations = 100000
    value_target = [0]*3
    value_behavior = [0]*3
    for i in range(iterations):
        value_target = get_value(target, value_target)
        value_behavior = get_value(behavior, value_behavior)
    print("Target Value: ", value_target)
    print("Behavior Value: ", value_behavior)

    print("Part 2")
    accuracy_level = 0.1
    R_max = 1
    discount = 0.95
    numerator = math.log(accuracy_level*(1-discount) / R_max)
    denominator = math.log(discount)
    effective_horizon = math.ceil(numerator/denominator)
    print("timesteps required: ", effective_horizon)

    print("Part 3")
    discount = 0.95
    num_traj = 50
    sum_rhoG = 0.0
    sum_rho  = 0.0
    sum_rhoG_ordinary = 0.0

    for _ in range(num_traj):
        S, A, R = simulate_trajectory_from_s1(behavior, effective_horizon)
        G0 = discounted_return(R, discount)
        rho = trajectory_ratio(S, A, target, behavior)
        sum_rhoG += rho * G0
        sum_rho  += rho

    V_hat_ordinary = sum_rhoG / num_traj
    V_hat_weighted = sum_rhoG / sum_rho if sum_rho > 0 else 0.0

    print("Off-policy MC (ordinary IS):", V_hat_ordinary)
    print("Off-policy MC (weighted  IS):", V_hat_weighted)

    print("Part 4")
    print("Error between Estimated MC (Weighted) and True Value Function: ", abs(V_hat_weighted - value_target[0]))
