import random
import matplotlib.pyplot as plt
import numpy as np

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
            temp_val = 0
            for state in range(3):
                if action == 0:
                    temp_val += transition_g[i][state] * (reward[i][action] + discount*value[state])
                elif action == 1:
                    temp_val += transition_h[i][state] * (reward[i][action] + discount*value[state])
            temp_val *= policy[i][action]
        new_value[i] = temp_val
    return new_value

if __name__ == "__main__":
    print("Part 1")
    iterations = 10000
    value_target = [0]*3
    value_behavior = [0]*3
    for i in range(iterations):
        value_target = get_value(target, value_target)
        value_behavior = get_value(behavior, value_behavior)
    print("Target Value: ", value_target)
    print("Behavior Value: ", value_behavior)
