import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson as scipy_poisson

DISCOUNT_FACTOR = 0.9
MOVE_COST = 20
RENTAL_REWARD = 100
MAX_CARS = 20
MIN_CARS_MOVE = 5
MAX_CARS_MOVE = 5
ACTIONS = np.arange(-MIN_CARS_MOVE, MAX_CARS_MOVE + 1)
MAX_POISSON_STEP = 11
FIRST_RENTAL_MEAN = 3
SECOND_RENTAL_MEAN = 4
FIRST_RETURN_MEAN = 3
SECOND_RETURN_MEAN = 2
poisson_cache = {}
SIMPLIFIED = False
DEBUG = False


def poisson(n, k):
    if (n, k) in poisson_cache:
        return poisson_cache[(n, k)]
    poisson_cache[(n, k)] = scipy_poisson.pmf(n, k)
    return poisson_cache[(n, k)]


def plot_array(arr):
    fig, ax = plt.subplots(1, 1)
    ax.matshow(arr, cmap='Blues')
    for (i, j), z in np.ndenumerate(arr):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', size='x-small')
    plt.show()


def get_reward(first_car_num, second_car_num, action, state_value):
    # impossible actions
    if action > first_car_num or -action > second_car_num:
        return -np.inf
    total_reward = - np.abs(action) * MOVE_COST
    first_after_action = min(first_car_num - action, MAX_CARS)
    second_after_action = min(second_car_num + action, MAX_CARS)
    # iterate over number of rented cars from two rentals
    for first_rental_num in range(MAX_POISSON_STEP):
        for second_rental_num in range(MAX_POISSON_STEP):
            rental_prob = poisson(first_rental_num, FIRST_RENTAL_MEAN) * poisson(second_rental_num, SECOND_RENTAL_MEAN)
            first_possible_num = min(first_after_action, first_rental_num)
            first_after_rental = first_after_action - first_possible_num
            second_possible_num = min(second_after_action, second_rental_num)
            second_after_rental = second_after_action - second_possible_num
            rental_reward = (first_possible_num + second_possible_num) * RENTAL_REWARD
            if SIMPLIFIED:  # number of returns is const
                first_return_num = FIRST_RENTAL_MEAN
                second_return_num = SECOND_RETURN_MEAN
                first_after_return = min(first_after_rental + first_return_num, MAX_CARS)
                second_after_return = min(second_after_rental + second_return_num, MAX_CARS)
                future_reward = state_value[first_after_return, second_after_return]
                total_reward += rental_prob * (rental_reward + DISCOUNT_FACTOR * future_reward)
            else:
                for first_return_num in range(MAX_POISSON_STEP):
                    for second_return_num in range(MAX_POISSON_STEP):
                        return_prob = poisson(first_return_num, FIRST_RETURN_MEAN) * poisson(second_return_num,
                                                                                             SECOND_RETURN_MEAN)
                        first_after_return = min(first_after_rental + first_return_num, MAX_CARS)
                        second_after_return = min(second_after_rental + second_return_num, MAX_CARS)
                        future_reward = state_value[first_after_return, second_after_return]
                        total_reward += return_prob * rental_prob * (rental_reward + DISCOUNT_FACTOR * future_reward)
    return total_reward


def print_strategy(strategy):
    strategy = strategy.copy().astype(int)
    for row in strategy:
        for move in row:
            print("{:2d} ".format(move), end='')
        print()


def main():
    state_value = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
    new_state = np.random.rand(MAX_CARS + 1, MAX_CARS + 1)
    strategy = np.zeros(state_value.shape)
    new_strategy = np.zeros(state_value.shape)
    while not np.sum(np.abs(state_value - new_state)) < 0.0001:
        changed = 0
        if DEBUG:
            print('round', round, 'difference', np.sum(np.abs(state_value - new_state)), 'last:\n',
                  state_value[-3:, -3:])
        state_value = new_state.copy()
        strategy = new_strategy.copy()
        for pos, _ in np.ndenumerate(state_value):
            if DEBUG:
                print(pos, end='\r')
            sys.stdout.flush()
            first_num, second_num = pos
            tab = [get_reward(first_num, second_num, action, state_value) for action in ACTIONS]
            new_strategy[pos] = ACTIONS[np.argmax(tab)]
            if DEBUG and new_strategy[pos] != strategy[pos]:
                changed += 1
            new_state[pos] = DISCOUNT_FACTOR * np.max(tab)
        if DEBUG:
            print('changed', changed)
    if DEBUG:
        np.savetxt('strategy.txt', strategy, fmt='%i', delimiter=' ')
        np.savetxt('state_value.txt', state_value, fmt='%.2f', delimiter=' ')
        plot_array(state_value)
        plot_array(strategy)
    print_strategy(strategy)
    return state_value, strategy


if __name__ == '__main__':
    main()
