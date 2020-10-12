import gym
import numpy as np
env = gym.make('FrozenLake-v0')


def test_performance(policy, nb_episodes=100):
    sum_returns = 0
    for i in range(nb_episodes):
        state  = env.reset()
        done = False
        while not done:
            action = policy(state)
            state, reward, done, info = env.step(action)
            if done:
                sum_returns += reward
    return sum_returns/nb_episodes


def random_episode(env,policy):
    states_actions_rewards = []
    state = env.reset()
    reward = 0
    done = False
    while not done:
        action = policy(state)
        states_actions_rewards.append((state,action,reward))
        new_state,reward,done,info = env.step(action)
        state = new_state
    states_actions_rewards.append((state,action,reward))
    env.reset()
    return states_actions_rewards


def calc_return(seq,discount_factor):
    G = 0
    for [state,action,reward] in seq:
        G = reward + (G * discount_factor)
    return G


def first_visit_evaluation(env,policy,discount_factor=1.0):
    V = np.random.rand(164)
    returns = [ []*1 for i in range(16)]
    for i in range(10000):
        episode = random_episode(env,policy)
        for index,[state,_,_] in enumerate(episode):
            visited_states = set()
            if state not in visited_states:
                G = calc_return(episode[index:],discount_factor)
                returns[state].append(G)
                V[state] = np.mean(returns[state])
                visited_states.add(state)
    return V


def every_visit_evaluation(env,policy,discount_factor=1.0):
    V = np.random.rand(16)
    returns = [ []*1 for i in range(16)]
    for i in range(10000):
        episode = random_episode(env,policy)
        for index,[state,_,_] in enumerate(episode):
            visited_states = set()
            G = calc_return(episode[index:],discount_factor)
            returns[state].append(G)
            V[state] = np.mean(returns[state])
            visited_states.add(state)
    return V

def first_visit_q_evaluation(env,policy,discount_factor=1.0):
    Q = np.random.rand(16,4)
    k_s = np.zeros((16,4))
    for i in range(100000):
        episode = random_episode(env,policy)
        for index,[state,action,_] in enumerate(episode):
            visited_state_action_pairs = set()
            if (state,action) not in visited_state_action_pairs:
                k_s[state,action] += 1
                G = calc_return(episode[index:],discount_factor)
                Q[state][action] = Q[state][action] + 1/k_s[state][action] * (G-Q[state][action])
                visited_state_action_pairs.add((state,action))
    return Q

def every_visit_q_evaluation(env,policy,discount_factor=1.0):
    Q = np.random.rand(16,4)
    k_s = np.zeros((16,4))
    for i in range(100000):
        episode = random_episode(env,policy)
        for index,[state,action,_] in enumerate(episode):
            visited_state_action_pairs = set()
            k_s[state,action] += 1
            G = calc_return(episode[index:],discount_factor)
            Q[state][action] = Q[state][action] + 1/k_s[state][action] * (G-Q[state][action])
            visited_state_action_pairs.add((state,action))
    return Q

def control(env,)

if __name__ == '__main__':
    print("Sizes")
    print("------")
    print("Action Space: ", env.action_space)
    print("Observation space: ", env.observation_space)

    policy_dict = {0: 1, 1: 2, 2: 1, 3: 0, 4: 1, 6: 1, 8: 2, 9: 0, 10: 1, 13: 2, 14: 2}  # random policy
    policy = lambda s: policy_dict[s]

    # calling dictionary with [] and function with ()

    print("Mean reward:", test_performance(policy))

    policy_dict = {0: 1, 1: 2, 2: 1, 3: 0, 4: 1, 6: 1, 8: 2, 9: 0, 10: 1, 13: 2, 14: 2}  # random
    policy = lambda s: policy_dict[s]

    print(first_visit_q_evaluation(env, policy))
    print("")
    print(every_visit_q_evaluation(env,policy))