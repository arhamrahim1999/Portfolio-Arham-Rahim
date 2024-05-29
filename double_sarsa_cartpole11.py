import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

def plot_q_value_heatmap(q_table, action_index, title):
    # extract Q-values for a specific action from the Q-table at position (5,5) for all angle and angular velocity bins
    q_values = q_table[5, 5, :, :, action_index]
    # create a heatmap using seaborn for the Q-values
    ax = sns.heatmap(q_values, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title(title)
    plt.xlabel('Angular Velocity Bins')
    plt.ylabel('Angle Bins')
    plt.show()

def plot_policy(q_table, pos_index, vel_index, title):
    # derive the policy by finding the action with the maximum Q-value at each state
    policy = np.argmax(q_table[pos_index, vel_index, :, :, :], axis=2)
    # create a heatmap for the policy using seaborn
    ax = sns.heatmap(policy, annot=True, cmap='viridis')
    plt.title(title)
    plt.xlabel('Angular Velocity Bins')
    plt.ylabel('Angle Bins')
    plt.show()

def plot_reward_distribution(rewards_per_episode):
    # plot the distribution of rewards using a histogram
    plt.figure(figsize=(10, 5))
    sns.histplot(rewards_per_episode, bins=30, kde=True)
    plt.title('Distribution of Rewards')
    plt.xlabel('Rewards')
    plt.ylabel('Frequency')
    plt.show()

def plot_moving_average(rewards, window_size=100, title='Moving Average of Rewards'):
    # calculate and plot the moving average of rewards to observe trends over time
    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    plt.figure(figsize=(10, 5))
    plt.plot(moving_avg)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (last 100 episodes)')
    plt.savefig('double_sarsa_moving_average_rewards.png')
    plt.show()

def plot_episode_lengths(lengths, title='Episode Length Over Time'):
    # plot the lengths of each episode over time
    plt.figure(figsize=(10, 5))
    plt.plot(lengths)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Length of Episode')
    plt.savefig('double_sarsa_episode_lengths.png')
    plt.show()

def run_double_sarsa(is_training=True, render=False):
    # initialize the environment, state space discretization
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    pos_space = np.linspace(-2.4, 2.4, 10)
    vel_space = np.linspace(-4, 4, 10)
    ang_space = np.linspace(-.2095, .2095, 10)
    ang_vel_space = np.linspace(-4, 4, 10)

    if is_training:
        # initialize two separate Q-tables for Double SARSA
        q1 = np.zeros((len(pos_space)+1, len(vel_space)+1, len(ang_space)+1, len(ang_vel_space)+1, env.action_space.n))
        q2 = np.zeros_like(q1)
    else:
        # load Q-tables from file if not training
        with open('double_sarsa_cartpole.pkl', 'rb') as f:
            q1, q2 = pickle.load(f)

    learning_rate = 0.1
    discount_factor = 0.6
    epsilon = 1
    epsilon_decay_rate = 0.00001

    rewards_per_episode = []
    episode_lengths = []
    mean_rewards = []
    i = 0

    while True:
        # start a new episode
        state = env.reset()[0]
        # digitize the continuous state variables to discretize the state space
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        state_a = np.digitize(state[2], ang_space)
        state_av = np.digitize(state[3], ang_vel_space)

        terminated = False
        rewards = 0
        step_count = 0

        while not terminated and rewards < 10000:
            # choose action based on policy derived from Q-values
            combined_q = (q1 + q2) / 2
            if is_training and np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(combined_q[state_p, state_v, state_a, state_av, :])

            # take action and observe new state and reward
            new_state, reward, terminated, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)
            new_state_a = np.digitize(new_state[2], ang_space)
            new_state_av = np.digitize(new_state[3], ang_vel_space)

            if is_training:
                # update Q-tables alternately
                if np.random.rand() < 0.5:
                    new_action = np.argmax(q1[new_state_p, new_state_v, new_state_a, new_state_av, :])
                    q1[state_p, state_v, state_a, state_av, action] += learning_rate * (
                        reward + discount_factor * q2[new_state_p, new_state_v, new_state_a, new_state_av, new_action]
                        - q1[state_p, state_v, state_a, state_av, action]
                    )
                else:
                    new_action = np.argmax(q2[new_state_p, new_state_v, new_state_a, new_state_av, :])
                    q2[state_p, state_v, state_a, state_av, action] += learning_rate * (
                        reward + discount_factor * q1[new_state_p, new_state_v, new_state_a, new_state_av, new_action]
                        - q2[state_p, state_v, state_a, state_av, action]
                    )

            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            state_a = new_state_a
            state_av = new_state_av

            rewards += reward
            step_count += 1

        # track rewards and episode lengths
        rewards_per_episode.append(rewards)
        episode_lengths.append(step_count)
        mean_rewards.append(np.mean(rewards_per_episode[max(0, i-100):(i+1)]))

        if is_training and i % 100 == 0:
            # print progress every 100 episodes
            print(f'Episode: {i}, Rewards: {rewards}, Epsilon: {epsilon:.2f}, Mean Rewards: {mean_rewards[-1]:.1f}')

        # break training loop if average reward target is achieved
        if mean_rewards[-1] >= 195:
            break

        # decrement epsilon for epsilon-greedy policy
        epsilon = max(epsilon - epsilon_decay_rate, 0)
        i += 1

    env.close()

    if is_training:
        # save Q-tables to file after training
        with open('double_sarsa_cartpole.pkl', 'wb') as f:
            pickle.dump((q1, q2), f)

    # visualization calls
    plot_q_value_heatmap(q1, 0, 'Q-value Heatmap for Action 0 (Q1)')
    plot_q_value_heatmap(q2, 0, 'Q-value Heatmap for Action 0 (Q2)')
    plot_policy((q1+q2)/2, 5, 5, 'Learned Policy at Center Position and Zero Velocity')
    plot_reward_distribution(rewards_per_episode)
    plot_moving_average(rewards_per_episode)
    plot_episode_lengths(episode_lengths)

if __name__ == '__main__':
    run_double_sarsa(is_training=True, render=False)
