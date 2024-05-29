import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

def plot_q_value_heatmap(q_table, action_index, title):
    # plot a heatmap for the Q-values for a specific action across angle and angular velocity bins at a fixed position
    q_values = q_table[5, 5, :, :, action_index]
    ax = sns.heatmap(q_values, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title(title)
    plt.xlabel('Angular Velocity Bins')
    plt.ylabel('Angle Bins')
    plt.show()

def plot_policy(q_table, pos_index, vel_index, title):
    # plot the policy heatmap showing the best action for each state at a fixed position and velocity
    policy = np.argmax(q_table[pos_index, vel_index, :, :, :], axis=2)
    ax = sns.heatmap(policy, annot=True, cmap='viridis')
    plt.title(title)
    plt.xlabel('Angular Velocity Bins')
    plt.ylabel('Angle Bins')
    plt.show()

def plot_reward_distribution(rewards_per_episode):
    # visualize the distribution of total rewards per episode to assess performance spread
    plt.figure(figsize=(10, 5))
    sns.histplot(rewards_per_episode, bins=30, kde=True)
    plt.title('Distribution of Rewards')
    plt.xlabel('Rewards')
    plt.ylabel('Frequency')
    plt.show()

def plot_moving_average(rewards, window_size=100, title='Moving Average of Rewards'):
    # calculate and plot moving average of rewards to smooth out variations and highlight trends
    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    plt.figure(figsize=(10, 5))
    plt.plot(moving_avg)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (last 100 episodes)')
    plt.savefig('moving_average_rewards.png')
    plt.show()

def plot_episode_lengths(lengths, title='Episode Length Over Time'):
    # plot the lengths of episodes over time to track changes in the number of steps before termination
    plt.figure(figsize=(10, 5))
    plt.plot(lengths)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Length of Episode')
    plt.savefig('episode_lengths.png')
    plt.show()

def run(is_training=True, render=False, learning_rate_a=0.1, epsilon_decay_rate=0.00001, discount_factor_g=0.99):
    # initialize the environment and discretize the state space
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    pos_space = np.linspace(-2.4, 2.4, 10)
    vel_space = np.linspace(-4, 4, 10)
    ang_space = np.linspace(-.2095, .2095, 10)
    ang_vel_space = np.linspace(-4, 4, 10)

    if is_training:
        # initialize Q-table for storing the action values
        q = np.zeros((len(pos_space)+1, len(vel_space)+1, len(ang_space)+1, len(ang_vel_space)+1, env.action_space.n))
    else:
        # load the Q-table from a file if not training
        with open('cartpole.pkl', 'rb') as f:
            q = pickle.load(f)

    rng = np.random.default_rng()  # create a random number generator instance
    rewards_per_episode = []
    episode_lengths = []
    mean_rewards = []
    i = 0
    epsilon = 1  # initialize epsilon for epsilon-greedy strategy

    while True:
        # reset environment and discretize initial state
        state = env.reset()[0]
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        state_a = np.digitize(state[2], ang_space)
        state_av = np.digitize(state[3], ang_vel_space)

        terminated = False
        rewards = 0
        step_count = 0  # initialize the step counter for the current episode

        while not terminated and rewards < 10000:
            # select action using epsilon-greedy strategy
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_p, state_v, state_a, state_av, :])

            # perform action and observe result
            new_state, reward, terminated, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)
            new_state_a = np.digitize(new_state[2], ang_space)
            new_state_av= np.digitize(new_state[3], ang_vel_space)

            if is_training:
                # update Q-value for the action taken using the Bellman equation
                q[state_p, state_v, state_a, state_av, action] += learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state_p, new_state_v, new_state_a, new_state_av, :]) - q[state_p, state_v, state_a, state_av, action]
                )

            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            state_a = new_state_a
            state_av = new_state_av
            rewards += reward
            step_count += 1  # increment step count

        # store rewards and episode length
        rewards_per_episode.append(rewards)
        episode_lengths.append(step_count)

        if is_training:
            # compute and store mean rewards to monitor progress
            mean_reward = np.mean(rewards_per_episode[max(0, i-100):(i+1)])
            mean_rewards.append(mean_reward)
            # logging progress every 100 episodes
            if i % 100 == 0:
                print(f'Episode: {i}, Rewards: {rewards}, Epsilon: {epsilon:.2f}, Mean Rewards: {mean_reward:.1f}')

            # check if learning goal is met
            if mean_reward >= 195:
                break

            # update epsilon for the epsilon-greedy policy
            epsilon = max(epsilon - epsilon_decay_rate, 0)
            i += 1

    env.close()

    if is_training:
        # save the updated Q-table
        with open('cartpole.pkl', 'wb') as f:
            pickle.dump(q, f)

    # visualization of training results
    plt.figure(figsize=(10, 5))
    plt.plot(mean_rewards)
    plt.title('Training Progress')
    plt.xlabel('Episodes')
    plt.ylabel('Mean Rewards (last 100 episodes)')
    plt.savefig('cartpole_training_progress.png')

    # visualizations for further analysis
    plot_q_value_heatmap(q, 0, 'Q-value Heatmap for Action 0')
    plot_q_value_heatmap(q, 1, 'Q-value Heatmap for Action 1')
    plot_policy(q, 5, 5, 'Learned Policy at Center Position and Zero Velocity')
    plot_reward_distribution(rewards_per_episode)
    plot_moving_average(rewards_per_episode)
    plot_episode_lengths(episode_lengths)

if __name__ == '__main__':
    run(is_training=True, render=True)
