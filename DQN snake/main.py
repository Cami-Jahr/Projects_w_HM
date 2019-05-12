from game import Game
from model import DQN
from replaymemory import DualReplayMemory, Transition, MinorStateMemory
import numpy as np
from random import random, randrange
from matplotlib import get_backend
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from sys import argv

# set up matplotlib and global vars
is_ipython = 'inline' in get_backend()
if is_ipython:
    from IPython import display
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = T.Compose([T.ToPILImage(), T.ToTensor()])


def get_screen():
    """
    Get current game state image, transform it to PIL
    Remove the last matrix representing blue colour due to no use
    """
    screen = game.get_image()
    return transform(torch.as_tensor(screen))[:2].unsqueeze(0).to(device)


def plot_screen(screen, score: (int, str) = "Unknown"):
    plt.figure()
    screen = screen.squeeze(0).cpu()
    screen = torch.cat((screen, torch.zeros((1, width, width)))).permute(1, 2, 0)  # Add blue matrix
    plt.imshow(screen.numpy())
    plt.title('End screen with score {:.2f}'.format(score))
    plt.show()


def get_command_line_params() -> None:
    """
    Allows the user to set width by passing -w, such as "-w 50"
    Allows the user to set number of episodes by passing -w, such as "-n 1000"
    """
    global width, num_episodes
    for i in range(1, len(argv) - 1):
        if argv[i] == "-w":
            if argv[i + 1].isdigit():
                width = int(argv[i + 1])
        elif argv[i] == "-n":
            if argv[i + 1].isdigit():
                num_episodes = int(argv[i + 1])


def select_best_action(state_history):
    return policy_net(state_history).max(1)[1].view(1, 1)


def select_action(state_history):
    global steps_done
    sample = random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
    # print(eps_threshold)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # Return best action
            return policy_net(state_history).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[randrange(valid_actions)]], device=device, dtype=torch.long)


def plot_durations():
    plt.figure(1)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose into named tuple
    batch = Transition(*zip(*transitions))
    # print(batch)
    # print(transitions, batch)
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.long)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)
    # print(state_batch)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # print("A", action_batch.shape)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    # print("NFNS", non_final_next_states.shape)
    next_state_values[non_final_mask] = policy_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    # print(next_state_values)
    # print("R", reward_batch)
    # print((next_state_values * GAMMA))
    # print((next_state_values * GAMMA) + reward_batch)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def cold_start():
    print("Running cold start")
    game.restart()
    done = False
    state = get_screen()
    prev_states.empty()
    prev_states.append(state)
    prev_series_of_states = prev_states.get()
    s = 0
    while s < num_cold_start_steps or len(memory) < BATCH_SIZE:
        s += 1
        if done:
            game.restart()
        action = torch.tensor([[randrange(valid_actions)]], device=device, dtype=torch.long)
        done, reward = game(action.item())
        reward = torch.tensor([reward], device=device, dtype=torch.float)
        state = get_screen()

        prev_states.append(state)
        new_series_of_states = prev_states.get()
        memory.push(prev_series_of_states, action, new_series_of_states, reward, primary_buffer=reward.item() > 0.5)
        prev_series_of_states = new_series_of_states


def main():
    print("Running main")
    for i_episode in range(num_episodes):
        game.restart()
        state = get_screen()
        prev_states.empty()
        prev_states.append(state)
        prev_series_of_states = prev_states.get()
        t = 0
        done = False
        while not done:
            t += 1
            action = select_action(prev_series_of_states)
            done, reward = game(action.item())
            reward = torch.tensor([reward], device=device, dtype=torch.float)
            state = get_screen()

            prev_states.append(state)
            new_series_of_states = prev_states.get()
            memory.push(prev_series_of_states, action, new_series_of_states, reward, primary_buffer=reward.item() > 0.5)
            prev_series_of_states = new_series_of_states

            optimize_model()
        episode_durations.append(t + 1)
        plot_durations()

    print('Complete')
    print("Close plot to display test run of network")
    plt.show()


def display_result():
    global game
    old_game = game
    game = Game(width=width, render=True)
    state = get_screen()
    done = reward = False
    prev_states.empty()
    prev_states.append(state)
    while not done:
        action = select_best_action(prev_states.get())
        done, reward = game(action.item())
        state = get_screen()
    plot_screen(state, reward)
    game = old_game


# Initialization params
BATCH_SIZE = 16
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
width = 12  # == height
num_cold_start_steps = 500
num_episodes = 250
replay_size = 1000000

# State params
steps_done = 0
episode_durations = []
memory = DualReplayMemory(replay_size)
prev_states = MinorStateMemory(4 * 2)
if __name__ == '__main__':
    get_command_line_params()
    print("Initilizing with width={}, number of episodes={} and device={}".format(width, num_episodes, device))
    game = Game(width=width)

    # DQN initialization
    valid_actions = game.get_amount_of_legal_actions()
    policy_net = DQN(width, valid_actions).to(device)
    optimizer = optim.Adam(policy_net.parameters())
    cold_start()

    main()

    display_result()
