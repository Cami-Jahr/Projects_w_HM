from game import Game
from model import DQN
from replaymemory import DualReplayMemory, Transition, MinorStateMemory
from random import random, randrange
from matplotlib import get_backend
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from sys import argv
from math import ceil, floor


def get_screen():
    """
    Get current game state image, transform it to PIL
    Remove the last matrix representing blue colour due to no use
    """
    screen = game.get_image()
    return transform(torch.as_tensor(screen))[:2].unsqueeze(0).to(device)


def plot_screen(screen, score="Unknown"):
    plt.figure()
    screen = screen.squeeze(0).cpu()
    screen = torch.cat((screen, torch.zeros((1, WIDTH, WIDTH)))).permute(1, 2, 0)  # Add blue matrix
    plt.imshow(screen.numpy())
    plt.title('End screen with score {}'.format(score))
    plt.show()


def get_command_line_params() -> None:
    """
    Allows the user to set WIDTH by passing -w, such as "-w 50"
    Allows the user to set number of episodes by passing -w, such as "-n 1000"
    """
    global WIDTH, NUM_EPISODES
    for i in range(1, len(argv) - 1):
        if argv[i] == "-w":
            if argv[i + 1].isdigit():
                WIDTH = int(argv[i + 1])
        elif argv[i] == "-n":
            if argv[i + 1].isdigit():
                NUM_EPISODES = int(argv[i + 1])


def select_action(state_history, _type="NORMAL"):
    """
    :param state_history:
    :param _type: May be "NORMAL", "RANDOM" or "OPTIMAL", defaults to "NORMAL"
    :return: best action as int in [0, 3] according to type selection rule
    """
    if _type == "OPTIMAL":
        return model(state_history).max(1)[1].view(1, 1)
    elif _type == "RANDOM":
        return torch.tensor([[randrange(valid_actions)]], device=device, dtype=torch.long)
    else:
        if random() > (EPSILON_START - EPSILON_END) - EPSILON_decay * turn_counter:
            return model(state_history).max(1)[1].view(1, 1)
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


def plot_scores():
    plt.figure(2)
    plt.clf()
    scores_t = torch.tensor(episode_scores, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Scores')
    plt.plot(scores_t.numpy())
    # Take 100 episode averages and plot them too
    if len(scores_t) >= 10:
        means = scores_t.unfold(0, 10, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(9), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:  # Should not be possible unless warm-up is removed
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose into named tuple
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state).to(device)  # SJ
    action_batch = torch.cat(batch.action).float().to(device)  # A
    reward_batch = torch.cat(batch.reward).to(device)  # R
    next_state_batch = torch.cat(batch.next_state).to(device)  # SJ1
    terminal_batch = torch.cat(batch.terminal).to(device)  # SJ1

    current_prediction_batch = model(state_batch)
    next_prediction_batch = model(next_state_batch)
    y_batch = torch.tensor(
        tuple(map(lambda inputs: inputs[0] if inputs[1] else inputs[0] + GAMMA * torch.max(inputs[2]),
                  zip(reward_batch, terminal_batch, next_prediction_batch))), device=device, dtype=torch.float)
    q_batch = torch.sum(current_prediction_batch * action_batch, dim=1)
    optimizer.zero_grad()
    loss = criterion(q_batch, y_batch)
    loss.backward()
    optimizer.step()


def calculate_delay_punishment_maximum(length):
    return floor(.7 * length) + WIDTH


def calc_time_after_apple_of_no_learning(length):
    return (WIDTH // 2) if length < K else ceil(P * length + q)


def main():
    global turn_counter

    def perform_round(cold_start=False):
        game.restart()
        state = get_screen()
        prev_states.empty()
        prev_states.append(state)
        prev_series_of_states = prev_states.get()
        steps = 0
        turns_since_apple = 0  # Start at inf?, start at 0?
        done = False
        max_time_before_delay_punishment = calculate_delay_punishment_maximum(LENGTH)
        post_apple_no_learn = calc_time_after_apple_of_no_learning(LENGTH)
        while not done:
            steps += 1
            action = select_action(prev_series_of_states, _type="RANDOM" if cold_start else "NORMAL")
            # Note: While params returned from the game could be deducted from state,
            # this is foregone here for the sake of ease of writing, readability and performance
            done, ate, length, reward = game(action.item())
            turns_since_apple += 1
            if ate:
                # if turns_since_apple < post_apple_no_learn:
                #     reward = 0
                turns_since_apple = 0
                max_time_before_delay_punishment = calculate_delay_punishment_maximum(length)
                post_apple_no_learn = calc_time_after_apple_of_no_learning(LENGTH)
            elif done:
                # Don't want to skip collide learning
                pass
            elif turns_since_apple < post_apple_no_learn:
                # Do not remember nor learn if recently picket up an apple, to avoid
                # punishing behaviour after successful task
                continue
            elif turns_since_apple == max_time_before_delay_punishment:
                # Reduce reward for all max_time_before_delay_punishment previous turns
                memory.reduce_score_of_previous_n_by_p(max_time_before_delay_punishment - 1, 0.5 / length)
                reward -= 0.5 / length
            elif turns_since_apple > max_time_before_delay_punishment:
                reward -= 0.5 / length

            if reward < -1:
                reward = -1
            elif reward > 1:
                reward = 1

            reward = torch.tensor([reward], device=device, dtype=torch.float)
            state = get_screen()
            terminal = torch.tensor([done], device=device, dtype=torch.float)
            prev_states.append(state)
            new_series_of_states = prev_states.get()
            memory.push(prev_series_of_states, action, new_series_of_states, terminal, reward,
                        primary_buffer=abs(reward.item()) >= M1_THRESHOLD)
            prev_series_of_states = new_series_of_states
            if not cold_start:
                optimize_model()
        return steps

    print("Running warm-up to counteract cold start")
    s = 0
    while s < NUM_COLD_START_STEPS or len(memory) < BATCH_SIZE:
        s += perform_round(cold_start=True)

    print("Running main")
    for i_episode in range(NUM_EPISODES):
        episode_durations.append(perform_round())
        episode_scores.append(game.get_final_score())
        memory.iterate_ratio()
        print(turn_counter, (EPSILON_START - EPSILON_END) - EPSILON_decay * turn_counter)
        turn_counter += 1

    plot_durations()
    plot_scores()

    print('Complete')
    print("Close plot to display test run of network")
    plt.show()


def show_a_run_current_network():
    global game
    old_game = game
    game = Game(width=WIDTH, length=LENGTH, render=True)
    state = get_screen()
    done = False
    prev_states.empty()
    prev_states.append(state)
    while not done:
        action = select_action(prev_states.get(), _type="OPTIMAL")
        done, *_ = game(action.item())
        state = get_screen()
        prev_states.append(state)
    plot_screen(state, game.get_final_score())
    game = old_game


# Initialization params uppercase should be changed to empirically
BATCH_SIZE = 64
LR = 1e-6
NUM_COLD_START_STEPS = 5000
NUM_EPISODES = 1000
REPLAY_SIZE = 1000000
WIDTH = 12  # == height
LENGTH = 3
P = .4
K = 10
M1_M2_WEIGHT_INITIAL = 0.8
M1_M2_WEIGHT_FINAL = 0.5
M1_M2_WEIGHT_ITERATIONS = NUM_EPISODES // 2
M1_THRESHOLD = 0.5
EPSILON_START = 0.7
EPSILON_END = 0
NUM_EPSILON_EPISODES = int(NUM_EPISODES * 0.8)  # Use optimal for past 20% of runs
GAMMA = .99
NR_OF_PLOTS = 1

# Calculated, state or constant params
q = 6 - K * P
is_ipython = 'inline' in get_backend()
if is_ipython:
    from IPython import display
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = T.Compose([T.ToPILImage(), T.ToTensor()])
episode_durations = []
episode_scores = []
memory = DualReplayMemory(REPLAY_SIZE, initial_weight=M1_M2_WEIGHT_INITIAL, final_weight=M1_M2_WEIGHT_FINAL,
                          iterations=M1_M2_WEIGHT_ITERATIONS,
                          history_size=calculate_delay_punishment_maximum(WIDTH ** 2))
prev_states = MinorStateMemory(4 * 2)
EPSILON_decay = (EPSILON_START - EPSILON_END) / NUM_EPSILON_EPISODES
turn_counter = 0

if __name__ == '__main__':
    get_command_line_params()
    print("Initializing with width={}, number of episodes={} and device={}".format(WIDTH, NUM_EPISODES, device))
    game = Game(width=WIDTH, length=LENGTH, render=True)

    # DQN initialization
    valid_actions = game.get_amount_of_legal_actions()
    model = DQN(4 * 2, WIDTH, valid_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()
    main()

    show_a_run_current_network()
