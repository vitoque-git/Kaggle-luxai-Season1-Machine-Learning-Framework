import numpy as np
import json
from pathlib import Path
import os
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# episode_dir = '../input/lux-ai-episodes-score1800'
episode_dir = 'C:/git/luxai/episodes/archive'


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def to_label(action, units):
    strs = action.split(' ')
    unit_id = strs[1]
    if strs[0] == 'm':
        label = {'c': None, 'n': 0, 's': 1, 'w': 2, 'e': 3}[strs[2]]
    elif strs[0] == 'bcity':
        label = 4
    elif strs[0] == 't':
        from_pos = units[unit_id]
        to_pos = units[strs[2]]
        if from_pos[1]  - 1 == to_pos[1]:
            label = 5 #n
        elif from_pos[1]  + 1 == to_pos[1]:
            label = 6 #s
        if from_pos[0] - 1 == to_pos[0]:
            label = 7  # w
        elif from_pos[0] + 1 == to_pos[0]:
            label = 8  # e
    else:
        label = None
    return unit_id, label

 # def translate(self, direction, units=1) -> 'Position':
 #        if direction == DIRECTIONS.NORTH:
 #            return Position(self.x, self.y - units)
 #        elif direction == DIRECTIONS.EAST:
 #            return Position(self.x + units, self.y)
 #        elif direction == DIRECTIONS.SOUTH:
 #            return Position(self.x, self.y + units)
 #        elif direction == DIRECTIONS.WEST:
 #            return Position(self.x - units, self.y)
 #        elif direction == DIRECTIONS.CENTER:
 #            return Position(self.x, self.y)


def depleted_resources(obs):
    for u in obs['updates']:
        if u.split(' ')[0] == 'r':
            return False
    return True


def create_dataset_from_json(episode_dir, team_name='Toad Brigade', set_sizes=[]):
    obses = {}
    samples = []
    append = samples.append
    episodes = [path for path in Path(episode_dir).glob('*.json') if
                ('output' not in path.name and '_info' not in path.name)]

    print('create_dataset_from_json')
    for filepath in episodes:

        with open(filepath) as f:
            json_load = json.load(f)

        ep_id = json_load['info']['EpisodeId']
        index = np.argmax([r or 0 for r in json_load['rewards']])
        if json_load['info']['TeamNames'][index] != team_name:
            continue

        if len(set_sizes) != 0:
            this_size = json_load['steps'][0][0]['observation']['height']
            if this_size not in set_sizes:
                continue

        for i in range(len(json_load['steps']) - 1):
            if json_load['steps'][i][index]['status'] == 'ACTIVE':
                actions = json_load['steps'][i + 1][index]['action']

                obs = json_load['steps'][i][0]['observation']

                if depleted_resources(obs):
                    break

                obs['player'] = index
                obs = dict([
                    (k, v) for k, v in obs.items()
                    if k in ['step', 'updates', 'player', 'width', 'height']
                ])
                obs_id = f'{ep_id}_{i}'
                obses[obs_id] = obs

                units = {}
                for update in obs['updates']:
                    strs = update.split(' ')
                    input_identifier = strs[0]

                    if input_identifier == 'u':
                        x = int(strs[4])
                        y = int(strs[5])
                        unit_id = strs[3]
                        units[unit_id] = (x, y)

                for action in actions:
                    unit_id, label = to_label(action, units)
                    if label is not None:
                        append((obs_id, unit_id, label))

    return obses, samples

CHANNELS = 25

# Input for Neural Network
def make_input(obs, unit_id, size=32):
    width, height = obs['width'], obs['height']
    x_shift = (size - width) // 2
    y_shift = (size - height) // 2
    cities = {}

    turn = obs['step']
    MAX_DAYS = 360
    DAY_LENGTH = 30
    NIGHT_LENGTH = 10
    FULL_LENTH = DAY_LENGTH + NIGHT_LENGTH

    all_night_turns_lef = ((MAX_DAYS - 1 - turn) // FULL_LENTH + 1) * NIGHT_LENGTH

    turns_to_night = (DAY_LENGTH - turn) % FULL_LENTH
    turns_to_night = 0 if turns_to_night > 30 else turns_to_night

    turns_to_dawn = FULL_LENTH - turn % FULL_LENTH
    turns_to_dawn = 0 if turns_to_dawn > 10 else turns_to_dawn

    if turns_to_night == 0:
        all_night_turns_lef -= (10 - turns_to_dawn)

    steps_until_night = 30 - turn % 40
    next_night_number_turn = min(10, 10 + steps_until_night)

    b = np.zeros((CHANNELS, size, size), dtype=np.float32)

    for update in obs['updates']:
        strs = update.split(' ')
        input_identifier = strs[0]

        if input_identifier == 'u':
            x = int(strs[4]) + x_shift
            y = int(strs[5]) + y_shift
            wood = int(strs[7])
            coal = int(strs[8])
            uranium = int(strs[9])
            fuel = wood + coal * 10 + uranium * 40
            if unit_id == strs[3]:
                # Position and Cargo
                b[:3, x, y] = (
                    1,
                    (wood + coal + uranium) / 100,
                    fuel / 4000
                )
            else:
                # Units
                team = int(strs[2])
                cooldown = float(strs[6])
                idx = 3 + (team - obs['player']) % 2 * 3
                b[idx:idx + 3, x, y] = (
                    1,
                    cooldown / 6,
                    (wood + coal + uranium) / 100
                )
        elif input_identifier == 'ct':
            # CityTiles
            team = int(strs[1])
            city_id = strs[2]
            x = int(strs[3]) + x_shift
            y = int(strs[4]) + y_shift
            idx = 9 + (team - obs['player']) % 2 * 4
            b[idx:idx + 4, x, y] = (
                1,
                cities[city_id][0],
                cities[city_id][1],
                cities[city_id][2]
            )
        elif input_identifier == 'r':
            # Resources
            r_type = strs[1]
            x = int(strs[2]) + x_shift
            y = int(strs[3]) + y_shift
            amt = int(float(strs[4]))
            b[{'wood': 17, 'coal': 18, 'uranium': 19}[r_type], x, y] = amt / 800
        elif input_identifier == 'rp':
            # Research Points
            team = int(strs[1])
            rp = int(strs[2])
            b[20 + (team - obs['player']) % 2, :] = min(rp, 200) / 200
        elif input_identifier == 'c':
            # Cities
            city_id = strs[2]
            fuel = float(strs[3])
            lightupkeep = float(strs[4])
            autonomy = int(fuel) // int(lightupkeep)
            will_live = autonomy >= all_night_turns_lef
            will_live_next_night = autonomy >= next_night_number_turn
            cities[city_id] = (
                int(will_live_next_night),
                int(will_live),
                min(autonomy, 10) / 10)

    # Day/Night Cycle
    b[22, :] = obs['step'] % 40 / 40
    # Turns
    b[23, :] = obs['step'] / 360
    # Map Size
    b[24, x_shift:size - x_shift, y_shift:size - y_shift] = 1

    return b


class LuxDataset(Dataset):
    def __init__(self, obses, samples, make_input_size=32):
        self.obses = obses
        self.samples = samples
        self.make_input_size = make_input_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs_id, unit_id, action = self.samples[idx]
        obs = self.obses[obs_id]
        state = make_input(obs, unit_id, self.make_input_size)

        return state, action


import matplotlib.pyplot as plt


########################################
#       Plotting the Graph             #
########################################

def plot_graphs(train_loss, valid_loss, epochs):
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(20, 4))
    ax = fig.add_subplot(1, 2, 1)
    plt.title("Train Loss")
    plt.plot(list(np.arange(epochs) + 1), train_loss, label='train')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('train_loss', fontsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(loc='best')

    ax = fig.add_subplot(1, 2, 2)
    plt.title("Validation Loss")
    plt.plot(list(np.arange(epochs) + 1), valid_loss, label='test')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('vaidation _loss', fontsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(loc='best')


# Neural Network for Lux AI
class BasicConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bn):
        super().__init__()
        self.conv = nn.Conv2d(
            input_dim, output_dim,
            kernel_size=kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2)
        )
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h) if self.bn is not None else h
        return h


class LuxNet(nn.Module):
    def __init__(self, filt=32):
        super().__init__()
        layers, filters = 12, filt
        self.conv0 = BasicConv2d(CHANNELS, filters, (3, 3), True)
        self.blocks = nn.ModuleList([BasicConv2d(filters, filters, (3, 3), True) for _ in range(layers)])
        self.head_p = nn.Linear(filters, 9, bias=False)

    def forward(self, x):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))
        h_head = (h * x[:, :1]).view(h.size(0), h.size(1), -1).sum(-1)
        p = self.head_p(h_head)
        return p


from datetime import datetime


def get_time():
    now = datetime.now()

    return now.strftime("%H:%M:%S")


def train_model(model, dataloaders_dict, criterion, optimizer, scheduler, num_epochs, map_size=32,
                skip_first_train=True):
    best_acc = 0.0

    num_train = len(dataloaders_dict['train'])
    num_val = len(dataloaders_dict['val'])
    print(get_time(),f'LR: {optimizer.param_groups[0]["lr"]} Epochs {num_epochs} | #train{num_train} #val{num_val}')

    for epoch in range(num_epochs):
        model.cuda()

        for phase in ['train', 'val']:
            if phase == 'train':
                if epoch == 0 and skip_first_train:
                    continue

                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_acc = 0

            dataloader = dataloaders_dict[phase]

            # for item in tqdm(dataloader, leave=False):
            for item in dataloader:

                states = item[0].cuda().float()
                actions = item[1].cuda().long()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    policy = model(states)
                    loss = criterion(policy, actions)
                    _, preds = torch.max(policy, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * len(policy)
                    epoch_acc += torch.sum(preds == actions.data)

            data_size = len(dataloader.dataset)
            epoch_loss = epoch_loss / data_size
            epoch_acc = epoch_acc.double() / data_size

            print(get_time(),
                  f'LR: {optimizer.param_groups[0]["lr"]} Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

        if epoch_acc > best_acc:
            traced = torch.jit.trace(model.cpu(), torch.rand(1, CHANNELS, map_size, map_size))
            traced.save('model.pth')
            print(
                f'Saved model.pth from epoch {epoch + 1} as it is the most accurate so far: Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            best_acc = epoch_acc

        traced = torch.jit.trace(model.cpu(), torch.rand(1, CHANNELS, map_size, map_size))
        traced.save(f'model_{epoch + 1}.pth')

        scheduler.step(epoch_loss)


def main():
    print('Start!')
    seed = 42
    seed_everything(seed)

    # map size to analyse
    filters = 32
    map_size = 16
    dataset_sizes = [12,16]

    make_input_size = map_size
    do_print = True
    obses, samples = create_dataset_from_json(episode_dir, set_sizes=dataset_sizes)
    if do_print:
        print('obses:', len(obses), 'samples:', len(samples))

    labels = [sample[-1] for sample in samples]
    actions = ['north', 'south', 'west', 'east', 'bcity', 't_north', 't_south', 't_west', 't_east']
    for value, count in zip(*np.unique(labels, return_counts=True)):
        if do_print:
            print(f'{actions[value]:^6}: {count:>3}')

    train, val = train_test_split(samples, test_size=0.1, random_state=42, stratify=labels)
    batch_size = 64
    train_loader = DataLoader(
        LuxDataset(obses, train, make_input_size=make_input_size),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        LuxDataset(obses, val, make_input_size=make_input_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    dataloaders_dict = {"train": train_loader, "val": val_loader}
    criterion = nn.CrossEntropyLoss()

    max_size_of_dataset = 32
    if len(dataset_sizes)>0:
        max_size_of_dataset = max (dataset_sizes)

    if map_size<max_size_of_dataset:
        print(f'map size {map_size} is set smaller than dataset size',dataset_sizes)
        exit()

        # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=10)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model = LuxNet(filt=filters); skip_first = False
    #model = torch.jit.load('./model4_32_7809.pth'); skip_first = False


    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-03)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    train_model(model, dataloaders_dict, criterion, optimizer, scheduler, num_epochs=4, map_size=map_size,
                skip_first_train=skip_first)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-04)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    train_model(model, dataloaders_dict, criterion, optimizer, scheduler, num_epochs=3, map_size=map_size,
                skip_first_train=skip_first)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-04)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    train_model(model, dataloaders_dict, criterion, optimizer, scheduler, num_epochs=3, map_size=map_size,
                skip_first_train=skip_first)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-05)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)
    train_model(model, dataloaders_dict, criterion, optimizer, scheduler, num_epochs=3, map_size=map_size,
                skip_first_train=skip_first)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-05)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)
    train_model(model, dataloaders_dict, criterion, optimizer, scheduler, num_epochs=20, map_size=map_size,
                skip_first_train=skip_first)

if __name__ == '__main__':
    main()
