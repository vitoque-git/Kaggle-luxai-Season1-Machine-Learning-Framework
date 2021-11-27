import traceback
from sys import exit

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

from datetime import datetime
import time


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
        label = {'c': 9, 'n': 0, 's': 1, 'w': 2, 'e': 3}[strs[2]]
    elif strs[0] == 'bcity':
        label = 4
    elif strs[0] == 't':
        from_pos = units[unit_id]
        if strs[2] in units:
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
    elif strs[0] == 'p':
        #pillage
        label = None
    else:
        if strs[0] not in ['r','bw','bc']:
            print("Unexpected no acton from",strs)
        label = None
    return unit_id, label



def depleted_resources(obs):
    for u in obs['updates']:
        if u.split(' ')[0] == 'r':
            return False
    return True


def create_dataset_from_json(episode_dir, team_name='', set_sizes=[], exclude_turns_on_after=350):
    if team_name=='':
        print('Need to specify a team name')
    samples = {}
    num_samples = 0
    num_episodes = 0
    num_actions=0
    num_non_actions=0
    num_cannot_work = 0

    episodes = [path for path in Path(episode_dir).glob('*.json') if
                ('output' not in path.name and '_info' not in path.name)]

    print('create_dataset_from_json,',team_name,':', episode_dir)
    for filepath in episodes:
        episode_samples = []
        episode_obses = {}
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
        num_episodes += 1
        for i in range(len(json_load['steps']) - 1):
            if json_load['steps'][i][index]['status'] == 'ACTIVE':
                actions = json_load['steps'][i + 1][index]['action']

                obs = json_load['steps'][i][0]['observation']

                # do not add samples after the cutoff turn. For example 350 if we do not want the last night
                if i >= exclude_turns_on_after:
                    break

                #do not add samples in which there are no resources, because they have no value for training (noise)
                if depleted_resources(obs):
                    break

                obs['player'] = index
                obs = dict([
                    (k, v) for k, v in obs.items()
                    if k in ['step', 'updates', 'player', 'width', 'height']
                ])
                obs_id = f'{ep_id}_{i}'
                episode_obses[obs_id] = obs

                units = {}
                unit_can_work = []
                for update in obs['updates']:
                    strs = update.split(' ')
                    input_identifier = strs[0]

                    if input_identifier == 'u':
                        x = int(strs[4])
                        y = int(strs[5])
                        tyep = int(strs[1])
                        team = int(strs[2])
                        if team == index:
                            #our unit!
                            unit_id = strs[3]

                            if type == 1:
                                print('turn',obs['step'], unit_id , 'is a cart...')


                            units[unit_id] = (x, y)
                            cooldown = float(strs[6])
                            if cooldown == 0:
                                unit_can_work.append(unit_id)
                            else:
                                num_cannot_work += 1


                unit_that_worked = []
                for action in actions:
                    unit_id, label = to_label(action, units)
                    unit_that_worked.append(unit_id) # if we move this below "if label", then we consider pillage a stay
                    if label is not None:
                        num_actions += 1
                        episode_samples.append((obs_id, unit_id, label))

                #those units could have worked but it didn't, it is an important to record those
                lazy_units = [u for u in unit_can_work if u not in unit_that_worked]
                # if obs['step'] <=4:
                #     print('turn',obs['step'],len(unit_can_work), '-', len(unit_that_worked),'=',len(lazy_units))
                #     print('turn', obs['step'], unit_can_work, '-', unit_that_worked, '=', lazy_units)
                for unit_id in lazy_units:
                    num_non_actions += 1
                    episode_samples.append((obs_id, unit_id, 9))

        samples[ep_id] = (episode_obses, episode_samples)
        num_samples += len(episode_samples)

    print(episode_dir,'num_episodes=',num_episodes)
    print("non_actions",num_non_actions,";actions",num_actions,";cannotwork",num_cannot_work)
    return samples, num_samples

CHANNELS = 25

# Input for Neural Network
def make_input(obs, unit_id, size=32):
    width, height = obs['width'], obs['height']
    x_shift = (size - width) // 2
    y_shift = (size - height) // 2

    check_invalid = False
    if width > size:
        # need to shift to get the main unit inside
        check_invalid = True
        for update in obs['updates']:
            strs = update.split(' ')
            if strs[0] == 'u' and unit_id == strs[3]:
                unit_x = int(strs[4])
                unit_y = int(strs[5])
                x_shift = get_shift_when_map_bigger_array(size, unit_x, width, x_shift)
                y_shift = get_shift_when_map_bigger_array(size, unit_y, height, y_shift)

                break

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
                # Main Unit
                if invalid_size(x, y, size):
                    print(f'WARN invalid xy', unit_id,x,y,'orig',strs[4],strs[5],'shift',x_shift, y_shift)
                    exit()
                # Position and Cargo
                b[:3, x, y] = (
                    1,
                    (wood + coal + uranium) / 100,
                    fuel / 4000
                )
            else:
                # Other Units
                team = int(strs[2])
                cooldown = float(strs[6])
                idx = 3 + (team - obs['player']) % 2 * 3
                if check_invalid:
                    if invalid_size(x, y, size):
                        continue

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
            if check_invalid:
                if invalid_size(x, y, size):
                    continue
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
            if check_invalid:
                if invalid_size(x, y, size):
                    continue
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
            excess_fuel = 0
            if will_live:
                excess_fuel = (1 + int(fuel) - (int(lightupkeep) * all_night_turns_lef)) / 4000
            will_live_next_night = autonomy >= next_night_number_turn
            cities[city_id] = (
                int(will_live_next_night),
                excess_fuel,
                turns_it_will_live(autonomy, steps_until_night) / 360)

    # Day/Night Cycle
    b[22, :] = obs['step'] % 40 / 40
    # Turns
    b[23, :] = obs['step'] / 360
    # Map Size
    b[24, x_shift:size - x_shift, y_shift:size - y_shift] = 1

    return b

def turns_it_will_live(autonomy, steps_until_night,_next_night_number_turn=-1) ->int:
    autonomy=max(0,autonomy)
    if _next_night_number_turn == -1:
        next_night_number_turn = min(10, 10 + steps_until_night)
    else:
        next_night_number_turn = _next_night_number_turn

    turn_to_night= max(0,steps_until_night)
    # print('turn_to_night',turn_to_night, 'next_night_number_turn',next_night_number_turn, 'aut',autonomy)
    if autonomy>=next_night_number_turn:
       return turns_it_will_live(autonomy-next_night_number_turn,turn_to_night+40,10)
    else:
       return autonomy + turn_to_night

def get_shift_when_map_bigger_array(size_array, unit_coordinate, size_map, shift):
    if unit_coordinate - (size_array // 2) <= 0:
        shift = 0
    elif unit_coordinate - (size_map // 2) <= 0:
        shift = (size_array // 2) - unit_coordinate
    elif unit_coordinate + (size_array // 2) > size_map:
        shift = size_array - size_map
    elif unit_coordinate + (size_map // 2) >= size_map:
        shift = (size_array // 2) - unit_coordinate + 1

    # print("XY5", unit_coordinate, shift)

    return shift



def invalid_size(x,y,size):
    return (x>=size) or (y>=size) or (x<0) or (y<0)

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
        self.head_p = nn.Linear(filters, len(actions), bias=False)

    def forward(self, x):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))
        h_head = (h * x[:, :1]).view(h.size(0), h.size(1), -1).sum(-1)
        p = self.head_p(h_head)
        return p



def get_time():
    now = datetime.now()

    return now.strftime("%H:%M:%S")

number_train_cycle = 0
def train_model(model, dataloaders_dict, criterion, optimizer, scheduler, num_epochs, map_size=32,
                skip_first_train=True, Save=True):
    global number_train_cycle
    try:
        number_train_cycle += 1
    except NameError:
        number_train_cycle = 1

    best_acc = 0.0

    num_train = len(dataloaders_dict['train'])
    num_val = len(dataloaders_dict['val'])

    print(get_time(),f' {number_train_cycle} LR: {optimizer.param_groups[0]["lr"]} Epochs {num_epochs} | #train{num_train} #val{num_val}')

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
                  f'LR: {optimizer.param_groups[0]["lr"]} Epoch {epoch + 1}/{num_epochs} of  {number_train_cycle} | {phase:^5}'
                  f' | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

            if not Save:
                r = (epoch_loss, float (f'{epoch_acc:.6f}'))
                return r

        if epoch_acc > best_acc and Save:
            traced = torch.jit.trace(model.cpu(), torch.rand(1, CHANNELS, map_size, map_size))
            traced.save('model.pth')
            print(
                f'Saved model.pth from epoch {epoch + 1} as it is the most accurate so far: Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            best_acc = epoch_acc

        if Save:
            traced = torch.jit.trace(model.cpu(), torch.rand(1, CHANNELS, map_size, map_size))
            suffix = datetime.now().strftime('%H%M')
            traced.save(f'model_{number_train_cycle}_{epoch + 1}_{suffix}.pth')

        scheduler.step(epoch_loss)




def my_train_test_split(samples, divide_by):
    loop_counter = 0
    obses_train, obses_eval = ({}, {})
    samples_train, samples_eval = ([], [])
    for obs, sample in samples.values():
        loop_counter += 1
        if loop_counter == divide_by:
            # 1/divide_by of samples go to eval
            obses_eval.update(obs)
            samples_eval.extend(sample)
            loop_counter = 0
        else:
            # rest go to train
            obses_train.update(obs)
            samples_train.extend(sample)
    return obses_eval, obses_train, samples_eval, samples_train

def samples_to_obs_sample_list(input_samples):
    obses  = {}
    samples = []
    for obs, sample in input_samples.values():
        obses.update(obs)
        samples.extend(sample)
    return obses, samples

def do_train(criterion, dataloaders_dict, map_size, model, num_epochs, lr,
             scheduler_factor=.5, scheduler_patience=-1, skip_first=False):
    if scheduler_patience == -1:
        scheduler_patience = num_epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=scheduler_factor, patience=scheduler_patience)
    train_model(model, dataloaders_dict, criterion, optimizer, scheduler, num_epochs=num_epochs, map_size=map_size,
                skip_first_train=skip_first)

def show_eval(dataloaders_dict, map_size, model):

    optimizer = torch.optim.AdamW(model.parameters(), lr=1)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=1)
    return train_model(model, dataloaders_dict, nn.CrossEntropyLoss(), optimizer, scheduler, num_epochs=1, map_size=map_size,
                skip_first_train=True, Save=False)


def show_accuracy_by_map(map_size,model_path, batch_size = 64):
    model = torch.jit.load(model_path);
    results = {}
    for dataset_sizes in [[12],[16],[24],[32],[]]:
        print("Loading with mapsize(s)=",dataset_sizes)
        ep_samples_eval, num_samples_eval = create_dataset_from_json(episode_eval, team_name=team_name,set_sizes=dataset_sizes)
        obses_eval, samples_eval  = samples_to_obs_sample_list(ep_samples_eval)
        make_input_size = map_size
        val_loader = DataLoader(
            LuxDataset(obses_eval, samples_eval, make_input_size=make_input_size),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        dataloaders_dict = {"train": val_loader, "val": val_loader}
        results[str(dataset_sizes)] = show_eval(dataloaders_dict, map_size, model)

    print('------------------------------------------------------------')
    for r in results.items():
        epoch_loss = r[1][0]
        epoch_acc = r[1][1]
        print(r[0],':',f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')


actions = ['north', 'south', 'west', 'east', 'bcity', 't_north', 't_south', 't_west', 't_east', 'stay']

# APPROACH 1, load episodes from one directory and then split
episode_dir = 'C:/git/luxai/episodes/all'

# APPROACH 2, load episodes from two different directories
episode_eval = 'C:/git/luxai/episodes/RL/work/1127_split/eval'
episode_train = 'C:/git/luxai/episodes/RL/work/1127_split/train'

# team_name = 'Toad Brigade'
team_name = 'RL is all you need'

def main():
    print('Start!', datetime.now().strftime('%H:%M'))

    seed = 42
    seed_everything(seed)

    # map size to analyse
    filters = 36
    map_size = 32
    dataset_sizes = []

    make_input_size = map_size

    # show performance
    # show_accuracy_by_map(map_size=map_size, model_path='./model.pth')
    # exit()

    # APPROACH 1, load episodes from one directory and then split
    # samples, num_samples = create_dataset_from_json(episode_dir, team_name=team_name, set_sizes=dataset_sizes)
    # print('episodes:', len(samples), 'samples:', num_samples)

    # # train, val = train_test_split(samples, test_size=0.1, random_state=42, stratify=labels)
    # #poor man split of data, based on episodees to avoid inter-episode contamination
    # obses_eval, obses_train, samples_eval, samples_train = my_train_test_split(samples, divide_by=11)

    # APPROACH 2, load episodes from two different directories
    ep_samples_eval, num_samples_eval = create_dataset_from_json(episode_eval, team_name=team_name,set_sizes=dataset_sizes)
    ep_samples_train, num_samples_train = create_dataset_from_json(episode_train, team_name=team_name, set_sizes=dataset_sizes)

    obses_eval, samples_eval  = samples_to_obs_sample_list(ep_samples_eval)
    obses_train, samples_train= samples_to_obs_sample_list(ep_samples_train)

    ep_samples_eval = None
    ep_samples_train = None
    print('eval  episodes:', len(samples_eval), 'samples:', num_samples_eval)
    # labels = [sample[-1] for sample in samples_eval]
    # for value, count in zip(*np.unique(labels, return_counts=True)):
    #     print(f'{actions[value]:^6}: {count:>3}')

    print('train episodes:', len(samples_train), 'samples:', num_samples_train)
    # labels = [sample[-1] for sample in samples_train]
    # for value, count in zip(*np.unique(labels, return_counts=True)):
    #     print(f'{actions[value]:^6}: {count:>3}')

    print('Train observations:', len(obses_train), 'samples:', len(samples_train))
    print('Eval  observations:', len(obses_eval), 'samples:', len(samples_eval))
    print(f'Ratio  observations: {len(obses_eval) / len(obses_train) :.4f}'
          f' samples:{len(samples_eval) / len(samples_train) :.4f}')

    # import os, psutil; print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

    batch_size = 64
    train_loader = DataLoader(
        LuxDataset(obses_train, samples_train, make_input_size=make_input_size),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        LuxDataset(obses_eval, samples_eval, make_input_size=make_input_size),
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
        print(f'WARN map size {map_size} is set smaller than dataset size',dataset_sizes)
        # exit()

        # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=10)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model = LuxNet(filt=filters); print('Starting new model filters=',filters); skip_first = False
    model_path='./model.pth'; model = torch.jit.load(model_path); print('Loading',model_path);skip_first = True

    # lr = 2e-03
    # for i in range(0,18):
    #     do_train(criterion, dataloaders_dict, map_size, model, num_epochs=1, lr=lr)
    #     lr = lr * .8

    do_train(criterion, dataloaders_dict, map_size, model, num_epochs=3, lr=1e-03)
    do_train(criterion, dataloaders_dict, map_size, model, num_epochs=3, lr=5e-04)
    do_train(criterion, dataloaders_dict, map_size, model, num_epochs=3, lr=1e-04)
    do_train(criterion, dataloaders_dict, map_size, model, num_epochs=3, lr=5e-05)
    do_train(criterion, dataloaders_dict, map_size, model, num_epochs=1, lr=1e-05)

    # do_train(criterion, dataloaders_dict, map_size, model, num_epochs=3, lr=2e-05)
    # do_train(criterion, dataloaders_dict, map_size, model, num_epochs=4, lr=1e-05)
    # do_train(criterion, dataloaders_dict, map_size, model, num_epochs=4, lr=5e-06)
    # do_train(criterion, dataloaders_dict, map_size, model, num_epochs=1, lr=1e-06)

if __name__ == '__main__':
    main()

