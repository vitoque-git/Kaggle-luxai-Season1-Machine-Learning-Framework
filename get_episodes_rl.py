import pandas as pd
import numpy as np
import os
import requests
import json
import datetime
import time
import glob
import collections

## You should configure these to your needs. Choose one of ...
# 'hungry-geese', 'rock-paper-scissors', santa-2020', 'halite', 'google-football'
COMP = 'lux-ai-2021'
MAX_CALLS_PER_DAY = 800 # Kaggle says don't do more than 3600 per day and 1 per second

MATCH_DIR = '../working/'
base_url = "https://www.kaggle.com/requests/EpisodeService/"
get_url = base_url + "GetEpisodeReplay"
BUFFER = 1
COMPETITIONS = {
    'lux-ai-2021': 30067,
    'hungry-geese': 25401,
    'rock-paper-scissors': 22838,
    'santa-2020': 24539,
    'halite': 18011,
    'google-football': 21723
}

# RELOAD

os.chdir('C:/git/luxai/episodes')
epagents_df = pd.read_csv("./EpisodeAgents_RL.csv")

sub_to_score_top = [23825143,23825329,23825266,23825370,23825224,23770016,23769678,23770123,23939664,23939632]
BASE_OUTPUT_DIRECTORY = 'C:/Users/vito/Dropbox/Exchange/luxai/episodes/'
SUFFIX_DIRECTORY = 'RL'

print(f'EpisodeAgents.csv: {len(epagents_df)} rows before filtering for {sub_to_score_top}.')
epagents_df = epagents_df[epagents_df.SubmissionId.isin(sub_to_score_top)]
print(f'EpisodeAgents.csv: {len(epagents_df)} rows after filtering for {sub_to_score_top}.')

episodes_df = pd.read_csv("./Episodes_30067.csv")
print(f'Episodes.csv: {len(episodes_df)} rows before filtering.')
episodes_df = episodes_df[episodes_df.Id.isin(epagents_df.EpisodeId)]
print(f'Episodes.csv: {len(episodes_df)} rows after filtering.')

# Prepare dataframes


episodes_df = episodes_df.set_index(['Id'])
episodes_df['CreateTime'] = pd.to_datetime(episodes_df['CreateTime'])
episodes_df['EndTime'] = pd.to_datetime(episodes_df['EndTime'])

epagents_df.fillna(0, inplace=True)
epagents_df = epagents_df.sort_values(by=['Id'], ascending=False)


# Get episodes for these submissions
print('Get episodes for these submissions')
sub_to_episodes = collections.defaultdict(list)
for key in sub_to_score_top:
    eps = sorted(epagents_df[epagents_df['SubmissionId'].isin([key])]['EpisodeId'].values, reverse=True)
    sub_to_episodes[key] = eps

candidates = len(set([item for sublist in sub_to_episodes.values() for item in sublist]))
print(f'{candidates} episodes for these {len(sub_to_score_top)} submissions')

global num_api_calls_today
num_api_calls_today = 0
all_files = []
for root, dirs, files in os.walk(MATCH_DIR, topdown=False):
    all_files.extend(files)
seen_episodes = [int(f.split('.')[0]) for f in all_files
                      if '.' in f and f.split('.')[0].isdigit() and f.split('.')[1] == 'json']
remaining = np.setdiff1d([item for sublist in sub_to_episodes.values() for item in sublist],seen_episodes)
print(f'{len(remaining)} of these {candidates} episodes not yet saved')
print('Total of {} games in existing library'.format(len(seen_episodes)))


def create_info_json(epid):
    create_seconds = int((episodes_df[episodes_df.index == epid]['CreateTime'].values[0]).item() / 1e9)
    end_seconds = int((episodes_df[episodes_df.index == epid]['CreateTime'].values[0]).item() / 1e9)

    agents = []
    for index, row in epagents_df[epagents_df['EpisodeId'] == epid].sort_values(by=['Index']).iterrows():
        agent = {
            "id": int(row["Id"]),
            "state": int(row["State"]),
            "submissionId": int(row['SubmissionId']),
            "reward": int(row['Reward']),
            "index": int(row['Index']),
            "initialScore": float(row['InitialScore']),
            "initialConfidence": float(row['InitialConfidence']),
            "updatedScore": float(row['UpdatedScore']),
            "updatedConfidence": float(row['UpdatedConfidence']),
            "teamId": int(99999)
        }
        agents.append(agent)

    info = {
        "id": int(epid),
        "competitionId": int(COMPETITIONS[COMP]),
        "createTime": {
            "seconds": int(create_seconds)
        },
        "endTime": {
            "seconds": int(end_seconds)
        },
        "agents": agents
    }

    return info


def saveEpisode(directory,epid) -> bool:
    # request, return whether used the API call
    path_ep = get_path(directory,epid)
    if os.path.exists(path_ep):
        print('File already exist',path_ep)
        return False

    re = requests.post(get_url, json={"EpisodeId": int(epid)})
    if not os.path.exists(str(directory)):
        os.makedirs(str(directory))
    # save replay
    with open(path_ep, 'w') as f:
        f.write(re.json()['result']['replay'])

    # save match info
    # info = create_info_json(epid)
    # with open(MATCH_DIR + '{}_info.json'.format(epid), 'w') as f:
    #     json.dump(info, f)
    return True

def get_path(directory,epid):
    return '{}/{}.json'.format(directory,epid)

r = BUFFER;

start_time = datetime.datetime.now()
se = 0
for key in sub_to_score_top:
    if num_api_calls_today <= MAX_CALLS_PER_DAY:
        print('')
        remaining = sorted(np.setdiff1d(sub_to_episodes[key], seen_episodes), reverse=True)
        print(
            f'submission={key}, matches={len(set(sub_to_episodes[key]))}, still to save={len(remaining)}')

        for epid in remaining:
            directory = BASE_OUTPUT_DIRECTORY + SUFFIX_DIRECTORY + '/' + str(key)
            if epid not in seen_episodes and num_api_calls_today <= MAX_CALLS_PER_DAY:
                if not saveEpisode(directory,epid):
                    continue # file already existed, we have not used our API call, we do not need to check, next
                r += 1
                se += 1
                try:
                    size = os.path.getsize(get_path(directory,epid)) / 1e6
                    print(str(num_api_calls_today) + f': saved episode #{epid}')
                    seen_episodes.append(epid)
                    num_api_calls_today += 1
                except:
                    print('  file {}.json did not seem to save'.format(epid),':', get_path(directory,epid))
                if r > (datetime.datetime.now() - start_time).seconds:
                    time.sleep(r - (datetime.datetime.now() - start_time).seconds)
            if num_api_calls_today > (min(3600, MAX_CALLS_PER_DAY)):
                break
print('')
print(f'Episodes saved: {se}')

