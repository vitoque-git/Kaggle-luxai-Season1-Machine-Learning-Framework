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
MAX_CALLS_PER_DAY = 1000 # Kaggle says don't do more than 3600 per day and 1 per second
LOWEST_SCORE_THRESH = 1700

ROOT ="../working/"
META = "../input/meta-kaggle/"
META = "../input/d/kaggle/meta-kaggle/"

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

os.chdir('C:/git/luxai/episodes')

# Load Episodes
episodes_df = pd.read_csv("./Episodes_30067.csv")
print(f'Episodes.csv: {len(episodes_df)} rows before filtering.')
# Load EpisodeAgents
epagents_df = pd.read_csv("./EpisodeAgents.csv")


print(f'EpisodeAgents.csv: {len(epagents_df)} rows before filtering.')

episodes_df = episodes_df[episodes_df.CompetitionId == COMPETITIONS[COMP]]
epagents_df = epagents_df[epagents_df.EpisodeId.isin(episodes_df.Id)]

print(f'Episodes.csv: {len(episodes_df)} rows after filtering for {COMP}.')
print(f'EpisodeAgents.csv: {len(epagents_df)} rows after filtering for {COMP}.')

# episodes_df.to_csv('episodes_df.csv')
epagents_df.to_csv('epagents_df.csv')

# Prepare dataframes

episodes_df = episodes_df.set_index(['Id'])
episodes_df['CreateTime'] = pd.to_datetime(episodes_df['CreateTime'])
episodes_df['EndTime'] = pd.to_datetime(episodes_df['EndTime'])

epagents_df.fillna(0, inplace=True)
epagents_df = epagents_df.sort_values(by=['Id'], ascending=False)

# Get top scoring submissions# Get top scoring submissions
max_df = (epagents_df.sort_values(by=['EpisodeId'], ascending=False).groupby('SubmissionId').head(1).drop_duplicates().reset_index(drop=True))
max_df = max_df[max_df.UpdatedScore>=LOWEST_SCORE_THRESH]
max_df = pd.merge(left=episodes_df, right=max_df, left_on='Id', right_on='EpisodeId')
sub_to_score_top = pd.Series(max_df.UpdatedScore.values,index=max_df.SubmissionId).to_dict()
print(f'{len(sub_to_score_top)} submissions with score over {LOWEST_SCORE_THRESH}')

print("sub_to_score_top----------------")
print(sub_to_score_top)
max_df.to_csv('max_df.csv')




