from pathlib import Path
import os

# TB
# episode_dir = 'C:/git/luxai/episodes/23297953'
# episode_dir = 'C:/git/luxai/episodes/23692494'
episode_dir = 'C:/git/luxai/episodes/23281649'

# RL
episode_dir = 'C:/git/luxai/episodes/RL/23769678'
episode_dir = 'C:/git/luxai/episodes/RL/23770016'
episode_dir = 'C:/git/luxai/episodes/RL/23770123'
episode_dir = 'C:/git/luxai/episodes/RL/23825143'
episode_dir = 'C:/git/luxai/episodes/RL/23825224'
episode_dir = 'C:/git/luxai/episodes/RL/23825266'
episode_dir = 'C:/git/luxai/episodes/RL/23825329'
episode_dir = 'C:/git/luxai/episodes/RL/23825370'
episode_dir = 'C:/Users/vito/Dropbox/Exchange/luxai/episodes/RL/23939664'
episode_dir = 'C:/Users/vito/Dropbox/Exchange/luxai/episodes/RL/23939632'

episodes = [path for path in Path(episode_dir).glob('*.json') if
                ('output' not in path.name and '_info' not in path.name)]

episode_list= []
for filepath in episodes:
    stem = filepath.stem
    episode_list.append(stem)

episode_list.sort()
print(len(episode_list))
truncated= episode_list[0:15]
print(len(truncated))
print('episodes_to_exclude.extend('+str(truncated)+')')