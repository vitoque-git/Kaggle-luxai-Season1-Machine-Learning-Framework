import pathlib
from pathlib import Path
import os
from shutil import copyfile, copy, copy2

episode_dir = 'C:/git/luxai/episodes/all_26112021'
episode_out = 'C:/git/luxai/episodes/all_split/'


episode_out_eval = episode_out+'eval/'
episode_out_train = episode_out+'train/'

def copy_file(filepath_src, dst_folder):
    filename = filepath.name
    dst_string = dst_folder+filename
    copy2(filepath_src, dst_string)


def ensure_dir(file_path):
    print(file_path)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
        print('created',os.path.dirname(file_path))
    else:
        print('already exist',os.path.dirname(file_path))

ensure_dir(episode_out)
ensure_dir(episode_out_eval)
ensure_dir(episode_out_train)

episodes = [path for path in Path(episode_dir).glob('*.json') if
                ('output' not in path.name and '_info' not in path.name)]

print('episodes', len(episodes))
episodes_to_exclude = []
episodes_to_exclude.extend(['28478317', '28479107', '28479301', '28479498', '28479691', '28479883', '28479969', '28480078', '28480272', '28480469', '28480660', '28480855', '28481048', '28481243', '28481441'])
episodes_to_exclude.extend(['30415549', '30416663', '30416938', '30417213', '30417487', '30417760', '30418034', '30418308', '30418582', '30418856', '30419130', '30419404', '30419678', '30419952', '30420226'])
episodes_to_exclude.extend(['28407441', '28408404', '28408593', '28408782', '28408973', '28409165', '28409355', '28409545', '28409735', '28409925', '28410114', '28410303', '28410492', '28410681', '28410873'])

loop_counter = 0
divide_by = 12
excluded = 0
num_eval  = 0
num_train  = 0

for filepath in episodes:
    stem = filepath.stem
    if stem in episodes_to_exclude:
        excluded += 1
        print("exluded", stem)
    loop_counter += 1
    if loop_counter == divide_by:
        # 1/divide_by of samples go to eval
        num_eval += 1
        copy_file(filepath,episode_out_eval)
        loop_counter = 0
    else:
        # rest go to train
        num_train += 1
        copy_file(filepath, episode_out_train)
        pass

print('num_eval',num_eval,'; num_train',num_train,'; excluded',excluded,'of',len(episodes_to_exclude))
print('sum',num_eval+num_train+excluded)
