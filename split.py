import pathlib
from pathlib import Path
import os
from shutil import copyfile, copy, copy2

episode_dir = 'C:/Users/vito/Dropbox/Exchange/luxai/episodes/TB/work/1127_all'
episode_out = 'C:/Users/vito/Dropbox/Exchange/luxai/episodes/TB/work/1127_split/'


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
# TB
episodes_to_exclude.extend(['28478317', '28479107', '28479301', '28479498', '28479691', '28479883', '28479969', '28480078', '28480272', '28480469', '28480660', '28480855', '28481048', '28481243', '28481441'])
episodes_to_exclude.extend(['30415549', '30416663', '30416938', '30417213', '30417487', '30417760', '30418034', '30418308', '30418582', '30418856', '30419130', '30419404', '30419678', '30419952', '30420226'])
episodes_to_exclude.extend(['28407441', '28408404', '28408593', '28408782', '28408973', '28409165', '28409355', '28409545', '28409735', '28409925', '28410114', '28410303', '28410492', '28410681', '28410873'])
# RL
episodes_to_exclude.extend(['30782136', '30782149', '30782437', '30782725', '30783014', '30783303', '30783593', '30783882', '30784170', '30784458', '30784750', '30785039', '30785157', '30785330', '30785620'])
episodes_to_exclude.extend(['30783290', '30783592', '30783881', '30784169', '30784457', '30784749', '30785038', '30785329', '30785619', '30785910', '30786199', '30786487', '30786776', '30787067', '30787356'])
episodes_to_exclude.extend(['30783579', '30783880', '30784168', '30784456', '30784748', '30785037', '30785328', '30785618', '30785909', '30786198', '30786486', '30786775', '30787066', '30787355', '30787644'])
episodes_to_exclude.extend(['30970698', '30971010', '30971311', '30971609', '30971911', '30972209', '30972507', '30972808', '30973106', '30973404', '30973700', '30974000', '30974214', '30974298', '30974593'])
episodes_to_exclude.extend(['30970997', '30971310', '30971607', '30971909', '30972208', '30972504', '30972806', '30973104', '30973399', '30973697', '30973999', '30974292', '30974590', '30974887', '30975184'])
episodes_to_exclude.extend(['30971294', '30971604', '30971907', '30972205', '30972501', '30972801', '30973101', '30973397', '30973693', '30973993', '30974290', '30974588', '30974884', '30975182', '30975479'])
episodes_to_exclude.extend(['30971591', '30971903', '30972201', '30972497', '30972798', '30973097', '30973396', '30973692', '30973991', '30974288', '30974581', '30974883', '30975181', '30975478', '30975778'])
episodes_to_exclude.extend(['30971890', '30971906', '30972204', '30972797', '30973095', '30973391', '30973688', '30973989', '30974285', '30974582', '30974877', '30975176', '30975474', '30975776', '30976077'])




loop_counter = 0
divide_by = 10
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
