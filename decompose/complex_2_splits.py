### Given a file id-label, it produces the split where each time a class is used a test set and the others as a train set

import os 
import random 
import json 
import numpy as np 

random.seed(42)

ratio_val_train = 1/15 # the ration of the traiset that gonna compose the validationset 

labels_martialarts = ["kickbox", "jab", "karate", "sword", "boxe", "boxing", "martial art"] 
labels_dance = ["danc", "salsa", "cha cha"]
labels_ballsports = ["basket", "basketball", "basket ball", "tennis", "golf", "football", "foot ball","soccer", "cricket", "volley", "volleyball", "pitch",  "hockey", "baseball"] # ball
labels_music = ["drum", "guitar", "piano", "violin", "viola", "instrument", "saxophone", "flute", "bongos"]
lables_others = ["skat", "breast stroke", "breaststroke"] 

labels = labels_martialarts + labels_dance + labels_ballsports + labels_music + lables_others

dataset = "humanml3d" # "humanml3d"
save_dir_path = f"datasets/annotations/{dataset}/splits/me/"
json_path = f"datasets/annotations/{dataset}/annotations.json"
motion_dir_path = "datasets/motions/AMASS_20.0_fps_nh_smpljoints_neutral_nobetas"

annotations_file = json.load(open(json_path, 'r'))

test = {}
train_val = {}
for annots_id, annots in annotations_file.items():

    train = True
    for annotation in annots["annotations"]:
        if any([l in annotation["text"].lower() for l in labels]):
            train = False
            test[annots_id] = {k:v for k,v in annots.items() if k != "annotations"}
            test[annots_id]["annotations"] = [annotation]
            break
    if train:
        train_val[annots_id] = annots
    

# Divide train - validation
num_trainval = len(train_val.keys())
num_val = int(num_trainval * ratio_val_train)
num_train = num_trainval - num_val

train_keys = random.sample(list(train_val.keys()), k=num_train)
val_keys = list(set(train_val.keys()) - set(train_keys))
assert len(train_keys)==num_train
assert len(val_keys)==num_val

#for key, value in test.items():
#    print(f"{key} - {value}")

print(f"Total elements trainset: {num_train}")
print(f"Total elements validationset: {num_val}")
print(f"Total elements testset: {len(test.keys())}")

if not os.path.exists(save_dir_path): 
    os.makedirs(save_dir_path) 

### train test val keys files
with open(f'{save_dir_path}/test.txt', 'w') as test_file:
    for sample_id in test.keys():
        test_file.write(f'{sample_id}\n')
with open(f'{save_dir_path}/train.txt', 'w') as train_file:
    for sample_id in train_keys:
        train_file.write(f'{sample_id}\n')
with open(f'{save_dir_path}/val.txt', 'w') as val_file:
    for sample_id in val_keys:
        val_file.write(f'{sample_id}\n')
### TINY
with open(f'{save_dir_path}/test_tiny.txt', 'w') as test_file:
    for sample_id in list(test.keys())[:10]:
        test_file.write(f'{sample_id}\n')

### README
with open(f'{save_dir_path}/README.txt', 'w') as info_file:
    info_file.write(f"Excluded elements containing: {labels}")
### Annotations
with open(f'{save_dir_path}/annotations_test.json', 'w', encoding='utf-8') as json_file:
    json.dump(test, json_file, ensure_ascii=False, indent=4)
with open(f'{save_dir_path}/annotations_train.json', 'w', encoding='utf-8') as json_file:
    json.dump({k:v for k,v in train_val.items() if k in train_keys}, json_file, ensure_ascii=False, indent=4)
with open(f'{save_dir_path}/annotations_val.json', 'w', encoding='utf-8') as json_file:
    json.dump({k:v for k,v in train_val.items() if k in val_keys}, json_file, ensure_ascii=False, indent=4)    