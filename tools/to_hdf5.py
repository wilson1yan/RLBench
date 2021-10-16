import argparse
import h5py
import os.path as osp
import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
import pickle

def int_from_fname(fname):
    return int(osp.basename(fname).split('.')[0])


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_file', type=str, required=True)
parser.add_argument('-p', '--prop_train', type=float, default=0.9)
parser.add_argument('-o', '--output_file', type=str, required=True)
args = parser.parse_args()

root = args.data_file
episodes = glob.glob(osp.join(root, '**', 'episodes', 'episode*'), recursive=True)[:15]
print(f'Found {len(episodes)} episodes')

images, masks = [], []
actions = []
idxs = [0]
for path in tqdm(episodes):
    rgb_images = glob.glob(osp.join(path, 'front_rgb', '*.png'))
    rgb_images.sort(key=int_from_fname)
    rgb_images = [np.array(Image.open(p)) for p in rgb_images]
    rgb_images = np.stack(rgb_images, axis=0)
    images.append(rgb_images)
    idxs.append(idxs[-1] + len(rgb_images))

    mask_images = glob.glob(osp.join(path, 'front_mask', '*.png'))
    mask_images.sort(key=int_from_fname)
    mask_images = [np.array(Image.open(p))[:, :, 0] for p in mask_images]
    mask_images = np.stack(mask_images, axis=0)
    masks.append(mask_images)
    
    low_dim_obs = pickle.load(open(osp.join(path, 'low_dim_obs.pkl'), 'rb'))
    vels = [o.joint_velocities for o in low_dim_obs._observations][:-1]
    vels.insert(0, np.zeros_like(vels[0]))
    vels = np.stack(vels, axis=0)
    actions.append(vels)

    assert rgb_images.shape[0] == mask_images.shape[0] == vels.shape[0]
    

images = np.concatenate(images, axis=0)
masks = np.concatenate(masks, axis=0).astype(np.int32)
actions = np.concatenate(actions, axis=0).astype(np.float32)
idxs = np.array(idxs[:-1], dtype=np.int64)

t = int(len(idxs) * args.prop_train)
train_images, test_images = images[:idxs[t]], images[idxs[t]:]
train_masks, test_masks = masks[:idxs[t]], masks[idxs[t]:]
train_actions, test_actions = actions[:idxs[t]], actions[idxs[t]:]
train_idxs, test_idxs = idxs[:t], idxs[t:]
test_idxs -= test_idxs[0]

f = h5py.File(args.output_file, 'a')
f.create_dataset('train_data', data=train_images)
f.create_dataset('train_action', data=train_actions)
f.create_dataset('train_idx', data=train_idxs)
f.create_dataset('train_mask', data=train_masks)

f.create_dataset('test_data', data=test_images)
f.create_dataset('test_action', data=test_actions)
f.create_dataset('test_idx', data=test_idxs)
f.create_dataset('test_mask', data=test_masks)
f.close()

print('Saved data to:', args.output_file)



