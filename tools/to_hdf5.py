import argparse
import h5py
import os.path as osp
import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
import pickle
from pyquaternion import Quaternion

def int_from_fname(fname):
    return int(osp.basename(fname).split('.')[0])


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_file', type=str, required=True)
parser.add_argument('-p', '--prop_train', type=float, default=0.9)
parser.add_argument('-o', '--output_file', type=str, required=True)
parser.add_argument('-s', '--skip', type=int, default=1)
args = parser.parse_args()

root = args.data_file
episodes = glob.glob(osp.join(root, '**', 'episodes', 'episode*'), recursive=True)
print(f'Found {len(episodes)} episodes')

images, masks = [], []
actions = []
idxs = [0]
for path in tqdm(episodes):
    rgb_images = glob.glob(osp.join(path, 'front_rgb', '*.png'))
    rgb_images.sort(key=int_from_fname)
    rgb_images = [np.array(Image.open(p)) for p in rgb_images]
    rgb_images = np.stack(rgb_images, axis=0)
    rgb_images = rgb_images[::args.skip]
    images.append(rgb_images)
    idxs.append(idxs[-1] + len(rgb_images))

    mask_images = glob.glob(osp.join(path, 'front_mask', '*.png'))
    mask_images.sort(key=int_from_fname)
    mask_images = [np.array(Image.open(p))[:, :, 0] for p in mask_images]
    mask_images = np.stack(mask_images, axis=0)
    mask_images = mask_images[::args.skip]
    masks.append(mask_images)
    
    low_dim_obs = pickle.load(open(osp.join(path, 'low_dim_obs.pkl'), 'rb'))
    acts = np.stack([
        np.concatenate((o.gripper_pose, [o.gripper_open]))
        for o in low_dim_obs._observations
    ], axis=0)
    new_acts = []
    for i in range(args.skip, acts.shape[0], args.skip):
        a1, a2 = acts[i - args.skip], acts[i]

        t1, t2 = a1[:3], a2[:3]
        t_diff = t2 - t1

        q1, q2 = a1[3:7], a2[3:7]
        q1 = Quaternion(q1[3], q1[0], q1[1], q1[2])
        q2 = Quaternion(q2[3], q2[0], q2[1], q2[2])
        q_diff = q2 / q1
        q_diff = np.array([q_diff.x, q_diff.y, q_diff.z, q_diff.w])

        g_diff = a2[7]

        new_acts.append(np.concatenate((t_diff, q_diff, [g_diff])))
    new_acts.append(np.zeros_like(new_acts[-1]))
    acts = np.stack(new_acts, axis=0)
    actions.append(acts)

    assert rgb_images.shape[0] == mask_images.shape[0] == acts.shape[0] 

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



