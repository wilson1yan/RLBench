import argparse
import h5py
import os.path as osp
import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
import pickle
from pyquaternion import Quaternion
import multiprocessing as mp

def int_from_fname(fname):
    return int(osp.basename(fname).split('.')[0])


def read(path):
    rgb_images = glob.glob(osp.join(path, 'front_rgb', '*.png'))
    rgb_images.sort(key=int_from_fname)
    rgb_images = [np.array(Image.open(p)) for p in rgb_images]
    rgb_images = np.stack(rgb_images, axis=0)
    rgb_images = rgb_images[::args.skip]

    mask_images = glob.glob(osp.join(path, 'front_mask', '*.png'))
    mask_images.sort(key=int_from_fname)
    mask_images = [np.array(Image.open(p))[:, :, 0] for p in mask_images]
    mask_images = np.stack(mask_images, axis=0)
    mask_images = mask_images[::args.skip]
    
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
    assert rgb_images.shape[0] == mask_images.shape[0] == acts.shape[0] 

    return rgb_images, mask_images, acts


def read_data(f, split, episodes):
    episode_lens = [len(glob.glob(osp.join(path, 'front_rgb', '*.png')))
                    for path in episodes]
    image_size = Image.open(osp.join(episodes[0], 'front_rgb', '0.png')).width # assume width == height
    n_frames = sum(episode_lens)
    idx = np.cumsum([0] + episode_lens[:-1])
    f.create_dataset(f'{split}_data', (n_frames, image_size, image_size, 3), dtype=np.uint8)
    f.create_dataset(f'{split}_mask', (n_frames, image_size, image_size), dtype=np.int32)
    f.create_dataset(f'{split}_action', (n_frames, action_dim), dtype=np.float32)
    f.create_dataset(f'{split}_idx', data=idx)

    pool = mp.Pool(args.workers)
    pbar = tqdm(total=len(episodes))
    for i in range(0, len(episodes), args.load_chunk_size):
        start_ep = i
        end_ep = min(len(episodes), i + args.load_chunk_size)
        out = pool.map(read, episodes[start_ep:end_ep])
        out = list(zip(*out))
        rgb_images, mask_images, acts = [np.concatenate(o) for o in out]

        start_idx = idx[start_ep]
        end_idx = idx[end_ep] if end_ep < len(episodes) else n_frames
        f[f'{split}_data'][start_idx:end_idx] = rgb_images
        f[f'{split}_mask'][start_idx:end_idx] = mask_images
        f[f'{split}_action'][start_idx:end_idx] = acts
        pbar.update(end_ep - start_ep)


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_file', type=str, required=True)
parser.add_argument('-p', '--prop_train', type=float, default=0.9)
parser.add_argument('-o', '--output_file', type=str, required=True)
parser.add_argument('-s', '--skip', type=int, default=1)
parser.add_argument('-w', '--workers', type=int, default=64)
parser.add_argument('-l', '--load_chunk_size', type=int, default=512)
args = parser.parse_args()

if __name__ == '__main__':
    root = args.data_file
    episodes = glob.glob(osp.join(root, '**', 'episodes', 'episode*'), recursive=True)
    print(f'Found {len(episodes)} episodes')
    t = int(len(episodes) * args.prop_train)
    train_episodes, test_episodes = episodes[:t], episodes[t:]
    action_dim = 8 # 3 for translation, 4 for gripper rotation, 1 for gripper open / close

    f = h5py.File(args.output_file, 'a')
    read_data(f, 'train', train_episodes)
    read_data(f, 'test', test_episodes)
    f.close()

    print('Saved data to:', args.output_file)



