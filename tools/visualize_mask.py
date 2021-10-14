from PIL import Image
import numpy as np
import glob
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib.cm as cm

root = osp.join('data', 'stack_blocks', 'variation0', 'episodes', 'episode0')
rgb_images = glob.glob(osp.join(root, 'front_rgb', '*.png'))
rgb_images.sort()
mask_images = glob.glob(osp.join(root, 'front_mask', '*.png'))
mask_images.sort()

assert len(rgb_images) == len(mask_images)

i = np.random.choice(len(rgb_images))
print(i)
rgb_image = np.array(Image.open(rgb_images[i]))
mask_image = np.array(Image.open(mask_images[i]))[:, :, 0]
unique_handles = np.unique(mask_image)

H, W = rgb_image.shape[:2]
for i, h in enumerate(unique_handles):
    mask = mask_image == h
    mask = mask.astype(float)[:, :, None]
    plt.imsave(f'handle_{h}.png', rgb_image.astype(float) / 255 * mask)

plt.imsave('origina_image.png', rgb_image)