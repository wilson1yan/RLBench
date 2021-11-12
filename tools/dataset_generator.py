from multiprocessing import Process, Manager

from pyrep.const import RenderMode

from rlbench import ObservationConfig
from rlbench.action_modes import ActionMode
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment
import rlbench.backend.task as rltask

from tqdm import tqdm
import os
import pickle
from PIL import Image
from rlbench.backend import utils
from rlbench.backend.const import *
import numpy as np

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('save_path',
                    '/tmp/rlbench_data/',
                    'Where to save the demos.')
flags.DEFINE_list('task', None, 'The task to collect')
flags.DEFINE_list('image_size', [128, 128],
                  'The size of the images tp save.')
flags.DEFINE_enum('renderer',  'opengl3', ['opengl', 'opengl3'],
                  'The renderer to use. opengl does not include shadows, '
                  'but is faster.')
flags.DEFINE_integer('processes', 1,
                     'The number of parallel processes during collection.')
flags.DEFINE_integer('episodes', 10,
                     'The number of episodes')
flags.DEFINE_integer('variations', -1,
                     'Number of variations to collect per task. -1 for all.')


def check_and_make(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

        
def compute_chunk(worker_id, n_workers, episodes, num_variations):
    eps_per_variation = [episodes // num_variations + (i < episodes % num_variations)
                         for i in range(num_variations)]
    assert sum(eps_per_variation) == episodes and len(eps_per_variation) == num_variations

    eps_per_worker = [episodes // n_workers + (i < episodes % n_workers)
                      for i in range(n_workers)]
    assert sum(eps_per_worker) == episodes and len(eps_per_worker) == n_workers
    
    chunks = [[] for _ in range(n_workers)]
    for i in range(n_workers):
        leftover_worker = eps_per_worker[i]
        for j in range(num_variations):
            leftover_var = eps_per_variation[j]
            if leftover_var <= 0:
                continue

            if leftover_worker <= leftover_var:
                chunks[i].append((j, leftover_worker))
                leftover_var -= leftover_worker
                eps_per_variation[j] = leftover_var
                break
            else:
                chunks[i].append(j, leftover_var)
                leftover_worker -= leftover_var
                eps_per_variation[j] = 0
    
    assert all([sum([ci[1] for ci in c]) == w for c, w in zip(chunks, eps_per_worker)])
    assert sum([c[1] for c in sum(chunks, [])]) == episodes

    start_ids = np.cumsum([0] + eps_per_worker[:-1])

    return chunks[worker_id], start_ids[worker_id]



def save_demo(demo, example_path):

    # Save image data first, and then None the image data, and pickle
#    left_shoulder_rgb_path = os.path.join(
#        example_path, LEFT_SHOULDER_RGB_FOLDER)
#    left_shoulder_depth_path = os.path.join(
#        example_path, LEFT_SHOULDER_DEPTH_FOLDER)
#    left_shoulder_mask_path = os.path.join(
#        example_path, LEFT_SHOULDER_MASK_FOLDER)
#    right_shoulder_rgb_path = os.path.join(
#        example_path, RIGHT_SHOULDER_RGB_FOLDER)
#    right_shoulder_depth_path = os.path.join(
#        example_path, RIGHT_SHOULDER_DEPTH_FOLDER)
#    right_shoulder_mask_path = os.path.join(
#        example_path, RIGHT_SHOULDER_MASK_FOLDER)
#    overhead_rgb_path = os.path.join(
#        example_path, OVERHEAD_RGB_FOLDER)
#    overhead_depth_path = os.path.join(
#        example_path, OVERHEAD_DEPTH_FOLDER)
#    overhead_mask_path = os.path.join(
#        example_path, OVERHEAD_MASK_FOLDER)
#    wrist_rgb_path = os.path.join(example_path, WRIST_RGB_FOLDER)
#    wrist_depth_path = os.path.join(example_path, WRIST_DEPTH_FOLDER)
#    wrist_mask_path = os.path.join(example_path, WRIST_MASK_FOLDER)
    front_rgb_path = os.path.join(example_path, FRONT_RGB_FOLDER)
#    front_depth_path = os.path.join(example_path, FRONT_DEPTH_FOLDER)
    front_mask_path = os.path.join(example_path, FRONT_MASK_FOLDER)

#    check_and_make(left_shoulder_rgb_path)
#    check_and_make(left_shoulder_depth_path)
#    check_and_make(left_shoulder_mask_path)
#    check_and_make(right_shoulder_rgb_path)
#    check_and_make(right_shoulder_depth_path)
#    check_and_make(right_shoulder_mask_path)
#    check_and_make(overhead_rgb_path)
#    check_and_make(overhead_depth_path)
#    check_and_make(overhead_mask_path)
#    check_and_make(wrist_rgb_path)
#    check_and_make(wrist_depth_path)
#    check_and_make(wrist_mask_path)
    check_and_make(front_rgb_path)
#    check_and_make(front_depth_path)
    check_and_make(front_mask_path)

    for i, obs in enumerate(demo):
#        left_shoulder_rgb = Image.fromarray(obs.left_shoulder_rgb)
#        left_shoulder_depth = utils.float_array_to_rgb_image(
#            obs.left_shoulder_depth, scale_factor=DEPTH_SCALE)
#        left_shoulder_mask = Image.fromarray(
#            (obs.left_shoulder_mask * 255).astype(np.uint8))
#        right_shoulder_rgb = Image.fromarray(obs.right_shoulder_rgb)
#        right_shoulder_depth = utils.float_array_to_rgb_image(
#            obs.right_shoulder_depth, scale_factor=DEPTH_SCALE)
#        right_shoulder_mask = Image.fromarray(
#            (obs.right_shoulder_mask * 255).astype(np.uint8))
#        overhead_rgb = Image.fromarray(obs.overhead_rgb)
#        overhead_depth = utils.float_array_to_rgb_image(
#            obs.overhead_depth, scale_factor=DEPTH_SCALE)
#        overhead_mask = Image.fromarray(
#            (obs.overhead_mask * 255).astype(np.uint8))
#        wrist_rgb = Image.fromarray(obs.wrist_rgb)
#        wrist_depth = utils.float_array_to_rgb_image(
#            obs.wrist_depth, scale_factor=DEPTH_SCALE)
#        wrist_mask = Image.fromarray((obs.wrist_mask * 255).astype(np.uint8))
        front_rgb = Image.fromarray(obs.front_rgb)
#        front_depth = utils.float_array_to_rgb_image(
#            obs.front_depth, scale_factor=DEPTH_SCALE)
        front_mask = Image.fromarray((obs.front_mask * 255).astype(np.uint8))

#        left_shoulder_rgb.save(
#            os.path.join(left_shoulder_rgb_path, IMAGE_FORMAT % i))
#        left_shoulder_depth.save(
#            os.path.join(left_shoulder_depth_path, IMAGE_FORMAT % i))
#        left_shoulder_mask.save(
#            os.path.join(left_shoulder_mask_path, IMAGE_FORMAT % i))
#        right_shoulder_rgb.save(
#            os.path.join(right_shoulder_rgb_path, IMAGE_FORMAT % i))
#        right_shoulder_depth.save(
#            os.path.join(right_shoulder_depth_path, IMAGE_FORMAT % i))
#        right_shoulder_mask.save(
#            os.path.join(right_shoulder_mask_path, IMAGE_FORMAT % i))
#        overhead_rgb.save(
#            os.path.join(overhead_rgb_path, IMAGE_FORMAT % i))
#        overhead_depth.save(
#            os.path.join(overhead_depth_path, IMAGE_FORMAT % i))
#        overhead_mask.save(
#            os.path.join(overhead_mask_path, IMAGE_FORMAT % i))
#        wrist_rgb.save(os.path.join(wrist_rgb_path, IMAGE_FORMAT % i))
#        wrist_depth.save(os.path.join(wrist_depth_path, IMAGE_FORMAT % i))
#        wrist_mask.save(os.path.join(wrist_mask_path, IMAGE_FORMAT % i))
        front_rgb.save(os.path.join(front_rgb_path, IMAGE_FORMAT % i))
#        front_depth.save(os.path.join(front_depth_path, IMAGE_FORMAT % i))
        front_mask.save(os.path.join(front_mask_path, IMAGE_FORMAT % i))

        # We save the images separately, so set these to None for pickling.
        obs.left_shoulder_rgb = None
        obs.left_shoulder_depth = None
        obs.left_shoulder_point_cloud = None
        obs.left_shoulder_mask = None
        obs.right_shoulder_rgb = None
        obs.right_shoulder_depth = None
        obs.right_shoulder_point_cloud = None
        obs.right_shoulder_mask = None
        obs.overhead_rgb = None
        obs.overhead_depth = None
        obs.overhead_point_cloud = None
        obs.overhead_mask = None
        obs.wrist_rgb = None
        obs.wrist_depth = None
        obs.wrist_point_cloud = None
        obs.wrist_mask = None
        obs.front_rgb = None
        obs.front_depth = None
        obs.front_point_cloud = None
        obs.front_mask = None

    # Save the low-dimension data
    with open(os.path.join(example_path, LOW_DIM_PICKLE), 'wb') as f:
        pickle.dump(demo, f)


def run(worker_id, n_workers, episodes, task):
    """Each thread will choose one task and variation, and then gather
    all the episodes_per_task for that variation."""
    # Initialise each thread with random seed
    np.random.seed(None)
    img_size = list(map(int, FLAGS.image_size))

    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.set_all_low_dim(True)
    obs_config.front_camera.set_all(True)
    obs_config.front_camera.image_size = img_size

    # We want to save the masks as rgb encodings.
    obs_config.front_camera.masks_as_one_channel = False

    if FLAGS.renderer == 'opengl':
        obs_config.front_camera.render_mode = RenderMode.OPENGL

    rlbench_env = Environment(
        action_mode=ActionMode(),
        obs_config=obs_config,
        headless=True)
    rlbench_env.launch()

    task_env = rlbench_env.get_task(task)
    if FLAGS.variations >= 0:
        num_variations = min(FLAGS.variations, task_env.variation_count())
    else:
        num_variations = task_env.variation_count()
    chunks, ep_id = compute_chunk(worker_id, n_workers, episodes, num_variations)
    print('Process', worker_id, 'chunks:', chunks)

    if worker_id == 0:
        pbar = tqdm(total=sum([c[1] for c in chunks]))
    for var_id, n_eps in chunks:
        task_env.set_variation(var_id)
        obs, descriptions = task_env.reset()

        variation_path = os.path.join(
            FLAGS.save_path, task_env.get_name(),
            VARIATIONS_FOLDER % var_id)
        episodes_path = os.path.join(variation_path, EPISODES_FOLDER)

        abort_variation = False
        for ex_idx in range(ep_id, ep_id + n_eps):
            #print('Process', worker_id, '// Task:', task_env.get_name(),
            #      '// Variation:', var_id, '// Demo:', ex_idx)
            attempts = 10
            while attempts > 0:
                try:
                    # TODO: for now we do the explicit looping.
                    demo, = task_env.get_demos(
                        amount=1,
                        live_demos=True)
                except Exception as e:
                    attempts -= 1
                    if attempts > 0:
                        continue
                    problem = (
                        'Process %d failed collecting task %s (variation: %d, '
                        'example: %d). Skipping this task/variation.\n%s\n' % (
                            worker_id, task_env.get_name(), var_id, ex_idx,
                            str(e))
                    )
                    print(problem)
                    abort_variation = True
                    break
                episode_path = os.path.join(episodes_path, EPISODE_FOLDER % ex_idx)
                save_demo(demo, episode_path)
                if worker_id == 0:
                    pbar.update(1)
                break
            if abort_variation:
                break
    if worker_id == 0:
        pbar.close()

    rlbench_env.shutdown()


def main(argv):

    task_files = [t.replace('.py', '') for t in os.listdir(rltask.TASKS_PATH)
                  if t != '__init__.py' and t.endswith('.py')]

    if FLAGS.task[0] not in task_files:
        raise ValueError('Task %s not recognised!.' % t)
    task_file = FLAGS.task[0]

    task = task_file_to_task_class(task_file)

    check_and_make(FLAGS.save_path)

    #rlbench_env = Environment(
    #    action_mode=ActionMode(),
    #    headless=True)
    #rlbench_env.launch()
    #task_env = rlbench_env.get_task(task)
    #if FLAGS.variations >= 0:
    #    num_variations = min(FLAGS.variations, task_env.variation_count())
    #else:
    #    num_variations = task_env.variation_count()
    num_variations = 1
        
    for var_id in range(num_variations):
        variation_path = os.path.join(
            FLAGS.save_path, 'lamp_on',#task_env.get_name(),
            VARIATIONS_FOLDER % var_id)
        check_and_make(variation_path) 

        episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
        check_and_make(episodes_path)

    processes = [Process(
        target=run, args=(
            i, FLAGS.processes, FLAGS.episodes, task))
        for i in range(FLAGS.processes)]
    print('Starting processes')
    [t.start() for t in processes]
    [t.join() for t in processes]

    print('Data collection done!')


if __name__ == '__main__':
  app.run(main)
