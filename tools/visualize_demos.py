from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *
import numpy as np
import sys


task_name = sys.argv[1]


# To use 'saved' demos, set the path below, and set live_demos=False
live_demos = True
DATASET = ''

obs_config = ObservationConfig()
obs_config.set_all(False)
obs_config.front_camera.set_all(True)

action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
env = Environment(
    action_mode, DATASET, obs_config, False)
env.launch()

task = env.get_task(eval(task_name))

demos = task.get_demos(20, live_demos=live_demos)  # -> List[List[Observation]]

print('Done')
env.shutdown()
