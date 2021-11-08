import numpy as np
from functools import partial
from typing import List
from rlbench.backend.task import Task
from rlbench.backend.conditions import JointCondition
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint


class LampOn(Task):

    def init_task(self) -> None:
        self.bulb_glass_visual = Shape('bulb')
        self.bulb_glass_visual.set_color([0, 0, 0])
        self.joint = Joint('target_button_joint')
        self.condition = JointCondition(self.joint, 0.003)
        # self.register_waypoint_ability_start(1, self._add_waypoint_noise)

    def init_episode(self, index: int) -> List[str]:
        self.bulb_glass_visual.set_color([0, 0, 0])
        self.register_success_conditions([self.condition])
        return ['turn on the light',
                'press the button to turn on the lamp',
                'press the light switch',
                'turn the lamp on',
                'close the gripper and press on the button until the light '
                'turns on']

    def variation_count(self) -> int:
        return 1

    def step(self) -> None:
        if self.condition.condition_met() == (True, True):
            self.bulb_glass_visual.set_color([1, 1, 1])
        
    # def base_rotation_bounds(self):
    #     return (0.0, 0.0, -np.pi), (0.0, 0.0, 0.0)

    # def _add_waypoint_noise(self, waypoint):
    #     waypoint = waypoint._waypoint
    #     print(waypoint.get_position())
    #     if np.random.rand() < 0.25:
    #         eps = np.zeros(3)
    #     else:
    #         eps = np.random.randn(3) * 0.025
    #     waypoint.set_position(waypoint.get_position() + eps)