import numpy as np
from functools import partial
from typing import List
from rlbench.backend.task import Task
from rlbench.backend.conditions import JointCondition
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint

COUNT_SUCCESS = True

class LampOnBig(Task):

    def init_task(self) -> None:
        self.bulb_glass_visual = Shape('bulb')
        self.bulb_glass_visual.set_color([0, 0, 0])
        self.joint = Joint('target_button_joint')
        self.condition = JointCondition(self.joint, 0.003)
        self.register_waypoint_ability_start(1, self._add_waypoint_noise)

        if COUNT_SUCCESS:
            self.total = 0
            self.success_count = 0 
            self.turned_on = False

    def init_episode(self, index: int) -> List[str]:
        if COUNT_SUCCESS:
            self.turned_on = False
            self.total += 1
        
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
        if self.condition.condition_met()[0]:
            if COUNT_SUCCESS:
                if not self.turned_on:
                    self.turned_on = True 
                    self.success_count += 1
                    print(f'Success count: {self.success_count}/{self.total}')
            
            self.bulb_glass_visual.set_color([1, 1, 1])
        
    def base_rotation_bounds(self):
        return (0.0, 0.0, -np.pi / 4), (0.0, 0.0, np.pi / 4)

    def _add_waypoint_noise(self, waypoint):
        waypoint = waypoint._waypoint
        eps = np.random.randn(3) * 0.01
        eps = np.clip(eps, -0.02, 0.02)
        waypoint.set_position(waypoint.get_position() + eps)
