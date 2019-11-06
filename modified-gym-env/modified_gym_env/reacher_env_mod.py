from pybulletgym.envs.roboschool.envs.env_bases import BaseBulletEnv
from pybulletgym.envs.roboschool.scenes.scene_bases import SingleRobotEmptyScene
import numpy as np
# from pybulletgym.envs.roboschool.robots.manipulators.reacher import Reacher
from modified_gym_env.reacher_fixed_goal import Reacher


class ReacherBulletEnv(BaseBulletEnv):
    def __init__(self, rand_init=True):
        self.robot = Reacher(rand_init)
        BaseBulletEnv.__init__(self, self.robot)

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=0.0, timestep=0.0165, frame_skip=1)

    def step(self, a):
        assert (not self.scene.multiplayer)
        self.robot.apply_action(a)
        self.scene.global_step()
        done = False

        state = self.robot.calc_state()  # sets self.to_target_vec

        self.potential = self.robot.calc_potential()*0.01

        electricity_cost = (
             - 0.1 * (a[0]**2 + a[1]**2)  # stall torque require some energy
        )
        self.rewards = [float(self.potential), float(electricity_cost)]
        if abs(self.potential) <0.01:
            done = True
            self.reward = [0]
        self.HUD(state, a, False)
        return state, sum(self.rewards) , done, {}

    def camera_adjust(self):
        x, y, z = self.robot.fingertip.pose().xyz()
        x *= 0.5
        y *= 0.5
        self.camera.move_and_look_at(0.3, 0.3, 0.3, x, y, z)
