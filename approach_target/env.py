import gym
from gym import spaces
import numpy as np


class BasicEnv(gym.Env):
    # metadata = {"render.modes": ["human"]}

    def __init__(self, task_name, action_type, physics_dt=1./60,
                 rendering_dt=1./60, seed=0, headless=True):
        super().__init__()

        self.task_name = task_name
        self.action_type = action_type
        self.headless = headless
        self.num_episode = 0
        self.num_steps = 0
        self.num_broken = 0
        self.num_complete = 0

        # Number of steps for take the predicted action
        self.step_len = 101

        # The number of maximum step for one episode
        self.max_step_episode = 200
        # Initialize siaac sim
        self._simulation_app = self._init_isaac_sim(headless)

        ########################################
        # Setup scene
        ########################################
        from omni.isaac.core import World

        # Create the world
        self.world = World(physics_dt=physics_dt,
                           rendering_dt=rendering_dt,
                           stage_units_in_meters=0.01)
        # Add the task (robot, objects, and environment)
        _task = self._init_task(self.task_name, stiffness=None)
        self.world.add_task(_task)
        # Reset the world after adding assets (make sure everything exists)
        self.world.reset()

        # Record robot and object
        self.complete_thr = _task.complete_thr
        task_params = _task.get_params()
        self.workspace = task_params['workspace']['value']
        self.controled_joints = task_params['controled_joints']['value']
        self.franka = self.world.scene.get_object(task_params["robot_name"]["value"])
        self.target = self.world.scene.get_object(task_params["target_name"]["value"])

        # Setup camera (Only for visualization)
        self.camera_position = [40, -150, 150]
        self.camera_target = [40, 0, 40]
        self.sd_helper, self.viewport = self._set_camera(
            self.camera_position, self.camera_target)
        # Initialize whole world for starting
        self.reset()

        ########################################
        # Define action and observation space
        ########################################
        self.action_space = self._init_action_space()
        self.observation_space = self._init_observation_space()

        self.seed(seed)

    def _init_isaac_sim(self, headless=False, height=720, width=1024):
        """ Initialize Issac Sim. """

        from omni.isaac.kit import SimulationApp
        config = {"headless": headless,
                  "height": height,
                  "width": width,
                  "renderer": "RayTracedLighting",
                  "anti_aliasing": 0}
        return SimulationApp(config)

    def _init_task(self, task_name, stiffness=None):
        """ [_Rewrite_] Define task. """

        from approach_target.task import ApproachTargetTask
        task = ApproachTargetTask(name=task_name, stiffness=stiffness)
        return task

    def _init_observation_space(self):
        """ [_Rewrite_] Define observation space. """
        pass

    def _init_action_space(self):
        """ [_Rewrite_] Define action space. """
        pass

    def _create_controller(self):
        """ [_Rewrite_] Create a controller to control the robot
            (e.g., pose -> joint positions).
        """
        pass

    def _reward_function(self):
        """ [_Rewrite_] Calculate reward. (Large is better) """
        pass

    def _is_done(self):
        """ [_Rewrite_] Return `True` to reset. """
        pass

    def _get_observations(self):
        """ [_Rewrite_] Get observations. """
        pass

    def _set_camera(self, camera_position, camera_target):
        """ [_Rewrite_] Set camera to capture RGB-D images.
            This is for the default camera view.
        """

        if not self.headless:
            # Set default camera to visualize
            from omni.isaac.core.utils.viewports import set_camera_view
            set_camera_view(eye=np.array(camera_position),
                            target=np.array(camera_target))
        return None, None

    def reset(self):
        """ [**Required**] Reset world (Robot, object, and environment) and return
            observation.

            Returns:
                observation (np.float32)
        """

        self.num_episode += 1
        self.num_steps = 0
        # Initialize the objects and robot
        self.world.reset()

        # Observation
        current_observations = self._get_observations()

        for _ in range(5):
            self.render()
        return current_observations

    def step(self, action):
        """ [**Required**] [_Rewrite_]
            Args:
                action (np.array)
            Returns:
                observation (np.float32):
                reward      (float)     :
                done        (boolean)   : Execute reset() if it's True.
                info        (dict)      : Debugging information
        """
        pass

    def render(self, mode='human'):
        """ [**Required**] Render the simulation and physics engine. """

        self.world.step(render=True)
        return

    def close(self):
        """ [**Required**] Close IsaacSim. """

        self.world.stop()
        self._simulation_app.close()
        return

    def seed(self, seed=None):
        """ Select specific random seed. """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]


class ApproachTargetEnv(BasicEnv):
    def __init__(self, task_name, action_type, physics_dt=1./60,
                 rendering_dt=1./60, seed=0, headless=True, visualize=False):

        super().__init__(task_name, action_type, physics_dt, rendering_dt,
                         seed, headless)

        x_low, y_low, z_low = self.workspace['low']
        x_high, y_high, z_high = self.workspace['high']
        self.norm_distance = np.linalg.norm(
            np.array([x_high, y_high, z_high]) -
            np.array([x_low, y_low, z_low]))

        # Visualize will stop for 30 frames when arrive the target
        self.visualize = visualize

    def _init_observation_space(self):
        # Approaching target postion vector
        x_low, y_low, z_low = self.workspace['low']
        x_high, y_high, z_high = self.workspace['high']

        high = np.array([x_high-x_low,
                         y_high-y_low,
                         z_high-z_low], dtype=np.float32)
        low = -high
        approach_vector = spaces.Box(low=low, high=high,
                                     shape=(3,),
                                     dtype=np.float32)

        # Joint state (joint position)
        num_joints = len(self.controled_joints)
        joint_state = spaces.Box(low=-4, high=4,
                                 shape=(num_joints,),
                                 dtype=np.float32)
        # return approach_vector
        return spaces.Dict({'vector': approach_vector,
                            'joint_state': joint_state})

    def _init_action_space(self):
        num_joints = len(self.controled_joints)

        if self.action_type == 'positions':
            action_space = spaces.Box(
                low=-0.1, high=0.1, shape=(num_joints,), dtype=np.float32)
        elif self.action_type == 'velocities':
            action_space = spaces.Box(
                low=-2, high=2, shape=(num_joints,), dtype=np.float32)
        elif self.action_type == 'torques':
            action_space = spaces.Box(
                low=-100, high=100, shape=(num_joints,), dtype=np.float32)
        else:
            raise NotImplementedError(f"Action space `{self.action_type}` not exist.")
        return action_space

    def _get_observations(self):
        self.world.render()

        observations = self.world.get_observations()
        approach_vector = \
            observations['target_position'] - observations['current_position']

        joint_state = self.franka.get_joint_positions()[self.controled_joints]

        current_observations = {
            'vector': approach_vector,
            'joint_state': joint_state
        }
        return current_observations

    def _reward_function(self, previous_position, current_position,
                         target_position, broken):
        reward = 0
        complete = False

        if broken:
            return reward, complete

        # Approach target
        previous_dist_to_target = np.linalg.norm(
            target_position - previous_position)
        current_dist_to_target = np.linalg.norm(
            target_position - current_position)
        reward = previous_dist_to_target - current_dist_to_target

        reward += -(current_dist_to_target) / self.norm_distance

        # Success
        if current_dist_to_target < self.complete_thr:
            complete = True
            reward += 1
            if self.visualize:
                for _ in range(30):
                    self.world.render()

        return reward, complete

    def _is_done(self, complete):
        done = False
        if complete:
            self.num_complete += 1
            done = True

        elif self.num_steps >= self.max_step_episode:
            done = True

        return done

    def _control_robot(self, action):
        """ Set status of joints to control the robot and then step render and
            physics.
        """

        # Way 1
        # from omni.isaac.core.utils.types import ArticulationAction
        # articulation = ArticulationAction(joint_velocities=action * 10.0)
        # self.franka.apply_action(actuation, indices=self.controled_joints)

        # Way 2
        # position
        if self.action_type == 'positions':
            a0 = self.franka.get_joint_positions()[self.controled_joints]
            actuation = a0 + action
            self.franka.set_joint_positions(positions=actuation,
                                            indices=self.controled_joints)
        # Velocity
        elif self.action_type == 'velocities':
            a0 = self.franka.get_joint_velocities()[self.controled_joints]
            actuation = action
            self.franka.set_joint_velocities(velocities=actuation,
                                             indices=self.controled_joints)
        # Torques
        elif self.action_type == 'torques':
            a0 = self.franka.get_joint_efforts()[self.controled_joints]
            # The USD of the franka arm has a stage scale of cm
            actuation = action * (100**2)
            self.franka.set_joint_efforts(efforts=actuation,
                                          indices=self.controled_joints)

        # Move the robot
        for _ in range(self.step_len):
            self.world.step(render=self.visualize)
        self.world.render()

    def step(self, action):
        # Apply action
        observations = self.world.get_observations()
        previous_position = observations['current_position']

        self._control_robot(action)

        # Current position of end-effector (for calculating reward)
        observations = self.world.get_observations()
        current_position = observations['current_position']
        target_position = observations['target_position']

        # Avoid observation occurs nan
        broken = \
            np.isnan(list(self.franka.get_joint_positions())).any() or \
            np.isnan(current_position).any()
        if broken:
            self.num_broken += 1
            self.reset()

        # Observation
        current_observation = self._get_observations()

        # Reward
        reward, complete = self._reward_function(
            previous_position, current_position, target_position,
            broken)

        # Done
        done = self._is_done(complete)

        # Info
        info = {'reward': reward, 'action': list(action),
                'num_complete': self.num_complete,
                'num_episode': self.num_episode,
                'num_broken': self.num_broken}

        self.num_steps += 1
        return current_observation, reward, done, info
