# from turtle import distance
import numpy as np
from omni.isaac.core.objects import VisualSphere
from omni.isaac.core.tasks import BaseTask
from omni.isaac.franka import Franka
from pxr import Sdf, UsdLux, UsdPhysics


class ApproachTargetTask(BaseTask):
    """ Describe the task includes robot, objects, and environment.
        The target in this task is control robot to approach the target
        position.
    """

    def __init__(self, name, offset=None, stiffness=None):
        super().__init__(name, offset)
        self.stiffness = stiffness

        # Target (Visual)
        self.target_radius = 5.
        self.target_color = np.array([.8, .1, .1])
        self.approach_color = np.array([.1, .8, .1])

        # Worksapce, target only appear in the workspace (cm)
        self.workspace = {
            'low': np.array([10, -30, 10]) + self.target_radius,
            'high': np.array([70, 30, 70]) - self.target_radius}
        # The completed condition (distance between end-effector and target)
        self.complete_thr = 5

        # Robot
        self.robot_position = np.array([0, 0, 0])
        # Indices, only control these joints
        self.controled_joints = [0, 1, 2, 3, 4, 5]
        self.target_position = None

    def set_up_scene(self, scene):
        """ [_Rewrite_] Setup all the assets.
            (Called by `world.reset()` at first time)
        """
        super().set_up_scene(scene)

        # Add ground plane
        scene.add_default_ground_plane()

        # Add a distant light
        self._init_light(scene)

        # Target position
        position = self._random_position()
        self.target_position = position
        self.target = VisualSphere(prim_path="/World/Target",
                                   name="target",
                                   position=position,
                                   radius=self.target_radius,
                                   color=self.target_color)
        scene.add(self.target)

        # Robot
        self.franka = Franka(prim_path="/World/Franka",
                             name="franka",
                             position=self.robot_position)
        # Original stifness is about 1000
        if self.stiffness is not None:
            self._set_joint_stiffness(self.franka.prim, stiffness=0)
        scene.add(self.franka)

    def post_reset(self):
        """ [_Rewrite_] Called while doing a `.reset()` on the world. """

        # Reset target position
        position = self._random_position()
        self.target_position = position
        self.target.set_world_pose(position)
        self.target.get_applied_visual_material().set_color(self.target_color)

    def get_observations(self):
        """ [_Rewrite_] Called while doing `world.get_observations()`. """

        # Robot position
        current_position, _ = self.franka.end_effector.get_world_pose()

        # Target position
        target_position, _ = self.target.get_world_pose()

        observations = dict()
        observations['current_position'] = current_position
        observations['target_position'] = target_position
        return observations

    def pre_step(self, control_index, simulation_time):
        """ [_Rewrite_] Called before each physics step. """

        observations = self.get_observations()
        distance = np.linalg.norm(
            observations['target_position'] - observations['current_position'])
        if distance < self.complete_thr:
            self.target.get_applied_visual_material().set_color(
                self.approach_color)
        return

    def _init_light(self, scene):
        """ Add a distant light. """

        light_prim = UsdLux.DistantLight.Define(scene.stage,
                                                Sdf.Path("/DistantLight"))
        light_prim.CreateIntensityAttr(500)

    def _random_position(self):
        x, y, z = np.random.randint(self.workspace['low'],
                                    self.workspace['high'],
                                    size=3).astype(np.float32)

        return np.array([x, y, z])

    def _set_joint_stiffness(self, robot_prim, stiffness=0):
        """ This is important for velocities control. """

        for prim in robot_prim.GetAllChildren():
            if prim.GetTypeName() != 'Xform':
                continue
            for p in prim.GetAllChildren():
                prim_type = p.GetTypeName()
                if prim_type in ["PhysicsRevoluteJoint",
                                 "PhysicsPrismaticJoint"]:
                    if prim_type == "PhysicsRevoluteJoint":
                        drive = UsdPhysics.DriveAPI.Get(p, "angular")
                    else:
                        drive = UsdPhysics.DriveAPI.Get(p, "linear")
                    drive.GetStiffnessAttr().Set(stiffness)
                    # drive.GetDampingAttr().Set(0)

    def get_params(self):
        params = dict()
        params["target_name"] = {"value": self.target.name,
                                 "modifiable": False}
        params["robot_name"] = {"value": self.franka.name, "modifiable": False}
        params["workspace"] = {"value": self.workspace, "modifiable": False}
        params["controled_joints"] = {"value": self.controled_joints,
                                      "modifiable": False}
        return params
