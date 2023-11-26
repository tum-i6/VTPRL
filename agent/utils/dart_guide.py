"""
Help code for:
              1. how to retrieve transforamtion matrices of chains of rigidbodies in target (dart)

              2. how to generate reachable targets for the end-effector pose

DART uses the Eigen library - Geometry module documentation: https://eigen.tuxfamily.org/dox/group__Geometry__Module.html

body_node.getTransform().translation() --> Cartesian position (x,y,z) of the center of that body_node
body_node.getTransform().rotation()    --> Orientation of the body_node in a 3*3 rotation matrix
body_node.getTransform().quaternion()  --> Orientation of the body_node in a unit quaternion form (w,x,y,z)
dart.math.logMap()                     --> calculates an angle-axis representation of a rotation matrix

background details:   https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation
further code details: https://github.com/dartsim/dart/blob/main/dart/math/Geometry.cpp
"""

import dartpy as dart
import numpy as np

def get_pos_error(self):
    ee = self.dart_sim.chain.getBodyNode('iiwa_link_ee')  # The end-effector rigid-body node
    target = self.dart_sim.target.getBodyNode(0)          # The target rigid-body node
    position_error = target.getTransform().translation() - ee.getTransform().translation()

    return position_error

def get_pos_distance(self):
    distance = np.linalg.norm(self.get_pos_error())

    return distance

def get_rot_error(self):
    ee = self.dart_sim.chain.getBodyNode('iiwa_link_ee')  # The end-effector rigid-body node
    target = self.dart_sim.target.getBodyNode(0)          # The target rigid-body node
    quaternion_error = target.getTransform().quaternion().multiply(ee.getTransform().quaternion().inverse())
    orientation_error = dart.math.logMap(quaternion_error.rotation()) # angle-axis x, y, z

    return orientation_error

def get_rot_distance(self):
    distance = np.linalg.norm(self.get_rot_error())

    return distance

def get_ee_pos(self):
    """
        return the position of the ee in dart cords

        :return: x, y, z position of the ee in dart coords system
    """
    x, y, z = self.dart_sim.chain.getBodyNode('iiwa_link_ee').getTransform().translation()

    return x, y, z

def get_ee_orient(self):
    """
        return the orientation of the ee in angle-axis

        :return: rx, ry, rz orientation of the ee in angle-axis in dart coords system
    """
    rot_mat = self.dart_sim.chain.getBodyNode('iiwa_link_ee').getTransform().rotation()
    rx, ry, rz = dart.math.logMap(rot_mat)

    return rx, ry, rz

def _random_target_gen_joint_level(self):
    """
        generate a rechable target in task-space for the manipulator

        :return: rx, ry, rz, x, y, z pose the ee in dart coords system (angle-axis)
    """

    # backing up joint positions of the arm in radians
    positions = self.dart_sim.chain.getPositions()

    # If your gym observation state is different than the default, #
    # please adapt the self.observation_indices['joint_pos']       #
    joint_indices_start = self.observation_indices['joint_pos']

    while True:
        # generating a random valid observation including joint positions
        random_state = self.observation_space.sample()

        # selecting the randomly generated joint positions from the agent state                               #
        # Note: the agent state containts additional information other than the joints positions of the robot #
        initial_positions = random_state[joint_indices_start:joint_indices_start + self.n_links]

        # computing the forward kinematics to retrieve end-effector pose
        self.dart_sim.chain.setPositions(initial_positions)
        ee = self.dart_sim.chain.getBodyNode('iiwa_link_ee')
        rx, ry, rz = dart.math.logMap(ee.getTransform().rotation())  # SO(3) to so(3): Angle * Axis of Rotation
        x, y, z = ee.getTransform().translation()

        # conditions for the generated targets - adapt it for your task
        if z > 0.1:
            break

    # reverting the joint positions back to the original ones
    self.dart_sim.chain.setPositions(positions)

    # resetting the orientation vector if it is not controlled
    if not self.dart_sim.orientation_control:
        rx, ry, rz = 0.0, 0.0, 0.0

    # initial_positions vector is a valid IK solution to the [rx, ry, rz, x, y, z]

    return rx, ry, rz, x, y, z
