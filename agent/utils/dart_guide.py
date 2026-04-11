"""
Help code for:
              1. how to retrieve transforamtion matrices of chains of rigidbodies in target (dart)

              2. how to generate reachable targets for the end-effector pose

              3. legacy coordinate transformations

DART uses the Eigen library - Geometry module documentation: https://eigen.tuxfamily.org/dox/group__Geometry__Module.html

body_node.getTransform().translation() --> Cartesian position (x,y,z) of the center of that body_node
body_node.getTransform().rotation()    --> Orientation of the body_node in a 3*3 rotation matrix
body_node.getTransform().quaternion()  --> Orientation of the body_node in a unit quaternion form (w,x,y,z)
dart.math.logMap()                     --> calculates an angle-axis representation of a rotation matrix

background details:   https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation
further code details: https://github.com/dartsim/dart/blob/main/dart/math/Geometry.cpp

Convert functions for legacy simulator versions:
Note: In legacy versions, coordinates in the Unity simulator were different from the ones in DART.
The mapping was [X, Y, Z] of Unity was [-y, z, x] of DART.
    _convert_vector_dart_to_unity(vector) -- transforms a [x, y, z] position vector
    _convert_vector_unity_to_dart(vector)
    _convert_rotation_dart_to_unity(matrix) -- transforms a 3*3 rotation matrix
    _convert_rotation_unity_to_dart(matrix)
    _convert_quaternion_dart_to_unity(quaternion) -- transforms a [w, x, y, z] quaternion vector
    _convert_quaternion_unity_to_dart(quaternion)
    _convert_angle_axis_dart_to_unity(vector) -- transforms a [rx, ry, rz] logmap representation of an Angle-Axis
    _convert_angle_axis_unity_to_dart(vector)
    _convert_pose_dart_to_unity(dart_pose, unity_in_deg=True) -- transforms a pose -- DART pose order [rx, ry, rz, x, y, z] -- Unity pose order [X, Y, Z, RX, RY, RZ]
"""

import dartpy as dart
import numpy as np

def get_pos_error(self):
    """Compute end-effector Cartesian position error in DART coordinates.

    Args:
        self: Environment/controller instance that owns ``dart_sim``.

    Returns:
        NumPy vector ``[dx, dy, dz]`` from end-effector to target.
    """
    ee = self.dart_sim.chain.getBodyNode('iiwa_link_ee')  # The end-effector rigid-body node
    target = self.dart_sim.target.getBodyNode(0)          # The target rigid-body node
    position_error = target.getTransform().translation() - ee.getTransform().translation()

    return position_error

def get_pos_distance(self):
    """Compute Euclidean distance between end-effector and target positions.

    Args:
        self: Environment/controller instance that owns ``dart_sim``.

    Returns:
        Scalar Cartesian distance in metres.
    """
    distance = np.linalg.norm(self.get_pos_error())

    return distance

def get_rot_error(self):
    """Compute orientation error between target and end-effector.

    Args:
        self: Environment/controller instance that owns ``dart_sim``.

    Returns:
        Angle-axis error vector ``[rx, ry, rz]`` from end-effector to target.
    """
    ee = self.dart_sim.chain.getBodyNode('iiwa_link_ee')  # The end-effector rigid-body node
    target = self.dart_sim.target.getBodyNode(0)          # The target rigid-body node
    quaternion_error = target.getTransform().quaternion().multiply(ee.getTransform().quaternion().inverse())
    orientation_error = dart.math.logMap(quaternion_error.rotation()) # angle-axis x, y, z

    return orientation_error

def get_rot_distance(self):
    """Compute magnitude of orientation error in angle-axis space.

    Args:
        self: Environment/controller instance that owns ``dart_sim``.

    Returns:
        Scalar rotational distance (radians in so(3) norm).
    """
    distance = np.linalg.norm(self.get_rot_error())

    return distance

def get_ee_pos(self):
    """
        Return the end-effector position in DART coordinates.

        Args:
            self: Environment/controller instance that owns ``dart_sim``.

        Returns:
            Tuple ``(x, y, z)`` position of the end-effector in DART coordinates.
    """
    x, y, z = self.dart_sim.chain.getBodyNode('iiwa_link_ee').getTransform().translation()

    return x, y, z

def get_ee_orient(self):
    """
        Return the end-effector orientation in angle-axis representation.

        Args:
            self: Environment/controller instance that owns ``dart_sim``.

        Returns:
            Tuple ``(rx, ry, rz)`` orientation in angle-axis form.
    """
    rot_mat = self.dart_sim.chain.getBodyNode('iiwa_link_ee').getTransform().rotation()
    rx, ry, rz = dart.math.logMap(rot_mat)

    return rx, ry, rz

def _random_target_gen_joint_level(self):
    """
        Generate a reachable task-space target from random valid joint states.

        Args:
            self: Environment/controller instance that owns observation and DART state.

        Returns:
            Tuple ``(rx, ry, rz, x, y, z)`` representing a reachable end-effector
            pose in DART coordinates.
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

def _convert_vector_dart_to_unity(self, vector):
    """
        transforms a vector from DART coordinate system to Unity coordinate system.
        the mapping is [X, Y, Z] of Unity is [-y, z, x] of DART.

        :return: an array of the transformed vector
    """
        
    return -vector[1], vector[2], vector[0]

def _convert_vector_unity_to_dart(self, vector):
    """
        transforms a vector from DART coordinate system to Unity coordinate system.
        the mapping is [X, Y, Z] of Unity is [-y, z, x] of DART.

        :return: an array of the transformed vector
    """
        
    return vector[2], -vector[0], vector[1]

def _convert_rotation_dart_to_unity(self, matrix):
    """
        transforms a rotation matrix from DART coordinate system to Unity coordinate system.
        the mapping is [X, Y, Z] of Unity is [-y, z, x] of DART.

        :return: a 3*3 array of the transformed rotation matrix
    """

    axis_i = matrix[:, 0]
    axis_j = matrix[:, 1]
    axis_k = matrix[:, 2]

    axis_i_prime = np.array([self._convert_vector_dart_to_unity(axis_i)])
    axis_j_prime = np.array([self._convert_vector_dart_to_unity(axis_j)])
    axis_k_prime = np.array([self._convert_vector_dart_to_unity(axis_k)])

    matrix_prime = np.zeros(matrix.shape)
    matrix_prime[:, 0] = -axis_j_prime
    matrix_prime[:, 1] = +axis_k_prime
    matrix_prime[:, 2] = +axis_i_prime
        
    return matrix_prime

def _convert_rotation_unity_to_dart(self, matrix):
    """
        transforms a rotation matrix from Unity coordinate system to DART coordinate system.
        the mapping is [X, Y, Z] of Unity is [-y, z, x] of DART.

        :return: a 3*3 array of the transformed rotation matrix
    """

    axis_i = matrix[:, 0]
    axis_j = matrix[:, 1]
    axis_k = matrix[:, 2]

    axis_i_prime = np.array([self._convert_vector_unity_to_dart(axis_i)])
    axis_j_prime = np.array([self._convert_vector_unity_to_dart(axis_j)])
    axis_k_prime = np.array([self._convert_vector_unity_to_dart(axis_k)])

    matrix_prime = np.zeros(matrix.shape)
    matrix_prime[:, 0] = +axis_k_prime
    matrix_prime[:, 1] = -axis_i_prime
    matrix_prime[:, 2] = +axis_j_prime
        
    return matrix_prime

def _convert_quaternion_dart_to_unity(self, quaternion):
    """
        transforms a quaternion vector from DART coordinate system to Unity coordinate system.
        the mapping is [X, Y, Z] of Unity is [-y, z, x] of DART.

        :return: an array of the transformed quaternion vector
    """

    rotation = dart.math.Quaternion(quaternion).rotation()
    rotation_prime = self._convert_rotation_dart_to_unity(rotation)
    quaternion_prime = dart.math.Quaternion(rotation_prime).wxyz()

    return quaternion_prime

def _convert_quaternion_unity_to_dart(self, quaternion):
    """
        transforms a quaternion vector from Unity coordinate system to DART coordinate system.
        the mapping is [X, Y, Z] of Unity is [-y, z, x] of DART.

        :return: an array of the transformed quaternion vector
    """

    rotation = dart.math.Quaternion(quaternion).rotation()
    rotation_prime = self._convert_rotation_unity_to_dart(rotation)
    quaternion_prime = dart.math.Quaternion(rotation_prime).wxyz()

    return quaternion_prime

def _convert_angle_axis_dart_to_unity(self, vector):
    """
        transforms a logmap representation of an Angle-Axis from DART coordinate system to Unity coordinate system.
        the mapping is [X, Y, Z] of Unity is [-y, z, x] of DART.
        the angle should be also negated to consider the change between a left-handed and a right-handed system

        :return: an array of the transformed logmap vector
    """

    return self._convert_vector_dart_to_unity(-np.array(vector))

def _convert_angle_axis_unity_to_dart(self, vector):
    """
        transforms a logmap representation of an Angle-Axis from Unity coordinate system to DART coordinate system.
        the mapping is [X, Y, Z] of Unity is [-y, z, x] of DART.
        the angle should be also negated to consider the change between a left-handed and a right-handed system

        :return: an array of the transformed logmap vector
    """

    return self._convert_vector_unity_to_dart(-np.array(vector))

def _convert_pose_dart_to_unity(self, dart_pose, unity_in_deg=True):
    """
        transforms a pose from DART coordinate system to Unity coordinate system.
        the mapping is [X, Y, Z] of Unity is [-y, z, x] of DART.
        DART pose starts with orientation rx, ry, rz first, which are the logmap of angle-axis representation, and finishes with position values x, y, z.
        Unity pose starts with position X, Y, Z first, and finishes with RX, RY, RZ which are the extrinsic ZXY Euler angles.

        :return: an array of the transformed pose
    """
        
    X, Y, Z = self._convert_vector_dart_to_unity(dart_pose[3:])
    RX, RY, RZ = self._convert_angle_axis_dart_to_unity(dart_pose[:3])
    if unity_in_deg:
        RX, RY, RZ = np.rad2deg([RX, RY, RZ])
    unity_pose = [X, Y, Z, RX, RY, RZ]

    return unity_pose
