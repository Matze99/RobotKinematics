from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union, List, Tuple
from enum import Enum

from sympy import Symbol, Expr, symbols, Float

from math_utils import Matrix, Vector


class JointType(Enum):
    BASE = 0
    ROTATIONAL = 1
    PRISMATIC = 2
    END_EFFECTOR = 3


class Robot:
    '''
    Robot base class

    :param joints: List of joints of robot
    '''

    def __init__(self, joints: List[Joint]):
        self.joints = joints

    def transform_to_zero_from_i(self, vector: Vector, index: int) -> Vector:
        '''
        transforms the vector from frame i to frame zero
        :param vector: vector that should be transformed
        :param index: frame of vector
        :return: transformed vector
        '''
        assert len(vector) == 4
        if index == 0:
            return vector
        rot = Matrix(None, False, True)
        for joint in self.joints[1:index+1]:
            rot = rot @ joint.get_transformation()
        return rot @ vector

    def rotate_to_zero_from_i(self, vector: Vector, index: int) -> Vector:
        '''
        rotates the vector from frame i to frame zero
        :param vector: vector that should be rotated
        :param index: frame of vector
        :return: rotated vector
        '''
        if index == 0:
            return vector
        assert len(vector) == 3
        rot = Matrix(None, True, True)
        for joint in self.joints[1:index+1]:
            rot = rot @ joint.get_rotation()
        return rot @ vector


    def get_transformations_from_zero(self, print_result = True) -> List[Matrix]:
        '''
        calculates the transformation matrix from start frame to each other frame

        :param print_result: prints the calculated transformation matrices
        :return: list of transformation matrices
        '''
        rot = Matrix(None, False, True)
        transformations = []
        for joint in self.joints[1:]:
            rot = rot @ joint.get_transformation()
            transformations.append(rot)
            if print_result:
                print(rot)
        return transformations

    def get_zero_Z_axis(self, print_result = True) -> List[Vector]:
        '''
        calculates Z axis of each frame and transforms to zero frame

        :param print_result: prints the calculated axis
        :return: list of Z axis in zero frame
        '''

        rot = Matrix(None, True, True)
        z_axis = []
        for joint in self.joints[1:]:
            rot = rot @ joint.get_rotation()
            z_axis.append((rot @ Vector([0, 0, 1])))
            if print_result:
                print(z_axis[-1])
        return z_axis

    @staticmethod
    def joint_types_from_str(string: str) -> List[JointType]:
        '''
        converts a string into a list of jointtypes

        'rrp' -> [JointType.ROTATIONAL, JointType.ROTATIONAL, JointType. PRISMATIC]

        '2rp' -> [JointType.ROTATIONAL, JointType.ROTATIONAL, JointType. PRISMATIC]

        '2RP' -> [JointType.ROTATIONAL, JointType.ROTATIONAL, JointType. PRISMATIC]

        :param string: string definition of robot joint setup
        :return: list of JointTypes
        '''
        skip = False
        joint_types = []
        for i in range(len(string)):
            if skip:
                skip = False
                continue
            else:
                letter = string[i]
                try:
                    skips = int(letter)
                    if i+1 >= len(string):
                        raise ValueError(f'{string} is invalid joint configuration. A letter has to follow after a number.')
                    skip_type = Robot.joint_type_from_char(string[i+1])
                    for i in range(skips):
                        joint_types.append(skip_type)
                    skip = True
                except:
                    joint_types.append(Robot.joint_type_from_char(letter))

        return joint_types

    @staticmethod
    def joint_type_from_char(char: str) -> JointType:
        assert len(char) == 1
        if char == 'r' or char == 'R':
            return JointType.ROTATIONAL
        elif char == 'p' or char == 'P':
            return JointType.PRISMATIC
        else:
            raise ValueError(f'{char} is not a valid joint type')

    @staticmethod
    def from_dh_parameters(dh: List[List[Union[Symbol, Expr, float, int]]],
                           joint_types: Union[List[JointType], JointType, str],
                           control_variables: Union[List[Symbol], None] = None) -> Robot:
        '''
        creates a robot and all its joints from dh parameters

        if there is no transformation from the last link frame to the end effector add a row of zeros
        to the dh parameters for the end effector

        :param dh: dh parameters, each row is one joint
        :param joint_types: types of joints, if single type all joints are of that type
                similar to convention it is possible to just write RRP for a 2 rotation, 1 prismatic joint robot
                see the documentation of _joint_types_from_string for more details
        :param control_variables: sympy symbols of controllable variables of corresponding joint
        :return: robot with joints defined by JointType and dh
        '''
        if isinstance(joint_types, str):
            joint_types = Robot.joint_types_from_str(joint_types)

        assert all(len(joint_param) == 4 for joint_param in dh), "each joint needs four dh parameters"

        joints = [BaseJoint(id=0)]
        for i, joint_param in enumerate(dh):
            symbol = None if control_variables is None else control_variables[i]
            if i+1 == len(dh):
                joints.append(
                    Joint.create_joint(*joint_param, joint_type=JointType.END_EFFECTOR, symbol=symbol, id=i + 1))
            elif isinstance(joint_types, JointType):
                joints.append(Joint.create_joint(*joint_param, joint_types, symbol, i + 1))
            else:
                joints.append(Joint.create_joint(*joint_param, joint_types[i], symbol, i + 1))

        return Robot(joints)

    def get_transformations(self):
        '''
        prints all the transformation matrices of the robot
        :return:
        '''
        for joint in self.joints[1:]:
            print(joint.get_transformation())

    def get_rotations(self):
        '''
        prints all the transformation matrices of the robot
        :return:
        '''
        for joint in self.joints[1:]:
            print(joint.get_rotation())

    def get_end_rotation(self) -> Matrix:
        '''
        calculates the rotation from base frame to end_effector frame
        :return: base to end rotation matrix
        '''
        mat = self.joints[0].get_rotation()
        for joint in self.joints[1:]:
            mat = mat @ joint.get_rotation()
        return mat

    def get_end_transformation(self) -> Matrix:
        '''
        calculates the transformation from base to end_effector frame
        :return: transformation matrix from base to end
        '''
        mat = self.joints[0].get_transformation()
        for joint in self.joints[1:]:
            mat = mat @ joint.get_transformation()
        return mat

    def cal_velocities(self, print_result: bool = False) -> Tuple[List[Vector], List[Vector]]:
        '''
        calculates the angular and linear velocities of the joints
        see joints for details on the calculation

        :param print_result: True results are also printed
        :return: angular velocities, linear velocities of the joints
        '''
        ws, vs = [], []
        w, v = self.joints[0].cal_velocities(None, None)
        ws.append(w.simplify())
        vs.append(v.simplify())
        for i, joint in enumerate(self.joints[1:]):
            w, v = joint.cal_velocities(w, v)
            ws.append(w.simplify())
            vs.append(v.simplify())

            if print_result:
                print(f'joint {i + 1}:\n - w: {ws[-1]}\n - v: {vs[-1]}')

        return ws, vs

    def cal_static_kinematics(self, print_result: bool = False, simplify: bool = False) -> Tuple[
        List[Vector], List[Vector], List[Vector], List[Vector], List[Union[Expr, Symbol, Float, float]]]:
        '''
        calculates the static forward and backwards kinematics equations

        :param print_result: True results are being printed
        :param simplify: simplify each intermediate result
        :return: angular velocities, linear velocities, forces on the link, torques on the link, torques of joints
        '''
        ws, vs = self.cal_velocities(print_result)
        fs, ns, ts = [], [], []

        f, n = self.joints[-1].cal_inertial_force_torque_static(None, None)
        if simplify:
            fs.append(f.simplify())
            ns.append(n.simplify())
        else:
            fs.append(f)
            ns.append(n)
        if print_result:
            print('\nbackwards:\n')
            print(f'joint {len(self.joints) - 2}:\n - f: {f}\n - n: {n}')
            if simplify:
                print(f'simplified {len(self.joints) - 2}:\n - f: {fs[-1]}\n - n: {ns[-1]}')

        for i in range(len(self.joints) - 3):
            f, n = self.joints[-i - 2].cal_inertial_force_torque_static(fs[-1], ns[-1])
            if simplify:
                fs.append(f.simplify())
                ns.append(n.simplify())
            else:
                fs.append(f)
                ns.append(n)
            t = self.joints[-i - 2].cal_torque_static(fs[-1], ns[-1])
            if simplify:
                ts.append(t.simplify())
            else:
                ts.append(t)

            if print_result:
                print(f'joint {len(self.joints) - i - 3}:\n - f: {f}\n - n: {n}\n - tau: {t}')
                if simplify:
                    print(f'simplified {len(self.joints) - i - 3}:\n - f: {fs[-1]}\n - n: {ns[-1]}\n - tau: {ts[-1]}')

        return ws, vs, fs, ns, ts

    def add_cof(self, cof_points: List[Vector]) -> None:
        '''
        adds center points of mass for each link
        :param cof_points: per prismatic, rotational joint a vector with the coordinates of the center of mass
        :return:
        '''
        for i, joint in enumerate(self.joints[1:-1]):
            joint.cof = cof_points[i]

    def cal_newton_euler(self, print_result: bool = False, simplify: bool = False,
                         gravity: Vector = Vector([0, 0, 0])) -> \
            Tuple[
                List[Vector], List[Vector], List[Vector], List[Vector], List[Vector], List[Vector], List[Vector], List[
                    Vector],
                List[Vector]]:
        '''
        calculates the newton euler dynamics

        for details of the calculation see the joint methods:
        - `cal_velocity_linear_rotational_acceleration_dynamic'
        - `cal_linear_acceleration_cof`
        - `cal_force_torque_cof`
        - `cal_force_torque_joint_dynamic`

        :param print_result: print results
        :param simplify: simplify the results
        :param gravity: gravity vector
        :return: angular velocities, angular accelerations, linear acceleration, linear acceleration of the center of mass,
                fores on the center of mass, torques on the center of mass, torques that apply to the joints
        '''
        ws, dws, dvs, dvcs, Fs, Ns = [], [], [], [], [], []
        if print_result:
            print('----------------Forwards Newton Euler----------------\n\n')
        w, dw, dv = self.joints[0].cal_velocity_linear_rotational_acceleration_dynamic(gravity, None, None)
        ws.append(w.simplify())
        dws.append(dw.simplify())
        dvs.append(dv.simplify())
        for i, joint in enumerate(self.joints[1:]):
            w, dw, dv = joint.cal_velocity_linear_rotational_acceleration_dynamic(ws[-1], dws[-1], dvs[-1])
            ws.append(w.simplify())
            dws.append(dw.simplify())
            dvs.append(dv.simplify())

            if print_result:
                print(f'\njoint {i + 1}:\n - w: {w}\n - dw: {dw}\n - dv: {dv}')

                if simplify:
                    print(f'simplified {i + 1}:\n - w: {ws[-1]}\n - dw: {dws[-1]}\n - dv: {dvs[-1]}')

            if i < len(self.joints) - 2:
                dvc = joint.cal_linear_acceleration_cof(ws[-1], dws[-1], dvs[-1])
                dvcs.append(dvc.simplify())
                F, N = joint.cal_force_torque_cof(ws[-1], dws[-1], dvcs[-1])
                Fs.append(F.simplify())
                Ns.append(N.simplify())

                if print_result:
                    print(f'joint {i + 1}:\n - dvc: {dvc}\n - F: {F}\n - N: {N}')

                    if simplify:
                        print(f'simplified {i + 1}:\n - dvc: {dvcs[-1]}\n - F: {Fs[-1]}\n - N: {Ns[-1]}')

        if print_result:
            print('\n\n\n----------------Backwards Newton Euler----------------\n\n')

        prev_rot = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], True)
        prev_p = Vector([0, 0, 0])
        f = Vector([0, 0, 0])
        n = Vector([0, 0, 0])

        fs, ns, ts = [], [], []


        for i in range(len(self.joints) - 2):
            f, n, prev_rot, prev_p = self.joints[-i-2].cal_force_torque_joint_dynamic(f, Fs[-i-1], n, Ns[-i-1], prev_rot, prev_p)
            fs.append(f.simplify())
            ns.append(n.simplify())

            if print_result:
                print(f'joint {len(self.joints) - i - 2}:\n - f: {f}\n - n: {n}\n - tau: {1}')
                if simplify:
                    print(f'simplified {len(self.joints) - i - 2}:\n - f: {fs[-1]}\n - n: {ns[-1]}\n - tau: {1}')

            f, n = fs[-1], ns[-1]

            #ToDO calculate torques for joints

        return ws, dws, dvs, dvcs, Fs, Ns, fs, ns, ts

    def __str__(self) -> str:
        string = 'Robot[\n'
        for joint in self.joints:
            string += ' - '+joint.__str__()+'\n'
        string += ']'
        return string

class Joint(ABC):
    '''
    Joint base class

    :param type: type of joint (BaseJoint, RotationalJoint, PrismaticJoint, EndEffector)
    :param a: distance from :math:`Z_{i-1}` to :math:`Z_{i}` along :math:`X_{i-1}`
    :param alpha: twist, angle from :math:`Z_{i-1}` to :math:`Z_{i}` around :math:`X_{i-1}`
    :param d: distance from :math:`X_{i-1}` to :math:`X_{i}` along :math:`Z_{i}`
    :param theta: angle from :math:`X_{i-1}` to :math:`X_{i}` around :math:`Z_{i}`
    :param control_variable: sympy variable that describes the control variable of this joint
    :param id: id of joint
    '''

    def __init__(self, type: JointType = JointType.BASE, a: Union[Expr, Symbol, float] = 0.,
                 alpha: Union[Expr, Symbol, float] = 0., d: Union[Expr, Symbol, float] = 0,
                 theta: Union[Expr, Symbol, float] = 0, control_variable: Union[Symbol, None] = None,
                 id: int = -1):
        self._type: JointType = type
        self._a: Union[Expr, Symbol, float] = a
        self._alpha: Union[Expr, Symbol, float] = alpha
        self._d: Union[Expr, Symbol, float] = d
        self._theta: Union[Expr, Symbol, float] = theta
        self._control_variable: Union[Symbol, None] = control_variable
        self._id: int = id
        self._cof: Vector = Vector([0, 0, 0])
        self._cof_set: bool = False

    def __str__(self) -> str:
        return f'{self._joint_type_to_string()}\t [a: {self.a}, alpha: {self.alpha}, d: {self.d}, theta: {self.theta}]'

    def _joint_type_to_string(self) -> str:
        if self._type == JointType.BASE:
            return 'BASE joint'
        elif self._type == JointType.PRISMATIC:
            return 'PRISMATIC joint'
        elif self._type == JointType.ROTATIONAL:
            return 'ROTATIONAL joint'
        else:
            return 'END_EFFECTOR'

    @property
    def cof(self):
        return self._cof

    @cof.setter
    def cof(self, cof: Vector):
        self._cof = cof
        self._cof_set = True

    @abstractmethod
    def is_prismatic(self) -> bool:
        # return self._type == JointType.PRISMATIC
        pass

    @property
    def a(self) -> Union[Expr, Symbol, float]:
        return self._a

    @property
    def alpha(self) -> Union[Expr, Symbol, float]:
        return self._alpha

    @property
    def d(self) -> Union[Expr, Symbol, float]:
        return self._d

    @property
    def theta(self) -> Union[Expr, Symbol, float]:
        return self._theta

    def get_transformation(self) -> Matrix:
        '''
        calculates the transformation matrix of the current joint
        :return: transformation matrix
        '''
        return Matrix.transformation_from_joint(self)

    def get_rotation(self) -> Matrix:
        '''
        calculates the rotation matrix of the current joint
        :return: rotation matrix
        '''
        return Matrix.rotation_from_joint(self)

    @staticmethod
    def create_joint(a: Union[Expr, Symbol, float] = 0., alpha: Union[Expr, Symbol, float] = 0.,
                     d: Union[Expr, Symbol, float] = 0, theta: Union[Expr, Symbol, float] = 0,
                     joint_type: JointType = JointType.BASE, symbol: Union[Symbol, None] = None,
                     id: int = -1) -> Joint:
        '''
        creates a joint from dh parameters

        :param joint_type: type of joint (BaseJoint, RotationalJoint, PrismaticJoint, EndEffector)
        :param a: distance from :math:`Z_{i-1}` to :math:`Z_{i}` along :math:`X_{i-1}`
        :param alpha: twist, angle from :math:`Z_{i-1}` to :math:`Z_{i}` around :math:`X_{i-1}`
        :param d: distance from :math:`X_{i-1}` to :math:`X_{i}` along :math:`Z_{i}`
        :param theta: angle from :math:`X_{i-1}` to :math:`X_{i}` around :math:`Z_{i}`
        :param symbol: sympy control symbol
        :param id: id of joint
        :return: Joint with given parameters
        '''
        if joint_type == JointType.BASE:
            return BaseJoint(a, alpha, d, theta, None, 0)
        elif joint_type == JointType.ROTATIONAL:
            return RotationalJoint(a, alpha, d, theta, symbol, id)
        elif joint_type == JointType.PRISMATIC:
            return PrismaticJoint(a, alpha, d, theta, symbol, id)
        else:
            return EndEffector(a, alpha, d, theta, None, id)

    @abstractmethod
    def cal_velocities(self, prev_w: Vector, prev_v: Vector) -> Tuple[Vector, Vector]:
        pass

    @abstractmethod
    def cal_inertial_force_torque_static(self, prev_f: Union[Vector, None], prev_n: Union[Vector, None]) -> Tuple[
        Vector, Vector]:
        pass

    @abstractmethod
    def cal_torque_static(self, f: Union[Vector, None], n: Union[Vector, None]) -> Union[Expr, Symbol, Float, float]:
        pass

    @abstractmethod
    def cal_velocity_linear_rotational_acceleration_dynamic(self, prev_w: Union[Vector, None],
                                                            prev_dw: Union[Vector, None],
                                                            prev_dv: Union[Vector, None]) -> Tuple[
        Vector, Vector, Vector]:
        pass

    def cal_linear_acceleration_cof(self, w: Union[Vector, None], dw: Union[Vector, None],
                                    dv: Union[Vector, None]) -> Vector:
        '''
        calculates the linear acceleration of the center of mass

        the point of the center of mass :math:`P_{C_i}` has to be set before calling this function

        .. math::

            \dot{v}_{C_i} = \dot{w}_i x P_{C_i} + w_i x (w_i x P_{C_i}) + \dot{v}_i

        :param w: angular velocity of link
        :param dw: angular acceleration of link
        :param dv: linear acceleration of link
        :return: linear acceleration of center of mass
        '''
        if not self._cof_set:
            raise ValueError(f'center of mass not set for joint {self._id}')
        return dw * self._cof + w * (w * self._cof) + dv

    def cal_force_torque_cof(self, w: Union[Vector, None], dw: Union[Vector, None], dvc: Union[Vector, None]) -> Tuple[
        Vector, Vector]:
        '''
        calculates the forces and torques of the center of mass

        .. math::

            &F_i = m \dot{v}_{C_i}

            &N_i = I_{C_i} \dot{w}_i + w_i x (I_{C_i} w_i)

        The inertia tensor :math:`I_{C_i}` is set to a diagonal matrix with symbols :math:`I_xxi`,
        :math:`I_yyi`, :math:`I_zzi` as elements
        If you have specific values for the entries of the inertia tensor use the subs function

        :param w: angular velocity of link
        :param dw: angular acceleration of link
        :param dvc: linear acceleration of center of mass
        :return: force of center of mass, torque of center of mass
        '''
        if not self._cof_set:
            raise ValueError(f'center of mass not set for joint {self._id}')
        i_str = self._id.__str__()
        i_str = 'I_xx' + i_str + ' I_yy' + i_str + ' I_zz' + i_str
        I = Matrix.diagonal(symbols(i_str))
        F = dvc * symbols('m' + self._id.__str__())
        N = I @ dw + w * (I @ w)

        return F, N

    def cal_force_torque_joint_dynamic(self, prev_f: Vector, F: Vector, prev_n: Vector, N: Vector, pre_rot: Matrix,
                                       pre_p: Vector) -> Tuple[Vector, Vector, Matrix, Vector]:
        '''
        calculates the dynamic forces and torques acting on the link

        .. math::

            &f_i = R^i_{i+1} f_{i+1} + F_i

            &n_i = R^i_{i+1} n_{i+1} + N_i + P_{C_i} x F_i + P^i_{i+1} x (R^i_{i+1} f_{i+1})

        :param prev_f: force on previous link (this has to be applied backwards, e.g. n, n-1, n-2, ..., 1)
        :param F: force on center of mass of link
        :param prev_n: torque on previous link
        :param N: torque on center of mass of link
        :param pre_rot: rotation matrix of previous to current frame
        :param pre_p: origin of previous frame in current frame
        :return: force on joint, torque on joint, current rotation matrix, current origin in prev frame
        '''
        trans = self.get_transformation()
        rot = trans.rot

        p = (trans @ Vector([0, 0, 0, 1])).remove_last()

        f = pre_rot @ prev_f + F
        n = pre_rot @ prev_n + N + self._cof * F + pre_p * (pre_rot @ prev_f)
        return f,n, rot, p


class BaseJoint(Joint):
    '''
    BaseJoint base class to represent the world coordinate frame

    :param a: distance from :math:`Z_{i-1}` to :math:`Z_{i}` along :math:`X_{i-1}`
    :param alpha: twist, angle from :math:`Z_{i-1}` to :math:`Z_{i}` around :math:`X_{i-1}`
    :param d: distance from :math:`X_{i-1}` to :math:`X_{i}` along :math:`Z_{i}`
    :param theta: angle from :math:`X_{i-1}` to :math:`X_{i}` around :math:`Z_{i}`
    :param c: not use
    :param id: id of joint
    '''
    def __init__(self, a: Union[Expr, Symbol, float] = 0.,
                 alpha: Union[Expr, Symbol, float] = 0.,
                 d: Union[Expr, Symbol, float] = 0, theta: Union[Expr, Symbol, float] = 0, c: None = None,
                 id: int = -1):
        super().__init__(JointType.BASE, a, alpha, d, theta, None, id)

    def is_prismatic(self) -> bool:
        return False

    def cal_velocities(self, prev_w: Vector = None, prev_v: Vector = None) -> Tuple[Vector, Vector]:
        return Vector([0, 0, 0]), Vector([0, 0, 0])

    def cal_inertial_force_torque_static(self, prev_f: Union[Vector, None], prev_n: Union[Vector, None]) -> Tuple[
        Vector, Vector]:
        return prev_f, prev_n

    def cal_torque_static(self, f: Union[Vector, None], n: Union[Vector, None]) -> Union[Expr, Symbol, Float, float]:
        return 0.

    def cal_velocity_linear_rotational_acceleration_dynamic(self, prev_w: Union[Vector, None],
                                                            prev_dw: Union[Vector, None],
                                                            prev_dv: Union[Vector, None]) -> Tuple[
        Vector, Vector, Vector]:
        return Vector([0, 0, 0]), Vector([0, 0, 0]), prev_w

    def cal_linear_acceleration_cof(self, w: Union[Vector, None], dw: Union[Vector, None],
                                    dv: Union[Vector, None]) -> Vector:
        raise ValueError("Base Joint does not have linear acceleration")


class PrismaticJoint(Joint):
    '''
    PrismaticJoint base class to represent a prismatic joint

    :param a: distance from :math:`Z_{i-1}` to :math:`Z_{i}` along :math:`X_{i-1}`
    :param alpha: twist, angle from :math:`Z_{i-1}` to :math:`Z_{i}` around :math:`X_{i-1}`
    :param d: distance from :math:`X_{i-1}` to :math:`X_{i}` along :math:`Z_{i}`
    :param theta: angle from :math:`X_{i-1}` to :math:`X_{i}` around :math:`Z_{i}`
    :param control_variable: control variable of joint
    :param id: id of joint
    '''

    def __init__(self, a: Union[Expr, Symbol, float] = 0.,
                 alpha: Union[Expr, Symbol, float] = 0.,
                 d: Union[Expr, Symbol, float] = 0, theta: Union[Expr, Symbol, float] = 0,
                 control_variable: Union[Symbol, None] = None,
                 id: int = -1):
        super().__init__(JointType.PRISMATIC, a, alpha, d, theta, control_variable, id)

    def is_prismatic(self) -> bool:
        return True

    def cal_velocities(self, prev_w: Vector, prev_v: Vector) -> Tuple[Vector, Vector]:
        '''
        calculates the rotational and linear velocity of the joint in a static situation

        .. math::

            &w_{i+1} = R^{i+1}_i w_i

            &v_{i+1} = R^{i+1}_i ((w_i x P^i_{i+1}) + \dot(v)_i)
                        + \dot(d_{i+1}) \hat{Z}_{i+1}

        :param prev_w: angular velocity of previous joint
        :param prev_v: linear velocity of previous joint
        :return: angular, linear velocity
        '''
        trans = self.get_transformation()
        rot = trans.rot.T

        p = (trans @ Vector([0, 0, 0, 1])).remove_last()

        w = rot @ prev_w
        v = rot @ (prev_v + prev_w * p) + Vector([0, 0, symbols('dot_d' + self._id.__str__())])

        return w, v

    def cal_inertial_force_torque_static(self, prev_f: Union[Vector, None], prev_n: Union[Vector, None]) -> Tuple[
        Vector, Vector]:
        '''
        calculates the inertial force and torque on the link in the static case

        .. math::

            &f_i = R^i_{i+1} f_{i+1}

            &n_i = R^i_{i+1} n_{i+1} + P^i_{i+1} x f_i

        :param prev_f: force on previous link
        :param prev_n: torque on previous link
        :return: force, torque on current link
        '''
        trans = self.get_transformation()
        rot = trans.rot

        p = (trans @ Vector([0, 0, 0, 1])).remove_last()

        f = rot @ prev_f
        n = rot @ prev_n + p * f
        return f, n

    def cal_torque_static(self, f: Union[Vector, None], n: Union[Vector, None]) -> Union[Expr, Symbol, Float, float]:
        '''
        calculates torque that apply to the joint

        In this case, the last component of the force vector

        :param f: force acting on the link
        :param n: torque acting on the link
        :return: torque that apply to the joint
        '''
        return f @ Vector([0, 0, 1])

    def cal_velocity_linear_rotational_acceleration_dynamic(self, prev_w: Union[Vector, None],
                                                            prev_dw: Union[Vector, None],
                                                            prev_dv: Union[Vector, None]) -> Tuple[
        Vector, Vector, Vector]:
        '''
        calculates the rotational velocity and acceleration, and the linear acceleration in the dynamic case

        .. math::
            &w_{i+1} = R^{i+1}_i w_i

            &\dot{w}_{i+1} = R^{i+1}_i \dot{w}_i

            &\dot{v} = R^{i+1}_i (\dot{w}_i x P^i_{i+1} + w_i x (w_i x P^i_{i+1}) + \dot(v)_i)
                        + 2 \dot{w}_{i+1} x \dot(d_{i+1}) \hat{Z}_{i+1} + \ddot(d_{i+1}) \hat{Z}_{i+1}


        :param prev_w: angular velocity of previous link
        :param prev_dw: angular acceleration of previous link
        :param prev_dv: linear acceleration of previous link
        :return: angular velocity, angular acceleration, and linear acceleration of current joint
        '''
        trans = self.get_transformation()
        rot = trans.rot.T

        p = (trans @ Vector([0, 0, 0, 1])).remove_last()

        w = rot @ prev_w
        dw = rot @ prev_dw

        dv = rot @ (prev_dw * p + prev_w * (prev_w * p) + prev_dv) + (
                w * Vector([0, 0, symbols('dot_d' + self._id.__str__())])) * 2 + Vector(
            [0, 0, symbols('ddot_d' + self._id.__str__())])

        return w, dw, dv


class RotationalJoint(Joint):
    '''
    RotationalJoint base class to represent a prismatic joint

    :param a: distance from :math:`Z_{i-1}` to :math:`Z_{i}` along :math:`X_{i-1}`
    :param alpha: twist, angle from :math:`Z_{i-1}` to :math:`Z_{i}` around :math:`X_{i-1}`
    :param d: distance from :math:`X_{i-1}` to :math:`X_{i}` along :math:`Z_{i}`
    :param theta: angle from :math:`X_{i-1}` to :math:`X_{i}` around :math:`Z_{i}`
    :param control_variable: control variable of joint
    :param id: id of joint
    '''

    def __init__(self, a: Union[Expr, Symbol, float] = 0.,
                 alpha: Union[Expr, Symbol, float] = 0.,
                 d: Union[Expr, Symbol, float] = 0, theta: Union[Expr, Symbol, float] = 0,
                 control_variable: Union[Symbol, None] = None,
                 id: int = -1):
        super().__init__(JointType.ROTATIONAL, a, alpha, d, theta, control_variable, id)

    def is_prismatic(self) -> bool:
        return False

    def cal_velocities(self, prev_w: Vector, prev_v: Vector) -> Tuple[Vector, Vector]:
        '''
        calculates the rotational and linear velocity of the joint in a static situation

        .. math::

            &w_{i+1} = R^{i+1}_i w_i + \dot{theta}_{i+1} \hat{Z}_{i+1}

            &v_{i+1} = R^{i+1}_i ((w_i x P^i_{i+1}) + \dot(v)_i)

        :param prev_w: angular velocity of previous joint
        :param prev_v: linear velocity of previous joint
        :return: angular, linear velocity
        '''
        trans = self.get_transformation()
        rot = trans.rot.T

        p = (trans @ Vector([0, 0, 0, 1])).remove_last()

        w = rot @ prev_w + Vector([0, 0, symbols('dot_t' + self._id.__str__())])
        v = rot @ (prev_v + prev_w * p)

        return w, v

    def cal_inertial_force_torque_static(self, prev_f: Union[Vector, None], prev_n: Union[Vector, None]) -> Tuple[
        Vector, Vector]:
        '''
        calculates the inertial force and torque on the link in the static case

        .. math::

            &f_i = R^i_{i+1} f_{i+1}

            &n_i = R^i_{i+1} n_{i+1} + P^i_{i+1} x f_i

        :param prev_f: force on previous link
        :param prev_n: torque on previous link
        :return: force, torque on current link
        '''
        trans = self.get_transformation()
        rot = trans.rot

        p = (trans @ Vector([0, 0, 0, 1])).remove_last()

        f = rot @ prev_f
        n = rot @ prev_n + p * f
        return f, n

    def cal_torque_static(self, f: Union[Vector, None], n: Union[Vector, None]) -> Union[Expr, Symbol, Float, float]:
        '''
        calculates torque that apply to the joint

        In this case, the last component of the torque vector

        :param f: force acting on the link
        :param n: torque acting on the link
        :return: torque that apply to the joint
        '''
        return n @ Vector([0, 0, 1])

    def cal_velocity_linear_rotational_acceleration_dynamic(self, prev_w: Union[Vector, None],
                                                            prev_dw: Union[Vector, None],
                                                            prev_dv: Union[Vector, None]) -> Tuple[
        Vector, Vector, Vector]:
        '''
        calculates the rotational velocity and acceleration, and the linear acceleration in the dynamic case

        .. math::
            &w_{i+1} = R^{i+1}_i w_i + \dot{\theta}_{i+1} \hat{Z}_{i+1}

            &\dot{w}_{i+1} = R^{i+1}_i \dot{w}_i + R^{i+1}_i w_i x \dot{theta}_{i+1} \hat{Z}_{i+1} +
                        \ddot{theta}_{i+1} \hat{Z}_{i+1}

            &\dot{v} = R^{i+1}_i (\dot{w}_i x P^i_{i+1} + w_i x (w_i x P^i_{i+1}) + \dot{v}_i)


        :param prev_w: angular velocity of previous link
        :param prev_dw: angular acceleration of previous link
        :param prev_dv: linear acceleration of previous link
        :return: angular velocity, angular acceleration, and linear acceleration of current joint
        '''
        trans = self.get_transformation()
        rot = trans.rot.T

        p = (trans @ Vector([0, 0, 0, 1])).remove_last()

        dt_i = symbols('dot_t' + self._id.__str__())
        w = rot @ prev_w + Vector([0, 0, dt_i])
        dw = rot @ prev_dw + (rot @ prev_w) * Vector([0, 0, dt_i]) + Vector(
            [0, 0, symbols('ddot_t' + self._id.__str__())])

        dv = rot @ (prev_dw * p + prev_w * (prev_w * p) + prev_dv)

        return w, dw, dv


class EndEffector(Joint):
    '''
    EndEffector class to represent the frame of the end effector

    :param a: distance from :math:`Z_{i-1}` to :math:`Z_{i}` along :math:`X_{i-1}`
    :param alpha: twist, angle from :math:`Z_{i-1}` to :math:`Z_{i}` around :math:`X_{i-1}`
    :param d: distance from :math:`X_{i-1}` to :math:`X_{i}` along :math:`Z_{i}`
    :param theta: angle from :math:`X_{i-1}` to :math:`X_{i}` around :math:`Z_{i}`
    :param c: not use
    :param id: id of joint
    '''

    def __init__(self, a: Union[Expr, Symbol, float] = 0.,
                 alpha: Union[Expr, Symbol, float] = 0.,
                 d: Union[Expr, Symbol, float] = 0, theta: Union[Expr, Symbol, float] = 0, c: None = None,
                 id: int = -1):
        super().__init__(JointType.END_EFFECTOR, a, alpha, d, theta, None, id)

    def is_prismatic(self) -> bool:
        return False

    def cal_velocities(self, prev_w: Vector, prev_v: Vector) -> Tuple[Vector, Vector]:
        trans = self.get_transformation()
        rot = trans.rot.T

        p = (trans @ Vector([0, 0, 0, 1])).remove_last()

        w = rot @ prev_w
        v = rot @ (prev_v + prev_w * p)

        return w, v

    def cal_inertial_force_torque_static(self, prev_f: Union[Vector, None], prev_n: Union[Vector, None]) -> Tuple[
        Vector, Vector]:
        f1, f2, f3, n1, n2, n3 = symbols('F1, F2, F3, N1, N2, N3')
        f, n = Vector([f1, f2, f3]), Vector([n1, n2, n3])

        trans = self.get_transformation()
        p = (trans @ Vector([0, 0, 0, 1])).remove_last()

        n = n + p * f
        return f, n

    def cal_torque_static(self, f: Union[Vector, None], n: Union[Vector, None]) -> Union[Expr, Symbol, Float, float]:
        return 0.

    def cal_velocity_linear_rotational_acceleration_dynamic(self, prev_w: Union[Vector, None],
                                                            prev_dw: Union[Vector, None],
                                                            prev_dv: Union[Vector, None]) -> Tuple[
        Vector, Vector, Vector]:
        trans = self.get_transformation()
        rot = trans.rot.T

        p = (trans @ Vector([0, 0, 0, 1])).remove_last()

        return prev_w, prev_dw, rot @ (prev_dw * p + prev_w * (prev_w * p) + prev_dv)

    def cal_linear_acceleration_cof(self, w: Union[Vector, None], dw: Union[Vector, None],
                                    dv: Union[Vector, None]) -> Vector:
        raise ValueError("EndEffector does not have linear acceleration")
