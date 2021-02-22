import pytest
from typing import List, Union

from sympy import symbols, pi, Expr, Symbol, cos, sin

from robot import Robot, JointType, Joint, RotationalJoint, PrismaticJoint, EndEffector, BaseJoint
from math_utils import Vector, Matrix

t1, t2 = symbols('theta1 theta2')
l1, l2 = symbols('l1 l2')
d1, d2, d3, dot_d3 = symbols('d1 d2 d3 dot_d3')

d_t1, d_t2 = symbols('dot_t1 dot_t2')


@pytest.mark.parametrize(
    ('dh', 'desired_v', 'desired_w'),
    [
        (
                [
                    [0, 0, 0, t1],
                    [l1, pi / 2, 0, t2]
                ],
                [
                    [0, 0, 0],
                    [0, 0, -d_t1 * l1]
                ],
                [
                    [0, 0, d_t1],
                    [sin(t2) * d_t1, cos(t2) * d_t1, d_t2]
                ]
        )
    ]
)
def test_velocity_cal_rot(dh: List[List[Union[Expr, int, Symbol]]], desired_v: List[List[Union[Expr, int, Symbol]]],
                          desired_w: List[List[Union[Expr, int, Symbol]]]):
    robot = Robot.from_dh_parameters(dh, JointType.ROTATIONAL)
    desired_w = [Vector(w_i) for w_i in desired_w]
    desired_v = [Vector(v_i) for v_i in desired_v]

    w, v = robot.joints[0].cal_velocities(None, None)
    for i, joint in enumerate(robot.joints[1:]):
        w, v = joint.cal_velocities(w, v)
        assert w == desired_w[i]
        assert v == desired_v[i]


@pytest.mark.parametrize(
    ('joint', 'desired_w', 'desired_v', 'prev_w', 'prev_v'),
    [
        (
                RotationalJoint(0, 0, l1, t1, t1, 1),
                Vector([0, 0, d_t1]),
                Vector([0, 0, 0]),
                Vector([0, 0, 0]),
                Vector([0, 0, 0])
        ),
        (
                RotationalJoint(0, -pi / 2, l2, t2, t2, 2),
                Vector([-sin(t2) * d_t1, -cos(t2) * d_t1, d_t2]),
                Vector([-cos(t2) * d_t1 * l2, sin(t2) * l2 * d_t1, 0]),
                Vector([0, 0, d_t1]),
                Vector([0, 0, 0]),
        ),
        (
                PrismaticJoint(0, pi / 2, d3, 0, d3, 3),
                Vector([-sin(t2) * d_t1, d_t2, cos(t2) * d_t1]),
                Vector([-cos(t2) * d_t1 * l2 + d3 * d_t2, sin(t2) * d_t1 * d3, -sin(t2) * d_t1 * l2 + dot_d3]),
                Vector([-sin(t2) * d_t1, -cos(t2) * d_t1, d_t2]),
                Vector([-cos(t2) * d_t1 * l2, sin(t2) * l2 * d_t1, 0]),
        ),
        (
                EndEffector(0, 0, 0, 0, None, 4),
                Vector([-sin(t2) * d_t1, -cos(t2) * d_t1, d_t2]),
                Vector([-cos(t2) * d_t1 * l2, sin(t2) * l2 * d_t1, 0]),
                Vector([-sin(t2) * d_t1, -cos(t2) * d_t1, d_t2]),
                Vector([-cos(t2) * d_t1 * l2, sin(t2) * l2 * d_t1, 0]),
        )
    ]
)
def test_cal_static_forward_kinematics(joint: Joint, desired_w: Vector, desired_v: Vector, prev_w: Vector,
                                       prev_v: Vector):
    w, v = joint.cal_velocities(prev_w, prev_v)

    assert w == desired_w
    assert v == desired_v


g, dot_d2, ddot_d2 = symbols('g dot_d2 ddot_d2')
dd_t1 = symbols('ddot_t1')


@pytest.mark.parametrize(
    ('joint', 'prev_w', 'prev_dw', 'prev_dv', 'desired_w', 'desired_dw', 'desired_dv'),
    [
        (
                RotationalJoint(0, 0, 0, t1, t1, 1),
                Vector([0, 0, 0]),
                Vector([0, 0, 0]),
                Vector([g, 0, 0]),
                Vector([0, 0, d_t1]),
                Vector([0, 0, dd_t1]),
                Vector([g * cos(t1), -g * sin(t1), 0])
        ),
        (
                PrismaticJoint(l1, -pi / 2, d2, 0, d2, 2),
                Vector([0, 0, d_t1]),
                Vector([0, 0, dd_t1]),
                Vector([g * cos(t1), -g * sin(t1), 0]),
                Vector([0, -d_t1, 0]),
                Vector([0, -dd_t1, 0]),
                Vector([-d2 * dd_t1 - l1 * d_t1 ** 2 + g * cos(t1) - 2 * d_t1 * dot_d2, 0,
                        l1 * dd_t1 - d2 * d_t1 ** 2 - g * sin(t1) + ddot_d2]),
        ),
        (
                BaseJoint(0, 0, 0, 0, None, 1),
                Vector([g, 0, 0]),
                Vector([0, 0, 0]),
                Vector([0, 0, 0]),
                Vector([0, 0, 0]),
                Vector([0, 0, 0]),
                Vector([g, 0, 0]),
        ),
        (
                EndEffector(l1, -pi / 2, d2, 0, d2, 2),
                Vector([0, 0, d_t1]),
                Vector([0, 0, dd_t1]),
                Vector([g * cos(t1), -g * sin(t1), 0]),
                Vector([0, 0, d_t1]),
                Vector([0, 0, dd_t1]),
                Vector([-d2 * dd_t1 - l1 * d_t1 ** 2 + g * cos(t1), 0,
                        l1 * dd_t1 - d2 * d_t1 ** 2 - g * sin(t1)]),
        ),
    ]
)
def test_cal_dynamic_forward_kin1(joint: Joint, prev_w: Vector, prev_dw: Vector, prev_dv: Vector, desired_w: Vector,
                                  desired_dw: Vector, desired_dv: Vector):
    w, dw, dv = joint.cal_velocity_linear_rotational_acceleration_dynamic(prev_w, prev_dw, prev_dv)

    assert w == desired_w
    assert dw == desired_dw
    print(dv)
    print(desired_dv)
    assert dv == desired_dv


@pytest.mark.parametrize(
    ('joint', 'prev_w', 'prev_dw', 'prev_dv', 'cof', 'desired_dvc'),
    [
        (
                RotationalJoint(0, 0, 0, t1, t1, 1),
                Vector([0, 0, d_t1]),
                Vector([0, 0, dd_t1]),
                Vector([g * cos(t1), -g * sin(t1), 0]),
                Vector([l1 / 2, 0, 0]),
                Vector([-l1 / 2 * d_t1 ** 2 + g * cos(t1), l1 / 2 * dd_t1 - g * sin(t1), 0]),
        ),
        (
                PrismaticJoint(l1, -pi / 2, d2, 0, d2, 2),
                Vector([0, -d_t1, 0]),
                Vector([0, -dd_t1, 0]),
                Vector([-d2 * dd_t1 - l1 * d_t1 ** 2 + g * cos(t1) - 2 * d_t1 * dot_d2, 0,
                        l1 * dd_t1 - d2 * d_t1 ** 2 - g * sin(t1) + ddot_d2]),
                Vector([0, 0, l2]),
                Vector([-(l2 + d2) * dd_t1 - l1 * d_t1 ** 2 + g * cos(t1) - 2 * d_t1 * dot_d2, 0,
                        -(l2 + d2) * d_t1 ** 2 + l1 * dd_t1 - g * sin(t1) + ddot_d2]),
        ),
    ]
)
def test_cal_linear_acceleration_cof(joint: Joint, prev_w: Vector, prev_dw: Vector, prev_dv: Vector, cof: Vector,
                                     desired_dvc: Vector):
    joint.cof = cof
    dvc = joint.cal_linear_acceleration_cof(prev_w, prev_dw, prev_dv)

    assert dvc == desired_dvc


m1, m2 = symbols('m1 m2')
Izz1, Iyy2 = symbols('I_zz1 I_yy2')


@pytest.mark.parametrize(
    ('joint', 'prev_w', 'prev_dw', 'vc', 'cof', 'desired_F', 'desired_N'),
    [
        (
                RotationalJoint(0, 0, 0, t1, t1, 1),
                Vector([0, 0, d_t1]),
                Vector([0, 0, dd_t1]),
                Vector([-l1 / 2 * d_t1 ** 2 + g * cos(t1), l1 / 2 * dd_t1 - g * sin(t1), 0]),
                Vector([l1 / 2, 0, 0]),
                Vector([m1 * (-l1 / 2 * d_t1 ** 2 + g * cos(t1)), m1 * (l1 / 2 * dd_t1 - g * sin(t1)), 0]),
                Vector([0, 0, dd_t1 * Izz1])
        ),
        (
                PrismaticJoint(l1, -pi / 2, d2, 0, d2, 2),
                Vector([0, -d_t1, 0]),
                Vector([0, -dd_t1, 0]),
                Vector([-d2 * dd_t1 - l1 * d_t1 ** 2 + g * cos(t1) - 2 * d_t1 * dot_d2, 0,
                        l1 * dd_t1 - d2 * d_t1 ** 2 - g * sin(t1) + ddot_d2]),
                Vector([0, 0, l2]),
                Vector([m2 * (-d2 * dd_t1 - l1 * d_t1 ** 2 + g * cos(t1) - 2 * d_t1 * dot_d2), 0,
                        m2 * (l1 * dd_t1 - d2 * d_t1 ** 2 - g * sin(t1) + ddot_d2)]),
                Vector([0, -Iyy2 * dd_t1, 0]),
        ),
    ]
)
def test_cal_torque_force_cof(joint: Joint, prev_w: Vector, prev_dw: Vector, vc: Vector, cof: Vector, desired_F: Vector,
                              desired_N: Vector):
    joint.cof = cof
    F, N = joint.cal_force_torque_cof(prev_w, prev_dw, vc)

    assert F == desired_F
    assert N == desired_N


@pytest.mark.parametrize(
    ('joint', 'prev_f', 'prev_n', 'F', 'N', 'cof', 'desired_f', 'desired_n', 'rot', 'p'),
    [
        (
                RotationalJoint(0, 0, 0, t1, t1, 1),
                Vector([m2 * (-d2 * dd_t1 - l1 * d_t1 ** 2 + g * cos(t1) - 2 * d_t1 * dot_d2), 0,
                        m2 * (l1 * dd_t1 - d2 * d_t1 ** 2 - g * sin(t1) + ddot_d2)]),
                Vector(
                    [0, -Iyy2 * dd_t1 + m2 * l2 * (-d2 * dd_t1 - l1 * d_t1 ** 2 + g * cos(t1) - 2 * d_t1 * dot_d2), 0]),
                Vector([m1 * (-l1 / 2 * d_t1 ** 2 + g * cos(t1)), m1 * (l1 / 2 * dd_t1 - g * sin(t1)), 0]),
                Vector([0, 0, dd_t1 * Izz1]),
                Vector([l1 / 2, 0, 0]),
                Vector([m2 * (-d2 * dd_t1 - l1 * d_t1 ** 2 + g * cos(t1) - 2 * d_t1 * dot_d2) + m1 * (
                        -l1 / 2 * d_t1 ** 2 + g * cos(t1)),
                        m2 * (l1 * dd_t1 - d2 * d_t1 ** 2 - g * sin(t1) + ddot_d2) + m1 * (
                                l1 / 2 * dd_t1 - g * sin(t1)), 0]),
                # Vector([m1 * (-l1 / 2 * d_t1 ** 2 + g * cos(t1)), m1 * (l1 / 2 * dd_t1 - g * sin(t1)), 0]),
                Vector([0, 0, -(-Iyy2 * dd_t1 + m2 * l2 * (
                        -d2 * dd_t1 - l1 * d_t1 ** 2 + g * cos(t1) - 2 * d_t1 * dot_d2)) - d2 * (
                                m2 * (-d2 * dd_t1 - l1 * d_t1 ** 2 + g * cos(t1) - 2 * d_t1 * dot_d2)) + l1 * (m2 * (
                        l1 * dd_t1 - d2 * d_t1 ** 2 - g * sin(
                    t1) + ddot_d2)) + Izz1 * dd_t1 + m1 * l1 ** 2 / 4 * dd_t1 - g * m1 * l1 / 2 * sin(t1)]),
                Matrix([[1, 0, 0], [0, 0, 1], [0, -1, 0]], True),
                Vector([l1, d2, 0])
        ),
        (
                PrismaticJoint(l1, -pi / 2, d2, 0, d2, 2),
                Vector([0, 0, 0]),
                Vector([0, 0, 0]),
                Vector([m2 * (-d2 * dd_t1 - l1 * d_t1 ** 2 + g * cos(t1) - 2 * d_t1 * dot_d2), 0,
                        m2 * (l1 * dd_t1 - d2 * d_t1 ** 2 - g * sin(t1) + ddot_d2)]),
                Vector([0, -Iyy2 * dd_t1, 0]),
                Vector([0, 0, l2]),
                Vector([m2 * (-d2 * dd_t1 - l1 * d_t1 ** 2 + g * cos(t1) - 2 * d_t1 * dot_d2), 0,
                        m2 * (l1 * dd_t1 - d2 * d_t1 ** 2 - g * sin(t1) + ddot_d2)]),
                Vector(
                    [0, -Iyy2 * dd_t1 + m2 * l2 * (-d2 * dd_t1 - l1 * d_t1 ** 2 + g * cos(t1) - 2 * d_t1 * dot_d2), 0]),
                Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], True),
                Vector([0, 0, 0])

        ),
    ]
)
def test_cal_torque_force_joint_dynamic(joint: Joint, prev_f: Vector, prev_n: Vector, F: Vector, N: Vector, cof: Vector,
                                        desired_f: Vector, desired_n: Vector, rot: Matrix, p: Vector):
    joint.cof = cof

    f, n, next_rot, next_p = joint.cal_force_torque_joint_dynamic(prev_f, F, prev_n, N, rot, p)

    assert f == desired_f
    assert n == desired_n

#  TODO implement
# unittest for get_transformations_from_zero
# unittest for get_zero_Z_axis

# @pytest.mark.parametrize(
#     ('robot', 'cof_points', 'desired_ws', 'desired_dws', 'desired_dvs', 'desired_dvcs', 'desired_Fs', 'desired_Ns',
#      'desired_fs', 'desired_ns', 'desired_ts'),
#     [
#         (
#                 Robot.from_dh_parameters([[0, 0, 0, t1], [l1, -pi / 2, d2, 0], [0, 0, 0, 0]],
#                                          [JointType.ROTATIONAL, JointType.PRISMATIC], [t1, d2, 'end']),
#                 [Vector([l1 / 2, 0, 0]), Vector([0, 0, l2])],
#                 [
#                     Vector([0, 0, 0]),
#                     Vector([0, 0, d_t1]),
#                     Vector([0, -d_t1, 0]),
#                     Vector([0, -d_t1, 0]),
#                 ],
#                 [
#                     Vector([0, 0, 0]),
#                     Vector([0, 0, dd_t1]),
#                     Vector([0, -dd_t1, 0]),
#                     Vector([0, -dd_t1, 0]),
#                 ],
#                 [
#                     Vector([g, 0, 0]),
#                     Vector([g * cos(t1), -g * sin(t1), 0]),
#                     Vector([-d2 * dd_t1 - l1 * d_t1 ** 2 + g * cos(t1) - 2 * d_t1 * dot_d2, 0,
#                             l1 * dd_t1 - d2 * d_t1 ** 2 - g * sin(t1) + ddot_d2]),
#                     Vector([-d2 * dd_t1 - l1 * d_t1 ** 2 + g * cos(t1) - 2 * d_t1 * dot_d2, 0,
#                             l1 * dd_t1 - d2 * d_t1 ** 2 - g * sin(t1) + ddot_d2]),
#                 ],
#                 [
#                     Vector([-l1 / 2 * d_t1 ** 2 + g * cos(t1), l1 / 2 * dd_t1 - g * sin(t1), 0]),
#                     Vector([-(l2 + d2) * dd_t1 - l1 * d_t1 ** 2 + g * cos(t1) - 2 * d_t1 * dot_d2, 0,
#                             -(l2 + d2) * d_t1 ** 2 + l1 * dd_t1 - g * sin(t1) + ddot_d2]),
#                 ],
#                 [
#                     Vector([m1 * (-l1 / 2 * d_t1 ** 2 + g * cos(t1)), m1 * (l1 / 2 * dd_t1 - g * sin(t1)), 0]),
#                     Vector([m2 * (-d2 * dd_t1 - l1 * d_t1 ** 2 + g * cos(t1) - 2 * d_t1 * dot_d2 - dd_t1 * l2), 0,
#                             m2 * (l1 * dd_t1 - d2 * d_t1 ** 2 - g * sin(t1) + ddot_d2 - d_t1 ** 2 * l2)]),
#
#                 ],
#                 [
#                     Vector([0, 0, dd_t1 * Izz1]),
#                     Vector([0, -Iyy2 * dd_t1, 0]),
#                 ],
#                 [
#                     Vector([m2 * (-d2 * dd_t1 - dd_t1 * l2 - l1 * d_t1 ** 2 + g * cos(t1) - 2 * d_t1 * dot_d2), 0,
#                             m2 * (l1 * dd_t1 - d2 * d_t1 ** 2 - g * sin(t1) + ddot_d2 - d_t1 ** 2 * l2)]),
#                     Vector([m2 * (-d2 * dd_t1 - l1 * d_t1 ** 2 + g * cos(t1) - 2 * d_t1 * dot_d2) + m1 * (
#                             -l1 / 2 * d_t1 ** 2 + g * cos(t1)),
#                             m2 * (l1 * dd_t1 - d2 * d_t1 ** 2 - g * sin(t1) + ddot_d2) + m1 * (
#                                     l1 / 2 * dd_t1 - g * sin(t1)), 0])
#                 ],
#                 [
#                     Vector(
#                         [0, -Iyy2 * dd_t1 + m2 * l2 * (
#                                 - dd_t1 * l2 - d2 * dd_t1 - l1 * d_t1 ** 2 + g * cos(t1) - 2 * d_t1 * dot_d2),
#                          0]),
#                     Vector([0, 0, -(-Iyy2 * dd_t1 + m2 * l2 * (
#                             -d2 * dd_t1 - l1 * d_t1 ** 2 + g * cos(t1) - 2 * d_t1 * dot_d2)) - d2 * (
#                                     m2 * (- dd_t1*l2-d2 * dd_t1 - l1 * d_t1 ** 2 + g * cos(t1) - 2 * d_t1 * dot_d2)) + l1 * (
#                                     m2 * (l1 * dd_t1 - d2 * d_t1 ** 2 - g * sin(
#                                 t1) + ddot_d2)) + Izz1 * dd_t1 + m1 * l1 ** 2 / 4 * dd_t1 - g * m1 * l1 / 2 * sin(t1)]),
#                 ],
#                 []
#         )
#     ]
# )
# def test_cal_newton_euler(robot: Robot, cof_points: list, desired_ws, desired_dws, desired_dvs, desired_dvcs,
#                           desired_Fs, desired_Ns, desired_fs, desired_ns, desired_ts):
#     robot.add_cof(cof_points)
#     ws, dws, dvs, dvcs, Fs, Ns, fs, ns, ts = robot.cal_newton_euler(False, False, Vector([g, 0, 0]))
#
#     for i, F in enumerate(Fs):
#         print(f'length {len(fs)}')
#         print(ns[i])
#         print(desired_ns[i])
#         assert fs[i] == desired_fs[i]
#         assert ns[i] == desired_ns[i]
#
#         assert F == desired_Fs[i]
#         assert Ns[i] == desired_Ns[i]
#         assert dvcs[i] == desired_dvcs[i]
#
#     for i, w in enumerate(ws):
#         assert w == desired_ws[i]
#         assert dws[i] == desired_dws[i]
#         assert dvs[i] == desired_dvs[i]
