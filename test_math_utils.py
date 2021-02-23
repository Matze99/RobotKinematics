import pytest
from typing import List, Union

import numpy as np

from sympy import simplify, symbols, pi, Expr, Symbol, cos, sin

from robot import Robot, JointType, RotationalJoint, PrismaticJoint, Joint
from math_utils import Vector, Matrix

t1, t2 = symbols('theta1 theta2')
l1, d1 = symbols('l1 d1')


@pytest.mark.parametrize(
    ('dh', 'true_entries'),
    [
        (
                [[0, 0, 0, t1],
                  [l1, pi / 2, 0, t2]],
                [
                    [cos(t1) * cos(t2), -cos(t1) * sin(t2), sin(t1), l1 * cos(t1)],
                    [sin(t1) * cos(t2), -sin(t1) * sin(t2), -cos(t1), l1 * sin(t1)],
                    [sin(t2), cos(t2), 0, 0],
                    [0, 0, 0, 1]
                ]
        )
    ]
)
def test_mat_mul(dh: List[List[Union[Expr, int, Symbol]]], true_entries: List[List[Union[Expr, int, Symbol]]]):
    robot = Robot.from_dh_parameters(dh, JointType.ROTATIONAL)
    rot = robot.get_end_rotation()
    trans = robot.get_end_transformation()

    end_pos = trans @ Vector([0, 0, 0, 1])

    desired_mat = Matrix(true_entries)
    assert trans == desired_mat
    assert rot == desired_mat.rot

    for i in range(4):
        assert simplify(end_pos[i]-true_entries[i][-1]) == 0


@pytest.mark.parametrize(
    ('matrix'),
    [
        tuple(
            [
                [1, 1, 1],
                [0, 1, 0],
                [1, 1, 0]
            ]
        )
    ]
)
def test_transpose(matrix: List[List[Union[Expr, int, Symbol]]]):
    mat = Matrix(matrix, True)

    mat2 = mat.T.T

    assert mat == mat2


@pytest.mark.parametrize(
    ('matrix', 'vector', 'desired1', 'desired2'),
    [
        (
            [
                [1, 1, 1],
                [0, 1, 0],
                [1, 1, 0]
            ],
            [0, 1, 1],
            [2, 1, 1],
            [1, 2, 0]
        )
    ]
)
def test_vector_mat_matrix(matrix: List[List[Union[Expr, int, Symbol]]], vector: List[Union[Expr, int, Symbol]],
                           desired1: List[Union[Expr, int, Symbol]], desired2: List[Union[Expr, int, Symbol]]):
    mat = Matrix(matrix, is_rotation=len(matrix)==3)
    vec = Vector(vector)

    vec1 = mat @ vec
    vec2 = vec @ mat

    desired_vec1 = Vector(desired1)
    assert desired_vec1 == vec1
    desired_vec2 = Vector(desired2)
    assert desired_vec2 == vec2

@pytest.mark.parametrize(
    ('joint'),
    [
        RotationalJoint(0, 0, 0, t1, t1, 1),
        RotationalJoint(0, pi/2, 0, t1, t1, 1),
        RotationalJoint(0, pi/2, d1, 0, t1, 1)
    ]
)
def test_mat_inv(joint: Joint):
    trans_matrix = joint.get_transformation()
    rot_matrix = joint.get_rotation()

    desired_mat_rot = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], True, False)
    desired_mat_trans = Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], False, False)

    assert trans_matrix @ trans_matrix.inv() == desired_mat_trans
    assert rot_matrix @ rot_matrix.inv() == desired_mat_rot

    assert trans_matrix.inv() @ trans_matrix == desired_mat_trans
    assert rot_matrix.inv() @ rot_matrix == desired_mat_rot

@pytest.mark.parametrize(
    ('vector1', 'vector2'),
    [
        (
            Vector([1, 0, 2]),
            Vector([2, 2, 0])
        ),
        (
            Vector([2, 2, 0]),
            Vector([2, 2, 0])
        )
    ]
)
def test_cross_product(vector1: Vector, vector2:Vector):
    output = vector1 * vector2
    desired = Vector(np.cross(vector1.entries, vector2.entries))
    assert output == desired

@pytest.mark.parametrize(
    ('matrix', 'shape'),
    [
        (
            Matrix(np.ones((1,2,3))),
            (1,2,3)
        ),
        (
                Matrix(np.random.rand(1, 2, 3)),
                (1, 2, 3)
        )
    ]
)
def test_add(matrix: Matrix, shape):
    result = matrix + (-matrix)
    desired = Matrix(np.zeros(shape))
    assert desired == result