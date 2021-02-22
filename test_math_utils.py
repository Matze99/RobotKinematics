import pytest
from typing import List, Union

from sympy import simplify, symbols, pi, Expr, Symbol, cos, sin

from robot import Robot, JointType
from math_utils import Vector, Matrix

t1, t2 = symbols('theta1 theta2')
l1 = symbols('l1')


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
    mat = Matrix(matrix)

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