from __future__ import annotations
from typing import List, Union

import numpy as np

from sympy import Symbol, Integer, Float, cos, sin, Expr, simplify


class Matrix:
    '''
    TODO update Matrix and vector to be numpy arrays

    Rotation- and Transformation Matrix class

    supports multiplications transformations and other nessesary behavior of rotation and transformation
    matrices for robots

    :param entries: elements of the matrix
    :param is_rotation: True -> rotation matrix, False -> transformation matrix
    :param is_identity: True if identity matrix
    '''

    def __init__(self, entries: Union[List[List[Union[Symbol, Float, Integer, Expr, float, int, None]]], np.ndarray] = None,
                 is_rotation: bool = False, is_identity: bool = False):
        self._entries: np.ndarray = np.array(entries)
        self._is_rotation: bool = is_rotation
        self._is_identity: bool = is_identity
        if is_rotation and not is_identity:
            # TODO check validity
            pass
        elif is_identity:
            # TODO check velidity
            pass

    @staticmethod
    def transformation_from_joint(joint) -> Matrix:
        '''
        creates a transformation matrix from a joint

        .. math::

            [cos Theta_i   ,               -sin Theta_i,                 0 ,             a_{i-1}             ]

            [sin Theta_i cos Alpha_{i-1},  cos Theta_i cos Alpha_{i-1},   -sin Alpha_{i-1}, -sin Alpha_{i-1} d_i  ]

            [sin Theta_i sin Alpha_{i-1},  sin Theta_i cos Alpha_{i-1},   cos Alpha_{i-1},  cos Alpha_{i-1} d_i   ]

            [0                          ,  0,                          0         ,     1                   ]

        :param joint: current joint (index i)
        :type joint: Joint
        :return: transformation matrix of joint
        '''
        entries = [[cos(joint.theta), -sin(joint.theta), 0, joint.a],
                   [sin(joint.theta) * cos(joint.alpha), cos(joint.theta) * cos(joint.alpha), -sin(joint.alpha),
                    -sin(joint.alpha) * joint.d],
                   [sin(joint.theta) * sin(joint.alpha), cos(joint.theta) * sin(joint.alpha), cos(joint.alpha),
                    cos(joint.alpha) * joint.d],
                   [0, 0, 0, 1]]
        return Matrix(entries, is_rotation=False, is_identity=False)

    @staticmethod
    def rotation_from_joint(joint) -> Matrix:
        '''
        creates the rotation matrix of the joint

        .. math::

            [cos Theta_i,                  -sin Theta_i    ,             0               ]

            [sin Theta_i cos Alpha_{i-1},   cos Theta_i cos Alpha_{i-1},    -sin Alpha_{i-1}  ]

            [sin Theta_i sin Alpha_{i-1},   sin Theta_i cos Alpha_{i-1},    cos Alpha_{i-1}   ]

        :param joint: current joint (index i)
        :return: Rotation matrix of joint
        '''
        entries = [[cos(joint.theta), -sin(joint.theta), 0],
                   [sin(joint.theta) * cos(joint.alpha), cos(joint.theta) * cos(joint.alpha), -sin(joint.alpha)],
                   [sin(joint.theta) * sin(joint.alpha), cos(joint.theta) * sin(joint.alpha), cos(joint.alpha)]]
        return Matrix(entries, is_rotation=True, is_identity=False)

    @property
    def entries(self) -> np.ndarray:
        return self._entries

    @entries.setter
    def entries(self, entries: List[List[Union[Symbol, Float, Integer, Expr, float, int]]]):
        self._entries = np.array(entries)

    @property
    def is_rotation(self) -> bool:
        return self._is_rotation

    @is_rotation.setter
    def is_rotation(self, is_rotation: bool):
        self._is_rotation = is_rotation

    @property
    def is_identity(self) -> bool:
        return self._is_identity

    @is_rotation.setter
    def is_rotation(self, is_identity: bool):
        self._is_identity = is_identity

    def __getitem__(self, item1: int) -> List[Union[Symbol, Float, Integer, Expr, float, int]]:
        return self._entries[item1]

    def __matmul__(self, other: Union[Matrix, Vector]) -> Union[Matrix, Vector]:
        '''
        matrix multiplication function

        currently only multiplication with another matrix of the same type
        or a vector with the same length is supported


        :param other: multiplicative
        :return: result of multiplication of self @ other
        '''
        if isinstance(other, Matrix) and (other.is_rotation == self.is_rotation):
            if self.is_identity:
                return Matrix(other.entries, other.is_rotation, other.is_identity)
            elif other.is_identity:
                return Matrix(self.entries, self.is_rotation, self.is_identity)

            new_entries = []
            for i, row in enumerate(self._entries):
                new_entries.append([])
                for j, ele in enumerate(row):
                    new_entries[i].append(self._mult_row_column(i, j, other))

            return Matrix(new_entries, self.is_rotation)
        elif isinstance(other, Vector):
            new_entries = []

            for row in self._entries:
                new_entries.append(other @ Vector(row))
            return Vector(new_entries)

        else:
            if isinstance(other, Matrix):
                raise ValueError(f'has to be of the same type as self (self is rotation {self.is_rotation}, other is rotation {other.is_rotation}')
            else:
                raise ValueError(f'other has to be of type Matrix or Vector (is {type(other)})')

    def _mult_row_column(self, row: int, column: int, other: Matrix) -> Union[Symbol, Float, Integer, Expr]:
        '''
        Multiplies the row (row) of this matrix with the column (column) of the other matrix

        .. math::

            \text{out} = self.entries[row]^T other.entries[:][column]

        :param row: index of the row of this matrix that is being multiplied
        :param column: index of the column of other matrix that is being multiplied
        :param other: matrix this is being multiplied with
        :return: multiplication of row of this matrix and column of other matrix
        '''
        length = 3 if self._is_rotation else 4
        result = Float(0)
        for i in range(length):
            result += self._entries[row][i] * other.entries[i][column]
        return result

    def __str__(self) -> str:
        string = 'Matrix[\n'
        for row in self._entries:
            string += '\t'
            for ele in row:
                string += ele.__str__() + ',    '
            string += '\n'
        string += ']'
        return string

    @property
    def T(self) -> Matrix:
        '''
        transposes matrix

        :return: transposed matrix
        '''
        if self.is_identity:
            return self

        new_entries = [[], [], []] if self._is_rotation else [[], [], [], []]

        for i, row in enumerate(self._entries):
            for j, ele in enumerate(row):
                new_entries[j].append(ele)

        return Matrix(new_entries, is_rotation=self.is_rotation)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Matrix):
            return False
        elif self._is_rotation != other.is_rotation:
            return False
        else:
            return simplify(np.sum(self._entries-other.entries)) == 0


    @property
    def rot(self) -> Matrix:
        '''
        makes rotation matrix out of current transformation matrix

        :return: rotation matrix
        '''
        if self._is_rotation:
            return self
        else:
            new_entries = []
            for row in self._entries[:-1]:
                new_entries.append(row[:-1])
            return Matrix(new_entries, is_rotation=True, is_identity=self._is_identity)

    def __add__(self, other: Matrix) -> Matrix:
        if isinstance(other, float) or isinstance(other, int):
            new_entries = self.entries + other
        elif isinstance(other, Matrix):
            assert other.is_rotation == self._is_rotation
            new_entries = self._entries + other.entries
        else:
            raise ValueError(f'Matrix addition is not implemented for type {type(other)}')

        return Matrix(new_entries, self.is_rotation)

    def __radd__(self, other):
        return self.__add__(other)

    def simplify(self) -> Matrix:
        '''
        simplifies each entry in the matrix

        :return: simplified matrix
        '''
        new_entries = []
        for i, row in enumerate(self._entries):
            new_entries.append([])
            for j, ele in enumerate(row):
                new_entries[i].append(simplify(ele))
        return Matrix(new_entries, self.is_rotation)

    def subs(self, args: dict):
        '''
        substitutes values or other symbols defined by args dictionary

        :param args: dictionary with key (will be substituted for) value
        :return: matrix with substitutions applied
        '''
        new_entries = []
        for i, row in enumerate(self._entries):
            new_entries.append([])
            for j, ele in enumerate(row):
                new_entries[i].append(ele.subs(args))
        return Matrix(new_entries, self.is_rotation)

    @staticmethod
    def diagonal(vec: Union[tuple, list, Vector]) -> Matrix:
        '''
        creates a diagonal matrix with the elements of vec being the diagonal elements.

        :param vec: vector with diagonal elements, has to be either length 3 or 4
        :return: diagonal matrix of vec
        '''
        if not len(vec) == 3 and not len(vec) == 4:
            raise ValueError('only vectors of length 4 and 3 are supported')
        rotation = len(vec) == 3
        new_entries = []
        for i in range(len(vec)):
            new_entries.append([])
            for j in range(len(vec)):
                if i == j:
                    new_entries[i].append(vec[i])
                else:
                    new_entries[i].append(0)

        return Matrix(new_entries, rotation, False)

    def __neg__(self):
        return self * -1

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int) or isinstance(other, Symbol) or isinstance(other, Expr):
            new_entries = self._entries * other
            return Matrix(new_entries, self.is_rotation, self.is_identity)
        else:
            raise ValueError(f'Multiplication of matrix is not defined for type {type(other)}')

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, power, modulo=None):
        '''
        elementwise power

        :param power:
        :param modulo:
        :return:
        '''
        new_entries = self._entries ** 2
        return Matrix(new_entries, self.is_rotation, self.is_identity)

    def get_translation(self) -> Vector:
        '''
        calculates the translation of the current matrix

        :return:
        '''
        if self.is_rotation:
            raise ValueError('rotation matrix does not have a translation')
        else:
            return self @ Vector([0, 0, 0, 1])


    def inv(self):
        '''
        invert the matrix
        :return:
        '''
        if self._is_rotation:
            return self.T
        else:
            rot = self.rot
            trans = self.get_translation()

            inv_rot = rot.T
            inv_trans = -inv_rot @ trans.remove_last()

            new_entries = [
                [*inv_rot[0], inv_trans[0]],
                [*inv_rot[1], inv_trans[1]],
                [*inv_rot[2], inv_trans[2]],
                [0, 0, 0, 1]
            ]
            return Matrix(new_entries, False, self.is_identity)

    def __repr__(self):
        return self.__str__()

class Vector:
    def __init__(self, entries: Union[List[Union[Symbol, Float, Integer, Expr, int, float]], np.ndarray]):
        self.entries = np.array(entries)

    def __getitem__(self, item) -> Union[Symbol, Float, Integer, Expr, int, float]:
        return self.entries[item]

    def __len__(self):
        return len(self.entries)

    def __mul__(self, other) -> Vector:
        if isinstance(other, Vector):
            assert len(other) == self.__len__()
            assert self.__len__() == 3

            new_entries = []
            for index,(i, j) in enumerate([(1,2),(2,0),(0,1)]):
                new_entries.append(self.entries[i]*other[j]-self.entries[j]*other[i])

            return Vector(new_entries)

        else:
            new_entries = self.entries * other

            return Vector(new_entries)

    def __rmul__(self, other):
        if isinstance(other, Vector):
            return other.__mul__(self)
        return self.__mul__(other)

    def __matmul__(self, other):
        if isinstance(other, Matrix):
            return other.T @ self
        else:
            assert isinstance(other, Vector)
            assert len(other) == self.__len__()

            value = self.entries @ other.entries
            return value

    def __eq__(self, other) -> bool:
        if not isinstance(other, Vector):
            return False
        elif len(other) != len(self):
            return False
        else:
            return simplify(np.sum(self.entries - other.entries)) == 0

    def __add__(self, other: Vector) -> Vector:
        if isinstance(other, int) or isinstance(other, float):
            new_entries = self.entries + other
        elif isinstance(other, Vector):
            assert len(other) == len(self)
            new_entries = self.entries + other.entries
        else:
            raise ValueError(f'Vector addition not implemented for type {type(other)}')
        return Vector(new_entries)

    def __radd__(self, other):
        return self.__add__(other)

    def remove_last(self) -> Vector:
        '''
        removes the last element of the vector

        :return: vector without last element
        '''
        return Vector(self.entries[:-1])

    def __str__(self) -> str:
        string = 'Vector[\t'
        for ele in self.entries:
            string += ele.__str__() + ',  '
        string += ']'
        return string

    def simplify(self) -> Vector:
        '''
        simplifies each entry in the vector

        :return: simplified vector
        '''
        new_entries = []

        for i, ele in enumerate(self.entries):
            new_entries.append(simplify(ele))
        return Vector(new_entries)

    def subs(self, args):
        '''
        substitutes values or other symbols defined by args dictionary

        :param args: dictionary with key (will be substituted for) value
        :return: vector with substitutions applied
        '''
        new_entries = []

        for i, ele in enumerate(self.entries):
            new_entries.append(ele.subs(args))
        return Vector(new_entries)

    def __pow__(self, power, modulo=None):
        '''
        elementwise power

        :param power:
        :param modulo:
        :return:
        '''
        new_entries = self.entries**power
        return Vector(new_entries)

    def self_dot(self):
        '''
        multiplies each entry in the vector with itself and sums the multiplied entries up
        :return: sum of multiplication
        '''
        return self @ self

    def __repr__(self):
        return self.__str__()