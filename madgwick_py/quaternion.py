# -*- coding: utf-8 -*-
"""
    Copyright (c) 2015 Jonas Böer, jonas.boeer@student.kit.edu

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import numbers
import math


class Quaternion:
    """
    A simple class implementing basic quaternion arithmetic.
    """
    def __init__(self, w_or_q, x=None, y=None, z=None):
        """
        Initializes a Quaternion object
        :param w_or_q: A scalar representing the real part of the quaternion, another Quaternion object or a
                    four-element array containing the quaternion values
        :param x: The first imaginary part if w_or_q is a scalar
        :param y: The second imaginary part if w_or_q is a scalar
        :param z: The third imaginary part if w_or_q is a scalar
        """
        self._q = np.array([1, 0, 0, 0])

        if x is not None and y is not None and z is not None:
            w = w_or_q
            q = np.array([w, x, y, z])
        elif isinstance(w_or_q, Quaternion):
            q = np.array(w_or_q.q)
        else:
            q = np.array(w_or_q)
            if len(q) != 4:
                raise ValueError("Expecting a 4-element array or w x y z as parameters")

        self.q = q

    # Quaternion specific interfaces

    def conj(self):
        """
        Returns the conjugate of the quaternion
        :rtype : Quaternion
        :return: the conjugate of the quaternion
        """
        q = self._q
        return Quaternion(q[0], -q[1], -q[2], -q[3])

    def to_euler(self):
        """
        Returns the Euler angles (roll, pitch, yaw) of the quaternion. (from matlab)
        :return: roll, pitch, yaw
        """
        R_1_1 = 2 * self._q[0]**2 - 1 + 2 * self._q[1]**2
        R_2_1 = 2 * self._q[1] * self._q[2] - 2 * self._q[0] * self._q[3]
        R_3_1 = 2 * self._q[1] * self._q[3] + 2 * self._q[0] * self._q[2]
        R_3_2 = 2 * self._q[2] * self._q[3] - 2 * self._q[0] * self._q[1]
        R_3_3 = 2 * self._q[0]**2 - 1 + 2 * self._q[3]**2

        phi = math.atan2(R_3_2, R_3_3)
        theta = math.atan(R_3_1/ math.sqrt(1 - R_3_1**2))
        psi = math.atan2(R_2_1, R_1_1)
        return phi, theta, psi # in radians

    def to_angle_axis(self):
        """
        Returns the quaternion's rotation represented by an Euler angle and axis.
        If the quaternion is the identity quaternion (1, 0, 0, 0), a rotation along the x axis with angle 0 is returned.
        :return: rad, x, y, z
        """
        if self[0] == 1 and self[1] == 0 and self[2] == 0 and self[3] == 0:
            return 0, 1, 0, 0
        rad = np.arccos(self[0]) * 2
        imaginary_factor = np.sin(rad / 2)
        if abs(imaginary_factor) < 1e-8:
            return 0, 1, 0, 0
        x = self._q[1] / imaginary_factor
        y = self._q[2] / imaginary_factor
        z = self._q[3] / imaginary_factor
        return rad, x, y, z

    @staticmethod
    def from_angle_axis(rad, x, y, z):
        s = np.sin(rad / 2)
        return Quaternion(np.cos(rad / 2), x*s, y*s, z*s)

    def to_euler_angles(self):
        pitch = np.arcsin(2 * self[1] * self[2] + 2 * self[0] * self[3])
        if np.abs(self[1] * self[2] + self[3] * self[0] - 0.5) < 1e-8:
            roll = 0
            yaw = 2 * np.arctan2(self[1], self[0])
        elif np.abs(self[1] * self[2] + self[3] * self[0] + 0.5) < 1e-8:
            roll = -2 * np.arctan2(self[1], self[0])
            yaw = 0
        else:
            roll = np.arctan2(2 * self[0] * self[1] - 2 * self[2] * self[3], 1 - 2 * self[1] ** 2 - 2 * self[3] ** 2)
            yaw = np.arctan2(2 * self[0] * self[2] - 2 * self[1] * self[3], 1 - 2 * self[2] ** 2 - 2 * self[3] ** 2)
        return roll, pitch, yaw

    def to_euler123(self):
        roll = np.arctan2(-2 * (self[2] * self[3] - self[0] * self[1]), self[0] ** 2 - self[1] ** 2 - self[2] ** 2 + self[3] ** 2)
        pitch = np.arcsin(2 * (self[1] * self[3] + self[0] * self[1]))
        yaw = np.arctan2(-2 * (self[1] * self[2] - self[0] * self[3]), self[0] ** 2 + self[1] ** 2 - self[2] ** 2 - self[3] ** 2)
        return roll, pitch, yaw

    def mul_q(self, other):
        sq = self._q
        oq = other._q
        sw, sx, sy, sz = sq
        ow, ox, oy, oz = oq

        w = sw*ow - sx*ox - sy*oy - sz*oz
        x = sw*ox + sx*ow + sy*oz - sz*oy
        y = sw*oy - sx*oz + sy*ow + sz*ox
        z = sw*oz + sx*oy - sy*ox + sz*ow

        return Quaternion(w, x, y, z)

    def add_q(self, other):
        return Quaternion(self._q + other._q)

    def mul_scalar(self, scalar):
        q = self._q * scalar
        return Quaternion(q)


    # Implementing other interfaces to ease working with the class

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, q):
        self._q = q

    def __getitem__(self, item):
        return self._q[item]

    def __array__(self):
        return self._q
