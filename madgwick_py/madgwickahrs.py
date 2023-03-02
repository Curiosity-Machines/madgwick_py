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

import warnings
import numpy as np
from numpy.linalg import norm
from .quaternion import Quaternion


class MadgwickAHRS:
    samplePeriod = 1/256
    quaternion = Quaternion(1, 0, 0, 0)
    beta = 1
    zeta = 0

    def __init__(self, sampleperiod=None, quaternion=None, beta=None, zeta=None):
        """
        Initialize the class with the given parameters.
        :param sampleperiod: The sample period
        :param quaternion: Initial quaternion
        :param beta: Algorithm gain beta
        :param beta: Algorithm gain zeta
        :return:
        """
        if sampleperiod is not None:
            self.samplePeriod = sampleperiod
        if quaternion is not None:
            self.quaternion = quaternion
        if beta is not None:
            self.beta = beta
        if zeta is not None:
            self.zeta = zeta

    def update(self, gyroscope, accelerometer, magnetometer):
        """
        Perform one update step with data from a AHRS sensor array
        :param gyroscope: A three-element array containing the gyroscope data in radians per second.
        :param accelerometer: A three-element array containing the accelerometer data. Can be any unit since a normalized value is used.
        :param magnetometer: A three-element array containing the magnetometer data. Can be any unit since a normalized value is used.
        :return:
        """
        q = self.quaternion

        gyroscope = np.array(gyroscope, dtype=float).flatten()
        accelerometer = np.array(accelerometer, dtype=float).flatten()
        magnetometer = np.array(magnetometer, dtype=float).flatten()

        # Normalise accelerometer measurement
        if norm(accelerometer) == 0:
            warnings.warn("accelerometer is zero")
            return
        accelerometer /= norm(accelerometer)

        # Normalise magnetometer measurement
        if norm(magnetometer) == 0:
            warnings.warn("magnetometer is zero")
            return
        magnetometer /= norm(magnetometer)

        h = q * (Quaternion(0, magnetometer[0], magnetometer[1], magnetometer[2]) * q.conj())
        b = np.array([0, norm(h[1:3]), 0, h[3]])

        q_w = q[0]
        q_x = q[1]
        q_y = q[2]
        q_z = q[3]

        qxz = q_x * q_z
        qxy = q_x * q_y
        qyz = q_y * q_z
        qwx = q_w * q_x
        qwy = q_w * q_y
        qwz = q_w * q_z
        qx2 = q_x * q_x
        qy2 = q_y * q_y
        qz2 = q_z * q_z
        b1 = b[1]
        b2 = b[2]
        b3 = b[3]
        twoqx = 2 * q_x
        twoqy = 2 * q_y
        twoqz = 2 * q_z
        twoqw = 2 * q_w
        fourqx = 4 * q_x
        fourqy = 4 * q_y
        fourqz = 4 * q_z

        # Gradient descent algorithm corrective step
        f = np.array([
            2*(qxz - qwy) - accelerometer[0],
            2*(qwx + qyz) - accelerometer[1],
            2*(0.5 - qx2 - qy2) - accelerometer[2],
            2*b1*(0.5 - qy2 - qz2) + 2*b3*(qxz - qwy) - magnetometer[0],
            2*b1*(qxy - qwz) + 2*b3*(qwx + qyz) - magnetometer[1],
            2*b1*(qwy + qxz) + 2*b3*(0.5 - qx2 - qy2) - magnetometer[2]
        ])
        j = np.array([
            [-twoqy,                twoqz,                -twoqw,                  twoqx],
            [twoqx,                 twoqw,                 twoqz,                  twoqy],
            [0,                   -fourqx,               -fourqy,                  0],
            [-b3*twoqy,             b3*twoqz,             -b1*fourqy-b3*twoqw,        -b1*fourqz+b3*twoqx],
            [-b1*twoqz+b3*twoqx,      b1*twoqy+b3*twoqw,       b1*twoqx+b3*twoqz,        -b1*twoqw+b3*twoqy],
            [b1*twoqy,              b1*twoqz-b3*fourqx,       b1*twoqw-b3*fourqy,        b1*twoqx]
        ])

        step = j.T.dot(f)
        step /= norm(step)  # normalise step magnitude

        # Gyroscope compensation drift
        gyroscopeQuat = Quaternion(0, gyroscope[0], gyroscope[1], gyroscope[2])
        stepQuat = Quaternion(step.T[0], step.T[1], step.T[2], step.T[3])

        gyroscopeQuat = gyroscopeQuat + (q.conj() * stepQuat) * 2 * self.samplePeriod * self.zeta * -1

        # Compute rate of change of quaternion
        qdot = (q * gyroscopeQuat) * 0.5 - self.beta * step.T

        # Integrate to yield quaternion
        q += qdot * self.samplePeriod
        self.quaternion = Quaternion(q / norm(q))  # normalise quaternion

    def update_imu(self, gyroscope, accelerometer):
        """
        Perform one update step with data from a IMU sensor array
        :param gyroscope: A three-element array containing the gyroscope data in radians per second.
        :param accelerometer: A three-element array containing the accelerometer data. Can be any unit since a normalized value is used.
        """
        q = self.quaternion

        gyroscope = np.array(gyroscope, dtype=float).flatten()
        accelerometer = np.array(accelerometer, dtype=float).flatten()

        # Normalise accelerometer measurement
        if norm(accelerometer) == 0:
            warnings.warn("accelerometer is zero")
            return
        accelerometer /= norm(accelerometer)

        # Gradient descent algorithm corrective step
        f = np.array([
            2*(q[1]*q[3] - q[0]*q[2]) - accelerometer[0],
            2*(q[0]*q[1] + q[2]*q[3]) - accelerometer[1],
            2*(0.5 - q[1]**2 - q[2]**2) - accelerometer[2]
        ])
        j = np.array([
            [-2*q[2], 2*q[3], -2*q[0], 2*q[1]],
            [2*q[1], 2*q[0], 2*q[3], 2*q[2]],
            [0, -4*q[1], -4*q[2], 0]
        ])
        step = j.T.dot(f)
        step /= norm(step)  # normalise step magnitude

        # Compute rate of change of quaternion
        qdot = (q * Quaternion(0, gyroscope[0], gyroscope[1], gyroscope[2])) * 0.5 - self.beta * step.T

        # Integrate to yield quaternion
        q += qdot * self.samplePeriod
        self.quaternion = Quaternion(q / norm(q))  # normalise quaternion
