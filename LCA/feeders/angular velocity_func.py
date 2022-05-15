import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


class Skeleton:
    def __init__(self, kinematic_tree):
        # We first define the skeleton for H3.6m
        self._kinematic_tree = kinematic_tree

    def kinematic_tree(self):
        return self._kinematic_tree

    def plot(self, joints):
        plt.figure()
        limits = 1200
        ax = plt.axes(xlim=(-limits, limits), ylim=(-limits, limits), zlim=(-limits, limits), projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(elev=120, azim=-90)
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='black')
        colors = ['red', 'yellow', 'black', 'green', 'blue']
        for chain, color in zip(self._kinematic_tree, colors):
            ax.plot3D(joints[chain, 0], joints[chain, 1], joints[chain, 2], linewidth=2.0, color=color)


"""""""""Initialze Skeleton class with following parameters"""""""""

# Define a kinematic tree for the skeletal struture
# Left leg, Right leg, Spine, Left arm, Right arm

kinematic_tree = [[0, 1, 2, 3, 4, 5], [0, 6, 7, 8, 9, 10], [0, 12, 13, 14, 15], [13, 17, 18, 19, 22, 19, 21],
                  [13, 25, 26, 27, 30, 27, 29]]
human_skel = Skeleton(kinematic_tree)
njoints = 32
parents = np.array(
    [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12, 16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27,
     30])

# Load a sample 3D pose and plot

data = sio.loadmat(
    'F')[
    'joint_xyz']
nframes = len(data)
cur_frame = 0
# load the frames and switch the y and z axis
cur_joints_xyz = np.reshape(data[cur_frame], [32, 3])[:, [0, 2, 1]]
human_skel.plot(cur_joints_xyz)

"""""""""Computing the angular velocity 𝜔 and angular acceleration 𝛼"""""""""""




def findrot(u, v, M):
    """find the axis angle parameters to rotate vector u onto vector v"""
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    w = np.cross(u, v, axis=0)
    w_norm = np.linalg.norm(w)
    q = np.dot(u.T, v)
    qq = [q[0][0], q[1][1]]
    if w_norm < 1e-6:
        A = np.zeros([3, M])
    else:
        ww = w / w_norm
        qq = np.arccos(qq)
        A = ww * qq  # normal vector * angle scalar
    return A


def calcu_curl(data_numpy):
    T, V, C, M = data_numpy.shape  # CTVM
    u = np.zeros([T, V, 3, M])
    omega = np.zeros([T, V, 3, M])
    alpha = np.zeros([T, V, 3, M])
    angular_v = np.zeros([T, V, 3, M])
    angular_a = np.zeros([T, V, 3, M])

    parents = np.array(
        [0, 1, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11, 1, 13, 14, 15, 1, 17, 18, 19, 2, 8, 7, 12, 11])
    for j in range(T):
        for i in range(1, V):
            joints = data_numpy[j][:, [0, 2, 1]]  # change xyz to xzy of vertical coors  CTVM
            u[j][i] = joints[i] - joints[parents[i]]

    for j in range(T - 1):
        for i in range(1, V):
            if np.linalg.norm(u[j][i]) != 0 and np.linalg.norm(u[j + 1][i]) != 0:
                omega[j][i] = findrot(u[j][i], u[j + 1][i], T)
                angular_v[j][i] = np.cross(omega[j][i], u[j][i], axis=0)
        if j > 0:
            alpha[j - 1] = omega[j] - omega[j - 1]
            angular_a[j - 1] = u[j + 1] - 2 * u[j] + u[j - 1]
    curl = 2 * angular_a[:, :, [0, 2, 1]]
    div = -2 * (angular_v[:, :, [0, 2, 1]] ** 2)
    curl = np.where(curl == 0, 1, curl)
    curl = np.transpose(curl, (2, 0, 1, 3))
    div = np.where(div == 0, 1, div)
    div = np.transpose(div, (2, 0, 1, 3))
    return curl, div


def norm(dt):
    mu = np.mean(dt, axis=0)
    sigma = np.std(dt, axis=0)
    return (dt - mu) / sigma





