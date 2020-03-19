# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

# noinspection PyUnresolvedReferences
def render_animation(keypoints, poses, skeleton, input_frame,
                     show=False):
    size = 6
    azim = np.array(70., dtype=np.float32)
    plt.ioff()
    fig = plt.figure(figsize=(size * 2, size))
    ax_in = fig.add_subplot(1, 2, 1)
    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)
    ax_in.set_axis_off()
    ax_in.set_title('Input video')

    lines_3d = []
    radius = 1.7
    ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
    ax_3d.view_init(elev=15., azim=azim)
    ax_3d.set_xlim3d([-radius / 2, radius / 2])
    ax_3d.set_zlim3d([0, radius])
    ax_3d.set_ylim3d([-radius / 2, radius / 2])
    ax_3d.set_aspect('equal')
    ax_3d.set_xticklabels([])
    ax_3d.set_yticklabels([])
    ax_3d.set_zticklabels([])
    ax_3d.dist = 7.5
    ax_3d.set_title("Reconstruction")  # , pad=35

    initialized = False
    image = None
    lines = []
    points = None
    fps = 20
    fig.tight_layout()

    trajectories = poses[0, [0, 1]]
    parents = skeleton.parents()

    # noinspection PyUnresolvedReferences
    def update_video(i):
        nonlocal image, lines, points, initialized
        ax_3d.set_xlim3d([-radius / 2 + trajectories[i, 0], radius / 2 + trajectories[i, 0]])
        ax_3d.set_ylim3d([-radius / 2 + trajectories[i, 1], radius / 2 + trajectories[i, 1]])

        # Update 2D poses
        joints_right_2d = [2, 4, 6, 8, 10, 12, 14, 16]
        # noinspection PyTypeChecker
        colors_2d = np.full(keypoints.shape[1], 'black')
        colors_2d[joints_right_2d] = 'red'
        if not initialized:
            image = ax_in.imshow(input_frame, aspect='equal')

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                if len(parents) == keypoints.shape[1]:
                    # Draw skeleton only if keypoints match (otherwise we don't have the parents definition)
                    lines.append(ax_in.plot([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                                            [keypoints[i, j, 1], keypoints[i, j_parent, 1]], color='pink'))

                col = 'red' if j in skeleton.joints_right() else 'black'

                lines_3d.append(ax_3d.plot([poses[j, 0], poses[j_parent, 0]],
                                           [poses[j, 1], poses[j_parent, 1]],
                                           [poses[j, 2], poses[j_parent, 2]], zdir='z', c=col))

            points = ax_in.scatter(*keypoints[i].T, 10, color=colors_2d, edgecolors='white', zorder=10)

            initialized = True
        else:
            image.set_data(input_frame)

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                if len(parents) == keypoints.shape[1]:
                    lines[j - 1][0].set_data([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                                             [keypoints[i, j, 1], keypoints[i, j_parent, 1]])

                lines_3d[j - 1][0].set_xdata([poses[j, 0], poses[j_parent, 0]])
                lines_3d[j - 1][0].set_ydata([poses[j, 1], poses[j_parent, 1]])
                lines_3d[j - 1][0].set_3d_properties([poses[j, 2], poses[j_parent, 2]], zdir='z')

            points.set_offsets(keypoints[i])

    anim = FuncAnimation(fig, func=update_video, frames=1, interval=1000 / fps, repeat=False)

    if show:
        plt.show()
    else:
        if output.endswith('.mp4'):
            writer = writers['ffmpeg']
            writer = writer(fps=fps, metadata={}, bitrate=bitrate)
            anim.save(output, writer=writer)
        elif output.endswith('.gif'):
            anim.save(output, dpi=80, writer='imagemagick')
        else:
            raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
        plt.close()
