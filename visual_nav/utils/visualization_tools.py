import itertools

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import patches
import numpy as np


def heatmap(image, heat_map, alpha=0.6, cmap='Reds', ax=None):
    height = image.shape[0]
    width = image.shape[1]

    # resize heat map
    # heat_map_resized = transform.resize(heat_map, (height, width))
    heat_map_resized = np.zeros((height, width))
    heat_map_width = heat_map.shape[0]
    scale = int(width / heat_map_width)
    for i, j in itertools.product(range(heat_map_width), range(heat_map_width)):
        heat_map_resized[i * scale:(i + 1) * scale, j * scale:(j + 1) * scale] = heat_map[i][j]

    # normalize heat map
    max_value = np.max(heat_map_resized)
    min_value = np.min(heat_map_resized)
    normalized_heat_map = (heat_map_resized - min_value) / (max_value - min_value)

    # display
    if not ax:
        _, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.imshow(255 * normalized_heat_map, alpha=alpha, cmap=cmap)
    # plt.show()


def top_down_view(state, ax, in_view_humans):
    ax.clear()
    robot_radius = 0.3
    human_radius = 0.5
    robot = state.self_state
    humans = state.human_states

    # invert the y axis
    robot.py = - robot.py
    robot.vy = - robot.vy
    robot.theta = -robot.theta
    for human in humans:
        human.py = - human.py
        human.vy = - human.vy

    cmap = plt.cm.get_cmap('hsv', 10)
    robot_color = 'yellow'
    goal_color = 'red'
    arrow_color = 'red'
    arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

    # ax.tick_params(labelsize=12)
    ax.set_xlim(-1, 7)
    ax.set_ylim(-4, 4)
    # ax.set_xlabel('x(m)', fontsize=14)
    # ax.set_ylabel('y(m)', fontsize=14)

    # add robot and its goal
    goal = mlines.Line2D([0], [6], color=goal_color, marker='*', linestyle='None',
                         markersize=15, label='Goal')
    robot_circle = plt.Circle((robot.px, robot.py), robot_radius, fill=True, color=robot_color)
    ax.add_artist(robot_circle)
    ax.add_artist(goal)
    plt.legend([robot_circle, goal], ['Robot', 'Goal'])

    # add humans and their numbers
    human_circles = [plt.Circle((humans[i].px, humans[i].py), human_radius, fill=False)
                     for i in range(len(humans))]
    human_numbers = [plt.text(human_circles[i].center[0]-0.1, human_circles[i].center[1]-0.1, str(i), fontsize=8,
                              color='black') for i in range(len(humans))]
    for human_circle, human_number in zip(human_circles, human_numbers):
        ax.add_artist(human_circle)
        ax.add_artist(human_number)

    # add time annotation
    # time = plt.text(0.4, 0.9, 'Time: {}'.format(0), transform=ax.transAxes)
    # ax.add_artist(time)

    # visualize attention scores
    for i, (human_index, attention) in enumerate(in_view_humans):
        attention_score_text = plt.text(2, 6 - 0.5 * i, 'Human {}: {:.2f}'.format(human_index, attention))
        ax.add_artist(attention_score_text)

    # compute orientation in each step and use arrow to show the direction
    robot_orientation = ((robot.px, robot.py), (robot.px + robot_radius * np.cos(robot.theta),
                                                robot.py + robot_radius * np.sin(robot.theta)))
    orientations = [robot_orientation]
    for i, human in enumerate(humans):
        theta = np.arctan2(human.vy, human.vx)
        orientation = ((human.px, human.py), (human.px + human_radius * np.cos(theta),
                                              human.py + human_radius * np.sin(theta)))
        orientations.append(orientation)
    arrows = [patches.FancyArrowPatch(*orientation, color=arrow_color, arrowstyle=arrow_style)
              for orientation in orientations]
    for arrow in arrows:
        ax.add_artist(arrow)
