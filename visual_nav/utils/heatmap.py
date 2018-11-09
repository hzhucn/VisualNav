import itertools

import matplotlib.pyplot as plt
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