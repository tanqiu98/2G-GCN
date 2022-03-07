from itertools import groupby
from typing import Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def create_label_bar(label_ids: list, bar_height: int = 30, bar_width: int = 5):
    """Create a numpy array that represents the video segmentation.

    Arguments:
        label_ids - Video segmentation represented as a list of label IDs.
        bar_height - Height (or number of rows) of the returned numpy array.
        bar_width - Width of individual segments in the desired plot. The returned array has len(label_ids) * bar_width
            columns.
    Returns:
        A numpy array of shape (bar_height, len(label_ids) * bar_width) containing a representation of video
        segmentation.
    """
    label_bar = np.empty([bar_height, bar_width * len(label_ids)])
    for i, label in enumerate(label_ids):
        label_bar[:, i * bar_width:(i + 1) * bar_width] = label
    return label_bar


def determine_xlabels_and_xticks_positions(labels: list, bar_width: int):
    """Simplify segmentation labelling in case of frame-wise segmentation.

    From a list of frame-level labels, extract the unique labels and determine x-axis positions to plot them.

    Arguments:
        labels - Video segmentation as a list of labels.
        bar_width - Width of a single segment bar in the expected plot.
    Returns:
        Two lists. The first one contains the unique labels in labels, and the second contain the x-axis position to
        place the labels in the final segmentation plot.
    """
    unique_labels, xticks, cumulative_length = [], [], 0
    for k, v in groupby(labels):
        unique_labels.append(k)
        num_frames = len(list(v))
        if xticks:
            xticks.append(cumulative_length + (num_frames // 3))
        else:
            xticks.append(num_frames // 3)
        xticks[-1] *= bar_width
        cumulative_length += num_frames
    return unique_labels, xticks


def plot_segmentation(target: list, *output, class_id_to_label: Dict[int, str], save_file: str = None,
                      bar_height: int = 30, bar_width: int = 2000, xlabels_type: str = 'label'):
    """Plot ground-truth and predicted segmentations.

    Arguments:
        target - A list containing the ground-truth label IDs.
        output - Output predictions to compare against the target. Each element is a list containing the predicted
            labels IDs.
        class_id_to_label - Dictionary mapping label IDs to label names.
        save_file - Optional file to write out segmentation plot.
        bar_height - Height of the bars drawn.
        bar_width - Width of the bars drawn.
        xlabels_type - One of 'label', 'id', or None.
    """
    bar_width = int(bar_width / len(target))
    num_classes = len(class_id_to_label)
    plt.figure(figsize=(num_classes, 1))
    grid_spec = mpl.gridspec.GridSpec(1 + len(output), 1)
    grid_spec.update(wspace=0.5, hspace=0.01)
    for plt_idx, label_ids in enumerate([target, *output]):
        ax = plt.subplot(grid_spec[plt_idx])
        label_bar = create_label_bar(label_ids, bar_height=bar_height, bar_width=bar_width)
        label_bar = label_bar.astype(np.int8)
        plt.imshow(label_bar, cmap=plt.get_cmap('tab20'), vmin=0, vmax=num_classes - 1)
        ax.tick_params(axis='both', which='both', length=0)
        xlabels, xticks = determine_xlabels_and_xticks_positions(label_ids, bar_width)
        ax.set_xticks(xticks)
        fontsize = 'small'
        if xlabels_type == 'labels':
            xlabels, fontsize = [class_id_to_label[label_id] for label_id in xlabels], 'x-small'
        elif xlabels_type == 'id':
            xlabels = [str(label_id) for label_id in xlabels]
        else:
            xlabels = []
        ax.set_xticklabels(xlabels, fontsize=fontsize, horizontalalignment='left')
        ax.set_yticklabels([])
    if save_file is not None:
        plt.savefig(save_file, bbox_inches='tight', pad_inches=0, transparent=True)
    else:
        plt.show()
    plt.close()
