import cv2 as cv
import numpy as np


def draw_line(img, pt1, pt2, color, thickness=1, style='dotted', gap=10):
    """More general routine, compared to opencv's line, to draw a line in an image."""
    if style == 'original':
        cv.line(img, pt1, pt2, color=color, thickness=thickness)
        return
    distance = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    points = []
    for i in np.arange(0, distance, gap):
        r = i / distance
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        point = x, y
        points.append(point)
    if style == 'dotted':
        for point in points:
            cv.circle(img, point, radius=thickness, color=color, thickness=-1)
    elif style == 'rectangled':
        for i, (start_point, end_point) in enumerate(zip(points[:-1], points[1:])):
            if i % 2:
                cv.line(img, start_point, end_point, color=color, thickness=thickness)
    else:
        raise ValueError(f'Unknown style {style}. Please choose one of: original, dotted, or rectangled.')


def draw_keypoints(img, keypoints, connections=None, color: tuple = (0, 0, 0), dotted: bool = False):
    """Draw a set of keypoints on an image.

    Arguments:
        img: uint8 tensor of shape (height, width, 3) containing image pixel values.
        keypoints: Tensor of shape (num_keypoints, 2) containing the x and y coordinates of each keypoint. Any
            coordinates with a value of zero are considered missing and are ignored.
        connections: List of 2-tuples specifying pairs of keypoints that are connected. If None, only draw the
            keypoints without connecting lines.
        color: Colour of the keypoints and potential lines drawn between them. It should be specified as a BGR triplet.
        dotted: Whether to draw dotted connections between the keypoints or not. Only meaningful if the connections
            are specified.
    """
    for x, y in keypoints:
        if 0 in (x, y):
            continue
        center = int(round(x)), int(round(y))
        cv.circle(img, center=center, radius=4, color=color, thickness=-1)
    if connections is not None:
        for keypoint_id1, keypoint_id2 in connections:
            x1, y1 = keypoints[keypoint_id1]
            x2, y2 = keypoints[keypoint_id2]
            if 0 in (x1, y1, x2, y2):
                continue
            pt1 = int(round(x1)), int(round(y1))
            pt2 = int(round(x2)), int(round(y2))
            line_style = 'dotted' if dotted else 'original'
            draw_line(img, pt1=pt1, pt2=pt2, color=color, thickness=2, style=line_style, gap=5)
