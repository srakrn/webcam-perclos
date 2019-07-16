import cv2
import numpy as np


def find_eye_position(eye):
    """Finds an eye position
    
    Parameters:
    eye: a OpenCV-based NumPy array of the eye image
    
    Returns:
    (upper_coefs, lower_coefs): a tuple of tuples containing the
    polynomial-fitted coefficients for upper and power eyelids
    respectively
    """
    chan_diff = (eye[:, :, 2] - eye[:, :, 0]) * 3
    _, chan_th = cv2.threshold(chan_diff, 100, 255, cv2.THRESH_BINARY)
    chan_th = cv2.blur(chan_th, (6, 6))
    _, chan_th = cv2.threshold(chan_th, 100, 255, cv2.THRESH_BINARY)

    xt = np.arange(0, chan_th.shape[1])
    yt = np.argmax(chan_th, axis=0)
    upper_data_points = np.vstack([xt, yt]).T
    upper_data_points = upper_data_points[upper_data_points[:, 1] != 0]

    reversed_chan_th = chan_th[::-1, :]
    xb = np.arange(0, reversed_chan_th.shape[1])
    yb = chan_th.shape[0] - np.argmax(reversed_chan_th, axis=0)
    lower_data_points = np.vstack([xb, yb]).T
    lower_data_points = lower_data_points[lower_data_points[:, 1] != chan_th.shape[0]]

    if len(upper_data_points[:, 0]) > 0 and len(lower_data_points[:, 0]) > 0:
        at, bt, ct = np.polyfit(upper_data_points[:, 0], upper_data_points[:, 1], deg=2)
        ab, bb, cb = np.polyfit(lower_data_points[:, 0], lower_data_points[:, 1], deg=2)
        return ((at, bt, ct), (ab, bb, cb))
    return (None, None)


def solve_second_degree(a, b):
    """Solves a second degree polynomial
    
    Parameters:
    a (tuple): The first polynomial coefficients of the form
    a[0](x**2) + a[1](x) + a[2]x
    b (tuple): The second polynomial coefficients of the form
    b[0](x**2) + b[1](x) + b[2]x
    """
    at, bt, ct = a
    ab, bb, cb = b
    a = at - ab
    b = bt - bb
    c = ct - cb
    x1 = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    x2 = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    return x1, x2


def move_second_degree(coefs, dx, dy):
    """Moves a parabola (ax^2+bx+c)'s vertex (m, n) to (m+dx, n+dy)

    Parameters:
    coefs (tuple): The coefficient (a, b, c) of the parabola ax^2+bx+c
    dx (int, float): The distant of moving the vertex along +x axis
    dy (int, float): The distant of moving the vertex along +y axis

    Returns:
    Tuple (a, b, c) as the new coefficients
    """
    a, b, c = coefs
    h = -b / (2 * a)
    k = (4 * a * c - b ** 2) / (4 * a)
    h += dx
    k += dy
    return (a, -2 * a * h, a * h ** 2 + k)

