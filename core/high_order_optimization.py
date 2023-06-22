from typing import Callable, List

import numpy as np
import scipy.constants
from math import exp, floor, sqrt

from numpy.linalg import LinAlgError


def gauss_newton(residuals: List[Callable[[np.ndarray], float]],
                 gradients: List[Callable[[np.ndarray], np.ndarray]],
                 x0: np.ndarray,
                 termination_condition: Callable[[Callable, List[np.ndarray]], bool]):

    f = lambda x: sum((r(x) ** 2 for r in residuals))

    points = [x0]
    current = x0
    while not termination_condition(f, points):
        jac = np.array([grad(current) for grad in gradients])
        try:
            current = current - np.linalg.inv(np.transpose(jac) @  jac) @ np.transpose(jac) @ np.array([r(current) for r in residuals])
        except LinAlgError:
            # jacobian is zero, so there is a actual local minimum
            return points

        points.append(current)

    return points


def gauss_newton_with_approx_grad(residuals: List[Callable[[np.ndarray], float]],
                                  x0: np.ndarray,
                                  termination_condition: Callable[[Callable, List[np.ndarray]], bool]):
    n = x0.shape[0]

    def gradient(n, f: Callable[[np.ndarray], float]):
        eps = 0.001

        def standard_vector(n, i):
            vec = np.zeros(n)
            vec[i] = 1
            return vec

        return lambda x: np.array([(f(x + eps * standard_vector(n, i)) - f(x)) / eps for i in range(n)])

    gradients = [gradient(n, r) for r in residuals]
    return gauss_newton(residuals, gradients, x0, termination_condition)
