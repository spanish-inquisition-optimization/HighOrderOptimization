from typing import Callable, List
from abc import ABC, abstractmethod

import numpy as np
import scipy.constants
from math import exp, floor, sqrt

from numpy import newaxis
from numpy.linalg import LinAlgError

from utils import symmetric_hessian_computer


class InverseHessianController(ABC):
    def __init__(self):
        self.inv_hessian = None

    def absorb_initial_approximation(self, initial_hessian):
        self.inv_hessian = np.linalg.inv(initial_hessian)

    @abstractmethod
    def approximate_inverse_hessian(self, new_point, new_gradient):  # Supposed to return a positive definite matrix
        pass


class BFGSInverseHessianController(InverseHessianController):
    def __init__(self):
        super().__init__()
        self.old_gradient = None
        self.old_point = None

    def approximate_inverse_hessian(self, point, gradient):
        if self.old_point is not None:
            # Update hessian approximation
            s = point - self.old_point
            y = gradient - self.old_gradient
            rho = 1 / (s @ y)

            # Expand equation and put parentheses properly
            # to avoid matrix-matrix operations which are O(n^3)
            def pre_multiply(m):
                return s[:, newaxis] @ (y @ m)

            def post_multiply(m):
                return (m @ y[:, newaxis]) @ s

            self.inv_hessian += \
                - rho * post_multiply(self.inv_hessian) \
                - rho * pre_multiply(self.inv_hessian) \
                + rho ** 2 * post_multiply(pre_multiply(self.inv_hessian)) \
                + rho * s[:, newaxis] @ s

        self.old_gradient = gradient
        self.old_point = point
        return self.inv_hessian


class LBFGSInverseHessianController(InverseHessianController):
    def approximate_inverse_hessian(self, point, gradient):
        if self.old_point is not None:
            # Update hessian approximation
            pass

        self.old_gradient = gradient
        self.old_point = point
        return self.inv_hessian


class GivenInverseHessianController(InverseHessianController):
    def __init__(self, computer):
        super().__init__()
        self.hessian_computer = computer

    def approximate_inverse_hessian(self, new_point, new_gradient):
        return self.hessian_computer(new_point)

    @classmethod
    def numerically_computing(cls, f):
        return cls(symmetric_hessian_computer(f))


def newton_optimize():
    pass


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
            current = current - np.linalg.inv(np.transpose(jac) @ jac) @ np.transpose(jac) @ np.array(
                [r(current) for r in residuals])
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
