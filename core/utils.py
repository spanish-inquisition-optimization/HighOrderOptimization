from typing import Callable

import numpy as np
from numpy import newaxis


class CallsCount:
    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.fn(*args, **kwargs)


def fn_sum(*args):
    return lambda x: sum((f(x) for f in args))


def partially_vectorize(f, f_input_dims):
    def vectorized_f(t):
        if len(t.shape) == f_input_dims:
            return f(t)
        else:
            splitted = np.split(t, t.shape[-1], axis=-1)
            slices_along_last_axis = [splitted[i][..., 0] for i in range(t.shape[-1])]
            return np.concatenate([vectorized_f(s)[..., newaxis] for s in slices_along_last_axis], axis=-1)

    return vectorized_f


def supports_argument(f, smple_arg):
    try:
        f(smple_arg)
        return True
    except:
        return False


class AutoVectorizedFunction:
    def __init__(self, f, f_input_dims=None):
        self.f = f
        self.f_input_dims = f_input_dims

    def __call__(self, t):
        try:
            return self.f(t)
        except:
            assert self.f_input_dims is not None
            return partially_vectorize(self.f, self.f_input_dims)(t)


def coordinate_vector_like(coordinate_index: int, reference: np.ndarray):
    res = np.zeros_like(reference)
    res[coordinate_index] = 1
    return res


NUMERIC_GRADIENT_COMPUTING_PRECISION = 1e-5


def symmetric_gradient_computer(f: Callable[[np.ndarray], float], h: float = NUMERIC_GRADIENT_COMPUTING_PRECISION):
    def computer(x):
        # This trick only works on functions defined
        # in terms of scalar (or dimension-independent) np operations (aka ufuncs) which can thus be vectorized…
        # return (f(x[:, newaxis] + h * np.eye(n)) - f(x[:, newaxis] - h * np.eye(n))) / (2 * h)

        return np.array([
            (f(x + h * coordinate_vector_like(i, x)) - f(x - h * coordinate_vector_like(i, x))) / (2 * h)
            for i in range(x.size)
        ])

    return computer


def n_calls_mocker(f):
    def mocked(*args, **kwargs):
        mocked.n_calls += 1
        return f(*args, **kwargs)

    mocked.n_calls = 0
    return mocked


def smoothly_criminally_call(f, *args, **kwargs):
    res = f(*args, **kwargs)
    if hasattr(f, "n_calls"):
        f.n_calls -= 1
    return res


if __name__ == '__main__':
    @n_calls_mocker
    def f(x):
        print(f"Hello, {x}!")


    for i in range(5):
        f(i)

    smoothly_criminally_call(f, 12412)

    assert f.n_calls == 5
