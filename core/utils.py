class CallsCount:
    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.fn(*args, **kwargs)


def fn_sum(*args):
    return lambda x: sum((f(x) for f in args))
