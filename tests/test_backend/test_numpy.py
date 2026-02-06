import numpy as np

from spekk import ops
from spekk.ops._backend.included_backends.numpy import _scan


def test_scan():
    def f(carry, xs):
        return carry + np.sum(xs), [carry + x for x in xs]

    result = _scan(f, 0, (np.array([0, 1, 2]), np.array([10, 11, 12])))
    print(result)


if __name__ == "__main__":
    test_scan()
