import numpy as np


def ax(x: np.array) -> np.array:
    r = np.zeros_like(x)

    N = x.shape[0]
    m = np.int(np.sqrt(N))

    # main diagonal
    r[:] += 20*x[:]

    # upper diagonals
    r[:-1] += -4*x[1:]
    r[:1-m] += -1*x[m-1:]
    r[:-m] += -4*x[m:]
    r[:-m-1] += -1*x[m+1:]

    # lower diagonals
    r[1:] += -4*x[:-1]

    r[m-1:] += -1*x[:1-m]
    r[m:] += -4*x[:-m]
    r[m+1:] += -1*x[:-m-1]

    # fix non diagonals
    r[m-1:-1:m] += 4*x[m::m]
    r[:1-m:m] += 1*x[m-1::m]
    r[m-1:-m-1:m] += 1*x[2*m::m]

    r[m::m] += 4*x[m-1:-1:m]

    r[m-1::m] += 1*x[:1-m:m]
    r[2*m::m] += 1*x[m-1:-m-1:m]

    return r


def ax_eig_min(n: int) -> float:
    N = np.sqrt(n) + 1
    return 18 - 16*np.cos(np.pi/N) - 2*np.cos(2*np.pi/N)


def ax_eig_max(n: int) -> float:
    N = np.sqrt(n) + 1
    return 18 - 16*np.cos((N-1)*np.pi/N) - 2*np.cos((N-1)*2*np.pi/N)


def ax_conditional_number(n: int) -> float:
    return ax_eig_max(n)/ax_eig_min(n)


def conj_grad_err(n: int, k: int, e0: float = 1.0) -> float:
    ka = np.sqrt(ax_conditional_number(n))
    return 2 * np.power((ka-1)/(ka+1), k) * e0


def conj_grad(A, b, tol, k_max):
    """
    Solves Ax = b
    """
    x = np.zeros_like(b)
    r = b - A(x)
    p = np.copy(r)
    k = 0
    while k < k_max:
        if np.linalg.norm(p, np.inf) < tol:
            return x, k
        Ap = A(p)
        a = (r.T @ r) / (p.T @ Ap)
        x += a * p
        r1 = r - a * Ap
        b = (r1.T @ r1) / (r.T @ r)
        p = r1 + b * p

        k += 1
        r = r1

    return x, k
