import torch

##### linear conjugate gradient with Tikhonov regularization #####
def cg(
        A,
        b,
        x0: torch.Tensor = None,
        max_iter: int = None,
        tol: float = 1e-3,
        regularisation = 0.0
):
        def apply_A(x):
                return A*x + regularisation * x
        j = 0
        r = b - apply_A(x0)
        d = r
        delta = torch.sum(torch.conj(r) * r)
        delta_0 = delta
        for i in range(max_iter):
                if torch.abs(delta) < torch.abs(tol **2 * delta_0):
                        return x0, j
                q = apply_A(d)
                alpha = delta / torch.sum(torch.conj(d)* q)
                x0 = x0 + alpha * d
                if i % 50 == 0:
                        r = b - apply_A(x0)
                else:
                        r = r - alpha * q
                delta_old = delta
                delta = torch.sum(torch.conj(r) * r)
                beta = delta / delta_old
                d = r + beta * d
                j += 1

        return x0, j


##### non linear conjugate gradient with Tikhonov regularization and different beta methods #####


def cinner(x, y):
    return torch.sum(torch.conj(x) * y)  # Hermitian inner product

def tikhonov_grad(x, A, b, lam):
    # residual A x - b
    residual = A * x - b
    # f'(x) gradient
    g = torch.conj(A) * residual + lam**2 * x
    return g

def nonlinear_cg(D, b, lam, x0=None, max_iter=1000, tol=1e-6, reset=30, beta_ = "FR"):
    # init D is a diagonal operator in 
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    g = tikhonov_grad(x, D, b, lam)
    r = -g
    d = r.clone()
    rr = cinner(r, r)
    k = 0  # restart counter

    for i in range(max_iter):
        # stop if gradient small
        if torch.linalg.norm(g).real <= tol:
            break

        # exact line search: alpha = - Re(<g,d>) / Re(<d, H d>)
        # H d = A^H(A d) + lam^2 d
        Ad = D * d
        Hd = torch.conj(D) * Ad + lam**2 * d

        numer = torch.real(cinner(g, d))
        denom = torch.real(cinner(d, Hd))
        if denom <= 1e-30:
            break

        alpha = - numer / denom

        # update x
        x = x + alpha * d

        # new grad, residual, r
        g_new = tikhonov_grad(x, D, b, lam)
        r_new = -g_new
        rr_new = cinner(r_new, r_new)

        # new search direction 
        if "FR" == beta_:
            # Fletcher–Reeves
            beta = torch.real(rr_new / rr)
        elif "PR" == beta_:
            # Polak–Ribiere
            beta = cinner(r_new, (r_new - r) / rr)
        elif "HS" == beta_:
            # Hestenes-Stiefel
            y = r_new - r
            beta = cinner(r_new, y) / cinner(d, y)
        elif "DY" == beta_:
            # Dai-Yuan
            y = r_new - r
            beta = rr_new / cinner(d, y)
        else:
            raise ValueError(f"Unknown beta method {beta_}")
        
        # new direction
        d = r_new + beta * d

        # restart conditions: k rounds or not a descent
        if k == reset or torch.real(cinner(r_new, d)) <= 0:
            d = r_new.clone()
            k = 0
        else:
            k += 1

        # roll
        r, g, rr = r_new, g_new, rr_new

    return x, i+1