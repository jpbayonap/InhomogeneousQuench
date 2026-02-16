import numpy as np
import matplotlib.pyplot as plt

def f_zeta(zeta, J=1.0):
    a = zeta / (2.0 * J)
    a_clipped = np.clip(a, -1.0, 1.0)
    alpha = np.arcsin(a_clipped)
    return (J**2 / (2.0 * np.pi)) * (np.sin(2.0 * alpha) - 2.0 * alpha - np.pi)

J = 1.0
ell = 5.0
N = 2000

zeta = np.linspace(-ell, ell, N)
f = f_zeta(zeta, J=J)

plt.figure(figsize=(8, 4.5))
plt.plot(zeta, f)
plt.axvline(-2*J, ls="--")
plt.axvline( 2*J, ls="--")
plt.xlabel(r"$\zeta$")
plt.ylabel(r"$\frac{J^2}{2\pi}\,[\sin(2\alpha)-2\alpha-\pi]$")
plt.title(rf"$J={J}$,  $\alpha=\arcsin(\zeta/2J)$ (clipped)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("f_zeta.png", dpi=200)
