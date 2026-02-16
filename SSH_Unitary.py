#!/usr/bin/env python3
"""
Evolve an inhomogeneous initial state (left at vacuum, right at Groundstate)
under the Hamiltonian and plot q/J profiles. No analytic overlay.
"""
import sys
import time
import argparse
import os
import numpy as np
import scipy.sparse.linalg as spla
import scipy.linalg as la
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Directory check
def ensure_directory_exists(home):
    if not os.path.exists(home):
        os.makedirs(home)


# ============================================================
#  1. Hamiltonian SSH
# ============================================================


def Hamiltonian_SSH(L, J1, J2, Type):
    unit_cells = L
    # Define the submatrices
    S = np.array([[0,  J1], 
                  [ J1 , 0]])
    
    T = np.array([[0, 0], [ J2, 0]])

    # Create sparse diagonal matrices   
    
    TN_up = np.eye(unit , k=1)
    TN_down = np.eye(unit , k=-1)
    IN = np.eye(unit )

    # Use kron (Kronecker product) to create the tensor products
    IN_S = np.kron(IN, S)
    TN_T_up = np.kron(TN_up, T)
    TN_T_down = np.kron(TN_down, T.T)


    # Assemble the Hamiltonian
    # o= OBC
    if Type == 0:
        H = IN_S + TN_T_up + TN_T_down
    # PBC condition
    else:
        H= IN_S + TN_T_up + TN_T_down
        H[2*unit-2:,:2]= T
        H[:2,2*unit-2:]= T.T

    return  H



#  ============================================================
#  2.Von Neuman Entropy
# ============================================================
def xlogx(x):
    x = np.asarray(x)
    out = np.zeros_like(x, dtype=float)
    mask = x > 0
    out[mask] = -x[mask] * np.log(x[mask]) -(1-x[mask]) * np.log(1-x[mask])
    return out

def Bipartite_EE(rho_0, H, T):
    N= rho_0.shape[0]//2
    U_conj= LA.expm(1j*H.T*T)
    U_T= LA.expm(-1j*H.T*T)
    rho_t= U_conj@rho_0@U_T
    #Right side
    EIGen, _ = LA.eig(rho_t[N:,N:])
    #Left side
    EIGen, _ = LA.eig(rho_t[:N,:N])
    rho_log_rho = np.sum(xlogx(EIGen))
    return np.real(rho_log_rho)

#  ============================================================
#  3. Ground state 
# ============================================================
# TODO : Obtain the ground state of the SSH model for PBC and translate it into site coordinates
'''
H= -\sum_{j=1}^N(J_1 c_{A,j}^\dagger c_{B,j+1}  J_2 c_{B,j+1}^\dagger c_{A,j+2}+ h.c.)
Spectrum := \epsilon(k)= \pm \sqrt{d_1^2+d_2^2}
d_1= J_1+J_2 \cos(k)
d_2 = J_2 \sin(k)
Eigenvectors
u(k)= 1/(\sqrt{2}|\epsilon(k)|) [1, -e^{i\alpha}]
e^{i\alpha}= (J_1+J_2e^{ik})/(|\epsilon(k)|)
Ground state
|GS >= \Pi_{k<0} (\sum_{\beta=A,B} u(k)_\beta c^\dagger_{\alpha,k})
Where 
c_{\beta,k}= 1/\sqrt{N} \sum_{j=1}^N e^{ikj} c_{\beta,j}


def GS_SSH(L, J1,J2):

    return |GS><GS| covariance matrix
'''
#  ============================================================
#  3. Initial State
# ============================================================


def Gamma_0(L, t1, t2,  Type):
    '''
    Compute the initial inhomogeneous state
    |0>= |0>_L x|GS>_R
    |GS>_R := ground state of a ssh chain with N/2 sites
    |0>_L= \Pi_{i=-N/2}^{0} |0>
'''
    rho_0L = np.zeros(L, dtype=complex)
    rho_0R = GS_SSH(L, t1, t2)
    rho_0= la.block_diag(rho_0L), rho_0R)
    return rho_0



def main():
    return

if __name__ == "__main__":
    main()