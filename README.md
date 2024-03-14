# Qiskit Quantum Espresso Driver
[![DOI](https://zenodo.org/badge/742501230.svg)](https://zenodo.org/doi/10.5281/zenodo.10513424)

## Disclaimer:
**The code is broken**:x:. Since [qiskit_nature](https://qiskit-community.github.io/qiskit-nature/) does indirectly not support complex expansion coefficients (see [this issue](https://github.com/qiskit-community/qiskit-nature/issues/1351)) the provided code does currently not work with qiskit_nature. Since the Kohn-Sham orbitals are expanded in plane-waves with complex expansion coeffiecients, the overlap matrix is complex. As described in [this issue](https://github.com/qiskit-community/qiskit-nature/issues/1351), the overlap matrix is calculated the wrong way and needs to be casted to a real matrix. This is especially relavant if the qiskit_algorithm [NumPyMinimumEigensolver](https://qiskit-community.github.io/qiskit-algorithms/stubs/qiskit_algorithms.NumPyMinimumEigensolver.html) in combination with the [default filter criterion](https://qiskit-community.github.io/qiskit-nature/stubs/qiskit_nature.second_q.problems.ElectronicStructureProblem.html#qiskit_nature.second_q.problems.ElectronicStructureProblem.get_default_filter_criterion) is used. The default filter criterion enforces that the ground state has the correct [number of particles](https://qiskit-community.github.io/qiskit-nature/stubs/qiskit_nature.second_q.properties.ParticleNumber.html#particlenumber), a [magnetization](https://qiskit-community.github.io/qiskit-nature/stubs/qiskit_nature.second_q.properties.Magnetization.html#magnetization) of zero and an [angular momentum](https://qiskit-community.github.io/qiskit-nature/stubs/qiskit_nature.second_q.properties.AngularMomentum.html#angularmomentum) of zero. The angular momentum operator is defined by the overlap matrix between spin-up and spin-down molecular orbitals, i.e. Kohn-Sham orbitals in our case.

Besides that, spin-polarized DFT calculations result in a non-zero ($\sim10^{-8}$) angular momentum of the ground-state. This causes the [NumPyMinimumEigensolver](https://qiskit-community.github.io/qiskit-algorithms/stubs/qiskit_algorithms.NumPyMinimumEigensolver.html) in combination with the [default filter criterion](https://qiskit-community.github.io/qiskit-nature/stubs/qiskit_nature.second_q.problems.ElectronicStructureProblem.html#qiskit_nature.second_q.problems.ElectronicStructureProblem.get_default_filter_criterion) to not find the ground-state. After the above mentionend [issue](https://github.com/qiskit-community/qiskit-nature/issues/1351) is resolved, we will work on fixing spin-polarized DFT calculations.

## Introduction
This repository provides an interface between the [QuantumEspresso](https://www.quantum-espresso.org/) DFT software and [Qiskit](https://www.ibm.com/quantum/qiskit). QuantumEspresso uses a plane-wave basis to express the Kohn-Sham orbitals. We extract the Kohn-Sham orbitals from a QuantumEspresso calculation to construct a many-body hamiltonian in Qiskit by calculating the necessary matrix elements.

We start from a DFT calculation done with [QuantumEspresso](https://www.quantum-espresso.org/) and extract the Kohn-Sham orbitals expressed in the momentum basis. Then, we select a set of orbitals around the. We formulate the electronic structure problem in second quantization with the selected Kohn-Sham orbitals. For this we need to calculate one-electron and two-electron matrix elements $h_{ij}$ and $h_{ijkl}$, respectively. Botrh are calculated in Python using the [Numpy](https://numpy.org/) package. The calculated matrix elements are then used to create a electronic structure problem in [Qiskit](https://www.ibm.com/quantum/qiskit) which is solved with the variational quantum eigensolver (VQE) algorithm using different classical optimizers. For reference we solve the same electronic structure problem in a exact manner via exact diagonalization with the Python package [PySCF](https://pyscf.org/) with their full configuration interaction (FCI) solver. When using VQE to solve the problem we use the [unitary coupled cluster (UCC) ansatz](https://qiskit.org/ecosystem/nature/stubs/qiskit_nature.second_q.circuit.library.UCCSD.html), which is a physically motivated ansatz but results in a deep ansatz circuit.

## Requirements
- Tested Python version: `3.11.5`
- See [requirements.txt](requirements.txt) for Python packages.
- Tested QuantumEspresso version: `7.1` compiled with HDF5 `1.14.0` (HDF5 support is not needed)
- **Only normconserving pseudopotentials can be currently used because then the Kohn-Sham orbitals are orthonormal. For ultrasoft pseudopotentials a generalized eigenvalue problem is solved in DFT and the wavefunctions are only orthonormal w.r.t. to an overlap matrix.**

## Usage
1. Run a QuantumEspresso SCF DFT calculation with the [H2.scf.in](qe_files/H2.scf.in): `pw.x -i H2.scf.in > H2.scf.out`.
2. The [demo_nospin.py](demo_nospin.py) and [demo_spin.py](demo_spin.py) scripts created a qiskit `ElectronicStructureProblem` (see [qe_driver.py](qiskit_nature_qe/qe_driver.py)) by calculating the electron repulsion integrals via pair densities (see [eri_pair_densities.py](qiskit_nature_qe/eri_pair_densities.py)) and by calculating the one-electron part of the Hamiltonian (see [calc_matrix_elements.py](qiskit_nature_qe/calc_matrix_elements.py) and [wfc.py](qiskit_nature_qe/wfc.py)) using the QuantumEspresso DFT output. The ground state of the problem is then found with the numpy ground-state solver in qiskit.

Note that the `QE_Driver` class in [qe_driver.py](qiskit_nature_qe/qe_driver.py) also provides a function `solve_fci` which uses PySCF to perform exact diagonalization of the electronic structure Hamiltonian.

## Formulas
### ERIs via pair densities
We calculate the electron repulsion integrals (ERIs) given as
$$h_{tuvw} = \int \int \frac{\phi^\ast_t(r_1)  \phi^\ast_u(r_2) \phi_v(r_2)  \phi_w(r_1)}{|r_1-r_2|}dr_1dr_2$$
In [eri_pair_densities.py](eri_pair_densities.py) we calculate the ERIs via pair densities $\rho_{tu}(r)=\psi^\ast_t(r)\psi_u(r)$ of real space wavefunctions $\psi_t(r)$. Note that all real space coordinates $r$ are vectors but we ommit the vector arrow for brevity. With this the ERIs $h_{tuvw}$ in the Kohn-Sham basis can be written as
$$h_{tuvw} = 4\pi \sum_{\substack{p, p\neq 0}} \frac{\rho^\ast_{tw}(p) \rho_{uv}(p)}{|p|^2}$$
with $\rho_{tu}(p)=\int\rho_{tu}(r) e^{-ip\cdot r}\mathrm{d}r$ which is the Fourier transform of $\rho_{tu}(r)$. Therefore $\rho_{tu}(p)$ is the convolution between $\psi^\ast_t(p)$ and $\psi_u(p)$: $\rho_{tu}(p)=\psi^\ast_t(p)*\psi_u(p)$.

In  the following we present a derivation of the pair density representation of the ERIs. We start from the real space representation:
$$h_{tuvw}=\int \int \psi^\ast_t(r_1)\psi^\ast_u(r_2)\psi_v(r_2)\psi_w(r_1) \frac{1}{|r_1-r_2|} \mathrm{d}r_1 \mathrm{d}r_2$$
Using the definition of pair densities $\rho_{tu}(r)=\psi^\ast_t(r)\psi_u(r)$ and the Fourier transformation of the Coulomb potential
$$\frac{1}{|r_1-r_2|}=\int\frac{4\pi}{|p|^2}e^{ip\cdot (r_1-r_2)}\mathrm{d}p$$
we find
$$h_{tuvw}=\int \int \rho_{tw}(r_1) \rho_{uv}(r_2) \int\frac{4\pi}{|p|^2}e^{ip\cdot (r_1-r_2)}\mathrm{d}p \mathrm{d}r_1 \mathrm{d}r_2$$
Swapping integrals and using $e^{ip\cdot (r_1-r_2)}=e^{ip\cdot r_1}e^{-ip\cdot r_2}$ yields
$$h_{tuvw}=\int \frac{4\pi}{|p|^2} \int \rho_{tw}(r_1) e^{ip\cdot r_1} \mathrm{d}r_1 \int  \rho_{uv}(r_2) e^{-ip\cdot r_2} \mathrm{d}r_2 \mathrm{d}p $$
Using the Fourier transformation of the pair densities $\rho_{tu}(p)=\int\rho_{tu}(r) e^{-ip\cdot r}\mathrm{d}r$ results in
$$h_{tuvw}=\int \frac{4\pi}{|p|^2} \rho^\ast_{tw}(p) \rho_{uv}(p) \mathrm{d}p$$
For numerical calculation the integral turns into a sum over all momentum vector. To avoid the singularity at $p=0$ we ommit this momentum in the numerical summation:
$$h_{tuvw}=\sum_{\substack{p, p \neq 0}} \frac{4\pi}{|p|^2} \rho^\ast_{tw}(p) \rho_{uv}(p)$$

### One-Electron Integrals
The one-electron integrals
$$h_{tu}=\int \phi^\ast_t(r) \left( -\frac{1}{2} \nabla^2 - \sum_{I} \frac{Z_I}{R_I- r} + \sum_{I\lt J} \frac{Z_I Z_J}{|R_I-R_J|} \right) \phi_u(r)\mathrm{d}r$$
are also calculated in the plane-wave basis. We split the above formula into its three parts:
$$h_{tu}=t_{tu} - u_{tu} +c_{tu}$$
with
$$t_{tu}=\int \phi^\ast_t(r) \left(  -\frac{1}{2} \nabla^2 \right) \phi_u(r)\mathrm{d}r$$
$$u_{tu}=\int \phi^\ast_t(r) \left( \sum_{I} \frac{Z_I}{R_I- r} \right) \phi_u(r)\mathrm{d}r$$
$$c_{tu}=\int \phi^\ast_t(r)  \phi_u(r)\mathrm{d}r \left(  \sum_{I\lt J} \frac{Z_I Z_J}{|R_I-R_J|} \right)=\sum_{I\lt J} \frac{Z_I Z_J}{|R_I-R_J|}$$
where we assumed a orthonormal basis in the last step in defining $c_{tu}$.
In the momentum basis using the Fourier transformation of kinetic energy and the coulomb potential we find:
$$t_{tu}= \frac{1}{2}\int\phi_t(p)p^2 \phi_u(p)\mathrm{d}p$$
$$u_{tu} = \frac{4\pi}{\Omega} \sum_{p,q,p\neq q} \phi_t(p)^\ast \phi_t(q) \frac{1}{|q-p|^2} \sum_I   e^{-i (q-p) \cdot R_I} $$
where we again avoid the singularity at $p=q$ by ommitting these momenta.


**Notes:**
- The calculations of ERIs treat the singularity in the sum of momenta arising from the zero-momentum term by removing this term from the sum. For charge neutral systems this zero-momentum terms cancels with the zero-momentum term of the interaction between electrons and nuclei, and only contributes a constant term. See [Babbush, R. et al. (2018) ‘Low-Depth Quantum Simulation of Materials’, Physical Review X, 8(1).](https://doi.org/10.1103/PhysRevX.8.011044) and [Martin, R.M. (2020) Electronic Structure. Cambridge University Press.](https://doi.org/10.1017/9781108555586) for further information.
- We perform a spin-less DFT calculation. Therefore, the Kohn-Sham orbitals we use from the DFT calculation do not include spin. For the VQE and FCI calculation we use each Kohn-Sham orbital as a spin orbital which can hold two electrons, one with spin-up one with spin-down. Therefore 2 occupied Kohn-Sham orbitals correspond to 4 electrons, each occupying one spin orbital.
- The calculation of ERIs via pair densities is currently only implemented for $\Gamma$-point calculations.
- The implementation of calculating the ERIs via pair densities is inspired by [WEST](https://west-code.org/) and its implementation on [GitHub](https://github.com/west-code-development/West), especially the code in the [compute_eri_vc function](https://github.com/west-code-development/West/blob/master/Wfreq/solve_eri.f90#L327). Publications related when citing WEST: [Large Scale GW Calculations, M. Govoni and G. Galli, J. Chem. Theory Comput. 11, 2680 (2015)](https://pubs.acs.org/doi/10.1021/ct500958p) and [GPU Acceleration of Large-Scale Full-Frequency GW Calculations, V. Yu and M. Govoni, J. Chem. Theory Comput. 18, 4690 (2022)](https://pubs.acs.org/doi/10.1021/acs.jctc.2c00241). We note that, although inspiration was taken from the WEST implementation, no code from WEST was used.

## Authors
- [Erik Schultheis](mailto:erik.schultheis@dlr.de)

## Acknowledgements
This software was developed within the [QuantiCoM project](https://qci.dlr.de/quanticom/) which is part of the [DLR Quantum Computing Initiative (QCI)](https://qci.dlr.de/). We acknowledge the support of the DLR Quantum Computing Initiative (QCI).
