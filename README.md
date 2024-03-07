# Qiskit Quantum Espresso Driver
[![DOI](https://zenodo.org/badge/742501230.svg)](https://zenodo.org/doi/10.5281/zenodo.10513424)

## Introduction
This repository provides an interface between the [QuantumEspresso](https://www.quantum-espresso.org/) DFT software and [Qiskit](https://www.ibm.com/quantum/qiskit). QuantumEspresso uses a plane-wave basis to express the Kohn-Sham orbitals. We extract the Kohn-Sham orbitals from a QuantumEspresso calculation to construct a many-body hamiltonian in Qiskit by calculating the necessary matrix elements.

We start from a DFT calculation done with [QuantumEspresso](https://www.quantum-espresso.org/) and extract the Kohn-Sham orbitals expressed in the momentum basis. Then, we select a set of orbitals around the. We formulate the electronic structure problem in second quantization with the selected Kohn-Sham orbitals. For this we need to calculate one-electron and two-electron matrix elements $h_{ij}$ and $h_{ijkl}$, respectively. Botrh are calculated in Python using the [Numpy](https://numpy.org/) package. The calculated matrix elements are then used to create a electronic structure problem in [Qiskit](https://www.ibm.com/quantum/qiskit) which is solved with the variational quantum eigensolver (VQE) algorithm using different classical optimizers. For reference we solve the same electronic structure problem in a exact manner via exact diagonalization with the Python package [PySCF](https://pyscf.org/) with their full configuration interaction (FCI) solver. When using VQE to solve the problem we use the [unitary coupled cluster (UCC) ansatz](https://qiskit.org/ecosystem/nature/stubs/qiskit_nature.second_q.circuit.library.UCCSD.html), which is a physically motivated ansatz but results in a deep ansatz circuit.

## Requirements
- Tested Python version: `3.11.5`
- See [requirements.txt](requirements.txt) for Python packages.
- Tested QuantumEspresso version: `7.1` compiled with HDF5 `1.14.0`
- **Only normconserving pseudopotentials can be currently used because then the Kohn-Sham orbitals are orthonormal. For ultrasoft pseudopotentials a generalized eigenvalue problem is solved in DFT and the wavefunctions are only orthonormal w.r.t. to overlap matrix.**

## Usage
1. Run a QuantumEspresso SCF DFT calculation with the [H2.scf.in](qe_files/H2.scf.in): `pw.x -i H2.scf.in > H2.scf.out`. We ran the calculation twice, one with Quantum Espresso that outputs hdf5 files and one that output dat files.
2. The [main.py](main.py) script calculates the electron repulsion integrals via pair densities (see [eri_pair_densities.py](eri_pair_densities.py)) and calculates the one-electron part of the Hamiltonian (see [calc_matrix_elements.py](calc_matrix_elements.py) and [wfc.py](wfc.py)) using the QuantumEspresso DFT output. With this we define a Hamiltonian (see [hamiltonian.py](hamiltonian.py)), a Qiskit electronic structure problem and a FCI solver. The ground state of the Hamiltonian is then found with the VQE algorithm in Qiskit and the FCI solver in PySCF.

## Formulas
### ERIs via pair densities
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


**Notes:**
- The calculations of ERIs treat the singularity in the sum of momenta arising from the zero-momentum term by removing this term from the sum. For charge neutral systems this zero-momentum terms cancels with the zero-momentum term of the interaction between electrons and nuclei, and only contributes a constant term. See [Babbush, R. et al. (2018) ‘Low-Depth Quantum Simulation of Materials’, Physical Review X, 8(1).](https://doi.org/10.1103/PhysRevX.8.011044) and [Martin, R.M. (2020) Electronic Structure. Cambridge University Press.](https://doi.org/10.1017/9781108555586) for further information.
- We perform a spin-less DFT calculation. Therefore, the Kohn-Sham orbitals we use from the DFT calculation do not include spin. For the VQE and FCI calculation we use each Kohn-Sham orbital as a spin orbital which can hold two electrons, one with spin-up one with spin-down. Therefore 2 occupied Kohn-Sham orbitals correspond to 4 electrons, each occupying one spin orbital.
- The calculation of ERIs via pair densities is currently only implemented for $\Gamma$-point calculations.
- The implementation of calculating the ERIs via pair densities is inspired by [WEST](https://west-code.org/) and its implementation on [GitHub](https://github.com/west-code-development/West), especially the code in the [compute_eri_vc function](https://github.com/west-code-development/West/blob/master/Wfreq/solve_eri.f90#L327). Publications related when citing WEST: [Large Scale GW Calculations, M. Govoni and G. Galli, J. Chem. Theory Comput. 11, 2680 (2015)](https://pubs.acs.org/doi/10.1021/ct500958p) and [GPU Acceleration of Large-Scale Full-Frequency GW Calculations, V. Yu and M. Govoni, J. Chem. Theory Comput. 18, 4690 (2022)](https://pubs.acs.org/doi/10.1021/acs.jctc.2c00241). We note that, although inspiration was taken from the WEST implementation, no code from WEST was used.

## Authors
- [Erik Schultheis](mailto:erik.schultheis@dlr.de)

## Acknowledgements
This software was developed within the [QuantiCoM project](https://qci.dlr.de/quanticom/) which is part of the [DLR Quantum Computing Initiative (QCI)](https://qci.dlr.de/). We acknowledge the support of the DLR Quantum Computing Initiative (QCI).
