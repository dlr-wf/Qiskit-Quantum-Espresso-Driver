# Qiskit Quantum Espresso Driver
[![DOI](https://zenodo.org/badge/742501230.svg)](https://zenodo.org/doi/10.5281/zenodo.10513424)

## Introduction
This repository provides an interface between the [QuantumEspresso](https://www.quantum-espresso.org/) DFT software and [Qiskit](https://www.ibm.com/quantum/qiskit). QuantumEspresso uses a plane-wave basis to express the Kohn-Sham orbitals. We extract the Kohn-Sham orbitals from a QuantumEspresso calculation to construct a many-body hamiltonian in Qiskit by calculating the necessary matrix elements.

We start from a DFT calculation done with [QuantumEspresso](https://www.quantum-espresso.org/) and extract the Kohn-Sham orbitals expressed in the momentum basis. Then, we select a set of orbitals around the. We formulate the electronic structure problem in second quantization with the selected Kohn-Sham orbitals. For this we need to calculate one-electron and two-electron matrix elements $h_{ij}$ and $h_{ijkl}$, respectively. The former is calculated in Python using the [Numpy](https://numpy.org/) package to perform matrix multiplications. The latter is more computatially expensive and an implementation using CUDA and C++ as well as an implementaion using Rust is provided. The calculated matrix elements are then used to create a electronic structure problem in [Qiskit](https://www.ibm.com/quantum/qiskit) which is solved with the variational quantum eigensolver (VQE) algorithm using different classical optimizers. For reference we solve the same electronic structure problem in a exact manner via exact diagonalization with the Python package [PySCF](https://pyscf.org/) with their full configuration interaction (FCI) solver. When using VQE to solve the problem we use the [unitary coupled cluster (UCC) ansatz](https://qiskit.org/ecosystem/nature/stubs/qiskit_nature.second_q.circuit.library.UCCSD.html), which is a physically motivated ansatz but results in a deep ansatz circuit.

## Requirements
- Tested Python version: `3.11.5`
- See [requirements.txt](requirements.txt) for Python packages.
- Tested `nvcc` version:  
`Cuda compilation tools, release 11.5, V11.5.50`  
`Build cuda_11.5.r11.5/compiler.30411180_0`
- Tested Rust version:  
`cargo version: cargo 1.70.0 (ec8a8a0ca 2023-04-25)`  
`rustc --version: rustc 1.70.0 (90c541806 2023-05-31)`
- Tested QuantumEspresso version: `7.1` compiled with HDF5 `1.14.0`
- **Only normconserving pseudopotential can be used because then the Kohn-Sham orbitals are orthonormal. For ultrasoft pseudopotential a generalized eigenvalue problem is solved in DFT and the wavefunctions are only orthonormal w.r.t. to overlap matrix.**

## Usage
1. Run a QuantumEspresso SCF DFT calculation with the [H2.scf.in](qe_files/H2.scf.in): `pw.x -i H2.scf.in > H2.scf.out`. We ran the calculation twice, one with Quantum Espresso that outputs hdf5 files and one that output dat files.
2. Save the momentum vectors $p$, Kohn-Sham coefficients $c_{i,p}$ and occupations with the [wfc_to_txt.py](wfc_to_txt.py) script to files. [wfc_to_txt.py](wfc_to_txt.py) can read both the dat and the hdf5 files.
3. These text files are used for the calculation of electron repulsion integrals with CUDA (see the [cuda folder](cuda_eri/)). The calculation is done in the [cuda source file](cuda_eri/eri_sym.cu) and can be compiled and executed manually with the `nvcc` compiler. The program writes the electron repulsion integrals in the Khon-Sham basis into a text output file. See an example of the [program output](eri/cuda_output.txt) and [electron repulsion integrals](eri/eri_sym_cu_0_4.txt) in the [eri](eri/) folder. More information in respective [README.md](cuda_eri/README.md).
4. Instead of using CUDA we also provide a RUST implementation for calculation the electron repulsion integrals in the [rust_eri](rust_eri/) folder. Instead of reading the momentum vectors $p$, coefficients $c_{i,p}$ and occupations from a text file, they are read from the Quantum Espresso xml and hdf5 output files. See an example of the [program output](eri/rust_output.txt) and [electron repulsion integrals](eri/eri_sym_rs_tuvw_0_4_f64.txt) in the [eri](eri/) folder. More information in respective [README.md](rust_eri/README.md)
5. The [main.py](main.py) script loads the electron repulsion integrals from the text file and uses the QuantumEspresso DFT output to calculate the one-electron part of the Hamiltonian (see [calc_matrix_elements.py](calc_matrix_elements.py) and [wfc.py](wfc.py)). With this we define a Hamiltonian (see [hamiltonian.py](hamiltonian.py)), a Qiskit electronic structure problem and a FCI solver. The ground state of the Hamiltonian is then found with the VQE algorithm in Qiskit and the FCI solver in PySCF.

**Notes:**
- We perform a spin-less DFT calculation. Therefore, the Kohn-Sham orbitals we use from the DFT calculation do not include spin. For the VQE and FCI calculation we use each Kohn-Sham orbital as a spin orbital which can hold two electrons, one with spin-up one with spin-down. Therefore 2 occupied Kohn-Sham orbitals correspond to 4 electrons, each occupying one spin orbital.

## Authors
- [Erik Schultheis](mailto:erik.schultheis@dlr.de)

## Acknowledgements
This software was developed within the [QuantiCoM project](https://qci.dlr.de/quanticom/) which is part of the [DLR Quantum Computing Initiative (QCI)](https://qci.dlr.de/). We acknowledge the support of the DLR Quantum Computing Initiative (QCI).
