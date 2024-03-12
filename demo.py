import os
import numpy as np
import xmltodict
from qiskit_nature_qe import calc_matrix_elements, eri_pair_densities, wfc, qe_driver
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.properties import ParticleNumber
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
import logging

from qiskit_nature_qe.calc_matrix_elements import (
    check_symmetry_one_body_matrix,
    check_symmetry_two_body_matrix,
)
from qiskit_nature.second_q.operators.symmetric_two_body import S1Integrals

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    hdf5_file = os.path.join("qe_files", "out_H2", "H2.save", "wfc1.hdf5")
    dat_file = os.path.join("qe_files", "out_H2", "H2.save", "wfc1.dat")
    dat_up_file = os.path.join("qe_files", "out_H2", "H2.save", "wfcup1.dat")
    dat_dw_file = os.path.join("qe_files", "out_H2", "H2.save", "wfcdw1.dat")
    xml_file = os.path.join("qe_files", "out_H2", "H2.save", "data-file-schema.xml")
    # xml_file = os.path.join(
    #     "qe_files", "out_H2 copy", "H2.save", "data-file-schema.xml"
    # )

    wfc_obj = wfc.Wfc.from_file(dat_file, xml_file)

    with open(xml_file, "r", encoding="utf-8") as file:
        xml_dict = xmltodict.parse(file.read())

    # driver = qe_driver.QE_Driver(dat_file, xml_file)
    driver = qe_driver.QE_Driver([dat_up_file, dat_dw_file], xml_file)

    problem = driver.run()

    # Calculate matrix elements
    from qiskit_nature.second_q.formats.qcschema_translator import (
        _reshape_2,
        _reshape_4,
    )

    h_ij_up, h_ij_dw = driver.calc_h_ij()
    eri_up, eri_dw, eri_dw_up, eri_up_dw = driver.calc_eri()

    # eri_up = eri_up.swapaxes(1, 2).swapaxes(1, 3)
    # eri_dw = eri_dw.swapaxes(1, 2).swapaxes(1, 3)
    # eri_dw_up = eri_dw_up.swapaxes(1, 2).swapaxes(1, 3)
    # eri_up_dw = eri_up_dw.swapaxes(1, 2).swapaxes(1, 3)

    from qiskit_nature.second_q.operators.tensor_ordering import find_index_order

    print(f"eri_up order: {find_index_order(eri_up)}")
    print(f"eri_dw order: {find_index_order(eri_dw)}")
    print(f"eri_dw_up order: {find_index_order(eri_dw_up)}")
    print(f"eri_up_dw order: {find_index_order(eri_up_dw)}")

    print(f"eri_up symmetries: {check_symmetry_two_body_matrix(eri_up)}")
    print(f"eri_dw symmetries: {check_symmetry_two_body_matrix(eri_dw)}")
    print(f"eri_dw_up symmetries: {check_symmetry_two_body_matrix(eri_dw_up)}")
    print(f"eri_up_dw symmetries: {check_symmetry_two_body_matrix(eri_up_dw)}")

    # h_ij_up = h_ij_up.ravel().tolist()
    # h_ij_dw = h_ij_dw.ravel().tolist()
    # eri_up = eri_up.ravel().tolist()
    # eri_dw = eri_dw.ravel().tolist()
    # eri_dw_up = eri_dw_up.ravel().tolist()
    # eri_up_dw = eri_up_dw.ravel().tolist()

    # n_orb = driver.wfc_up_obj.nbnd
    # h_ij_up = _reshape_2(h_ij_up, n_orb)
    # h_ij_dw = _reshape_2(h_ij_dw, n_orb)
    # eri_up = _reshape_4(eri_up, n_orb)
    # eri_dw = _reshape_4(eri_dw, n_orb)
    # eri_dw_up = _reshape_4(eri_dw_up, n_orb)
    # eri_up_dw = _reshape_4(eri_up_dw, n_orb)

    # h_ij_up = np.array(h_ij_up).reshape(4, 4)
    # h_ij_dw = np.array(h_ij_dw).reshape(4, 4)
    # eri_up = np.array(eri_up).reshape(4, 4, 4, 4)
    # eri_dw = np.array(eri_dw).reshape(4, 4, 4, 4)
    # eri_dw_up = np.array(eri_dw_up).reshape(4, 4, 4, 4)
    # eri_up_dw = np.array(eri_up_dw).reshape(4, 4, 4, 4)

    # h_ij_up = np.zeros_like(h_ij_up)
    # h_ij_up[0, 0] = 3
    # h_ij_dw = np.zeros_like(h_ij_dw)
    # h_ij_dw[0, 0] = -3
    # eri_up = np.zeros_like(eri_up)
    # eri_up[0, 0, 0, 0] = 2
    # eri_dw = np.zeros_like(eri_dw)
    # eri_dw[0, 0, 0, 0] = -2
    # eri_dw_up = np.zeros_like(eri_dw_up)
    # eri_dw_up[0, 0, 0, 0] = -1
    # eri_up_dw = np.zeros_like(eri_up_dw)
    # eri_up_dw[0, 0, 0, 0] = 1

    nucl_repulsion = calc_matrix_elements.nuclear_repulsion_energy(
        driver.wfc_up_obj.atoms, driver.wfc_up_obj.cell_volume
    )

    # num_particles = (
    #     int(np.sum(driver.occupations_up)),
    #     int(np.sum(driver.occupations_dw)),
    # )

    from qiskit_nature.second_q.operators.symmetric_two_body import S1Integrals

    # # Qiskit calculation
    from qiskit_nature.second_q.operators import ElectronicIntegrals

    from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
    from qiskit_nature.second_q.problems import ElectronicStructureProblem

    # from_raw_integrals transforms integrals to PolynomialTensor objects
    # which use physicsits index order. The index order of the
    # given integrals is determined from eri_up (h2_aa) if auto_index_order is True
    # else they are assumed to be in physicsits index order
    # while S1Integrals need chemists index order
    # -- The following with np.zeros_like(eri_dw_up) as h2_ba
    #    gives the same result as pyscf.fci.direct_uhf.FCI Solver
    #    with same inputs --
    # -- The following with eri_up_dw as h2_ba
    #    gives the same result as pyscf.fci.direct_uhf.FCI Solver
    #    with same inputs but eri_dw_up as eri_ab (=2nd element in eri tuple) --
    integrals = ElectronicIntegrals.from_raw_integrals(
        h_ij_up,  # a (a=up, b=down)
        eri_up,  # aa
        h_ij_dw,  # b
        eri_dw,  # bb
        eri_dw_up,  # ba
        auto_index_order=True,
        validate=True,
    )

    from qiskit_nature.second_q.operators import Tensor, PolynomialTensor

    tensor_up = Tensor(eri_up)  # Given that eri_up is in physicists' order
    tensor_up.label_template = "+_{{0}} +_{{1}} -_{{2}} -_{{3}}"
    alpha = PolynomialTensor({"+-": h_ij_up, "++--": tensor_up})
    tensor_dw = Tensor(eri_dw)  # Given that eri_up is in physicists' order
    tensor_dw.label_template = "+_{{0}} +_{{1}} -_{{2}} -_{{3}}"
    beta = PolynomialTensor({"+-": h_ij_dw, "++--": tensor_dw})
    tensor_dw_up = Tensor(eri_dw_up)  # Given that eri_up is in physicists' order
    tensor_dw_up.label_template = "+_{{0}} +_{{1}} -_{{2}} -_{{3}}"
    beta_alpha = PolynomialTensor({"++--": tensor_dw_up})
    tensor_up_dw = Tensor(eri_up_dw)  # Given that eri_up is in physicists' order
    tensor_up_dw.label_template = "+_{{0}} +_{{1}} -_{{2}} -_{{3}}"
    alpha_beta = PolynomialTensor({"++--": tensor_up_dw})
    # integrals = ElectronicIntegrals(alpha, beta, beta_alpha)

    qiskit_energy = ElectronicEnergy(integrals)
    qiskit_energy.nuclear_repulsion_energy = nucl_repulsion
    qiskit_problem = ElectronicStructureProblem(qiskit_energy)

    # # number of particles for spin-up, spin-down
    # qiskit_problem.num_particles = num_particles
    # qiskit_problem.num_spatial_orbitals = driver.wfc_up_obj.nbnd

    # qiskit_problem.reference_energy = driver.reference_energy
    # problem = qiskit_problem
    # problem = driver.to_qiskit_problem_old()

    hamiltonian = problem.hamiltonian
    # np.allclose(hamiltonian.electronic_integrals.alpha_beta["++--"], eri_dw_up)
    # np.allclose(
    #     hamiltonian.electronic_integrals.beta_alpha["++--"],
    #     hamiltonian.electronic_integrals.alpha_beta["++--"],
    # )
    # hamiltonian.electronic_integrals.equiv(integrals)
    # hamiltonian.electronic_integrals.alpha.equiv(integrals.alpha)
    # Not equivalent because _label_template is None for integrals obj
    # hamiltonian.electronic_integrals.alpha["++--"]._label_template
    # integrals.alpha["++--"]._label_template
    # np.allclose(hamiltonian.electronic_integrals.alpha_beta["++--"], integrals.beta_alpha["++--"])

    # problem.hamiltonian.second_q_op()
    # qiskit_problem.hamiltonian.second_q_op()

    second_q_op = hamiltonian.second_q_op()

    # for i in range(4):
    #     for j in range(4):
    #         ij = hamiltonian.second_q_op()[f"+_{i} -_{j}"]
    #         ij_plus_4 = hamiltonian.second_q_op()[f"+_{i+4} -_{j+4}"]

    #         assert ij == ij_plus_4, f"i|j: {i}|{j}, ij: {ij}, ij_plus_4: {ij_plus_4}"

    # for i in range(4):
    #     for j in range(4):
    #         for k in range(4):
    #             for l in range(4):
    #                 ijkl = hamiltonian.second_q_op()[f"+_{i} +_{j} -_{k} -_{l}"]
    #                 ijkl_plus_4 = hamiltonian.second_q_op()[
    #                     f"+_{i+4} +_{j+4} -_{k+4} -_{l+4}"
    #                 ]

    #                 assert (
    #                     ijkl == ijkl_plus_4
    #                 ), f"i|j|k|l: {i}|{j}|{k}|{l}, ijkl: {ijkl}, ijkl_plus_4: {ijkl_plus_4}"

    print(hamiltonian.nuclear_repulsion_energy)

    print(problem.molecule)
    print(problem.reference_energy)
    print(problem.num_particles)
    print(problem.num_spatial_orbitals)
    print(problem.basis)

    print(problem.properties)
    print(problem.properties.particle_number)
    print(problem.properties.angular_momentum)
    print(problem.properties.magnetization)
    print(problem.properties.electronic_dipole_moment)

    # ------------------ Solve with NumPyMinimumEigensolver ------------------
    mapper = JordanWignerMapper()
    algo = NumPyMinimumEigensolver()
    algo.filter_criterion = problem.get_default_filter_criterion()

    solver = GroundStateEigensolver(mapper, algo)
    # result = solver.solve(problem)

    # print(f"Total ground state energy = {result.total_energies[0]:.4f}")

    # result = solver.solve(problem)
    # print(result)

    print("FCI START")
    driver.solve_fci(10)
    print("\n".join(str(x) for x in driver.fci_evs))
    print("FCI END")

    pauli_sum_op = mapper.map(second_q_op)
    h_mat = pauli_sum_op.to_matrix()
    if np.allclose(h_mat, h_mat.T.conj()) is False:
        print("!!!!!!!!!!!!! Hamiltonian is not hermitian !!!!!!!!!!!!!")

    # ------------------ Solve with Numpy eigh function ------------------
    eig_vals_all, eig_vecs_all = np.linalg.eigh(h_mat)
    # print(eig_vals_all[:5])
    print("# \t Eigenvalue \t\t #electrons\t S²\tS_z")

    num_op = ParticleNumber(pauli_sum_op.num_qubits // 2).second_q_ops()
    num_op = mapper.map(num_op["ParticleNumber"]).to_matrix()
    for idx, vals in enumerate(eig_vals_all):
        state = eig_vecs_all[:, idx]  # [:, idx] NOT [idx]
        num_op_exp_val = np.round(state.T @ num_op @ state, 3).real
        if num_op_exp_val == 2:
            print(f"{idx}\t{vals}\t{num_op_exp_val}")

    # Statevector(eig_vecs_all[:, 0]).draw("latex")

    # problem = h_ks.to_qiskit_problem()

    # fci_energy = h_ks.solve_fci()

    # h_ks.solve_vqe()
    # vqe_energy = h_ks.qiskit_elec_struc_result.total_energies

    # print(f"vqe energy: {vqe_energy}")
    # print(f"fci energy: {fci_energy}")
    # print(f"VQE Result:\n{h_ks.qiskit_elec_struc_result}")

    # print("Done!")

    # import pyscf
    # import pyscf.fci

    # h_ij_up, h_ij_dw = driver.calc_h_ij()
    # eri_up, eri_dw, eri_dw_up, eri_up_dw = driver.calc_eri()

    # # # Transform ERIs to chemist's index order
    # eri_up = eri_up.swapaxes(1, 2).swapaxes(1, 3)
    # eri_dw = eri_dw.swapaxes(1, 2).swapaxes(1, 3)
    # eri_dw_up = eri_dw_up.swapaxes(1, 2).swapaxes(1, 3)
    # eri_up_dw = eri_dw_up.swapaxes(1, 2).swapaxes(1, 3)

    # nucl_repulsion = calc_matrix_elements.nuclear_repulsion_energy(
    #     driver.wfc_up_obj.atoms, driver.wfc_up_obj.cell_volume
    # )

    # nelec = (int(np.sum(driver.occupations_up)), int(np.sum(driver.occupations_dw)))
    # print(f"nelec: {nelec}")

    # # FCI calculation
    # nroots = 1

    # norb = driver.wfc_up_obj.nbnd
    # fcisolver = pyscf.fci.direct_uhf.FCISolver()
    # fci_evs, fci_evcs = fcisolver.kernel(
    #     h1e=(h_ij_up.real, h_ij_dw.real),
    #     eri=(
    #         eri_up.real,
    #         eri_up_dw.swapaxes(0,2).swapaxes(2,3).real,
    #         eri_dw.real,
    #     ),  # aa, ab, bb (a=up, b=down)
    #     norb=norb,
    #     nelec=nelec,
    #     nroots=nroots,
    # )
    # print(fci_evs)

    # PYSCF alpha-beta two-body test
    # from pyscf import gto, scf, ci, ao2mo
    # from qiskit_nature.second_q.operators.symmetric_two_body import fold
    # mol = gto.M(
    #     atom = 'O 0 0 0; O 0 0 1.2',  # in Angstrom
    #     basis = 'sto-3g',
    #     spin = 2
    # )
    # mf = scf.HF(mol).run() # this is UHF
    # # myci = ci.CISD(mf).run() # this is UCISD
    # # print('UCISD total energy = ', myci.e_tot)

    # mo_coeff, mo_coeff_b = mf.mo_coeff
    # np.allclose(mo_coeff, mo_coeff_b)
    # # plt.imshow(mo_coeff-mo_coeff_b, cmap="coolwarm")
    # # plt.colorbar()

    # eri_mo_ba = fold(ao2mo.general(
    #         mol,
    #         [mo_coeff_b, mo_coeff_b, mo_coeff, mo_coeff]
    #     ))
    # eri_mo_bb = fold(ao2mo.full(mol, mo_coeff_b))
    # eri = mol.intor("int2e")
    # eri_mo = fold(ao2mo.full(mol, mo_coeff))
    # eri_mo.shape
    # # ao2mo.full(mol, mo_coeff_b).shape

    # eri_mo_ba._array.shape

    # # check_symmetry_two_body_matrix(eri_mo)

    # # mo_coeff.shape

    # # PYSCF driver test
    # from qiskit_nature.units import DistanceUnit
    # from qiskit_nature.second_q.drivers import PySCFDriver

    # driver = PySCFDriver(
    #     atom="O 0 0 0; O 0 0 1.2", basis="sto-3g", spin=2  # in Angstrom
    # )

    # problem = driver.run()

    # problem.hamiltonian.electronic_integrals.alpha["++--"].reshape((10,10,10,10))
