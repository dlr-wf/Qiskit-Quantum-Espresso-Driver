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

    hdf5_file = os.path.join("qe_files", "out_H2 spin", "H2.save", "wfc1.hdf5")
    dat_file = os.path.join("qe_files", "out_H2", "H2.save", "wfc1.dat")
    dat_up_file = os.path.join("qe_files", "out_H2 spin", "H2.save", "wfcup1.dat")
    dat_dw_file = os.path.join("qe_files", "out_H2 spin", "H2.save", "wfcdw1.dat")
    xml_file = os.path.join(
        "qe_files", "out_H2 spin", "H2.save", "data-file-schema.xml"
    )
    xml_file_copy = os.path.join(
        "qe_files", "out_H2", "H2.save", "data-file-schema.xml"
    )

    wfc_obj = wfc.Wfc.from_file(dat_file, xml_file)

    with open(xml_file, "r", encoding="utf-8") as file:
        xml_dict = xmltodict.parse(file.read())

    driver = qe_driver.QE_Driver(dat_file, xml_file_copy)
    # driver = qe_driver.QE_Driver([dat_up_file, dat_dw_file], xml_file)

    problem = driver.run()

    # anna
    # problem = driver.run(ovlp=True)
    # ovlp_true = problem.second_q_ops()[1]["AngularMomentum"]
    # problem = driver.run(ovlp=False)
    # ovlp_false = problem.second_q_ops()[1]["AngularMomentum"]
    # ovlp_false

    hamiltonian = problem.hamiltonian

    second_q_op = hamiltonian.second_q_op()

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
    # FIXME: Get numpy ground state solver running
    result = solver.solve(problem)

    # coeff_a.T @ overlap @ coeff_b IS NOT THE IDENTITY IF overlap is the idenitity, why?
    (driver.c_ip_up @ driver.c_ip_dw.T)  # IS NOT IDENTITY
    (driver.c_ip_up.conj() @ driver.c_ip_dw.T)  # IS IDENTITY
    # -> .conj() missing in qiskit_nature.second_q.formats.qcschema_translator.get_overlap_ab_from_qcschema?
    # coeff_a.T @ overlap @ coeff_b -> coeff_a.T.conj() @ overlap @ coeff_b
    # TODO: release code without overlap parameter and create issue regarding .conj() in
    #       get_overlap_ab_from_qcschema function

    # anna

    # print(f"Total ground state energy = {result.total_energies[0]:.4f}")

    # result = solver.solve(problem)
    print(result)

    driver.solve_fci(1)
    print("FCI eigenvalues:", "\n".join(str(x) for x in driver.fci_evs))

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

    Statevector(eig_vecs_all[:, 0]).draw("latex")

    anna

    # ------------------------------------------------------------------------
    h_ij_up, h_ij_dw = driver.calc_h_ij()
    eri_up, eri_dw, eri_dw_up, eri_up_dw = driver.calc_eri()
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

    # Overlap is identity check

    evc_up_dw = np.concatenate([driver.c_ip_dw, driver.c_ip_up])
    # evc_up_dw = driver.c_ip_up
    # overlap = np.einsum("ij, kj -> ik", evc_up_dw.conj(), evc_up_dw)
    # # overlap matrix has to be of shape (nao, nao) where nao is the
    # # number of atomic orbitals which are plane-waves in our case
    # overlap = np.einsum("ji, jk -> ik", evc_up_dw.conj(), evc_up_dw)
    overlap = evc_up_dw.T.conj() @ evc_up_dw
    mo_coeff_a = driver.c_ip_up.T
    mo_coeff_b = driver.c_ip_dw.T
    (evc_up_dw @ evc_up_dw.conj().T).shape
    # plt.imshow((mo_coeff_a.T @ overlap @ mo_coeff_b).real, vmin=-1.0, vmax=1.0)
    # plt.colorbar()
    # (mo_coeff_a.T @ overlap @ mo_coeff_b).max()

    # plt.imshow((driver.c_ip_dw.conj() @ evc_up_dw.T.conj() @ evc_up_dw @ driver.c_ip_dw.T).real)
    ovlp = (
        driver.c_ip_dw.conj() @ evc_up_dw.T.conj() @ evc_up_dw @ driver.c_ip_dw.T
    ).real  # is 2 times identity, why?
    # overlap calculated in qcschema_translator.get_overlap_ab_from_qcschema
    # does not apply .conj() on the first matrix
    # Note that coeff_a in qcschema_translator.get_overlap_ab_from_qcschema
    # has shape (nao, nmo) but driver.c_ip_dw has shape (nmo, nao)

    # driver.wfc_up_obj.get_overlaps()

    from pyscf import gto, scf, ci, ao2mo

    # from qiskit_nature.second_q.operators.symmetric_two_body import fold

    # # mol = gto.M(atom="H 0 0 0; H 0 0 1.2", basis="sto-3g", spin=2)  # in Angstrom
    # # mf = scf.HF(mol).run() # this is UHF
    # mf.get_ovlp().shape
    # # plt.imshow(mf.get_ovlp())
    # # mf.mo_coeff.shape

    # # plt.imshow(mf.mo_coeff[0].T @ mf.get_ovlp() @ mf.mo_coeff[1])
    # from qiskit_nature.second_q.drivers import PySCFDriver
    # from qiskit_nature.second_q.drivers import MethodType

    # x = PySCFDriver(
    #     atom="H 0 0 0; H 0 0 1.2",  # in Angstrom
    #     basis="sto-3g",
    #     spin=2, # mole.spin is the number of unpaired electrons 2S, i.e. the difference between the number of alpha and beta electrons.
    #     method=MethodType.UHF,
    # )
    # x.method
    # x.run()
    # x.to_problem().second_q_ops()[1]["AngularMomentum"]
    # # type(x.to_qcschema().wavefunction.scf_orbitals_b)
    # # x._calc.mo_coeff.shape
    # plt.imshow(mf.mo_coeff[0].T @ mf.get_ovlp() @ mf.mo_coeff[1])

    # # problem = x.run()

    # # mapper = JordanWignerMapper()
    # # algo = NumPyMinimumEigensolver()
    # # algo.filter_criterion = problem.get_default_filter_criterion()

    # # solver = GroundStateEigensolver(mapper, algo)
    # # result = solver.solve(problem)
    # # print(result)

    # from qiskit_nature.second_q.formats.qcschema_translator import get_overlap_ab_from_qcschema
    # get_overlap_ab_from_qcschema(x.to_qcschema())

    from pyscf import gto, scf, ci, ao2mo

    mol = gto.M(atom="H 0 0 0; H 0 0 1.2", basis="sto-3g", spin=0)  # in Angstrom
    mf = scf.UHF(mol).run()  # this is UHF
    overlap = mf.get_ovlp()
    # overlap = mf.mo_coeff[0] @ mf.mo_coeff[0].T
    mo_coeff_a = mf.mo_coeff[0]
    mo_coeff_b = mf.mo_coeff[1]
    # plt.imshow((mo_coeff_a.T @ overlap @ mo_coeff_b).real, vmin=-1.0, vmax=2.0)
    # plt.imshow((mo_coeff_b.T @ mo_coeff_b).real, vmin=-1.0, vmax=2.0)
    # plt.colorbar()
    (mo_coeff_b.T @ mo_coeff_b).real
    # mf.mo_coeff.shape
    # mol.nelec
