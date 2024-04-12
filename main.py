import os
from warnings import warn
import numpy as np
import xmltodict
from wfc import Wfc
import calc_matrix_elements
import hamiltonian
import eri_pair_densities
from eri_hashmap import EriHashmap, load_eri_from_file


if __name__ == "__main__":
    # hdf5_file = os.path.join("qe_files", "out_H2", "H2.save", "wfc1.hdf5")
    # dat_file = os.path.join("qe_files", "out_H2", "H2.save", "wfc1.dat")
    # xml_file = os.path.join("qe_files", "out_H2", "H2.save", "data-file-schema.xml")
    
    dat_file = os.path.join("qe_files", "out_Mg", "Mg.save", "wfc1.dat")
    xml_file = os.path.join("qe_files", "out_Mg", "Mg.save", "data-file-schema.xml")
    
    dat_file = os.path.join("qe_files", "out_Mg_paw", "Mg.save", "wfc1.dat")
    xml_file = os.path.join("qe_files", "out_Mg_paw", "Mg.save", "data-file-schema.xml")

    # Choose Kohn-Sham orbitals
    orbitals_indices = [0, 1] # H2
    orbitals_indices = [9, 10] # Mg normconserving
    orbitals_indices = [1, 2] # Mg paw

    wfc1_ncpp = Wfc.from_file(dat_file, xml_file)

    with open(xml_file, "r", encoding="utf-8") as file:
        xml_dict = xmltodict.parse(file.read())
    reference_energy = xml_dict["qes:espresso"]["output"]["total_energy"]["etot"]

    overlaps_ncpp = wfc1_ncpp.get_overlaps()
    
    if not np.allclose(overlaps_ncpp, np.eye(overlaps_ncpp.shape[0])):
        warn("The wavefunctions are not orthonormal!")

    p = wfc1_ncpp.k_plus_G  # shape (#waves, 3)
    c_ip = wfc1_ncpp.evc  # shape (#bands, #waves)

    occupations, c_ip_orbitals = wfc1_ncpp.get_orbitals_by_index(orbitals_indices)

    # Calculate matrix elements
    # Kinetic energy
    iTj_orbitals = calc_matrix_elements.iTj(p, c_ip_orbitals)

    # Nuclear repulsion
    pUq = calc_matrix_elements.pUq(p, wfc1_ncpp.atoms, wfc1_ncpp.cell_volume)
    iUj_orbitals = calc_matrix_elements.iUj(
        p, c_ip_orbitals, wfc1_ncpp.atoms, wfc1_ncpp.cell_volume
    )

    # # Load Electron repulsion integrals from Rust calculation
    # eri_file = os.path.join("eri", "eri_sym_rs_tuvw_0_4_f64.txt")

    # (
    #     n_elements_pqrs,
    #     n_states_pqrs,
    #     indices_pqrs,
    #     mat_pqrs,
    # ) = load_eri_from_file(eri_file, "tuvw")
    # eri_hashmap = EriHashmap()
    # eri_hashmap.update(indices_pqrs, mat_pqrs)
    # h_pqrs: np.ndarray = eri_hashmap.get_tuvw(orbitals_indices) / wfc1_ncpp.cell_volume

    # Cacluate ERIs via pair density instead of loading from Rust and CUDA output
    assert (
        wfc1_ncpp.gamma_only is True
    ), "Calculating ERIs via pair densities is only implemented for the gamma-point!"
    h_pqrs: np.ndarray = (
        eri_pair_densities.eri_gamma(p=p, c_ip=c_ip_orbitals, b=wfc1_ncpp.b, mill=wfc1_ncpp.mill) / wfc1_ncpp.cell_volume
    )

    h_pq = iTj_orbitals - iUj_orbitals

    nucl_repulsion = calc_matrix_elements.nuclear_repulsion_energy(
        wfc1_ncpp.atoms, wfc1_ncpp.cell_volume
    )

    h_ks = hamiltonian.second_quant_hamiltonian(
        h_pq, h_pqrs, nucl_repulsion, occupations, reference_energy=reference_energy
    )

    problem = h_ks.to_qiskit_problem()

    fci_energy = h_ks.solve_fci()

    h_ks.solve_vqe()
    vqe_energy = h_ks.qiskit_elec_struc_result.total_energies

    print(f"vqe energy: {vqe_energy}")
    print(f"fci energy: {fci_energy}")
    print(f"VQE Result:\n{h_ks.qiskit_elec_struc_result}")

    print("Done!")
