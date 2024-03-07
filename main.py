import os
import numpy as np
import xmltodict
from wfc import Wfc
import calc_matrix_elements
import hamiltonian
import eri_pair_densities


if __name__ == "__main__":
    hdf5_file = os.path.join("qe_files", "out_H2", "H2.save", "wfc1.hdf5")
    dat_file = os.path.join("qe_files", "out_H2", "H2.save", "wfc1.dat")
    xml_file = os.path.join("qe_files", "out_H2", "H2.save", "data-file-schema.xml")

    # Choose Kohn-Sham orbitals
    orbitals_indices = [0, 1]

    wfc1_ncpp = Wfc.from_file(hdf5_file, xml_file)

    with open(xml_file, "r", encoding="utf-8") as file:
        xml_dict = xmltodict.parse(file.read())
    reference_energy = xml_dict["qes:espresso"]["output"]["total_energy"]["etot"]

    overlaps_ncpp = wfc1_ncpp.get_overlaps()

    p = wfc1_ncpp.k_plus_G  # shape (#waves, 3)
    c_ip = wfc1_ncpp.evc  # shape (#bands, #waves)

    occupations, c_ip_orbitals = wfc1_ncpp.get_orbitals_by_index(orbitals_indices)

    # Calculate matrix elements
    # Kinetic energy
    iTj_orbitals = calc_matrix_elements.iTj(p, c_ip_orbitals)

    # Nuclear repulsion
    iUj_orbitals = calc_matrix_elements.iUj(
        p, c_ip_orbitals, wfc1_ncpp.atoms, wfc1_ncpp.cell_volume
    )

    # Cacluate ERIs via pair density
    assert (
        wfc1_ncpp.gamma_only is True
    ), "Calculating ERIs via pair densities is only implemented for the gamma-point!"
    h_pqrs: np.ndarray = (
        eri_pair_densities.eri_gamma(p=p, c_ip=c_ip_orbitals) / wfc1_ncpp.cell_volume
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
