from warnings import warn
import xmltodict
import numpy as np
import pyscf
import pyscf.fci
from qiskit_nature.second_q.drivers import ElectronicStructureDriver
from qiskit_nature.second_q.drivers.electronic_structure_driver import _QCSchemaData
from qiskit_nature.second_q.formats.qcschema import QCSchema
from qiskit_nature.second_q.formats.qcschema_translator import qcschema_to_problem
from qiskit_nature.second_q.problems import ElectronicBasis, ElectronicStructureProblem
from qiskit_nature.second_q.operators import ElectronicIntegrals
from qiskit_nature.second_q.operators.symmetric_two_body import S1Integrals
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from . import calc_matrix_elements
from . import eri_pair_densities
from . import wfc


class QE_Driver(ElectronicStructureDriver):
    def __init__(self, wfc_files: str | list, xml_file: str) -> None:
        """QuantumEspresso (QE) driver class

        Args:
            wfc_files (str | list): QE wfc output file or list of files. If list of files, the first is for spin-up, the second for spin-down
            xml_file (str): QE xml output file "data-file-schema.xml"
        """
        super().__init__()

        if isinstance(wfc_files, list) is True:
            assert (
                len(wfc_files) == 2
            ), f"Did not provide two but {len(wfc_files)} wfc files: {wfc_files}!"
            wfc_file_up = wfc_files[0]
            wfc_file_dw = wfc_files[1]
            self.wfc_up_obj = wfc.Wfc.from_file(wfc_file_up, xml_file)
            self.wfc_dw_obj = wfc.Wfc.from_file(wfc_file_dw, xml_file)

            err_msg = (
                "Corresponding wfc files do not seem to belong to the same calculation!"
            )
            assert np.allclose(
                self.wfc_up_obj.k_plus_G, self.wfc_dw_obj.k_plus_G
            ), err_msg
            assert self.wfc_up_obj.gamma_only == self.wfc_dw_obj.gamma_only, err_msg
            # assert self.wfc_up_obj.atoms == self.wfc_dw_obj.atoms, err_msg
            assert self.wfc_up_obj.cell_volume == self.wfc_dw_obj.cell_volume, err_msg
            assert self.wfc_up_obj.nbnd == self.wfc_dw_obj.nbnd, err_msg

            self.nspin = 2
        else:
            wfc_file = wfc_files
            self.wfc_up_obj = wfc.Wfc.from_file(wfc_file, xml_file)
            self.wfc_dw_obj = wfc.Wfc.from_file(wfc_file, xml_file)
            self.nspin = 1
            if self.wfc_up_obj.spin != self.nspin:
                warn(
                    "The xml file belongs to a spin-polarized calculation but only one wfc files is provided!"
                )

        with open(xml_file, "r", encoding="utf-8") as file:
            xml_dict = xmltodict.parse(file.read())

        # Extract information from xml file
        self.reference_energy = float(
            xml_dict["qes:espresso"]["output"]["total_energy"]["etot"]
        )
        self.creator = xml_dict["qes:espresso"]["general_info"]["creator"]["@NAME"]
        self.version = xml_dict["qes:espresso"]["general_info"]["creator"]["@VERSION"]
        self.basis = xml_dict["qes:espresso"]["input"]["basis"]
        self.symbols = [
            x["@name"]
            for x in xml_dict["qes:espresso"]["input"]["atomic_structure"][
                "atomic_positions"
            ]["atom"]
        ]
        self.atom_positions = [
            np.fromstring(x["#text"], sep=" ", dtype=np.float32).tolist()
            for x in xml_dict["qes:espresso"]["input"]["atomic_structure"][
                "atomic_positions"
            ]["atom"]
        ]
        self.atom_positions = [x for xs in self.atom_positions for x in xs]
        self.charge = float(xml_dict["qes:espresso"]["input"]["bands"]["tot_charge"])
        self.nelec = float(
            xml_dict["qes:espresso"]["output"]["band_structure"]["nelec"]
        )

        if (
            xml_dict["qes:espresso"]["output"]["convergence_info"]["scf_conv"][
                "convergence_achieved"
            ]
            != "true"
        ):
            warn("The QuantumEspresso SCF calculation is not converged")

        self.p = self.wfc_up_obj.k_plus_G  # shape (#waves, 3)

        self.occupations_up = self.wfc_up_obj.occupations_up
        self.occupations_dw = self.wfc_dw_obj.occupations_dw
        self.c_ip_up = self.wfc_up_obj.evc
        self.c_ip_dw = self.wfc_dw_obj.evc

    def run(self) -> ElectronicStructureProblem:
        return self.to_problem()

    def to_problem(
        self, basis: ElectronicBasis = ElectronicBasis.MO, include_dipole: bool = False
    ) -> ElectronicStructureProblem:
        basis: ElectronicBasis = ElectronicBasis.MO
        include_dipole: bool = False

        qcschema = self.to_qcschema(include_dipole=include_dipole)

        problem = qcschema_to_problem(
            qcschema, basis=basis, include_dipole=include_dipole
        )

        if include_dipole and problem.properties.electronic_dipole_moment is not None:
            problem.properties.electronic_dipole_moment.reverse_dipole_sign = True

        return problem

    def to_qcschema(self, *, include_dipole: bool = True) -> QCSchema:
        include_dipole: bool = False
        # Calculate matrix elements
        h_ij_up, h_ij_dw = self.calc_h_ij()
        # All ERIs are given in physicists' order
        eri_up, eri_dw, eri_dw_up, eri_up_dw = self.calc_eri()

        # Transform ERIs to chemist's index order to define S1Integrals object
        # Do not use qiskit_naute function to_chemist_ordering since
        # it tries to determine the index order of the ERIs
        # which fails for eri_up_dw and eri_dw_up because ERIs connecting different
        # spins do not satisfy all symmetries and therefore a index order
        # cannot be detemined based on symmetries
        eri_up = eri_up.swapaxes(1, 2).swapaxes(1, 3)
        eri_dw = eri_dw.swapaxes(1, 2).swapaxes(1, 3)
        eri_dw_up = eri_dw_up.swapaxes(1, 2).swapaxes(1, 3)
        eri_up_dw = eri_up_dw.swapaxes(1, 2).swapaxes(1, 3)

        print(f"h_ij up-down equal: {np.allclose(h_ij_dw, h_ij_up)}")
        print(f"eri up-down equal: {np.allclose(eri_dw, eri_up)}")
        print(f"eri up-(down-up) equal: {np.allclose(eri_dw_up, eri_up)}")
        print(f"eri (up-down)-(down-up) equal: {np.allclose(eri_up_dw, eri_dw_up)}")

        nucl_repulsion = calc_matrix_elements.nuclear_repulsion_energy(
            self.wfc_up_obj.atoms, self.wfc_up_obj.cell_volume
        )

        # overlap = self.wfc_up_obj.get_overlaps()
        # FIXME: Is the following correct for calculating the overlap matrix?
        evc_up_dw = np.concatenate([self.c_ip_up, self.c_ip_dw])
        overlap = np.einsum("ij, kj -> ik", evc_up_dw.conj(), evc_up_dw)

        # Molecular orbitals (MOs) are the Kohn-Sham orbitals
        # Atomic orbitals (AOs) are the plane-waves
        # Up=a, down=b
        data = _QCSchemaData()
        # data.hij
        # data.hij_b
        # data.eri
        data.hij_mo = h_ij_up
        data.hij_mo_b = h_ij_dw

        data.eri_mo = S1Integrals(eri_up)
        data.eri_mo_ba = S1Integrals(eri_dw_up)
        data.eri_mo_bb = S1Integrals(eri_dw)

        data.e_nuc = nucl_repulsion
        data.e_ref = self.reference_energy
        data.overlap = overlap

        # FIXME: Is the following correct for calculating the mo coefficients?
        data.mo_coeff = np.einsum("ij, kj -> ik", self.c_ip_up.conj(), self.c_ip_up)
        data.mo_coeff_b = np.einsum("ij, kj -> ik", self.c_ip_dw.conj(), self.c_ip_dw)

        data.mo_energy = self.wfc_up_obj.ks_energies
        data.mo_energy_b = self.wfc_dw_obj.ks_energies
        data.mo_occ = self.occupations_up
        data.mo_occ_b = self.occupations_dw
        # data.dip_x
        # data.dip_y
        # data.dip_z
        # data.dip_mo_x_a
        # data.dip_mo_y_a
        # data.dip_mo_z_a
        # data.dip_mo_x_b
        # data.dip_mo_y_b
        # data.dip_mo_z_b
        # data.dip_nuc
        # data.dip_ref
        data.symbols = self.symbols
        data.coords = self.atom_positions
        data.multiplicity = self.nspin + 1  # Spin + 1
        data.charge = self.charge
        # data.masses
        # data.method
        data.basis = self.basis
        data.creator = self.creator
        data.version = self.version
        # data.routine
        data.nbasis = self.p.shape[0]
        data.nmo = self.c_ip_up.shape[0]
        data.nalpha = int(np.sum(self.occupations_up))
        data.nbeta = int(np.sum(self.occupations_dw))
        # data.keywords

        return self._to_qcschema(data, include_dipole=include_dipole)

    def calc_h_ij(self):
        # Kinetic energy
        iTj_up = calc_matrix_elements.iTj(self.p, self.c_ip_up)
        iTj_dw = calc_matrix_elements.iTj(self.p, self.c_ip_dw)

        # Nuclear repulsion
        iUj_up = calc_matrix_elements.iUj(
            self.p,
            self.c_ip_up,
            self.wfc_up_obj.atoms,
            self.wfc_up_obj.cell_volume,
        )
        iUj_dw = calc_matrix_elements.iUj(
            self.p,
            self.c_ip_dw,
            self.wfc_dw_obj.atoms,
            self.wfc_dw_obj.cell_volume,
        )

        h_ij_up = iTj_up - iUj_up
        h_ij_dw = iTj_dw - iUj_dw

        return h_ij_up, h_ij_dw

    def calc_eri(self):
        # Calculate ERIs via pair density
        assert (
            self.wfc_up_obj.gamma_only is True
        ), "Calculating ERIs via pair densities is only implemented for the gamma-point!"
        assert (
            self.wfc_dw_obj.gamma_only is True
        ), "Calculating ERIs via pair densities is only implemented for the gamma-point!"

        eri_up: np.ndarray = (
            eri_pair_densities.eri_gamma(p=self.p, c_ip_up=self.c_ip_up)
            / self.wfc_up_obj.cell_volume
        )
        eri_dw: np.ndarray = (
            eri_pair_densities.eri_gamma(p=self.p, c_ip_up=self.c_ip_dw)
            / self.wfc_dw_obj.cell_volume
        )
        eri_dw_up: np.ndarray = (
            eri_pair_densities.eri_gamma(
                p=self.p, c_ip_up=self.c_ip_up, c_ip_dw=self.c_ip_dw
            )
            / self.wfc_dw_obj.cell_volume
        )
        # eri_up_dw: np.ndarray = (
        #     eri_pair_densities.eri_gamma(
        #         p=self.p, c_ip_up=self.c_ip_dw, c_ip_dw=self.c_ip_up
        #     )
        #     / self.wfc_dw_obj.cell_volume
        # )
        eri_up_dw: np.ndarray = eri_dw_up.swapaxes(0, 1).swapaxes(2, 3)

        return eri_up, eri_dw, eri_dw_up, eri_up_dw

    def to_qiskit_problem_old(self):
        # Calculate matrix elements
        h_ij_up, h_ij_dw = self.calc_h_ij()
        eri_up, eri_dw, eri_dw_up, eri_up_dw = self.calc_eri()

        nucl_repulsion = calc_matrix_elements.nuclear_repulsion_energy(
            self.wfc_up_obj.atoms, self.wfc_up_obj.cell_volume
        )

        num_particles = (
            int(np.sum(self.occupations_up)),
            int(np.sum(self.occupations_dw)),
        )

        # Qiskit calculation
        integrals = ElectronicIntegrals.from_raw_integrals(
            h_ij_up,
            eri_up,
            h_ij_dw,
            eri_dw,
            eri_up_dw,
            auto_index_order=True,
            validate=True,
        )
        qiskit_energy = ElectronicEnergy(integrals)
        qiskit_energy.nuclear_repulsion_energy = nucl_repulsion
        qiskit_problem = ElectronicStructureProblem(qiskit_energy)

        # number of particles for spin-up, spin-down
        qiskit_problem.num_particles = num_particles
        qiskit_problem.num_spatial_orbitals = self.wfc_up_obj.nbnd

        qiskit_problem.reference_energy = self.reference_energy

        return qiskit_problem

    def solve_fci(self, n_energies=1):
        # Calculate matrix elements
        h_ij_up, h_ij_dw = self.calc_h_ij()
        eri_up, eri_dw, eri_dw_up, eri_up_dw = self.calc_eri()

        # Transform ERIs to chemist's index order
        eri_up = eri_up.swapaxes(1, 2).swapaxes(1, 3)
        eri_dw = eri_dw.swapaxes(1, 2).swapaxes(1, 3)
        eri_dw_up = eri_dw_up.swapaxes(1, 2).swapaxes(1, 3)
        eri_up_dw = eri_up_dw.swapaxes(1, 2).swapaxes(1, 3)

        nucl_repulsion = calc_matrix_elements.nuclear_repulsion_energy(
            self.wfc_up_obj.atoms, self.wfc_up_obj.cell_volume
        )

        nelec = (int(np.sum(self.occupations_up)), int(np.sum(self.occupations_dw)))
        # FCI calculation
        nroots = n_energies  # number of states to calculate

        norb = self.wfc_up_obj.nbnd

        self.fcisolver = pyscf.fci.direct_uhf.FCISolver()
        # Ordering of parameters from direct_uhf.make_hdiag
        self.fci_evs, self.fci_evcs = self.fcisolver.kernel(
            h1e=(h_ij_up.real, h_ij_dw.real),  # a, b (a=up, b=down)
            eri=(
                eri_up.real,
                eri_up_dw.real,
                eri_dw.real,
            ),  # aa, ab, bb (a=up, b=down)
            norb=norb,
            nelec=nelec,
            nroots=nroots,
        )

        fci_energy = self.fci_evs + nucl_repulsion

        self.fci_energy = fci_energy
        return fci_energy
