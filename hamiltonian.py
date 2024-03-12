from warnings import warn
import numpy as np
from qiskit_nature.second_q.problems.electronic_structure_result import (
    ElectronicStructureResult,
)
from qiskit.algorithms.minimum_eigensolvers.vqe import VQEResult
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.operators import ElectronicIntegrals
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper
import qiskit.algorithms.optimizers as optim
from qiskit.circuit import QuantumCircuit
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit.primitives import Estimator
import pyscf.fci
from qiskit_nature_qe import calc_matrix_elements


class second_quant_hamiltonian:
    """
    Defines a second-quanitzation hamiltonian of the form
    .. math::
        H = \sum_{pq} h_{pq} a_p^\dagger a_q + 1/2 \sum_{pqrs} h_{pqrs} a_p^\dagger a_q^\dagger a_r a_s
    """

    def __init__(
        self,
        h_pq: np.ndarray,
        h_pqrs: np.ndarray,
        nuclear_repulsion_energy: float,
        occupations: np.ndarray,
        frozen_core_energy: float = 0.0,
        reference_energy: float = 0.0,
    ):
        assert h_pq.ndim == 2
        assert h_pqrs.ndim == 4

        assert (
            h_pq.shape[0]
            == h_pq.shape[1]
            == h_pqrs.shape[0]
            == h_pqrs.shape[1]
            == h_pqrs.shape[2]
            == h_pqrs.shape[3]
        )

        assert occupations.ndim == 1
        assert occupations.shape[0] == h_pq.shape[0]

        self.nspin = 1
        self.norb = h_pq.shape[0]
        self.occupations = occupations

        if calc_matrix_elements.check_symmetry_one_body_matrix(h_pq) is False:
            warn("h_pq does not obey one body matrix symmetries (hermiticity)!")

        if False in (
            sym := calc_matrix_elements.check_symmetry_two_body_matrix(h_pqrs)
        ):
            warn(
                f"h_pqrs does not obey two body matrix symmetries (swap symmetry {'fulfilled' if sym[0] else 'not fulfilled'}, "
                f"hermiticity {'fulfilled' if sym[1] else 'not fulfilled'}, hermiticity+swap {'fulfilled' if sym[2] else 'not fulfilled'})!"
            )

        self.nstates = h_pq.shape[0]

        self.h_pq = h_pq
        self.h_pqrs = h_pqrs
        self.nuclear_repulsion_energy = nuclear_repulsion_energy
        self.frozen_core_energy = frozen_core_energy
        self.reference_energy = reference_energy

        self.fci_energy = np.nan
        self.fci_solver: pyscf.fci.direct_spin1.FCIBase = (
            pyscf.fci.direct_spin1.FCIBase()
        )
        self.fci_evs: float = np.nan
        self.fci_evcs: list = []
        self.qiskit_elec_struc_result = ElectronicStructureResult()
        self.qiskit_vqe_result = VQEResult()
        self.qiskit_problem = None
        self.qiskit_ansatz = QuantumCircuit()
        self.qiskit_pauli_sum_op = SparsePauliOp(["I"])
        self.qiskit_vqe_counts = {}
        self.qiskit_vqe_values = {}
        self.vqe_ansatz = {}
        self.vqe_solver = {}

    def to_qiskit_problem(self):
        num_particles = (int(np.sum(self.occupations)), int(np.sum(self.occupations)))

        # Qiskit calculation
        integrals = ElectronicIntegrals.from_raw_integrals(
            self.h_pq, self.h_pqrs, auto_index_order=True
        )
        qiskit_energy = ElectronicEnergy(integrals)
        qiskit_energy.nuclear_repulsion_energy = (
            self.nuclear_repulsion_energy
        )  # *num_particles_sum
        qiskit_energy.constants["frozen_core_energy"] = self.frozen_core_energy
        qiskit_problem = ElectronicStructureProblem(qiskit_energy)

        # number of particles for spin-up, spin-down
        qiskit_problem.num_particles = num_particles
        qiskit_problem.num_spatial_orbitals = self.nstates

        qiskit_problem.reference_energy = self.reference_energy

        self.qiskit_problem = qiskit_problem

        return qiskit_problem

    def solve_vqe(
        self,
        mapper: FermionicMapper | None = JordanWignerMapper(),
    ):
        problem = self.to_qiskit_problem()
        problem.reference_energy = self.fci_energy

        if mapper is None:
            mapper = JordanWignerMapper()
        energy = problem.hamiltonian
        fermionic_op = energy.second_q_op()
        pauli_sum_op = mapper.map(fermionic_op)

        # Qubit mapping
        initial_state = HartreeFock(
            num_spatial_orbitals=problem.num_spatial_orbitals,
            num_particles=problem.num_particles,
            qubit_mapper=mapper,
        )
        ansatz = UCCSD(
            problem.num_spatial_orbitals,
            problem.num_particles,
            mapper,
            initial_state=initial_state,
        )

        # initial_state.draw(
        #     "mpl", filename=os.path.join("results", "UCCSD_initial_state")
        # )

        # ansatz.draw("mpl", filename=os.path.join("results", "vqe_ansatz"))

        optimizer = optim.COBYLA()  # Classical optimizer
        estimator = Estimator()

        counts = []
        values = []

        solver = VQE(
            estimator,
            ansatz,
            optimizer,
        )
        print("VQE defined")
        print("Solving VQE...")
        vqe_result = solver.compute_minimum_eigenvalue(pauli_sum_op)
        print("Solved")
        elec_struc_result = problem.interpret(vqe_result)

        self.vqe_solver = solver
        self.vqe_results = vqe_result

        self.vqe_ansatz = solver.ansatz

        self.qiskit_vqe_counts = counts
        self.qiskit_vqe_values = values

        self.qiskit_elec_struc_result = elec_struc_result
        self.qiskit_vqe_result = vqe_result
        self.qiskit_problem = problem
        self.qiskit_ansatz = ansatz
        self.qiskit_pauli_sum_op = pauli_sum_op

        return self.vqe_results

    def solve_fci(self, n_energies=1):
        if self.h_pq.shape[0] == 0:
            return self.frozen_core_energy + self.nuclear_repulsion_energy

        nelec = 0
        nelec = (int(np.sum(self.occupations)), int(np.sum(self.occupations)))

        # FCI calculation
        nroots = n_energies  # number of states to calculate
        self.fci_solver = pyscf.fci.direct_spin1.FCISolver()
        h_pq_real = self.h_pq.real
        h_pq = h_pq_real
        h_pqrs_real = self.h_pqrs.real

        self.fci_evs, self.fci_evcs = self.fci_solver.kernel(
            h1e=h_pq,
            eri=h_pqrs_real,
            norb=self.norb,
            nelec=nelec,
            nroots=nroots,
        )

        fci_energy = (
            self.fci_evs + self.frozen_core_energy + self.nuclear_repulsion_energy
        )

        self.fci_energy = fci_energy
        return fci_energy
