import os
from qiskit_nature_qe import qe_driver
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.mappers import JordanWignerMapper
import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    hdf5_file = os.path.join("qe_files", "out_H2", "H2.save", "wfc1.hdf5")
    xml_file = os.path.join("qe_files", "out_H2", "H2.save", "data-file-schema.xml")

    driver = qe_driver.QE_Driver(hdf5_file, xml_file)

    problem = driver.run()

    # ------------------ Solve with NumPyMinimumEigensolver ------------------
    mapper = JordanWignerMapper()
    algo = NumPyMinimumEigensolver()
    algo.filter_criterion = problem.get_default_filter_criterion()

    solver = GroundStateEigensolver(mapper, algo)
    result = solver.solve(problem)

    print(result)
