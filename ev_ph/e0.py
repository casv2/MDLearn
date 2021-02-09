import ase
import numpy as np
from quippy.potential import Potential

calc = Potential("TB NRL-TB", param_filename="./quip_params.xml")

at = ase.Atoms("Ti", positions=[[0,0,0]], cell=np.eye(3)*100, pbc=True)
at.set_calculator(calc)

print(at.get_potential_energy())


