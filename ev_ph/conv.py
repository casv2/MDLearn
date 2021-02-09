from ase.io import read, write
import ase

al = read("./FLD_hcp_bcc_s-4+3p.xyz", ":")
al2 = [] 

for at in al:
    at2 = ase.Atoms(at.get_chemical_symbols(), positions=at.get_positions(), cell=at.get_cell())
    at2.info["energy"] = at.info["energy_TB"]
    at2.info["config_type"] = at.info["config_type"]
    at2.arrays["force"] = at.arrays["force"]
    #at2.info["virial"] = at.info["virial"]
    al2.append(at2)

write("./FLD_hcp_bcc_s-4+3p_conv.xyz", al2)
