import mdtraj as md
print(md.version.version)

traj = md.load_hdf5('/media/bmohr/Backup/POSTDOC/WCHG/MTD/DNAWC/mtd_DNAWC.h5')
print(traj)
# atoms = [atom.name for atom in traj.topology.atoms]
# print(atoms)
