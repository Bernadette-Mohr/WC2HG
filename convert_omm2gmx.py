import argparse
import mdtraj as md
from pathlib import Path


def convert_traj(sys_name, dir_path):
    """
    Convert OMM H5MD trajectory to GROMACS XTC format for visualization.
    :param sys_name: Name of the system, used to identify trajectory file and output file names.
    :param dir_path: Path to the directory containing the trajectory file.
    :return: None. Automatically saves the converted trajectory and PDB files in the specified directory.
    """
    traj_name = list(dir_path.glob(f"{sys_name}/*.h5"))[0]
    print(traj_name)

    m_traj = md.load_hdf5(traj_name)
    align_atoms = m_traj.topology.select("all")
    m_traj_aligned = m_traj.superpose(reference=m_traj, frame=0, atom_indices=align_atoms)
    if sys_name == 'DNAWC':
        anchor_molecules = [
            set(m_traj_aligned.topology.chain(0).atoms),
            set(m_traj_aligned.topology.chain(1).atoms),
        ]
    elif sys_name == 'DNAWC2MAT_B':
        anchor_molecules = [
            set(m_traj_aligned.topology.chain(0).atoms),
            set(m_traj_aligned.topology.chain(1).atoms),
            set(m_traj_aligned.topology.chain(2).atoms),
        ]
    else:
        anchor_molecules = [
            set(m_traj_aligned.topology.chain(0).atoms),
            set(m_traj_aligned.topology.chain(1).atoms),
            set(m_traj_aligned.topology.chain(2).atoms),
            set(m_traj_aligned.topology.chain(3).atoms),
        ]
    md.Trajectory.image_molecules(m_traj_aligned, anchor_molecules=anchor_molecules)
    m_traj_aligned.remove_solvent(inplace=True)
    m_traj_aligned[0].save_pdb(dir_path / f"{sys_name}/{sys_name}_nosolv.pdb")
    m_traj_aligned.save_xtc(dir_path / f"{sys_name}/{sys_name}_transition_100fs.xtc")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert OMM trajectory to GROMACS format.")
    parser.add_argument('-sys', "--system", type=str, help="System name (e.g., DNAWC, DNAWC2MAT_B, etc.)")
    parser.add_argument('-dir', '--directory', type=Path, default=Path.cwd(),
                        help="Directory path for the trajectory files")

    args = parser.parse_args()

    system = args.system
    directory = args.directory

    convert_traj(system, directory)

    print(f"Conversion completed for {system}. Files saved in {directory}/{system}.")