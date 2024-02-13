#center of mass function by Ids Teepe
import numpy as np
import mdtraj as md


def get_center_of_mass(trajectory, atoms, shift=True):
    """
    Returns the center of mass of a group of atoms.
    Atoms is a list containing the indices of the coordinates,

    shift uses the function 'shift to middle'
    to place the first atom of the selection in the middle of the box
    to account for periodic boundary conditions.
    """
    if shift:
        #Save old coordinates
        trajectory.xyz_tmp = trajectory.xyz
        # Shift first atom to the box center
        trajectory.xyz = shift_to_middle(trajectory, atoms[0])
        # A 3D matrix (n_frames : n_atomen : 3) of the coordinates of the chosen atoms
        coordinates = trajectory.xyz[:, atoms, :]
    # A list containing atom masses
    masses = np.array([trajectory.top.atom(i).element.mass for i in atoms])
    # multiply each coordinate with the mass
    coordinates_weighted = coordinates*masses[:,np.newaxis]
    # sum coordinates and divide by the sum of the masses
    coms = np.sum(coordinates_weighted, axis=1)/sum(masses)
    if shift:
        coordshift = trajectory.xyz_tmp[:,atoms[0],:]-trajectory.xyz[:,atoms[0],:]
        # shift center of mass to original coordinates
        coms += coordshift
        # restore original coordinates
        trajectory.xyz = trajectory.xyz_tmp
        # return 2D matrix (n_frames : 3) with COM coordinates
    return coms


def shift_to_middle(trajectory, at_middle: 'atom to place in middle' = None) -> np.ndarray:
    """
    Translates the system to put an atom in the middle of the box.
    at_middle: The atom to place in the middle. Can either be an integer
    or an array with shape (n_frames, 3) for the coordinates every frame.
    """
    if (isinstance(at_middle, int) or (isinstance(at_middle, np.int64))
        or len(at_middle) == 1): at_middle = \
        [trajectory.xyz[i][at_middle] for i in range(len(trajectory.xyz))]

    trajectory.xyz_shifted = np.asarray(
        [(trajectory.xyz[i] + (trajectory.unitcell_lengths[i]/2 - at_middle[i]))%trajectory.unitcell_lengths[i]
        for i in range(len(trajectory.xyz))])
    return trajectory.xyz_shifted


# angle functions by David Swenson
def angle_between_planes(a1, a2, a3, b1, b2, b3):
    a12=np.subtract(a1,a2)
    a23=np.subtract(a3,a2)
    norm_to_A = np.cross(a12, a23)
    b12=np.subtract(b1,b2)
    b23=np.subtract(b3,b2)
    norm_to_B = np.cross(b12, b23)
    return angle_between_vectors(norm_to_A, norm_to_B)


def angle_between_vectors(v1,v2):
    unit_dot = np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
    if (unit_dot > 1.0):
        unit_dot = 1.0
    elif (unit_dot < -1.0):
        unit_dot= -1.0
    return np.arccos(unit_dot)


def rolling_angle(traj, backbone, rollingbase):
    """
    Parameters
    ----------
    traj : mdtraj.Trajectory
        trajectory to analyze
    rolling_atoms : RollingAtoms
        info on which atoms to use; returned from get_baserolling_atoms
    """
    def normalize(vector):
        norms = np.linalg.norm(vector, axis=1)
        return vector / norms.reshape(len(norms), 1)
    def arccos_based_angle(v1, v2):
        return np.arccos(np.clip(np.dot(v1, v2), -1, 1))
    def atan2_based_angle(v1, v2):
        return np.arctan2(np.cross(v1, v2), np.dot(v1, v2))

    # get the vector associated with the rolling base
    bp_v21 = traj.xyz[:,rollingbase[0]] - traj.xyz[:,rollingbase[1]]
    bp_v23 = traj.xyz[:,rollingbase[2]] - traj.xyz[:,rollingbase[1]]
    bp_vector = normalize(np.cross(bp_v21, bp_v23))

    # get the vector associated with the backbone
    bb_vector = traj.xyz[:,backbone[1]] - traj.xyz[:,backbone[0]]
    bb_vector = normalize(bb_vector)
    # calculate angle
    return np.degrees([
        atan2_based_angle(bb, bp)
        for (bb, bp) in zip(bb_vector, bp_vector)
    ])