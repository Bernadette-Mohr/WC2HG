import numpy as np

class CollectiveVariable:
    def __init__(self):
        pass

    @classmethod
    def angle_between_vectors(cls, v1, v2, angle=False):
        # Chose whether to calculate the angle as arctan2 [-180°, 180°] or arccos [0°, 180°]
        if angle:
            normal = np.cross(v1, v2)
            # Use the sign of the z coordinate of the normal to determine if the angle is rotated (counter-)clockwise
            # and reflect the full angle range from -180° to 180° in the 3D case.
            angle = np.degrees(
                np.arctan2(np.linalg.norm(normal), np.dot(v1, v2))
            ) * np.sign(np.dot(normal, np.array([0.0, 0.0, 1.0])))
        else:
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            angle = np.degrees(
                np.arccos(np.clip(np.divide(dot_product, (norm_v1 * norm_v2)), -1.0, 1.0))
            )

        return angle

    @classmethod
    def base_opening_angle(
            cls, snapshot, comI_cv, comII_cv, comIII_cv, comIV_cv, angle_between_vectors_cv, angle
    ):
        """
        Parameters:
        :param snapshot:
        :param comI_cv:
        :param comII_cv:
        :param comIII_cv:
        :param comIV_cv:
        :param angle_between_vectors_cv:
        :param angle:
        :return:
        """
        comI = comI_cv(snapshot)
        comII = comII_cv(snapshot)
        comIII = comIII_cv(snapshot)
        comIV = comIV_cv(snapshot)

        vec_21 = np.subtract(comI, comII)
        vec_23 = np.subtract(comIII, comII)
        vec_24 = np.subtract(comIV, comII)
        norm1 = np.cross(vec_21, vec_23)
        norm2 = np.cross(vec_24, vec_23)

        return angle_between_vectors_cv(norm1, norm2,
                                        angle)  # hard-coded negative sign in the code to Vreede et al., 2019

    @classmethod
    def base_rolling_angle(
            cls, snapshot, backbone_idx, rollingbase_idx, angle_between_vectors_cv, angle
    ):
        """
        Parameters
        ----------
        :param angle: selects whether the angle between two vectors is calculated as atan2 (True) or arccos (False).
        :param snapshot: ops trajectory frame
        :param rollingbase_idx: list of the indices of the N1, N3 and N7 atoms defining the vectors of the rolling base
        :param backbone_idx: list of the P atom indices defining the backbone vector
        :param angle_between_vectors_cv: function to calculate the angle between two vectors.
        """

        def normalize(vector):
            norm = np.linalg.norm(vector)
            if norm == 0:
                return vector
            return vector / norm

        # Get the vectors connecting atoms N3 and N1 and N3 and N7 in the rolling base.
        bp_v1 = np.subtract(
            snapshot.xyz[rollingbase_idx[0]], snapshot.xyz[rollingbase_idx[1]]
        )
        bp_v2 = np.subtract(
            snapshot.xyz[rollingbase_idx[2]], snapshot.xyz[rollingbase_idx[1]]
        )

        # Calculate the normal of the rolling-base vectors
        bp_vector = normalize(np.cross(bp_v1, bp_v2))

        # Get the vector associated with the backbone
        bb_vector = np.subtract(
            snapshot.xyz[backbone_idx[1]], snapshot.xyz[backbone_idx[0]]
        )
        bb_vector = normalize(bb_vector)

        # calculate angle
        return angle_between_vectors_cv(bb_vector, bp_vector, angle)

    # lambda = arctan2(dHG, dWC)
    @classmethod
    def lambda_CV(cls, snapshot, d_WC_cv, d_HG_cv):
        """
        Parameters:
        :param snapshot:
        :param d_WC_cv:
        :param d_HG_cv:
        :return: Single CV combining the hydrogen bond lengths of the WC and the HG pairing.
        """
        d_wc = d_WC_cv(snapshot)
        d_hg = d_HG_cv(snapshot)

        return np.arctan2(d_wc, d_hg)