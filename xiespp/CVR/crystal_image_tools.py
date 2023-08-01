# import sys
# from pathlib import Path
# sys.path.insert(0, "../formation-energy/")
# sys.path.append(str(Path(sys.argv[0]).resolve().parent.parent))

from .util_cvr import *
from ase import Atoms
from .box import BoxImage
from .channels import channels_gen
from .util_crystal import crystal_parser, atoms_find_min_dist
import typing


class ImageRequirementError(Exception):
    pass


class ThreeDImage:
    """
    The class of 3D crystal images
    Args:
        atoms: The crystal structure in the ASE.Atoms format
        box: Side length of the box in (A) preferably in BoxImage class format. e.g. BoxImage((25,25,25))
        n_bins:  Number of bins along each side of the box (Total voxels = n_bins**3)
        channels: The list of requested channels with order. The values are the columns names for "channels_gen"
            variable.
        filling: (str) An option for how the box is being filled with atoms.
            'vacuum': it doesn't replicate the crystal's unit cell.
            'fill-cut': it completely fills the box by atoms by replicating the unit cell and then removing atoms
                outside the box.
            'fill-no-cut': it replicates the cell but make sures no atoms goes locates outside the box,
                so no cutting is necessary.
            'reps': the number of replication is fixed
        image_id: An optional unique ID for each crystal
        point_cloud: An option to save the point cloud
            None: To not save the point cloud
            'save': To save the point cloud
        # **kwargs:
        #     old_image_method: If True, it uses the image creation method from the synthesizability
        #     prediction project.
    """

    def __init__(
            self,
            atoms,
            # box: BoxImage,
            box: typing.Union[BoxImage, typing.Dict[str, int]],
            channels: list,
            filling: str,
            image_id=None,
            point_cloud=None,
            **kwargs,
    ):
        self.atoms = convert_to_ase_atoms(atoms)
        self.image_id = image_id
        self.image3d = None
        self.point_cloud = point_cloud
        self.point_cloud_rotational = None
        # If box is not a BoxImage instance, create one
        if not isinstance(box, BoxImage):
            assert isinstance(box, dict) and "box_size" in box and "n_bins" in box, \
                "box must be a BoxImage instance or a dictionary with 'box_size' and 'n_bins' keys"
            box_size, n_bins = box['box_size'], box['n_bins']
            box = BoxImage(box_size=box_size, n_bins=n_bins)
        self.box = box
        self.box_rotational = None

        self.filling = filling
        self.channels = channels
        self.meet_requirements = None
        self.params = kwargs.get("image_params", {})

        # lengths_and_angles = ("a", "b", "c", "alpha", "beta", "gamma")
        # lengths_and_angles = dict(zip(lengths_and_angles, self.atoms.cell.cellpar()))
        self.info = {
            # "lengths_and_angles": lengths_and_angles,
            # "rotation": {"x": 0, "y": 0, "z": 0},
        }

        if self.image_id is None:
            if hasattr(self.atoms, "info"):
                if "material_id" in self.atoms.info:
                    self.image_id = self.atoms.info["material_id"]

        self.get_rotational_box()

    # def rotation_atom(self, angles=None, random_angles=True, inplace=False):
    #     """
    #     This function rotates the atoms and the cell of a crystal structure
    #     Args:
    #         angles: (optional) A dictionary with the angles to rotate the crystal. The keys are 'x', 'y', 'z' and the
    #             values are the angles in degrees.
    #         random_angles: If True, the angles will be random.
    #         If False, angles has to be given by the angle parameter.
    #         inplace: If True, the atoms will be rotated inplace. If False, a new atoms object will be returned.
    #     Returns:
    #         ASE.Atoms object of rotated crystal structure
    #     """
    #     if random_angles and angles is None:
    #         angles = (np.random.random(3) * 360).astype("float16")
    #         angles = dict(zip(("x", "y", "z"), angles))
    #     rotated_atoms = self.atoms.copy()
    #     rotated_atoms.info["rotation"] = angles
    #
    #     for axis, deg in angles.items():
    #         rotated_atoms.rotate(deg, axis, rotate_cell=True)
    #
    #     se = self
    #     if inplace:
    #         self.atoms = rotated_atoms
    #     else:
    #         se = copy.deepcopy(self)
    #         se.atoms = rotated_atoms
    #     return se.atoms

    def check_requirements(self, raise_error=False, return_errors=False):
        """
        It checks if a crystal structure meets the requirements of the model
        Args:
            raise_error: If True, it will raise error if it doesn't meet the requirements
            return_errors: If True, it will return the errors as a list

        Returns:
            Ture if it meets the requirement and False otherwise
        """
        # for the case of box=70A and n_bins=128 -> threshold=0.947A
        threshold = self.box.resolution_threshold
        errors = []

        # Check #1 - Check if atoms are too close to each other
        min_nearest_neighbor = -1
        if "min_nearest_neighbor" in self.atoms.info:
            min_nearest_neighbor = self.atoms.info["min_nearest_neighbor"]
        if min_nearest_neighbor < 0:
            min_nearest_neighbor = atoms_find_min_dist(self.atoms)
        if min_nearest_neighbor < threshold:
            errors.append(ImageRequirementError(f"The min nearest neighbor is {min_nearest_neighbor:.4f}"
                                                f"which less than threshold ({threshold} A)"))

        # Check #2 - Check if the crystal is fits in the box
        pos = self.atoms.get_positions()
        atoms_cube = np.max(pos, axis=0) - np.min(pos, axis=0)  # The smallest cube that contains the atoms
        if np.max(atoms_cube) > np.min(self.box.box_size):  # This doesn't work for non-cubic boxes
            errors.append(ImageRequirementError("The crystal does not fit in the box"))

        # Check #3 - Check if there is a strange element among the atomic types
        atomic_numbers = self.atoms.get_atomic_numbers()
        if np.any(atomic_numbers < 1) or np.any(atomic_numbers > 118):
            errors.append(ImageRequirementError("There is a strange element among the atomic types."))

        # Check #4 - Check if the crystal fits in the rotational box
        if np.max(atoms_cube) > np.min(self.box_rotational.box_size):
            errors.append(ImageRequirementError("The crystal does not fit in the rotational box"))

        # Check #5 - Check if the crystal fits in the small box
        if np.max(atoms_cube) > np.min(self.box.small_box.box_size) and self.params.get('req-fit-small-box'):
            errors.append(ImageRequirementError("The crystal does not fit in the small box"))

        if errors and raise_error:
            # raise ExceptionGroup("One or more image requirements were not met: ", errors)
            raise Exception(errors)
        if return_errors:
            return errors
        return True if not errors else False

    def get_point_cloud(self, random_rotation=False):
        """
        Prepares and store a point cloud
        :param random_rotation: randomly rotates atoms before making the point cloud
        :return:
        """
        if self.point_cloud is None and not random_rotation:
            self.point_cloud = SparseImage(label="main").compute_point_cloud(
                atoms=self.atoms,
                box=self.box,
                filling=self.filling,
                shift_atoms_to="middle",
            )
            # self.point_cloud.wrap_point_cloud(box=self.box, filling=self.filling)
            self.point_cloud.compute_voxel_indices(box=self.box)
        elif self.point_cloud is None and random_rotation:
            self.get_rotational_point_cloud()

        pc = self.point_cloud_rotational if random_rotation else self.point_cloud
        # pc['info']['image_id'] = self.image_id
        assert pc is not None
        return pc

    def get_rotational_point_cloud(self):
        """
        Prepares and store a point cloud that is large enough that if it is being rotated
        it still fills the original box.
        """
        if self.box_rotational is None:
            self.get_rotational_box()
        rpc = SparseImage(label=self.box_rotational.label).compute_point_cloud(
            atoms=self.atoms,
            box=self.box_rotational,
            filling=self.filling,
            # wrap_point_cloud=False if 'out-box-okay' in self.filling else True, # We should wrap it anyway
        )
        self.point_cloud_rotational = rpc
        return rpc

    def set_point_cloud(self, point_cloud):
        """
        Sets the point cloud for the image
        """
        assert point_cloud.label in ["main", "large", "small"]
        if point_cloud.label == "main":
            self.point_cloud = point_cloud
        if point_cloud.label == "large" or point_cloud.label == "small":
            self.point_cloud_rotational = point_cloud

    def get_image(self, normalization=True, random_rotation=False, **kwargs):
        """
        It returns the 3D image of the crystal in np.array format with size of (,,,channels)
        Args:
            normalization: If True, the values of image are normalized otherwise they're the real values
            random_rotation: If Ture, it will randomly rotate the crystals before making the image

        Returns:
            np.array
        """
        img = None
        for dtype in (np.float32, np.longdouble):
            try:
                img_sparse = self.point_cloud
                rpc = self.point_cloud_rotational

                # If we are using random rotation:
                if rpc is not None or random_rotation:
                    # This is needed to make sure that the large box is set
                    if self.box_rotational is None:
                        self.get_rotational_box()
                    # Rotation of the points with respect to the center of the large box
                    if random_rotation:
                        img_sparse = rpc.rotate(
                            center=self.box_rotational.center,
                            angles=self.info.get("rotation-fixed-angles", None),
                            inplace=False,
                            np_dot_dtype=dtype,
                        )
                    else:
                        img_sparse = copy.deepcopy(rpc)

                    # if 'vacuum' in self.filling:
                    #     self.find_the_best_shift(img_sparse)
                    # Wrap the point cloud to the original box
                    img_sparse.wrap_point_cloud(box=self.box)
                    # Creating the voxel grid
                    img_sparse.compute_voxel_indices(box=self.box)

                if img_sparse is None:
                    img_sparse = self.get_point_cloud(random_rotation=random_rotation)

                img = point_cloud_to_3d_voxel_image(
                    img_sparse,
                    self.box.n_bins,
                    channels_list=self.channels,
                    normalization=normalization,
                )
            except Exception as e:
                if dtype == np.float32:
                    warnings.warn("WARNING: Error in creating the image with float32. Trying with longdouble")
                    continue
                save_var(locals(), "tmp/exception_get_image.pkl", verbose=True)
                ColorPrint().red_print("Error in get_image: {}".format(e))
                raise e
        return img

    def get_rotational_box(self, filling: str = None, box=None, ):
        """
        Sets the box that is large enough to rotate the atoms and still fill the original box when filling is fill-cut
        Or it sets the box that is small enough to rotate the atoms and still fill the original box
        when filling is fill-no-cut
        Args:
            filling: (optional) The filling to be used for the rotational box. If None, the filling of
                the crystal will be used
            box: (optional) The box to be used for the rotational box. If None, the box of the crystal will be used

        """
        box = box or self.box
        filling = filling or self.filling

        box_rotational = None
        assert filling.split("_")[0] in ["fill-cut", "fill-no-cut", "vacuum"]

        if filling == "fill-cut":
            box_rotational = box.get_box(label='large')

        if filling == "fill-no-cut" or "vacuum" in filling:
            box_rotational = box.get_box(label='small')

        self.box_rotational = box_rotational
        return self.box_rotational

    # def find_the_best_shift(self, sparse_image=None, box: BoxImage = None, ):
    #     """
    #     Finds the best shift for the point cloud to display more atoms in the image
    #     :param sparse_image:
    #     :param box:
    #     :return:
    #     """
    #     # I should find the best shift of the original box before wrapping it
    #     # By checking the different shifts and check which one holds the most atoms
    #
    #     box = box or self.box
    #     shift_types = {
    #         'type': ['middle', 'mean', 'corner'],
    #         'highest_points': 0,
    #         'selected_shift': None,
    #     }
    #     points = sparse_image.positions
    #     for shift_type in shift_types['type']:
    #         p = box.shift_points(points, shift_type)
    #         keep = np.logical_not(box.check_points_outside_of_box(p))
    #         if np.sum(keep) > shift_types['highest_points']:
    #             shift_types['highest_points'] = np.sum(keep)
    #             shift_types['selected_shift'] = shift_type
    #             if shift_types['highest_points'] == len(points):
    #                 break
    #     sparse_image.positions = box.shift_points(points, shift_types['selected_shift'])

    # def __array__(self):
    #     return self.get_image()

    def __repr__(self):
        return f"3D Image - ID: {str(self.image_id)}"


class SparseImage:
    """
    This class stores crystal atoms as a point cloud.
    """
    def __init__(self, label=None, box=None, dtype=np.float32):
        self.label = label
        self.indices = self.atomic_numbers = self.positions = self.inside_box = None
        self.box = box
        self.dtype = dtype
        self.info = {}

    def compute_point_cloud(self, atoms, filling, box=None, shift_atoms_to="middle", wrap_point_cloud=True):
        box = box or self.box
        assert box is not None, "Box is not defined"

        try:
            d = atoms_to_unwrapped_point_cloud(
                atoms,
                filling,
                box,
                reps=None,
                shift_atoms_to=shift_atoms_to,
                dtype=self.dtype,
            )
        except AssertionError as e:
            save_var(locals(), "../utility/tmp/exception_image_tools__compute_point_cloud_.pkl")
            print(f"Atoms: {atoms}")
            print(f"ID: {self.label}", flush=True)
            raise e
        self.positions = d["positions"]
        self.atomic_numbers = d["atomic_numbers"]
        if wrap_point_cloud:
            self.wrap_point_cloud(box=box, filling=filling)
        return self

    def wrap_point_cloud(self, box: BoxImage, filling=None):
        eliminations = box.check_points_outside_of_box(self.positions)
        keep = np.logical_not(eliminations)

        assert (filling != "fill-no-cut") or (np.all(keep)), "Some atoms are outside the box while filling with no cut"

        self.inside_box = keep[keep]
        self.positions = self.positions[keep]
        self.atomic_numbers = self.atomic_numbers[keep]
        self.indices = None if self.indices is None else self.indices[keep]

        return self

    def compute_voxel_indices(self, box: BoxImage = None):
        box = box if box is not None else self.box
        assert box is not None, "Box is not defined"
        assert self.positions is not None, "Point cloud is not computed"

        indices = np_choose_optimal_dtype(box.positions_to_index(self.positions))
        self.indices = indices

        assert np.all(indices.max(axis=0) < box.n_bins), "Some indices are out of the box"
        assert np.all(indices.min(axis=0) >= 0), "Some indices are out of the box"
        # Checking if all atoms seating in distinct voxels
        unique = np.unique(indices, axis=0)
        if not len(unique) == len(indices):
            save_var(locals(), "tmp/exception_image_tools__compute_voxel_indices_.pkl")
            raise ImageRequirementError("Higher resolution or smaller grid is required")
        return self

    def rotate(
            self,
            angles=None,
            axis=None,
            center=(0, 0, 0),
            inplace=False,
            np_dot_dtype=np.float32,
    ):
        """
        Rotates the point cloud
        Args:
            angles: Angle in degrees
            axis: Axis to rotate around
            center: Center of rotation
            inplace: If True, the point cloud is rotated inplace otherwise a new point cloud is returned
            np_dot_dtype: The dtype of the dot product
        """
        if angles is None:
            angles = (np.random.random(3) * 360).astype("float16")
        if axis is None:
            axis = ("x", "y", "z")
        if isinstance(angles, (int, float)):
            angles = [angles]
            axis = [axis]
        assert self.positions is not None, "No positions to rotate"

        si = self if inplace else copy.deepcopy(self)
        si.info["rotation"] = {"angles": angles, "axis": axis, "center": center}

        for an, ax in zip(angles, axis):
            si.positions = rotate_points(
                points=si.positions,
                angle=an,
                vector=ax,
                center=center,
                np_dot_dtype=np_dot_dtype,
            )

        # ##### For debugging: #####
        # p = si.positions
        # p1 = rotate_points(points=p, angle=angles[0], vector=axis[0], center=center, dtype=self.dtype)
        # p2 = rotate_points(points=p1['p'], angle=angles[1], vector=axis[1], center=center, dtype=self.dtype)
        # p3 = rotate_points(points=p2['p'], angle=angles[2], vector=axis[2], center=center, dtype=self.dtype)
        # si.positions = p3['p']
        # return locals()

        return si

    def remove_positions(self):
        self.positions = None
        return self

    def set_box(self, box):
        self.box = box
        return self

    def get_voxel_indices(self):
        indices = self.indices
        atomic_numbers = self.atomic_numbers
        # indices = self.indices[self.inside_box] if only_inside_box else self.indices
        # atomic_numbers = self.atomic_numbers[self.inside_box] if only_inside_box else self.atomic_numbers
        return {"indices": indices, "atomic_numbers": atomic_numbers}

    # def __getitem__(self, key):
    #     return self.__dict__[key]

    def __len__(self):
        return len(self.positions)

    def __repr__(self):
        id_str = f" (ID: {self.label})" if self.label is not None else ""
        pos_str = (
            f"{self.positions.shape[0]} points"
            if self.positions is not None
            else "No points"
        )
        return f"Sparse Image{id_str}" + f": #Points[{pos_str}]"


def get_unit_cell_corner_points(cell):
    """
    It returns a unit cell corner points
    Args:
        cell: Unit cell vectors, 3*3 np.array, each row is a vector, e.g. Atoms.get_cell()

    Returns:
        corner points. np.array (8,3)
    """
    cell_points = np.concatenate((cell, np.array([[0, 0, 0]])), axis=0)
    cell_points = np.concatenate(
        (cell_points, np.array([cell[0, :] + cell[1, :]])), axis=0
    )
    cell_points = np.concatenate(
        (cell_points, np.array([cell[0, :] + cell[2, :]])), axis=0
    )
    cell_points = np.concatenate(
        (cell_points, np.array([cell[2, :] + cell[1, :]])), axis=0
    )
    cell_points = np.concatenate(
        (cell_points, np.array([cell[0, :] + cell[1, :] + cell[2, :]])), axis=0
    )
    return cell_points


def atoms_to_unwrapped_point_cloud(
        atoms_input: Atoms,
        filling: str,
        box: BoxImage,
        reps=None,
        shift_atoms_to="middle",
        dtype=np.float32,
):
    """
    Converts a crystal structure to a sparse 3D image.
    Args:
        atoms_input: ASE Atomic Object
        filling: (str) An option for how the box is being filled with atoms.
            'vacuum': it doesn't replicate the crystal's unit cell.
            'fill-cut': it completely fills the box by atoms by replicating the unit cell and then removing atoms
                outside the box.
            'fill-no-cut': it replicates the cell but make sures no atoms goes locates outside the box,
                so no cutting is necessary.
            'reps': the number of replication is fixed
        box: The box object created in (Angstrom)
        # n_bins: Number of bins along each side of the box (Total voxels = n_bins**3)
        reps: Number of repetitions in along each of the cell axis. default: None, it will be determined by the code
        shift_atoms_to: The position of the unit cell before replication
            'middle': It puts the first unit cell in the middle of the box
            'intact': It doesn't change the positions at all
            'corner': It starts replicating from 0,0,0 corner of the box
        dtype: The data type of the output
    Returns:
        A dictionary containing information of the point cloud
    """

    if filling == "reps":
        atoms_input = atoms_input.repeat(reps)

    # Move atoms to the right position in the box
    positions = atoms_input.get_positions().astype(dtype)
    positions = box.shift_points(positions, where=shift_atoms_to)

    # Filling the box
    df, positions, atom_types = replicate_in_box(
        cell=atoms_input.get_cell(),
        positions=positions,
        box=box,
        filling=filling,
        return_atoms=False,
        atoms_input=atoms_input,
    )

    voxel_indices = None

    # ### For development ###
    if "atoms" in df:
        # ### Compute the voxel indexes ###
        voxel_indices = np_choose_optimal_dtype(box.positions_to_index(positions))

        # ### Checking if the computations are valid ###
        unique, counts = np.unique(
            voxel_indices, axis=0, return_counts=True
        )  # Checking if all atoms seating in distinct voxels
        assert len(unique) == len(
            voxel_indices
        ), "Higher resolution or smaller grid is required"
        # assert (filling != 'fill-no-cut') or (np.all(keep)),
        # 'Some atoms are outside the box while filling with no cut'

        warnings.warn("return_atoms is True, Turn off if you don't need the atoms")
        eliminations = box.check_points_outside_of_box(positions)
        grid_ind_eli = np.arange(len(eliminations))[eliminations]
        atoms = Atoms()
        for a in df["atoms"].tolist():
            atoms += a
        del atoms[[atom for atom in grid_ind_eli]]
        unique, counts = np.unique(atoms.get_atomic_numbers(), return_counts=True)
        unique, counts2 = np.unique(atom_types, return_counts=True)
        assert np.array_equal(
            counts, counts2
        ), "Atoms object does not match with the sparse image"

    return {
        "n_repeats": reps,
        "positions": positions,
        "indices": voxel_indices,
        "atomic_numbers": atom_types,
    }


def replicate_in_box(
        cell,
        positions,
        box: BoxImage,
        filling,
        return_atoms=False,
        atoms_input=None,
        return_cell_ind=False,
):
    """
    It repeats the unit cell in the box to fill the box given the requested 'filling' type
    :param cell: The unit cell (e.g. Atoms.get_cell())
    :param positions: np.array of the positions (e.g. Atoms.get_positions())
    :param box: The defined box in which the cell is being replicated (e.g. BoxImage(20,64))
    :param filling: The type of filling. Refer to ThreeDImage class
    :param return_atoms: If Ture, returns the Atoms objects. In this case atoms_input should be given
    :param atoms_input: The Atoms object used to create all the replication if return_atoms is True
    :return: A data_frame in which a list of the replicated cell exist
    :param return_cell_ind: Adds the cell indexes to
    """

    # ### Algorithm ###
    '''
    Start from the first set of points (unit cell atoms) in the box
    Replicate the set of points based on the v_set (vectors set) and add them to the list of set points
    Repeat the process until the box is filled based on the filling type

    v_set: The set of vectors that are used to replicate the unit cell 
        plus the indices of the unit cell in case of replication
    max_reps: A guess for the maximum number of repetitions needed
    df: A data frame that contains the list of the replicated unit cells
        need_rep: A boolean array that indicates if the unit cell needs to be replicated
        all_in: A boolean array that indicates if all the atoms of the unit cell are inside the box
        all_out: A boolean array that indicates if all the atoms of the unit cell are outside the box
    cell_ind: The index of the unit cell in the df in a separate array
    positions: All the positions
    '''

    # Types of filling the box is limited to the following:
    assert filling.split('_')[0] in ["vacuum", "fill-cut", "fill-no-cut", "reps"]

    # cell = atoms_input.get_cell()
    """
    Reference from the Cell object in ASE:
    This object resembles a 3x3 array whose [i, j]-th element is the jth
    Cartesian coordinate of the ith unit vector.
    The three cell vectors: cell[0], cell[1], and cell[2].
    """

    # The cell vectors and their negative vectors
    v_set = np.concatenate((cell, -cell))
    # Adding the indices of the cell vectors
    v_set = np.concatenate((v_set, np.concatenate((np.identity(3), -np.identity(3)))), axis=1)

    # A guess on the maximum number of repetitions needed
    # Logic: the volume of (box side length + cell diagonal) divided by the cell volume
    # is the maximum number of repetitions
    # This is a guess to speed up the appending and can be exceeded if needed
    cd = np.sum((cell.max(axis=0) - cell.min(axis=0)) ** 2) ** 0.5  # Cell diagonal
    max_rep = int(np.prod(box.box_size + cd * 3) / cell.volume)
    if pd.Series(filling).str.contains("reps|vacuum").bool():
        max_rep = 1

    cond = box.check_points_outside_of_box(positions)
    n_init_p = len(positions)
    positions_init = positions

    positions = np.tile(positions_init, (max_rep, 1))
    cell_ind = np.array([(0, 0, 0)] * max_rep, dtype="int16")
    d_0 = {
        "need_rep": np.array(False).all(),
        "all_in": np.array(False).all(),
        "all_out": np.array(False).all(),
    }

    df = pd.DataFrame([d_0] * max_rep)
    if pd.Series(filling).str.contains("reps|vacuum").bool():
        df.loc[0, :] = [np.array(False).all(), ~cond.any(), cond.all()]
    else:
        df.loc[0, :] = [np.array(True).all(), ~cond.any(), cond.all()]

    i = 0  # The cell ind to replicate from
    r = 0  # The cell ind to store the replication to

    while i <= r:  # As long as there is a cell that needs replication
        if not df.loc[i, "need_rep"]:  # passing if the cell does not need replication
            i += 1
            continue
        df.loc[i, "need_rep"] = False  # Since we are going to replicate the cell, we set the need_rep to False

        for vi in range(6):  # Replicating the cell based on the 6 cell vectors
            n_i = (cell_ind[i] + v_set[vi, 3:])  # The index of the new cell

            # Checking if we already replicated this cell (n_i) and pass on it in this case
            if np.any(np.all(cell_ind[: r + 1] == np.array(n_i), axis=1)):
                continue
            r += 1  # Since we are replicating, we increase the cell ind to store the replication to
            if r == len(cell_ind):  # Expanding cell_ind before allocating more values
                # 100 is a guess for the number of needed cells
                cell_ind = np.concatenate((cell_ind, np.zeros((100, 3), dtype="int16")))
                df = pd.concat((df, pd.DataFrame([d_0] * 100)), axis=0, ignore_index=True)
                positions = np.concatenate((positions, np.tile(positions_init, (100, 1))))
                # raise Exception('The cell_ind array is not big enough to store all the replications')

            # The replication is accepted and storing in process
            p = positions[n_init_p * r: n_init_p * (r + 1)]
            p += np.dot(n_i, cell)
            # p = positions + np.dot(n_i, cell)
            cond = box.check_points_outside_of_box(p)
            # all_in = ~cond.any()
            all_out = cond.all()

            cell_ind[r, :] = tuple(n_i.astype(int).tolist())
            df.loc[r, ["need_rep", "all_in", "all_out"]] = [~all_out, ~cond.any(), all_out, ]
        i += 1

    # warnings.warn('Turn this off')
    # return_cell_ind = True
    if return_cell_ind:  # For debugging
        df["cell_ind"] = cell_ind.tolist()

    df = df.drop(df.index[r + 1:])
    positions = positions[: n_init_p * (r + 1)]
    assert len(positions) // n_init_p == r + 1 == len(df), "There is a bug in the code"
    # positions = np.concatenate([init_positions + np.dot(n_i, cell) for _, n_i in df['cell_ind'].iteritems()], axis=0)

    if "vacuum" in filling:
        assert len(df) == 1, "For vacuum we should not replicate the cell"
        assert df["all_in"].all() or ("out-box-okay" in filling), \
            "For vacuum all the atoms should be inside the box"

    if filling == "fill-no-cut":
        raise Exception("fill-no-cut is not implemented yet")
        # There is a problem with rotation.
        # df = df[df["all_in"]]
        # # TODO: Consider the condition when were in the smaller box and we don't
        # #  care if the main replica is out of the box
        # # but for the replicas we do care if they are out of the box
        # assert len(df) > 0, "For fill-no-cut we should have at least one cell that is fully inside the box"
        # ind = np.concatenate(
        #     [
        #         (i * n_init_p) + np.arange(n_init_p)
        #         for i in df[df["all_in"]].index.to_numpy()
        #     ]
        # )
        # positions = positions[ind]
        # assert ~box.check_points_outside_of_box(
        #     positions).any(), "For fill-no-cut we should not have any points outside of the box"

    # ### df is ready to be used at this point ###
    # Preparing the atom types based on the number of replications
    atom_types = np.tile(atoms_input.get_atomic_numbers(), len(df)).astype("uint8")

    if return_atoms:  # For debugging
        assert isinstance(atoms_input, Atoms)
        atoms = [atoms_input.copy() for _ in range(len(df))]
        # for a, p in zip(atoms, df['positions']):
        #     a.set_positions(p)
        for r in range(len(df)):
            atoms[r].set_positions(positions[r * n_init_p: (r + 1) * n_init_p])
        df["atoms"] = atoms

    """
    To test the results of the replicate_in_box function I run the following lines
    """
    # atoms = Atoms()
    # for a in df.loc[~df['all_out'], 'atoms'].tolist():
    #     atoms += a
    # all_translations = np.matmul(np.array(df.loc[~df['all_out'], 'cell_ind'].to_list()), cell)
    # all_positions = np.concatenate([i + atoms_input.get_positions() for i in all_translations])
    # np.abs(atoms.get_positions() - all_positions).max()  # Proof that the manual replications of atoms works
    # plt.figure()
    # plt.hist(atoms.get_positions().flatten())
    # plt.show()

    return df, positions, atom_types


def rotate_points(points, angle, vector, center=(0, 0, 0), np_dot_dtype=np.float32):
    """
    Rotates points around a vector by an angle
    :param points: np.array of shape (n, 3)
    :param angle: rotation angle in degrees
    :param vector: 'x', 'y', 'z'
    :param center: np.array of shape (3,)
    :param np_dot_dtype: dtype of the np.dot function
    :return: np.array of shape (n, 3)
    """
    # assert points.shape[1] == 3
    dtype = points.dtype
    center = np.array(center).astype(dtype)
    vector_dict = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}
    vector = (
        np.array(vector_dict[vector]).astype(dtype) if vector in vector_dict else vector
    )

    angle *= np.pi / 180
    v = vector / np.linalg.norm(vector)
    c = np.cos(angle)
    s = np.sin(angle)

    p = points - center
    p[:] = (c * p - np.cross(p, s * v)
            + np.outer(
                np.dot(p.astype(dtype=np_dot_dtype), v.astype(dtype=np_dot_dtype)).astype(dtype),
                (1.0 - c) * v,
            )
            + center)
    return p

    # p_center = points - center
    # p_c = c * p_center
    # p_cross = np.cross(p_center, s * v)
    # p_dot = np.dot(np.array(p_center, dtype=np.longdouble), np.array(v, dtype=np.longdouble)).as.type(dtype)
    # c_v = (1.0 - c) * v
    # # p_outer = np.outer(p_dot, c_v)
    # p_outer = np.outer(np.array(p_dot, dtype=np.longdouble), np.array(c_v, dtype=np.longdouble)).as.type(dtype)
    # p = np.array(p_c - p_cross + p_outer + center, dtype=dtype)
    # return locals()
    # p[:] = (c * p -
    #     np.cross(p, s * v) +
    #     np.outer(
    #         np.dot(p, v),
    #         (1.0 - c) * v) +
    #     center)


def point_cloud_to_3d_voxel_image(sparse_image: SparseImage, n_bin, channels_list, normalization, dtype="float32"):
    """
    Converting a single sparse image to a 3D numpy array image with
    Args:
        sparse_image: Sparse image, (The output of atoms_to_point_cloud)
        n_bin: Number of bins along each side of the box (Total voxels = n_bins**3)
        channels_list: The list of requested channels with order
        normalization: True, for outputting normalized images, False for outputting unnormalized images
        dtype: np.dtype of the 3d image
    Returns:
        A [n_bin, n_bin, n_bin, n_channels] numpy array as the 3D image
    """
    # assert pd.Series(channels_list).isin(['atomic_number', 'group', 'period']).all()
    mat2 = np.zeros((n_bin, n_bin, n_bin, len(channels_list)), dtype=dtype)

    channels_df = channels_gen.df
    if normalization:
        channels_df = channels_gen.df_norm
    spi = sparse_image.get_voxel_indices()
    indices = spi["indices"]

    df = pd.DataFrame({"atomic_index": spi["atomic_numbers"]})
    df = df.merge(channels_df, how="left", on="atomic_index")

    for i, c in enumerate(channels_list):
        mat2[indices[:, 0], indices[:, 1], indices[:, 2], i] = df[c].to_numpy()

    # mat3 = np.zeros((n_bin, n_bin, n_bin, len(channels_list)), dtype=dtype)
    # mat3[indices[:, 0], indices[:, 1], indices[:, 2], :] = df[channels_list].to_numpy()
    return mat2


def pymatgen_to_ase_atoms(data):
    """Converts pymatgen Structure to ase.Atoms"""
    try:
        from pymatgen.core import Structure  # , Lattice, PeriodicSite
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError:
        raise ImportError("pymatgen is required for this function")

    structure = None
    mp_data = data

    if (
            'MPDataDoc' in str(type(mp_data)) or
            'ComputedStructureEntry' in str(type(mp_data))
    ):
        structure = mp_data.structure

    if 'PeriodicSite' in str(type(mp_data)):
        structure = Structure.from_sites([mp_data])

    if 'Structure' in str(type(mp_data)):
        structure = mp_data

    atoms = AseAtomsAdaptor.get_atoms(structure)
    if hasattr(mp_data, 'material_id'):
        atoms.info['material_id'] = mp_data.material_id.__str__()

    return atoms


def convert_to_ase_atoms(atoms) -> Atoms:
    if isinstance(atoms, Atoms):
        return atoms
    if isinstance(atoms, (str, Path)):
        return crystal_parser(atoms)
    module_name = atoms.__class__.__module__
    if 'pydantic' in module_name or 'pymatgen' in module_name:
        return pymatgen_to_ase_atoms(atoms)
    raise TypeError(f'Cannot convert {atoms.__class__.__name__} to ase.Atoms')


if __name__ == "__main__":
    # from utility import utility_crystal
    from . import voxel_visualization as vv

    # from utility import util_plotly
    # util_plotly.pycharm_plotly_renderer()

    # file = os.path.expanduser('~/Downloads/1000007.cif')
    # atms = utility_crystal.crystal_parser(filepath=file)
    # df = pd.DataFrame(columns=['atomic_index', 'atomic_number', 'group', 'period'])
    # df.sample()
    # cif_file = util_cod.get_cif_from_cod_website('1100621')
    # cif_file = util_cod.get_cif_from_cod_website('1000007')
    # atms = utility_crystal.crystal_parser(string_input=cif_file)
    # atms = util_mp.mpr.get_atoms('mp-757137')[0]
    # atms = util_mp.mpr.get_atoms('mp-1174200')[0]
    # atms = util_mp.mpr.get_atoms('mp-1201941')[0]
    # atms = util_mp.mpr.get_atoms('mp-1102417')[0]
    # atms = util_mp.mpr.get_atoms('mp-559348')[0]
    # atms = load_var('tmp/sparse_image_error_evl.pkl')['atoms']
    # atms = util_mp.mpr.get_atoms('mp-644276')[0]
    # atms = util_mp.mpr.get_atoms('mvc-847')[0]
    # atms = util_mp.mpr.get_atoms('mp-1220229')[0]
    # atms = util_mp.mpr.get_atoms('mp-1102417')[0]
    # atms = util_mp.mpr.get_atoms('mp-1080319')[0]
    # atms = util_mp.mpr.get_atoms("mp-627334")[0]  # Has issue with vacuum condition
    # atms = util_mp.mpr.get_atoms("mp-1210439")[0]  # Vacuum issue: The rotational sparse image is empty
    # atms = util_mp.mpr.get_atoms("mp-1248716")[0]  # Vacuum issue: The rotational sparse image is empty
    atms = crystal_parser(Path('~/Downloads/0250004.cif').expanduser())
    atoms_find_min_dist(atms)

    # B = BoxImage(box_size=70 / 4, n_bins=128 // 4, label='Test')
    Bo = BoxImage(box_size=70, n_bins=128, label='Original')
    # rotations = {"x": 135, "y": 270, "z": 30}

    atoms3d = ThreeDImage(
        atoms=atms,
        box=Bo,
        channels=["atomic_number", "group", "period"],
        # filling="vacuum_out-box-okay",
        filling='fill-cut',
        cif_id=None,
        # image_params={'req-fit-small-box': True},
    )
    # r1 = atoms3d.check_requirements()

    atoms3d.get_rotational_box()

    atoms3d.get_rotational_point_cloud()

    output = vv.point_cloud_viewer(
        atoms3d.point_cloud_rotational,
        # types=pt.loc[atoms3d.point_cloud_rotational.atomic_numbers, 'name'].to_list(),
        show=False,
        box=Bo, box_outline=True,
        return_comprehensive=True,
    )

    fig = output['fig']
    scatter_colors = output['scatter_colors']

    # df = pt.loc[atoms3d.point_cloud_rotational.atomic_numbers, ['atomic number', 'name']]
    # df = df.merge(pd.DataFrame(scatter_colors, columns=['name', 'color']), how='left', on='name')
    # color_scale = [list(x) for x in df.drop_duplicates(subset='name')[['atomic number', 'color']].to_numpy()]
    #
    # img_1 = atoms3d.get_image()
    # fig = vv.voxel_interactive_plotly(img_1, box_outline=True, )
    # fig.update_traces(colorscale='Viridis')
    #
    # # atoms3d_2 = atoms3d
    # # img_sparse_1 = atoms3d_2.point_cloud_rotational.rotate(
    # #     center=atoms3d_2.box_rotational.box_size / 2,
    # #     inplace=True,
    # #     angles=list(rotations.values()),
    # #     axis=list(rotations.keys()),
    # # )
    #
    # # img_1 = atoms3d.get_image(normalization=False, random_rotation=True)
    #
    # atoms3d = ThreeDImage(
    #     atoms=atms,
    #     box=B,
    #     channels=["atomic_number", "group", "period"],
    #     filling="fill-cut",
    #     cif_id=None,
    # )
    # atoms3d.get_point_cloud(random_rotation=True)
    # atoms3d_2 = atoms3d
    # img_sparse_1 = atoms3d_2.point_cloud_rotational.rotate(
    #     center=atoms3d_2.box_rotational.box_size / 2,
    #     inplace=True,
    #     # angles=list(rotations.values()),
    #     # axis=list(rotations.keys()),
    # )
    #
    # img_2 = atoms3d.get_image(normalization=False, random_rotation=True)
    #
    # import plotly.io as pio
    #
    # pio.renderers.default = "browser"
    #
    # # fig4 = vv.voxel_interactive_plotly(
    # #     image3d=img_1,
    # #     box_outline=True,
    # #     box_filled=False,
    # #     channels="max",
    # #     # filename=Path('~/Downloads/test.html').expanduser(),
    # # )
    # # fig4.show()
    # fig5 = vv.voxel_interactive_plotly(
    #     image3d=img_2,
    #     box_outline=True,
    #     box_filled=False,
    #     channels="max",
    #     # filename=Path('~/Downloads/test.html').expanduser(),
    # )
    # fig5.show()
    print("end")
