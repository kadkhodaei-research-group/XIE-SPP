# from .util_cvr import *
import numpy as np
from collections.abc import Iterable
import itertools
import pandas as pd


class BoxImage:
    """
    A box embracing all the atoms. This object is capable to tell which atoms are inside or
    outside the box.
    param edge: in (A)
        number: Considers a cube with a side length of edge
        (a,b,c): Considers a rectangular cube with different side lengths
        n_bins: Number of bins along each edge of the box
    """

    def __init__(self, box_size: object, n_bins: int = None, label: str = 'original'):
        """
        Creates a box
        :param
        edge: in (A)
            number: Considers a cube with a side length of edge
            (a,b,c): Considers a rectangular cube with different side lengths
        n_bins: Number of bins along each edge of the box
        label: The name of the box. If 'original' is chosen, larger and smaller box are made automatically.
        """
        if not isinstance(box_size, Iterable):
            box_size = np.ones(3) * box_size
        assert len(box_size) == 3

        self.box_size = np.array(box_size).astype(np.float32)
        self.n_bins = n_bins

        self.shift = np.zeros(3)
        self.corners = None
        self.center = None
        self.resolution_threshold = None

        self.small_box = None
        self.large_box = None

        self.label = label
        self.compute_box_params()
        if self.label.lower() == 'original':
            self.make_rotational_boxes()
            self.image_resolution_threshold()
        self.info = {}

    def compute_box_params(self):
        """
        Computes the box parameters.
        This function is called when the box is created or when the box is shifted.
        :return:
        """
        unit_cube_corner_points = np.array([p for p in itertools.product([0, 1], repeat=3)])
        self.corners = unit_cube_corner_points * self.box_size + self.shift
        self.center = np.mean(self.corners, axis=0)

    def make_rotational_boxes(self):
        """
        Creates small and large boxes for rotating without losing atoms.
        Small box:
            The diagonal of the small box is the edge of the original box.
            This is to make sure no point is lost when rotating.
        Large box:
            The edge of the large box is the diagonal of the original box.
            This is to make sure the original box is inside the large box when rotating.
        :return:
        """
        if self.large_box or self.small_box:
            return self

        # The edge of the box is the diagonal of the original box
        large_box_size = (self.box_size ** 2).sum() ** 0.5
        # The diagonal of the box is the edge of the original box
        small_box_size = np.min(self.box_size / 3 ** 0.5)

        self.small_box = BoxImage(box_size=small_box_size, label='small')
        self.large_box = BoxImage(box_size=large_box_size, label='large')

        # The large box starts at (0,0,0) and the original and small boxes start at the center of the large box
        self.set_shift_box(shift_to=self.large_box.center - self.center)
        self.small_box.set_shift_box(shift_to=self.large_box.center - self.small_box.center)
        return self

    def get_box(self, label: str):
        """
        Returns the box with the given label.
        :param label:
        :return:
        """
        if not self.large_box:
            self.make_rotational_boxes()
        if label.lower() == 'small':
            return self.small_box
        if label.lower() == 'large':
            return self.large_box
        assert 'Wrong input label'

    def set_shift_box(self, shift_to):
        """
        Moves the box to a new location.
        :param shift_to: vector in np.array(x,y,z) form
        :return:
        """
        self.shift = np.array(shift_to)
        self.compute_box_params()
        return self

    def positions_to_index(self, positions):
        """
        Converts a list of positions (x,y,z) to indexes.
        param positions: np.array of the positions
            array[0]: The first point
        return: A np.array of the indexes
        """
        return np.floor(
            self.shift_points(positions.astype(np.float64))  # As.type is important, because it copies the array.
            # We don't want to change the values of the original array
            / self.box_size
            * self.n_bins
        ).astype(np.int16)

    def find_min_dist_cor(self, positions):  # This is not being used
        """
        Finds the maximum of (the minimum distance of the points to each corner)
        Args:
            positions: Positions of atoms

        Returns:

        """
        min_dist = 0
        for point in self.corners:
            # The min. distant of points to a corner
            dist = min(
                np.sum(
                    ((self.shift_points(positions.copy()) - point) ** 2).reshape((-1, 3)),
                    axis=1,
                )
                ** 0.5
            )
            min_dist = max(dist, min_dist)
        return min_dist

    def check_box_filling(self, positions):  # This is not being used
        """
        Checks if a box is completely filled with points (Atoms)
        Args:
            positions: Positions of atoms

        Returns:
        True if Box is full, False if replicating unit cell can fill the box
        """
        box_corner_points = self.corners
        # To check if any atom can be found beyond the corner points of the box
        positions = self.shift_points(positions.copy())

        for point in box_corner_points:
            # point = box_corner_points[i]
            check = np.array([True] * len(positions))
            # Going over x,y,z separately
            for j in range(3):
                if point[j] == 0:
                    check = (positions[:, j] < point[j]) & check
                else:
                    check = (positions[:, j] > point[j]) & check
            if not np.any(check):
                # No point found beyonds it, so there is an error in the structure
                return False
        # Check if passed the box limitations
        if np.min(np.max(positions, axis=0), axis=0) < np.amin(self.box_size, axis=0):
            return False
        # Box is full:
        return True

    def check_points_outside_of_box(self, positions, any_p=False) -> np.array:
        """
        Checks which points are located inside and outside the box
        Args:
            positions: Position of atoms
            any_p: if any point is outside the box returns Ture, otherwise False

        Returns:
            True for the points outside the box, and False for the points inside the box
        """
        upper = np.any(positions >= self.corners.max(axis=0), axis=1)
        lower = np.any(positions < self.corners.min(axis=0), axis=1)
        points_passed_box = np.logical_or(upper, lower)
        if any_p:
            return bool(np.any(points_passed_box))
        return points_passed_box

    def shift_points(self, points: np.ndarray, where: str = "shifted_box") -> np.ndarray:
        """
        It shifts the points (the atoms in from ASE.Atoms) in the box
        :param points: np.array(:,3) positions of the points
        :param where:
            'corner': Moves to the corner at (0,0,0)
            'middle': Moves the unit cell to the middle of the box based on their min and max values
            'mean': Moves the unit cell to the middle of the box based on their mean values
            'same': Does not move the unit cell
            'shifted_box': Returns the relative position of the points relative to the box.
        :return: The new positions
        """
        assert any(pd.Series(["corner", "middle", "mean", "same", "shifted_box"]).isin([where]))
        d_type = points.dtype
        d_type_accurate = np.float64
        if not np.can_cast(d_type, d_type_accurate):
            d_type_accurate = d_type

        b = c = np.zeros(3)
        # b: Where we want to move the points to
        # c: From where we want to move the points

        if where == "corner":
            # b = np.zeros(3)
            b = self.corners.min(axis=0)
            c = np.min(points, axis=0)

        if where == "middle":
            # b = self.corners.mean(axis=0)
            b = self.center
            c = (np.max(points, axis=0) + np.min(points, axis=0)) / 2

        if where == "mean":
            b = self.corners.mean(axis=0)
            c = np.mean(points, axis=0)

        if where == "shifted_box":
            c = self.shift

        shift = b.astype(d_type_accurate) - c.astype(d_type_accurate)
        # points += shift  # This changes the original array
        points = points + shift
        return points.astype(d_type)

    def image_resolution_threshold(self):
        """
        :return: Given the edges and n_bins returns the closest possible distance of atoms acceptable by model
        """
        threshold = np.sum((self.box_size / self.n_bins) ** 2) ** 0.5
        self.resolution_threshold = np.ceil(threshold * 1e4) / 1e4  # To avoid floating point errors
        return self.resolution_threshold

    # def rotate(
    #         self,
    #         angles=None,
    #         axis=None,
    #         center=None,
    #         inplace=False,
    #         np_dot_dtype=np.float32,
    # ):
    #     """
    #     Rotates the box
    #     Args:
    #         angles: Angle in degrees
    #         axis: Axis to rotate around
    #         center: Center of rotation
    #         inplace: If True, the point cloud is rotated inplace otherwise a new point cloud is returned
    #         np_dot_dtype: The dtype of the dot product
    #     """
    #     if axis is None:
    #         axis = ("x", "y", "z")
    #     if center is None:
    #         center = self.center
    #
    #     box_copy = self if inplace else copy.deepcopy(self)
    #     box_copy.info["rotation"] = {"angles": angles, "axis": axis, "center": center}
    #
    #     for an, ax in zip(angles, axis):
    #         box_copy.corners = rotate_points(
    #             points=box_copy.corners,
    #             angle=an,
    #             vector=ax,
    #             center=center,
    #             np_dot_dtype=np_dot_dtype,
    #         )
    #     return box_copy

    def __array__(self):
        return self.corners

    # def set_bins(self, bins):
    #     self.n_bins

    def __repr__(self):
        box_str = (
            f"{self.box_size[0]:.1f}"
            if np.all(self.box_size == self.box_size[0])
            else str(self.box_size)
        )
        return f"Image box: Box size={box_str}; # Bins={str(self.n_bins)}"
