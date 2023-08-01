import open3d as o3d
import numpy as np
from .crystal_image_tools import ThreeDImage


def o3d_visualization(image3d: ThreeDImage = None, threshold=0.0, cmap='cool', draw_box=True,
                      fill_empty_voxels=True):
    # import matplotlib.cm as cm
    # from matplotlib.colors import Normalize

    # preparing the point cloud
    point_cloud = image3d.get_point_cloud()
    points = point_cloud.indexes
    points_values = point_cloud.atomic_numbers

    filled_voxel_grid = prepare_o3d_voxel_grid(points, create_color_map(points_values, cmap=cmap), shift=1,
                                               threshold=threshold)
    box = o3d_draw_box(image3d.box.n_bins, shift=0.5) if draw_box else None
    empty_voxel_grid = None
    if fill_empty_voxels:
        empty_voxels_indexes = np.argwhere(image3d.get_image()[:, :, :, 0] == 0)
        colors = np.vstack([[0, 0, 0]] * len(empty_voxels_indexes))
        empty_voxel_grid = prepare_o3d_voxel_grid(empty_voxels_indexes, colors, shift=1, threshold=threshold)

    # visualizing the point cloud
    visualization_list = [i for i in [filled_voxel_grid, box, empty_voxel_grid] if i is not None]

    o3d.visualization.draw_geometries(visualization_list)


def create_color_map(values, cmap='cool'):
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    cmap = cm.get_cmap(cmap)
    norm = Normalize(vmin=values.min(), vmax=values.max())
    rgba_values = cmap(norm(values))[:, :3]
    return rgba_values


def prepare_o3d_voxel_grid(points, rgb_colors, voxel_size=1, shift=0.0, threshold=0.0):

    if threshold > 0:
        cond = ~np.all(points > threshold, axis=1)
        points = points[cond]
        rgb_colors = rgb_colors[cond]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points + shift)
    pcd.colors = o3d.utility.Vector3dVector(rgb_colors)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    return voxel_grid


def o3d_draw_box(box_size, shift=0.0):

    points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
    points = np.array(points) * box_size + shift
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [[1, 0, 0] for _ in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set
