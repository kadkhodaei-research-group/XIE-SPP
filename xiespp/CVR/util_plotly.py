import numpy as np
try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
except ImportError:
    px = None
    go = None
    pio = None
    pass


# Helpful link: https://github.com/dillondaudert/voxviz
class VoxelMesh:
    # The code is implemented from the following link and then modifications is made:
    # https://github.com/olive004/Plotly-voxel-renderer/blob/master/VoxelData.py

    def __init__(self, data, threshold=0.0, merge_of_neighbor_voxels=False):
        self.data = data
        self.threshold = threshold
        self.merge_of_neighbor_voxels = merge_of_neighbor_voxels

        self.triangles = np.zeros((np.size(np.shape(self.data)), 1))
        self.xyz = self.get_coords()
        self.x_length = np.size(data, 0)
        self.y_length = np.size(data, 1)
        self.z_length = np.size(data, 2)
        self.vert_count = 0
        self.vertices = self.make_edge_verts()
        self.triangles = np.delete(self.triangles, 0, 1)

        self.intensity = data[data > self.threshold].repeat(4 * 6)

    def get_coords(self):
        # indices = np.nonzero(self.data)
        indices = np.where(self.data > self.threshold)
        indices = np.stack((indices[0], indices[1], indices[2]))
        return indices

    def has_voxel(self, neighbor_coord):
        return self.data[neighbor_coord[0], neighbor_coord[1], neighbor_coord[2]]

    def get_neighbor(self, voxel_coords, direction):
        x = voxel_coords[0]
        y = voxel_coords[1]
        z = voxel_coords[2]
        offset_to_check = CubeData.offsets[direction]
        neighbor_coord = [x + offset_to_check[0], y + offset_to_check[1], z + offset_to_check[2]]

        # return 0 if neighbor out of bounds or nonexistent
        if (any(np.less(neighbor_coord, 0)) | (neighbor_coord[0] >= self.x_length) | (
                neighbor_coord[1] >= self.y_length) | (neighbor_coord[2] >= self.z_length)):
            return 0
        else:
            return self.has_voxel(neighbor_coord)

    def make_face(self, voxel, direction):
        voxel_coords = self.xyz[:, voxel]
        explicit_dir = CubeData.direction[direction]
        vert_order = CubeData.face_triangles[explicit_dir]

        next_i = [self.vert_count, self.vert_count]
        next_j = [self.vert_count + 1, self.vert_count + 2]
        next_k = [self.vert_count + 2, self.vert_count + 3]

        next_tri = np.vstack((next_i, next_j, next_k))
        self.triangles = np.hstack((self.triangles, next_tri))
        # self.triangles = np.vstack((self.triangles, next_triangles))

        face_verts = np.zeros((len(voxel_coords), len(vert_order)))
        for i in range(len(vert_order)):
            face_verts[:, i] = voxel_coords + CubeData.cube_verts[vert_order[i]]

        self.vert_count = self.vert_count + 4
        return face_verts

    def make_cube_verts(self, voxel):
        voxel_coords = self.xyz[:, voxel]
        cube = np.zeros((len(voxel_coords), 1))

        # only make a new face if there's no neighbor in that direction
        dirs_no_neighbor = []
        for direction in range(len(CubeData.direction)):
            if np.any(self.get_neighbor(voxel_coords, direction)) and self.merge_of_neighbor_voxels:
                continue
            else:
                dirs_no_neighbor = np.append(dirs_no_neighbor, direction)
                face = self.make_face(voxel, direction)
                cube = np.append(cube, face, axis=1)

        # remove cube initialization
        cube = np.delete(cube, 0, 1)
        return cube

    def make_edge_verts(self):
        # make only outer vertices
        edge_verts = np.zeros((np.size(self.xyz, 0), 1))
        num_voxels = np.size(self.xyz, 1)
        for voxel in range(num_voxels):
            cube = self.make_cube_verts(voxel)  # passing voxel num rather than
            edge_verts = np.append(edge_verts, cube, axis=1)
        edge_verts = np.delete(edge_verts, 0, 1)
        return edge_verts


class CubeData:
    # all data and knowledge from https://github.com/boardtobits/procedural-mesh-tutorial/blob/master/CubeMeshData.cs
    # for creating faces correctly by direction
    face_triangles = {
        'North': [0, 1, 2, 3],  # +y
        'East': [5, 0, 3, 6],  # +x
        'South': [4, 5, 6, 7],  # -y
        'West': [1, 4, 7, 2],  # -x
        'Up': [5, 4, 1, 0],  # +z
        'Down': [3, 2, 7, 6]  # -z
    }

    cube_verts = [
        [1, 1, 1],
        [0, 1, 1],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 0, 0],
        [0, 0, 0],
    ]

    direction = [
        'North',
        'East',
        'South',
        'West',
        'Up',
        'Down'
    ]

    opposing_directions = [
        ['North', 'South'],
        ['East', 'West'],
        ['Up', 'Down']
    ]

    # xyz direction corresponding to 'Direction'
    offsets = [
        [0, 1, 0],
        [1, 0, 0],
        [0, -1, 0],
        [-1, 0, 0],
        [0, 0, 1],
        [0, 0, -1],
    ]


def box_filled_mesh(box=None, point_1=None, point_2=None, opacity=0.1, color='#DC143C'):
    if box is not None:
        point_1 = box.corners.min(axis=0)
        point_2 = box.corners.max(axis=0)
    if point_2 is None:
        point_2 = [0, 0, 0]
    if isinstance(point_1, int) or isinstance(point_1, float):
        point_1 = [point_1, point_1, point_1]
    x = [point_1[0], point_2[0]]
    y = [point_1[1], point_2[1]]
    z = [point_1[2], point_2[2]]
    # x=[0, 32]
    # y=[0, 32]
    # z=[0, 32]
    mesh = go.Mesh3d(
        # 8 vertices of a cube
        x=[x[0], x[0], x[1], x[1], x[0], x[0], x[1], x[1]],
        y=[y[0], y[1], y[1], y[0], y[0], y[1], y[1], y[0]],
        z=[z[0], z[0], z[0], z[0], z[1], z[1], z[1], z[1]],

        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        opacity=opacity,
        color=color,
        flatshading=True
    )
    fig = go.Figure(data=[mesh])
    return fig


def box_outline(box=None, ):
    if not isinstance(box, list):
        box = [box]
    connectors = np.array([0, 1, 3, 2, 0, 4, 5, 7, 6, 4, 0, 2, 6, 4, 5, 1, 3, 7, 3, 1])
    figs = []

    for b in box:
        data = b.corners[connectors]
        fig = px.line_3d(x=data[:, 0], y=data[:, 1], z=data[:, 2])
        figs.append(fig)
    return figs


def save_or_show_figure(fig, filename=None, link=True, return_iframe=False, show=False):
    if filename is not None:
        fig.write_html(filename)
        # display_list = []
        if link:
            from IPython.display import display, HTML
            # display_list.append(HTML(f"""<a href="{filename}">Link to plot: {filename}</a>"""))
            display(HTML(f"""<a href="{filename}">Link to plot: {filename}</a>"""))
        if return_iframe:
            from IPython.display import IFrame, display
            width = 800
            height = 800
            # noinspection PyBroadException
            try:
                width = fig.layout.width * 1.1
                height = fig.layout.height * 1.1
            except Exception:
                pass

            i_frame = IFrame(filename, width=width, height=height)
            if show:
                display(i_frame)
            return fig, i_frame
    if show:
        # from IPython.display import display
        # print(f"Displaying plot: {filename}")
        # display(fig)
        fig.show()
    return fig


def prepare_plotly_for_display(renderer='jupyterlab'):
    renderer_dict = {
        'jupyterlab': 'iframe_connected',
        'browser': 'browser',
    }
    # import plotly.io as pio
    pio.renderers.default = renderer_dict[renderer]


def pycharm_plotly_renderer():
    # import plotly.io as pio
    pio.renderers.default = 'browser'


def multiple_figures(all_figures):
    import functools
    import operator
    fig = go.Figure(data=functools.reduce(operator.add, [_.data for _ in all_figures]))
    return fig


def apply_layout(fig, width=7, height=5, ):
    fig.update_layout(
        width=width * 80,
        height=height * 80,
        paper_bgcolor='rgba(0,0,0,0)',  # remove background color
        plot_bgcolor='rgba(0,0,0,0)',  # remove plot area background color
        font=dict(family="Helvetica", size=12),  # change font family and size
    )
    return fig


def remove_background(f, remove_axis=False, transparent_background=True):
    # https://community.plotly.com/t/scatter3d-background-plot-color/38838/2
    f.update_layout(
        scene=dict(
            xaxis=dict(
                backgroundcolor="rgba(0, 0, 0,0)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
                showline=False,
                showticklabels=False,
                ticks=""),
            yaxis=dict(
                backgroundcolor="rgba(0, 0, 0,0)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
                showline=False,
                showticklabels=False,
                ticks=""),
            zaxis=dict(
                backgroundcolor="rgba(0, 0, 0,0)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
                showline=False,
                showticklabels=False,
                ticks=""),
        ),
    )

    if remove_axis:
        f.update_layout(
            scene=dict(
                xaxis=dict(showticklabels=False, showline=False, ticks=""),
                yaxis=dict(showticklabels=False, showline=False, ticks=""),
                zaxis=dict(showticklabels=False, showline=False, ticks=""),
                xaxis_title='',
                yaxis_title='',
                zaxis_title='',
            )
        )

    if transparent_background:
        f.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")


def hex_to_rgb(hexa):
    rgb = tuple(str(int(hexa.lstrip('#')[i:i + 2], 16)) for i in (0, 2, 4))
    return 'rgb(' + ', '.join(rgb) + ')'
