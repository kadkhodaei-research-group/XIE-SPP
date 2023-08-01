# Because of VSCode issue:
# from sklearn.feature_extraction import img_to_graph
# import sys
# sys.path.insert(0, '../formation-energy/')

# from utility.utility_general import *
from .util_cvr import *
from . import crystal_image_tools
from . import util_plotly
# import plotly.express as px
import matplotlib.pyplot as plt
try:
    # import plotly.offline as py
    import plotly.graph_objs as go
    import plotly.express as px
    import plotly.io as pio
    # any other necessary plotly imports
    _PLOTLY_AVAILABLE = True
except ImportError:
    px = None
    go = None
    pio = None
    _PLOTLY_AVAILABLE = False


axis_names = {'x': 0, 'y': 1, 'z': 2, 0: 'x', 1: 'y', 2: 'z'}


def plot_2d_cross_sections(image3d, cmap='cool', cross_sections=None,
                           cross_sections_ind=None, cross_axis=None):
    # https://scikit-image.org/docs/stable/auto_examples/applications/plot_3d_image_processing.html
    if cross_axis is None:
        cross_axis = [0, 2]

    if isinstance(image3d, crystal_image_tools.ThreeDImage):
        image3d = image3d.get_image()[:, :, :, 0]
    image = image3d

    if cross_sections is None:
        cross_sections = [0, 25, 50, 75, 100]

    if cross_sections_ind is None:
        cross_sections_ind = (image.shape[0]) * np.array(cross_sections) // 100
    else:
        cross_sections_ind = np.array(cross_sections_ind)
        cross_sections = cross_sections_ind * 100 / (image.shape[0] - 1)

    assert (cross_sections_ind is not None) and (cross_sections is not None) and (cross_axis is not None)

    # warnings.warn("Remove the following line when the bug is fixed: ")
    # cross_sections_ind = [0, 1, 2, 3, 4]

    # Preparing the 3d image
    img_cross = get_2d_cross_sections_from_3d(image, cross_sections_ind, cross_axis)

    # Preparing the 2d image
    n_rows = len(img_cross)
    n_cols = len(list(img_cross.values())[0])
    _, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(1.5 * n_cols, 1.5 * n_rows))

    v_min = image.min()
    v_max = image.max()

    img_list = [x for xs in list(img_cross.values()) for x in xs]

    for ax, im in zip(axes.flatten(), img_list):
        ax.imshow(im, cmap=cmap, vmin=v_min, vmax=v_max)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax, ca in zip(axes[:, 0], cross_axis):
        ax.set_ylabel(f"Fixing {chr(120 + ca)} axis",
                      # fontsize=20,
                      )
    for r in range(n_rows):
        for c in range(n_cols):
            axes[r, c].set_xlabel(f"{chr(120 + cross_axis[r])} =  {cross_sections_ind[c]}, "
                                  f"({cross_sections[c]:.1f}%)",
                                  # fontsize=20,
                                  )

    # plt.suptitle('Cross section visualization of 3d image', fontsize=20)
    # fig.tight_layout()
    plt.show()
    return axes


def get_2d_cross_sections_from_3d(image3d: np.ndarray, cross_sections=None, cross_axis=None):
    image = image3d
    img_cross = {}

    for ca in cross_axis:
        ind = [0, 1, 2]
        ind.remove(ca)
        img_cross[ca] = []

        for c in cross_sections:
            tmp = None
            if ca == 0:
                tmp = image[c, :, :]
            elif ca == 1:
                tmp = image[:, c, :]
            elif ca == 2:
                tmp = image[:, :, c]
            img_cross[ca].append(tmp)
    return img_cross


def volume_slice_visualizer_plotly(image3d, colorscale=None, filename=None,
                                   link=False, show=False, return_iframe=False,
                                   ):
    # https://plotly.com/python/visualizing-mri-volume-slices/
    # import ipywidgets as widgets
    # from IPython.display import display
    # import time
    # import numpy as np
    # from skimage import io
    # import plotly.graph_objects as go

    if isinstance(image3d, crystal_image_tools.ThreeDImage):
        image3d = image3d.get_image()[:, :, :, 0]

    vol = image3d
    r = 4
    vol = vol.repeat(r, axis=0).repeat(r, axis=1)
    volume = vol.T

    r, c = volume[0].shape
    nb_frames = len(volume)

    # Create a figure with all the frames
    fig = go.Figure(frames=[go.Frame(data=go.Surface(
        z=((nb_frames - 1) - k) * np.ones((r, c)),
        surfacecolor=np.flipud(volume[nb_frames - 1 - k]),
        cmin=vol.min(), cmax=vol.max(),
    ),
        name=str(k)  # you need to name the frame for the animation to behave properly
    )
        for k in range(nb_frames)])

    # Add the initial frame
    fig.add_trace(go.Surface(
        z=(nb_frames - 1) * np.ones((r, c)),
        surfacecolor=np.flipud(volume[-1]),
        colorscale=colorscale,
        cmin=vol.min(), cmax=vol.max(),
        colorbar=dict(thickness=20, ticklen=4)
    ))

    # Add the animation
    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    # Add the slider
    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]

    fig.update_layout(
        title='2D scanning of volumetric data',
        width=600,
        height=600,
        scene=dict(
            # zaxis=dict(range=[-0.1, 6.8], autorange=False),
            zaxis=dict(range=[0, volume.shape[0]], autorange=False),
            aspectratio=dict(x=1, y=1, z=1),
        ),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, frame_args(50)],
                        "label": "&#9654;",  # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;",  # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
        sliders=sliders
    )

    fig = util_plotly.save_or_show_figure(fig, filename=filename, link=link,
                                          show=show, return_iframe=return_iframe)

    return fig


def solid_3d_visualizer(image3d, ):
    # from mpl_toolkits.mplot3d import Axes3D

    def make_ax(grid=False):
        fig = plt.figure()
        #     ax = fig.gca(projection='3d')
        axp = fig.add_subplot(111, projection="3d")
        axp.set_xlabel("x")
        axp.set_ylabel("y")
        axp.set_zlabel("z")
        axp.grid(grid)
        return axp

    from matplotlib import colors as mcolors

    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

    # Sort colors by hue, saturation, value and name.
    by_hsv = sorted(
        (tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
        for name, color in colors.items()
    )
    sorted_names = [name for hsv, name in by_hsv]
    np.random.seed(0)
    np.random.shuffle(sorted_names)
    color_names = sorted_names

    if isinstance(image3d, crystal_image_tools.ThreeDImage):
        image3d = image3d.get_image()[:, :, :, 0]
    filled = image3d

    # pl = sns.color_palette("dark")
    colors = np.empty(filled.shape, dtype=object)
    # colors = np.zeros(sphere.shape + (3,))

    filled_colors = filled
    if np.unique(filled).size > len(color_names):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=20, random_state=0).fit_predict(filled.flatten().reshape(-1, 1))
        filled_colors = kmeans.reshape(filled.shape)

    for i in np.unique(filled_colors):
        colors[filled_colors == i] = color_names.pop(0)

    ax = make_ax(True)
    ax.voxels(filled, edgecolors="gray", shade=False,
              facecolors=colors
              )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.show()


def image_slice_view_plotly(image3d, slice_along_axis_n=0, binary_string=False,
                            filename=None, link=False, show=False, return_iframe=False, channels=0,
                            split_channels=False):
    # https://scikit-image.org/docs/stable/auto_examples/applications/plot_3d_interaction.html
    # import plotly.express as px

    image3d = prepare_image3d_channels(image3d, channels)

    labels = {}

    channel_axis_to_split = None if image3d.ndim == 3 else 3
    if split_channels:
        channel_axis_to_split = image3d.ndim - 1

    if slice_along_axis_n is not None:
        labels['animation_frame'] = axis_names[slice_along_axis_n]
    if channel_axis_to_split is not None:
        labels['facet_col'] = 'channel'
    if channel_axis_to_split == -1:
        channel_axis_to_split = len(image3d.shape) - 1

    if channel_axis_to_split is not None:
        if image3d.shape[channel_axis_to_split] / 4 > 4:
            image3d = image3d[..., :16]

    fig = px.imshow(
        image3d,
        zmin=np.min(image3d), zmax=np.max(image3d),
        animation_frame=slice_along_axis_n,
        facet_col=channel_axis_to_split,
        binary_string=binary_string,
        labels=labels,
        facet_col_wrap=min(image3d.shape[-1], 4),
        # zmin=image3d.min(),
        # color_continuous_scale=px.colors.sequential.Jet,
    )
    fig.update_layout(title=f'Slices of a 3D array as 2D images, [channel(s)={channels}]')
    # plotly.io.show(fig)

    fig = util_plotly.save_or_show_figure(fig, filename=filename, link=link, show=show,
                                          return_iframe=return_iframe)

    return fig


def volume_visualizer_plotly(image3d, filename=None, link=False, show=False, return_iframe=False):
    # https://plotly.com/python/3d-volume-plots/
    # import plotly.graph_objects as go
    # import numpy as np

    if isinstance(image3d, crystal_image_tools.ThreeDImage):
        image3d = np.average(image3d.get_image(), axis=3)
    vol = image3d

    # X, Y, Z = np.mgrid[-1:1:32j, -1:1:32j, -1:1:32j]
    xx = np.linspace(0, vol.shape[0] - 1, vol.shape[0])
    yy = np.linspace(0, vol.shape[1] - 1, vol.shape[1])
    zz = np.linspace(0, vol.shape[2] - 1, vol.shape[2])
    x, y, z = np.meshgrid(xx, yy, zz, indexing='ij')
    values = vol

    fig = go.Figure(data=go.Volume(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=values.flatten(),
        isomin=values[values != 0].min(),
        # isomin=values.min(),
        isomax=values.max(),
        opacity=0.1,  # needs to be small to see through all surfaces
        surface_count=21,  # needs to be a large number for good volume rendering
    ))
    # fig.show()
    fig.update_layout(
        title='Volumetric visualization of a 3D array',
        # width=600,
        # height=600,
    )

    fig = util_plotly.save_or_show_figure(fig, filename=filename, link=link, show=show,
                                          return_iframe=return_iframe)

    return fig


def prepare_image3d_channels(image3d, channels=None) -> np.ndarray:
    if isinstance(image3d, crystal_image_tools.ThreeDImage):
        image3d = image3d.get_image()
    if len(image3d.shape) == 3:
        return image3d
    if len(image3d.shape) == 2:
        return image3d.reshape(image3d.shape + (1,))
    if len(image3d.shape) == 1:
        return image3d.reshape(image3d.shape + (1, 1))
    if channels is None:
        channels = 0
    if channels == -1 or channels == 'all':
        channels = range(image3d.shape[-1])
    if type(channels) is int or type(channels) is range:
        image3d = image3d[:, :, :, channels]
    if channels == 'average':
        image3d = np.average(image3d, axis=3)
    if channels == 'max':
        image3d = np.max(image3d, axis=3)
    return image3d


def voxel_interactive_plotly(image3d, filename=None, link=False, show=False, channels=0, return_iframe=False,
                             box_outline=False, box_filled=False, box=None, return_separately=False,
                             colors=None):
    # https://plotly.com/python/3d-volume-plots/
    # import plotly.graph_objects as go
    # import numpy as np

    image3d = prepare_image3d_channels(image3d, channels)

    if len(image3d) == 1:
        image3d = np.tile(image3d, (2, 2, 2))

    all_fig = []

    if box is None:
        box = crystal_image_tools.BoxImage(box_size=image3d.shape[0], n_bins=image3d.shape[0], label='box')
    if box_outline:
        all_fig += util_plotly.box_outline(box)
    if box_filled:
        all_fig.append(util_plotly.box_filled_mesh(box))

    if colors is None:
        voxels = util_plotly.VoxelMesh(image3d)
        mesh = go.Mesh3d(
            x=voxels.vertices[0],
            y=voxels.vertices[1],
            z=voxels.vertices[2],
            i=voxels.triangles[0],
            j=voxels.triangles[1],
            k=voxels.triangles[2],
            intensity=voxels.intensity,
            cmin=np.min(image3d),
        )
        all_fig.append(go.Figure(data=mesh))

    if colors is not None:
        color_num = np.unique(image3d)[1:]
        assert len(colors) == len(color_num)
        for i, color in enumerate(colors):
            voxels = util_plotly.VoxelMesh(image3d == color_num[i])
            mesh = go.Mesh3d(
                x=voxels.vertices[0],
                y=voxels.vertices[1],
                z=voxels.vertices[2],
                i=voxels.triangles[0],
                j=voxels.triangles[1],
                k=voxels.triangles[2],
                color=color
            )
            all_fig.append(go.Figure(data=mesh))
    # fig = go.Figure(data=mesh_fig.data + box[0].data + box[1].data)
    # import functools, operator
    # fig = go.Figure(data=functools.reduce(operator.add, [_.data for _ in all_fig]))

    if return_separately:
        return all_fig
    fig = util_plotly.multiple_figures(all_fig)
    fig.update_layout(title=f'Voxelized visualization of a 3D array, [channel(s)={channels}]')
    fig.update_layout(scene=dict(
        xaxis_showticklabels=False,
        yaxis_showticklabels=False,
        zaxis_showticklabels=False,
        xaxis_title=' ',
        yaxis_title=' ',
        zaxis_title=' ',
    ))

    fig = util_plotly.save_or_show_figure(fig, filename=filename, link=link, show=show,
                                          return_iframe=return_iframe)

    return fig


def scatter_plotly(image3d=None, indeces=None, color=None, box_outline=False, channels=None,
                   filename=None, link=False, show=False, return_iframe=False, ):
    image3d = prepare_image3d_channels(image3d, channels)
    if indeces is None:
        indeces = np.where(image3d != 0)
    if color is None:
        color = image3d[indeces]
    # import plotly.express as px
    all_fig = []
    fig = px.scatter_3d(x=indeces[0], y=indeces[1], z=indeces[2], color=color)
    all_fig.append(fig)
    if box_outline:
        assert image3d is not None, 'image3d must be provided'
        all_fig.append(util_plotly.box_outline(image3d.shape[0]))
    fig = util_plotly.multiple_figures(all_fig)

    fig.update_layout(title='Scatter visualization of a 3D array')

    fig = util_plotly.save_or_show_figure(fig, filename=filename, link=link, show=show,
                                          return_iframe=return_iframe)

    return fig


def point_cloud_viewer(points, types=None, box_outline=False, box=None, box_filled=False, filename=None, link=False,
                       show=False,
                       return_iframe=False, return_comprehensive=False, ):
    # from utility import crystal_image_tools
    # if isinstance(points, crystal_image_tools.SparseImage):
    if str(type(points)).find('SparseImage') != -1:
        sp_img = points
        points = sp_img.positions
        types = types or sp_img.atomic_numbers
        box = box or sp_img.box

    # points1 = box
    # points2 = None
    # if isinstance(box, crystal_image_tools.BoxImage):
    #     points1 = box.corners.min(axis=0)
    #     points2 = box.corners.max(axis=0)

    all_fig = []
    if box_outline:
        all_fig += util_plotly.box_outline(box=box)
    if box_filled:
        all_fig.append(util_plotly.box_filled_mesh(box=box))
    if types is None:
        types = np.zeros(points.shape[0]) + 1

    df = pd.DataFrame(points, columns=['x', 'y', 'z'])
    df['color'] = types
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='color',
                        color_continuous_scale=px.colors.sequential.Viridis)
    scatter_colors = [(i.legendgroup, i.marker.color) for i in fig.data]
    all_fig.append(fig)
    fig = util_plotly.multiple_figures(all_fig)

    fig.update_layout(title='Point cloud visualization')
    fig = util_plotly.save_or_show_figure(fig, filename=filename, link=link, show=show,
                                          return_iframe=return_iframe)
    if return_comprehensive:
        return {
            'fig': fig,
            'scatter_colors': scatter_colors,
            'all_fig': all_fig,
        }
    return fig


def image2d_slice_view(image3d, filename='Image 2D slice view', cmap=None):
    max_col = 8
    image2d = image3d

    if len(image2d.shape) == 1:
        all_primes = prime_factors(image2d.size)
        image2d = image2d.reshape(int(np.prod(all_primes[::2])), int(np.prod(all_primes[1::2])))
        image2d = image2d[..., np.newaxis]

    n_channels = image2d.shape[-1]

    n_cols = min(max_col, n_channels)
    n_rows = int(np.ceil(n_channels / n_cols))

    single_image_size = 2
    fig = plt.figure(figsize=(n_cols * single_image_size, n_rows * single_image_size))
    # Figure title
    fig.suptitle(filename, fontsize=14)

    # Create subplots
    ax = None
    for i in range(n_channels):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        ax.imshow(image2d[:, :, i], cmap=cmap)
        ax.axis('off')
        ax.set_title(f'Channel {i}')

    return fig, ax


def pycharm_plotly_renderer():
    # import plotly.io as pio
    pio.renderers.default = 'browser'


if __name__ == '__main__':
    # from utility import util_mp
    # import plotly.io as pio

    pio.renderers.default = "browser"
    # pio.renderers.default = 'png'

    # atoms = util_mp.mpr.get_atoms('mp-1953')[0]
    # atoms = util_mp.mpr.get_atoms('mp-628617')[0]

    # img = crystal_image_tools.ThreeDImage(atoms,
    #                                       box=crystal_image_tools.BoxImage(box_size=70 / 4, n_bins=128 // 4),
    #                                       channels=['atomic_number', 'group', 'period'],
    #                                       filling='fill-cut'
    #                                       )
    # pc = img.get_rotational_point_cloud()
    # point_cloud_viewer(pc, box_outline=True, box=img.box, show=True, box_filled=True)
    # img[img<0.99] = 0

    # o3d_visualization(img, threshold=19, cmap='cool', draw_box=True, fill_empty_voxels=True)
    # plot_2d_cross_sections(img, cmap='cool', cross_sections_ind=[0, 1, 2, 3, 4, 5], cross_axis=[0, 2])

    # fig1 = volume_slice_visualizer_plotly(
    #     image3d=img, 
    #     # colorscale="hot",
    #     # save_fig="tmp/test.html", link=True, show_as_html=False,
    # )
    # fig1.show()
    # img_all_chnnels = img.get_image()
    # img_all_chnnels = np.average(img_all_chnnels, axis=-1)
    # fig2 = image_slice_view_plotly(
    #     image3d=img,
    #     channels=-1,
    #     slice_along_axis_n=2,
    # )
    # fig2.show()
    # fig3 = volume_visualizer_plotly(
    #     image3d=img,
    # )
    # fig3.show()
    # fig4 = voxel_interactive_plotly(
    #     image3d=img, box_outline=True, box_filled=False,
    #     channels='max',
    #     # filename=Path('~/Downloads/test.html').expanduser(),
    # )
    # fig4.show()
    # fig5 = scatter_plotly(
    #     image3d=img,
    #     box_outline=True,
    # )
    # fig5.show()
    # img = np.random.random((32,32,32))
    # solid_3d_visualizer(img)
    pass
