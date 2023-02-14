import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.font_manager
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import os

try:
    matplotlib.font_manager._rebuild()
except AttributeError:
    pass


# from matplotlib import font_manager
# font_dir = os.path.expanduser('~/Apps/fonts/Helvetica Family/')
# font_files = font_manager.findSystemFonts(fontpaths=font_dir)
# for font_file in font_files:
#     font_manager.fontManager.addfont(font_file)

# from matplotlib import font_manager
# font_dir = os.path.expanduser('~/Apps/fonts/Helvetica Family/')
#
# Helvetica = font_manager.FontEntry(
#     fname=font_dir + 'Helvetica-Normal.ttf',
#     name='Helvetica')
# font_manager.fontManager.ttflist.insert(0, Helvetica)

def plot_format(size=None, font_size=9, nrows=1, ncols=1, palette=None, equal_axis=False, sharex=False,
                flip_size=False):
    # f, ax, fs = Plots.plot_format()
    # colors: https://python-graph-gallery.com/100-calling-a-color-with-seaborn/
    # Pallets https://chrisalbon.com/python/data_visualization/seaborn_color_palettes/
    # https://colorbrewer2.org/#type=diverging&scheme=BrBG&n=6
    # font size: https://stackoverflow.com/questions/12444716/
    # RC params:
    # how-do-i-set-the-figure-title-and-axes-labels-font-size-in-matplotlib?rq=1
    # global init_plot_format
    # if init_plot_format:
    #     print('Initializing plot font sizes by making a dummy plot')
    #     init_plot_format = False
    #     plot_format()
    #     plt.show()

    # Line colors: plt.rcParams['axes.prop_cycle'].by_key()['color']

    # subplots = (nrows, ncols)
    if size is None:
        size = 88

    if not (isinstance(size, tuple) or isinstance(size, np.ndarray)):
        golden_ratio = (5 ** .5 - 1) / 2
        if equal_axis:
            golden_ratio = 1
        size = (size, size * golden_ratio * (nrows / ncols))
    if flip_size:
        size = (size[1], size[0])

    # from matplotlib import font_manager
    # font_dir = os.path.expanduser('~/Apps/fonts/Helvetica Family/')
    # Helvetica = font_manager.FontEntry(
    #     fname=font_dir + 'Helvetica-Normal.ttf',
    #     name='Helvetica')
    # font_manager.fontManager.ttflist.insert(0, Helvetica)
    # matplotlib.rc('font', family=Helvetica.name)

    plt.rcParams['font.size'] = font_size
    matplotlib.rc('font', family='sans-serif')
    matplotlib.rc('font', serif='Helvetica')
    matplotlib.rc('text', usetex='false')
    matplotlib.rc('xtick', labelsize=font_size)
    matplotlib.rc('ytick', labelsize=font_size)

    matplotlib.rcParams['svg.fonttype'] = 'none'
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    params = {
        "legend.fontsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        'font.size': font_size,
        'axes.labelsize': font_size,
        'axes.titlesize': font_size + 2,
        # ax[i].grid(color='silver', linestyle='--', linewidth=1, alpha=0.25)
        'grid.color': 'silver',
        'grid.linestyle': '--',
        'grid.linewidth': 1,
        'grid.alpha': 0.45,
    }
    matplotlib.rcParams.update(params)

    matplotlib.rcParams['lines.linewidth'] = 2
    # matplotlib.rcParams['lines.color'] = 'r'

    # sns.set_style("whitegrid")
    # sns.set_context('paper')

    f, ax = plt.subplots(figsize=(size[0] / 100 * 3.93701, size[1] / 100 * 3.93701), nrows=nrows, ncols=ncols,
                         sharex=sharex)
    # sns.set_style("darkgrid")
    # sns.set(style="whitegrid")
    # sns.set_context("paper")
    plt.style.use('seaborn-paper')

    # if palette is None:
    #     # https://medium.com/@morganjonesartist/color-guide-to-seaborn-palettes-da849406d44f
    #     sns.set_palette('Paired_r')

    # if not isinstance(ax, np.ndarray):
    #     ax = [ax]
    # for a in np.nditer(ax):
    #     a.xaxis.set_tick_params(labelsize=font_size)
    #     a.yaxis.set_tick_params(labelsize=font_size)
    # if len(ax) == 1:
    #     ax = ax[0]

    return f, ax, font_size


def plot_save(name, path=None, formats=None, dpi=800, save_to_papers=False, transparent=False):
    if formats is None:
        formats = ['png', 'pdf']
    if isinstance(formats, str):
        formats = [formats]
    if save_to_papers:
        if path is None:
            path = []
        path += ['plots/paper/']
    if path is None:
        path = ['plots/']
    if isinstance(path, str):
        path = [path]
    for p in path:
        for f in formats:
            plt.savefig(p + name + f'.{f}', dpi=dpi, bbox_inches='tight',
                        # pad_inches=0,
                        format=f, transparent=transparent)
        print(f'Plot saved at: {p}{name}.{f}')


def plot_tick_formatter(ax=None, axis='y', style='sci'):
    # styles = sci, comma
    if ax is None:
        ax = plt.gca()
    if style == 'comma':
        ax.__dict__[f'{axis}axis'].set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
        # if axis == 'y':
        #     # instead of plt.gca().xaxis. use: ax.get_xaxis()
        #     ax.__dict__[f'{axis}axis'].set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
        # if axis == 'x':
        #     ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
    if style == 'sci':
        ax.ticklabel_format(axis=axis, style="sci", scilimits=(0, 0))
        # import matplotlib.ticker as mtick
        # if axis == 'y':
        #     ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        # if axis == 'x':
        #     ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))


def remove_ticks(ax, axis='x', keep_ticks=False):
    if ('x' in axis) and ('y' in axis):
        remove_ticks(ax, axis='x', keep_ticks=keep_ticks)
        remove_ticks(ax, axis='y', keep_ticks=keep_ticks)
    if keep_ticks:
        labels = []
        if axis == 'x':
            labels = [item.get_text() for item in ax.get_xticklabels()]
        if axis == 'y':
            labels = [item.get_text() for item in ax.get_yticklabels()]
        empty_string_labels = [''] * len(labels)
        if axis == 'x':
            ax.set_xticklabels(empty_string_labels)
        if axis == 'y':
            ax.set_yticklabels(empty_string_labels)
        return
    if axis == 'x':
        ax.set_xticks([])
    if axis == 'y':
        ax.set_yticks([])


def ylabel_right_side(ax, label):
    ax_2 = ax.twinx()
    ax_2.set_ylabel(label)
    ax_2.set_yticks([])
    return ax_2


def put_costume_text_instead_of_legends(ax, labels, handles=None, loc='best', remove_frame=False):
    if handles is None:
        import matplotlib.patches as mpl_patches
        handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white",
                                         lw=0, alpha=0)] * 2
    if isinstance(labels, str):
        labels = [labels]
    leg = ax.legend(handles, labels, loc=loc)
    if remove_frame:
        leg.get_frame().set_facecolor('none')
        leg.get_frame().set_linewidth(0.0)
    return leg


def add_hash_pattern(ax, pattern='//'):
    import itertools
    hatches = itertools.cycle(['/', '//', '+', '-', 'x', '\\', '*', 'o', 'O', '.'])
    if pattern is None:
        pattern = next(hatches)

    for _, bar in enumerate(ax.patches):
        bar.set_hatch(pattern)


def annotate_subplots_with_abc(axs, x=-0.2, y=1.1, start_from=0, bigger_fonts=1):
    for n, ax in enumerate(axs):
        import string
        ax.text(x, y, f'{string.ascii_lowercase[n + start_from]})', transform=ax.transAxes,
                size=matplotlib.rcParams['font.size'] + bigger_fonts,
                weight='bold')


def change_number_of_ticks(ax, n, dtype=int):
    ymin, ymax = ax.get_ylim()
    custom_ticks = np.linspace(ymin, ymax, n, dtype=dtype)
    ax.set_yticks(custom_ticks)
    ax.set_yticklabels(custom_ticks)


def periodic_table_heatmap(elemental_data, cbar_label="", cbar_label_size=14,
                           show_plot=False, cmap="YlOrRd", cmap_range=None, blank_color="grey",
                           value_format=None, max_row=9, figsize=None,
                           ax=None, fig=None, not_show_blank_value=False,
                           ):
    """
    A static method that generates a heat map overlayed on a periodic table.

    Args:
         elemental_data (dict): A dictionary with the element as a key and a
            value assigned to it, e.g. surface energy and frequency, etc.
            Elements missing in the elemental_data will be grey by default
            in the final table elemental_data={"Fe": 4.2, "O": 5.0}.
         cbar_label (string): Label of the colorbar. Default is "".
         cbar_label_size (float): Font size for the colorbar label. Default is 14.
         cmap_range (tuple): Minimum and maximum value of the colormap scale.
            If None, the colormap will autotmatically scale to the range of the
            data.
         show_plot (bool): Whether to show the heatmap. Default is False.
         value_format (str): Formatting string to show values. If None, no value
            is shown. Example: "%.4f" shows float to four decimals.
         cmap (string): Color scheme of the heatmap. Default is 'YlOrRd'.
            Refer to the matplotlib documentation for other options.
         blank_color (string): Color assigned for the missing elements in
            elemental_data. Default is "grey".
         max_row (integer): Maximum number of rows of the periodic table to be
            shown. Default is 9, which means the periodic table heat map covers
            the first 9 rows of elements.
    """

    from pymatgen.core.periodic_table import Element
    # Convert primitive_elemental data in the form of numpy array for plotting.
    if cmap_range is not None:
        max_val = cmap_range[1]
        min_val = cmap_range[0]
    else:
        max_val = max(elemental_data.values())
        min_val = min(elemental_data.values())

    max_row = min(max_row, 9)

    if max_row <= 0:
        raise ValueError("The input argument 'max_row' must be positive!")

    value_table = np.empty((max_row, 18)) * np.nan
    blank_value = min_val - 0.01
    if not_show_blank_value:
        blank_value = float('nan')

    for el in Element:
        if el.row > max_row:
            continue
        value = elemental_data.get(el.symbol, blank_value)
        value_table[el.row - 1, el.group - 1] = value

    # Initialize the plt object
    import matplotlib.pyplot as plt
    if ax is None:

        fig, ax = plt.subplots()
        if figsize is None:
            plt.gcf().set_size_inches(12, 8)
        else:
            plt.gcf().set_size_inches(figsize[0] / 100 * 3.93701, figsize[1] / 100 * 3.93701)

    # We set nan type values to masked values (ie blank spaces)
    data_mask = np.ma.masked_invalid(value_table.tolist())
    heatmap = ax.pcolor(data_mask, cmap=cmap, edgecolors='w', linewidths=1,
                        vmin=min_val - 0.001, vmax=max_val + 0.001)
    cbar = fig.colorbar(heatmap)

    # Grey out missing elements in input data
    cbar.cmap.set_under(blank_color)

    # Set the colorbar label and tick marks
    cbar.set_label(cbar_label, rotation=270, labelpad=25, size=cbar_label_size)
    cbar.ax.tick_params(labelsize=cbar_label_size)

    cbar.ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # Refine and make the table look nice
    ax.axis('off')
    ax.invert_yaxis()

    # Label each block with corresponding element and value
    for i, row in enumerate(value_table):
        for j, el in enumerate(row):
            if not np.isnan(el):
                symbol = Element.from_row_and_group(i + 1, j + 1).symbol
                # plt.text(j + 0.5, i + 0.25, symbol,
                #          horizontalalignment='center',
                #          verticalalignment='center', fontsize=14)
                ax.text(j + 0.5, i + 0.5, symbol,
                        horizontalalignment='center',
                        verticalalignment='center', fontsize=cbar_label_size)
                if el != blank_value and value_format is not None:
                    # plt.text(j + 0.5, i + 0.5, value_format % el,
                    #          horizontalalignment='center',
                    #          verticalalignment='center', fontsize=10)
                    ax.text(j + 0.5, i + 0.5, value_format % el,
                            horizontalalignment='center',
                            verticalalignment='center', fontsize=cbar_label_size)

    fig.tight_layout()

    if show_plot:
        plt.show()

    return plt, ax


def choice_of_markers():
    from matplotlib import markers
    all_shapes = list(markers.MarkerStyle.markers.keys())
    plt.figure()
    for i in range(len(all_shapes)):
        plt.scatter(i, i, marker=all_shapes[i])
    plt.show()

    print('End markers')


def ax_tick_locator(ax, axis='x', major_period=None, minor_period=None):
    if major_period is not None:
        ax.__dict__[f'{axis}axis'].set_major_locator(MultipleLocator(major_period))
    if minor_period is not None:
        ax.__dict__[f'{axis}axis'].set_minor_locator(MultipleLocator(minor_period))


def change_bar_width(ax, new_value):
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)


# sns.distplot(mp,
#              color='darkviolet', hist_kws=dict(edgecolor="k", linewidth=1),
#              bins=10, kde=None, label=f'MP (positive ratio={mp_acc:.2f})', ax=ax[i])

# ax_2 = ax[i].twinx()

colors_databases = {
    'cod': 'darkgreen',
    'mp': 'darkblue',
    'cspd': 'dimgray',
}

clf_names = {'MLPClassifier': 'Neural network',
             'RandomForestClassifier': 'Random forest',
             'BaggingClassifier': "SVM-B"}

if __name__ == '__main__':
    choice_of_markers()
