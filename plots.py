from utility.util_general import *
import seaborn as sns
import matplotlib
import matplotlib.pyplot as mpl
from crystal_tools import plot_lattice_parameters, distribution, atom2prop
from crystal_tools import *
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, \
    roc_curve, auc
import numpy as np
from utility.util_plot import plot_format, plot_save


# sns.set_style("darkgrid")


def atom_3d_visualizations():
    target_data = 'cod/data_sets/all/cif_chunks/'
    iterator = 1
    n_bins = 32 * iterator
    pad_len = 17.5 * iterator
    all_files = list_all_files(data_path + target_data, pattern="[0-9]*.pkl")
    N = 1520767
    filename = expanduser(f'~/Downloads/{N}.cif')
    atoms = cif_parser(filename)
    # view_atom_vmd(atoms)
    atoms_box = atom2mat(filename='', len_limit=pad_len, n_bins=n_bins, atoms_unit=atoms, return_atom=True)
    write(f'~/Downloads/{N}_box.cif', atoms_box)
    write(f'~/Downloads/{N}_box.xyz', atoms_box)

    print('End function')


def atom_3d_visualizations_2():
    target_data = 'cod/data_sets/all/cif_chunks/'
    iterator = 1 / 2
    n_bins = 32 * iterator
    pad_len = 17.5 * iterator
    all_files = list_all_files(data_path + target_data, pattern="[0-9]*.pkl")
    # N = 1010925
    N = 1000007
    filename = expanduser(f'~/Downloads/{N}.cif')
    atoms = cif_parser(filename)
    # view_atom_vmd(atoms)
    write(f'~/Downloads/{N}.xyz', atoms)
    atoms_box = atom2mat(filename='', len_limit=pad_len, n_bins=n_bins, atoms_unit=atoms, return_atom=True)
    atoms_box.set_positions(atoms_box.get_positions() + 0.25)
    write(f'~/Downloads/{N}_box.cif', atoms_box)
    write(f'~/Downloads/{N}_box.xyz', atoms_box)

    # VMD Commands
    # pbc set {16 16 16}
    # pbc box
    # rotate z to 45
    # rotate x to 45
    # rotate y to 45

    print('End function')


# init_plot_format = True

def plot_literature():
    path = data_path + f'cod/anomaly_cspd/cspd_cif_top_{108}/'
    n_top_comp = 15
    data = load_var(path + 'info.pkl')[:n_top_comp]
    data['formula'] = data['formula'].str.strip()
    data['hyp_cod'] = data['hyp'] + data['cod']

    cod = pd.DataFrame({'formula': data['formula'], 'n': data['cod'], 'type': ['Synthesized'] * len(data)})
    hyp = pd.DataFrame({'formula': data['formula'], 'n': data['hyp'], 'type': ['Anomaly'] * len(data)})
    data_new = pd.concat([cod, hyp])

    formula_list = []
    for i in range(len(data)):
        formula = data['formula'][i]
        formula = '$' + ''.join([f'_{j}' if j.isdigit() else j for j in formula]) + '$'
        formula_list.append(formula)
    data['formula'] = formula_list

    f, axs, fs = plot_format((88, 70), ncols=2)
    ax = axs[1]
    # sns.barplot(x="hyp_cod", y="formula", data=data,
    #             label="Anomaly",
    #             # palette="YlOrRd",
    #             # palette="Reds",
    #             color='firebrick',
    #             ax=ax,
    #             )
    # sns.barplot(x="cod", y="formula", data=data,
    #             label="Synthesized",
    #             # palette="BuGn_r",
    #             # palette="pastel",
    #             width=5,
    #             color='seagreen',
    #             ax=ax,
    #             )
    sns.barplot(x='n', y='formula', data=data_new, hue='type')
    ax.set_ylabel('')
    ax.set_xlabel('Found instances', fontsize=fs)
    ax.xaxis.set_ticks([0, 100, 200, 300])
    ax.yaxis.grid(True)
    # ax.xaxis.grid(True)
    # ax.grid('on')
    ax.grid()
    lines2, labels2 = ax.get_legend_handles_labels()
    ax.get_yaxis().set_visible(False)

    ax.legend(loc='lower right')

    # ax.get_yaxis().set_ticks([])
    #
    # f.legend([lines, lines2], labels=['labels', 'labels2'])

    # plot_save('fig-2b')
    # plt.show()

    # f, ax, fs = plot_format((95 / 2, 100), font_size=font_size)
    ax = axs[0]
    sns.set_style("darkgrid")
    sns.barplot(x="lit_per", y="formula", data=data,
                label="Study intensity",
                color='slateblue',
                # palette="GnBu_d",
                # palette="pastel",
                ax=ax,
                )

    ax.set_ylabel('')
    ax.set_xlabel('Prevalence %')
    ax.xaxis.set_ticks([0, 1, 2, 3, 4, 5])
    # ax.yaxis.grid(True)
    ax.grid()
    lines, labels = ax.get_legend_handles_labels()
    # plt.savefig(path + 'fig-2a.png', dpi=600)
    # plt.savefig(path + 'fig-2a.svg', dpi=600)
    sns.set_style("darkgrid")
    # ax.legend(lines + lines2, labels + labels2, loc='lower right')

    Plots.annotate_subplots_with_abc(axs, x=0, y=1.05, start_from=1)

    plot_save('literature_b_c_', save_to_papers=True)
    plt.show()


def plot_lattice_constants(fig_size=88 - 2, save_fig=True, plot_min_max=False, ax=None, show_plot=True,
                           lattice_parameters=None, xlim_max=25):
    if lattice_parameters is None:
        lattice_parameters = plot_lattice_parameters(return_data=True)

    if ax is None:
        f, ax, fs = plot_format(fig_size)
    else:
        show_plot = False
    # sns.set(style="darkgrid", rc={"lines.linewidth": 1}, color_codes=True)
    # sns.set_style("dark")
    bins = 15

    # sns.set_palette(sns.color_palette("BuGn", 5))
    if plot_min_max:
        sns.distplot(lattice_parameters['min_lattice'], bins=bins, ax=ax)
        sns.distplot(lattice_parameters['max_lattice'], bins=bins, ax=ax)
    axes = plt.gca()
    makedirs('plots/', exist_ok=True)
    for i in ['a', 'b', 'c']:
        sns.distplot(lattice_parameters[i][lattice_parameters[i] < xlim_max], bins=bins, ax=ax,
                     hist_kws=dict(edgecolor="k", linewidth=1), label=i)
    if plot_min_max:
        ax.legend(['Min. of Lattice Constants', 'Max. of Lattice Constants', 'a', 'b', 'c'])
    else:
        # ax.legend(['a', 'b', 'c'])
        ax.legend()
    axes.set_xlim([0, xlim_max])
    # ax.tick_params(labelsize=font_size)
    ax.set_xlabel("Lattice Constants $(\AA)$")
    ax.set_ylabel('')
    ax.grid()
    Plots.ylabel_right_side(ax, 'PDF')
    # plt.title('Lattice Parameters Distribution', fontsize=font_size)

    if save_fig:
        plot_save('lattice constants')
    if show_plot:
        plt.show()
    return ax


def plot_min_dist(fig_size=88 - 4.5, save_fig=True, ax=None, show_plot=True):
    data = distribution(
        data_sets=['all/cif_chunks/'],
        property_fn=atom2prop,
        prop='min_nearest_neighbor',
        previous_run=True,
        bins=[40, 10],
        x_lim_lo=0,
        x_lim_hi=200,
        return_data=True,
    )
    data_h_excluded = distribution(
        data_sets=['all/cif_chunks/'],
        property_fn=atom2prop,
        prop='min_nearest_neighbor',
        previous_run=True,
        bins=[40, 10],
        x_lim_lo=0,
        x_lim_hi=200,
        return_data=True,
        exclude_condition=['H'], exclude_name='_H_excluded'
    )
    data = data['all/cif_chunks/']['min_nearest_neighbor']
    data_h_excluded = data_h_excluded['all/cif_chunks/']['min_nearest_neighbor']
    data_h_excluded = np.array(data_h_excluded)

    if ax is None:
        f, ax, fs = plot_format(fig_size)
    else:
        show_plot = False

    ax2 = ax.twinx()
    # ax.tick_params(labelsize=fs)
    # ax2.tick_params(labelsize=fs)
    # font_size = fs
    bins = 50
    # sns.set_palette(sns.color_palette("BuGn", 5))
    # sns.distplot(data, bins=bins, ax=ax, kde=False)
    sns.distplot(data, bins=bins, ax=ax, kde=False, color='darkblue', label='All COD',
                 hist_kws=dict(edgecolor="k", linewidth=1))
    sns.distplot(data_h_excluded[data_h_excluded > 0.5], bins=bins, ax=ax2, kde=False, color='darkgreen',
                 label='COD w/o H', hist_kws=dict(edgecolor="k", linewidth=1))
    sns.distplot(data_h_excluded[data_h_excluded <= 0.5], bins=3, ax=ax2, kde=False, color='darkred',
                 label='Excluded', hist_kws=dict(edgecolor="k", linewidth=1))
    axes = plt.gca()
    axes.set_xlim([0, 4])

    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid()

    # ax.tick_params(labelsize=font_size)
    # ax.set_ylabel('Crystals inluding H')
    ax.set_ylabel('All COD crystals')
    ax2.set_ylabel('Crystals w/o H')
    ax.set_xlabel('Nearest neighbor distance ($\AA$)')
    # ax.xticks(rotation=45)
    # ax2.set_yticklabels(ax2.get_yticklabels(), rotation=45, ha='right')
    ax.get_yaxis().set_visible(True)

    lines2, labels2 = ax2.get_legend_handles_labels()
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper right')

    ax2.tick_params(axis='y', colors='darkgreen')
    ax.tick_params(axis='y', colors='darkblue')

    # plt.show()
    if save_fig:
        plot_save('min distance', save_to_papers=True)
        gl_run.save_plot(formats=['svg', 'pdf'])
    if show_plot:
        plt.show()
    print('End fn')


def plot_lattice_cnt_and_min_dist(save_fig=False):
    f, axs, fs = plot_format(80, ncols=1, nrows=2)
    plot_lattice_constants(save_fig=False, ax=axs[0], xlim_max=35)
    plot_min_dist(save_fig=False, ax=axs[1])
    axs[0].set_xlim((0, 35))

    for n, ax in enumerate(axs):
        import string
        ax.text(-0.2, 1.1, f'{string.ascii_lowercase[n]})', transform=ax.transAxes,
                # size=fs,
                weight='bold')

    Plots.plot_save('distributions', save_to_papers=True)

    plt.show()

    print('End fn')


def plot_pca(save_fig=True):
    def reset_df(df):
        dic = df.to_dict()
        for k in dic:
            l = []
            for kp in dic[k]:
                l.append(dic[k][kp])
            dic[k] = l
        return pd.DataFrame(dic)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # search = load_var('results/PCA/run_001/grid_search.pkl')
        # search2 = load_var('results/PCA/run_001/grid_search2.pkl')
        search = load_var('results/PCA/run_003/best_clfs_list.pkl')
        cum_var = load_var('results/PCA/run_002/cum_var_exp.pkl')
        # search3 = load_var('results/PCA/run_007/results.pkl')
        # search3 = load_var('results/PCA/run_012_comet-selected/results.pkl')
        search3 = load_var('results/PCA/pca_comet/results.pkl')
    components_col = 'param_pca__n_components'
    # gs = [search] + search2
    gs = search
    names = [
        'Neural network',
        'Random forest',
        # 'SVM'
    ]
    # Removing SVM
    gs.pop(2)

    f, axs, fs = plot_format(size=70, nrows=2)

    # ax2 = ax.twinx()
    ax = axs[0]
    ax2 = axs[1]
    # ax.tick_params(labelsize=fs)
    # ax2.tick_params(labelsize=fs)

    sns.lineplot(x=range(len(cum_var)), y=cum_var, ax=ax2, color='k', label='C.E.V.')

    # for i in gs:
    #
    #     if not isinstance(i, pd.DataFrame):
    #         results = pd.DataFrame(i.cv_results_)
    #         best_clfs = results.groupby(components_col).apply(
    #             lambda g: g.nlargest(1, 'mean_test_score'))
    #     else:
    #         best_clfs = i
    #     best_clfs = reset_df(best_clfs)
    #     best_clfs.reset_index()
    #     best_clfs['mean_test_score'] = best_clfs['mean_test_score'] * 100
    #     best_clfs['std_test_score'] = best_clfs['std_test_score'] * 100
    #     # sns.lineplot(x=components_col, y='mean_test_score', hue='std_test_score', ax=ax)
    #     best_clfs.plot(x=components_col, y='mean_test_score', yerr='std_test_score',
    #                    label=names.pop(0), ax=ax,
    #                    )
    components_col = 'comp'
    if not 'Set' in search3.columns:

        df = {components_col: [], 'Classifier': [], 'val': [], 'Set': []}
        for i in range(len(search3)):
            if not search3['name'][i].lower() in names.__str__().lower():
                continue
            for j in range(len(search3['train_all'][i])):
                df[components_col].append(search3[components_col][i])
                df['Classifier'].append(search3['name'][i])
                df['val'].append(search3['train_all'][i][j])
                df['Set'].append('Train')
            for j in range(len(search3['dev_all'][i])):
                df[components_col].append(search3[components_col][i])
                df['Classifier'].append(search3['name'][i])
                df['val'].append(search3['dev_all'][i][j])
                df['Set'].append('Test')
        df = pd.DataFrame(df)
    else:
        df = search3
    df.loc[:, 'val'] = df['val'] * 100

    # df = df[df[components_col] < 600]

    sns.lineplot(data=df, x=components_col, y='val', hue='Classifier', style='Set', ax=ax)

    # for i in range(len(names)):
    #     best_clfs = search3[search3['name'].str.lower() == names[i].lower()]
    #     best_clfs.reset_index(inplace=True)
    #     best_clfs.loc[:, 'dev'] = best_clfs['dev'] * 100
    #     best_clfs.loc[:, 'dev_std'] = best_clfs['dev_std'] * 100
    #
    #     # sns.lineplot(x=components_col, y='mean_test_score', hue='std_test_score', ax=ax)
    #     best_clfs.plot(x=components_col, y='dev', yerr='dev_std',
    #                    label=names[i] + ' (Test)', ax=ax,
    #                    )
    #     if 'train' in search3.columns:
    #         best_clfs.loc[:, 'train'] = best_clfs['train'] * 100
    #         best_clfs.loc[:, 'train_std'] = best_clfs['train_std'] * 100
    #         # best_clfs.plot(x=components_col, y='train', yerr='train_std',
    #         #                label=names[i] + ' (Train)', ax=ax,
    #         #                )
    #         ax.errorbar(best_clfs[components_col], best_clfs['train'], yerr=best_clfs['train_std'],
    #                     fmt='.k')
    #         ax.plot(best_clfs[components_col], best_clfs['train'],
    #                 label=names[i] + ' (Train)')

    # ax.set_xticks(best_clfs['comp'])
    ax2.set_xlabel('PCA components', fontsize=fs)

    # ax2.set_xticks(ax.get_xticks())
    ax2.tick_params(axis='x', rotation=45)

    ax.set_ylabel('AUC %', fontsize=fs)
    ax2.set_ylabel('C.E.V. %', fontsize=fs)
    ax.xaxis.label.set_visible(False)

    # ax.set_xticks(ax2.get_xticks())
    # Plots.remove_ticks(ax, keep_ticks=True)

    for i in range(2):

        ylim = axs[i].get_ylim()
        if i == 1:
            axs[i].plot([1000, 1000], [0., ylim[1]], '--', color='r', linewidth=1)
        axs[i].set_ylim(ylim)
        # axs[i].set_xlim((0, 2 ** 12))
        # axs[i].set_xlim((0, 2 ** 6))
        axs[i].grid()
    ax2.text(1200, np.mean(ylim), 'Effective features threshold', color='r',
             # rotation=90
             )
    ax2.get_xaxis().set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
    # ax2.get_xaxis().set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
    # ax.set_yticks([80, 85, 90, 95, 100])
    ax2.set_yticks([99, 99.5, 100])

    Plots.annotate_subplots_with_abc(axs)

    # ax.set_xlim([1, 1050])
    # ax2.set_xlim([1, 1050])

    if save_fig:
        # plot_save('PCA-analysis')
        plot_save('PCA-analysis-zoom')
    plt.show()
    print('End fn')


def plot_chemical_elements_distribution(save_fig=True):
    # from pymatgen.util.plotting import periodic_table_heatmap
    data = distribution(
        data_sets=['all/cif_chunks/'],
        prop='symbols',
        previous_run=True,
        return_data=True,
    )
    data = [item for sublist in data['all/cif_chunks/']['symbols'] for item in list(sublist)]
    data_symbol, data_count = np.unique(data, return_counts=True)
    str1 = ''
    elemental_data = {}
    for i in range(len(data_count)):
        str1 += f'{data_symbol[i]},{data_count[i]}\n'
        elemental_data.update({data_symbol[i]: data_count[i]})

    write_text('tmp/elements.csv', str1)

    top_n = 10
    ind = np.argsort(data_count)
    elemental_data_top = {}
    elemental_data_bottom = elemental_data.copy()
    for i in range(top_n):
        elemental_data_top.update(
            {data_symbol[ind[len(ind) - i - 1]]: elemental_data_bottom.pop(data_symbol[ind[len(ind) - i - 1]])})

    f, ax, fs = Plots.plot_format((180, 80))
    p = Plots.periodic_table_heatmap(elemental_data, cbar_label="",
                                     show_plot=False, cmap="Greens", cmap_range=None, blank_color="grey",
                                     value_format=None, max_row=9,
                                     cbar_label_size=9,
                                     ax=ax, fig=f, not_show_blank_value=True,
                                     )
    if save_fig:
        plot_save('elemental_distribution')
    plt.show()

    f, ax, fs = Plots.plot_format((180, 80))
    ax2 = ax.twinx()
    p = Plots.periodic_table_heatmap(elemental_data_bottom, cbar_label="",
                                     show_plot=False, cmap="winter", cmap_range=None, blank_color="grey",
                                     value_format=None, max_row=9,
                                     cbar_label_size=9,
                                     ax=ax, fig=f, not_show_blank_value=False,
                                     )
    # if save_fig:
    #     plot_save('elemental_distribution_bottom')
    # p.show()

    p = Plots.periodic_table_heatmap(elemental_data_top, cbar_label="",
                                     show_plot=False, cmap="YlOrRd", cmap_range=None, blank_color="grey",
                                     value_format=None, max_row=9,
                                     cbar_label_size=9,
                                     ax=ax2, fig=f, not_show_blank_value=True,
                                     )
    ax.set_title('All used COD crystals')
    if save_fig:
        plot_save('elemental_distribution_dual_ax')
    plt.show()

    print('End fn')


def plot_loss_fn_cae_comparision(mark_epoch=False):
    global gl_run
    fig, ax, fs = plot_format(180, nrows=3, ncols=4)
    cae_num = [
        43, 38, 39, 40,
        42, 49, 50, 51, 52, 53,
        41, 45,
        # 56, 57
    ]
    cae_batch_number = []
    for cae_n in range(len(cae_num)):

        filename = f'results/CAE/run_0{cae_num[cae_n]}/'
        run = load_var(filename + 'run.pkl')
        cae_batch_number.append(run.params['batch'])
        if run.params['samples_fraction'] != 0.05:
            red_print(f'Wrong sample fraction for: run_{cae_num[cae_n]:03d}')
        txt = read_text(filename + 'log.txt')
        txt = txt.split('\n')
        # loss = []
        # for l in txt:
        #     t = re.findall('- loss: [0-9.e\-]*', txt[201])
        #     if len(t) == 0:
        #         continue
        #     loss.append(float(t[0].split(':')[1]))
        # loss = np.array(loss)

        loss = []
        epoch = 0
        for i in range(len(txt)):
            if len(re.findall('Epoch [0-9]+/[0-9]+', txt[i])) == 1:
                loss += [[]]
                epoch += 1
            if len(re.findall('batch:.*loss:', txt[i])) == 1:
                l = float(re.findall('loss: [0-9.]+', txt[i])[0][6:])
                loss[-1].append(l)
        baches_n = len(loss[0])

        loss_flat = np.array([item for sublist in loss for item in sublist])

        batches_per_epoch = len(loss[0])

        # noiseless loss
        # n = 1
        # l = []
        # for i in range(len(loss_flat) - n):
        #     l.append(np.average(loss_flat[0:0 + n]))
        sns.lineplot(range(len(loss_flat)), loss_flat, label='Loss', ax=ax[np.unravel_index(cae_n, ax.shape)])

        loss_avg_in_epochs = [np.average(i) for i in np.split(loss_flat, len(loss_flat) // batches_per_epoch)][0]

        if mark_epoch:
            x = []
            y = []
            for i in range(epoch):
                x += list(np.array([1, 1]) * (i + 1) * batches_per_epoch)
                y += [0, 0.2]
            sns.lineplot(x, y, color='r', label='Epoch')

        sub_plt_pos = np.unravel_index(cae_n, ax.shape)
        if len(ax) > 0:
            if sub_plt_pos[0] + 1 != ax.shape[0]:
                Plots.remove_ticks(ax[sub_plt_pos], axis='x', keep_ticks=True)
            else:
                Plots.plot_tick_formatter(ax[sub_plt_pos], axis='x', style='sci')
            if sub_plt_pos[1] != 0:
                Plots.remove_ticks(ax[sub_plt_pos], axis='y', keep_ticks=True)
        # plt.xlabel('batches')
        # plt.ylabel('Loss: Binary Cross-Entropy')
        # plt.title(f'{filepath}')
        # plt.legend()
        plt.grid()
        legend = Plots.put_costume_text_instead_of_legends(ax[sub_plt_pos], f'CAE # {cae_n + 1}\n'
                                                                            f'Avg. loss={loss_avg_in_epochs:.5f}')
        legend.get_frame().set_facecolor('none')
        legend.get_frame().set_linewidth(0.0)

        # plt.savefig(run.last_run + 'Loss-batch.png')
        # plt.savefig(run.last_run + 'Loss-batch.svg')
    # ax[4, 2].set_visible(False)
    fig.text(0.5, -0.0, 'Number of batches', ha='center')
    fig.text(0.0, 0.5, 'Loss function', va='center', rotation='vertical')
    gl_run.save_plot('cae_loss_comp')
    plt.show()

    globals().update(locals())
    print('End fn')


def plot_loss_fn(mark_epoch=True):
    global gl_run
    fig, ax, fs = plot_format(88)
    cae_num = [60]
    cae_batch_number = []
    for cae_n in range(len(cae_num)):

        filename = f'results/CAE/run_0{cae_num[cae_n]}/'
        run = load_var(filename + 'run.pkl')
        cae_batch_number.append(run.params['batch'])
        # if run.params['samples_fraction'] != 0.05:
        #     red_print(f'Wrong sample fraction for: run_{cae_num[cae_n]:03d}')
        txt = read_text(filename + 'log.txt')
        txt = txt.split('\n')
        # loss = []
        # for l in txt:
        #     t = re.findall('- loss: [0-9.e\-]*', txt[201])
        #     if len(t) == 0:
        #         continue
        #     loss.append(float(t[0].split(':')[1]))
        # loss = np.array(loss)

        loss = []
        epoch = 0
        for i in range(len(txt)):
            if len(re.findall('Epoch [0-9]+/[0-9]+', txt[i])) == 1:
                loss += [[]]
                epoch += 1
            if len(re.findall('batch:.*loss:', txt[i])) == 1:
                l = float(re.findall('loss: [0-9.]+', txt[i])[0][6:])
                loss[-1].append(l)
        baches_n = len(loss[0])

        loss_flat = np.array([item for sublist in loss for item in sublist])

        batches_per_epoch = len(loss[0])

        # noiseless loss
        # n = 1
        # l = []
        # for i in range(len(loss_flat) - n):
        #     l.append(np.average(loss_flat[0:0 + n]))
        sns.lineplot(range(len(loss_flat)), loss_flat, label=None, ax=ax)

        loss_avg_in_epochs = [np.average(i) for i in np.split(loss_flat, len(loss_flat) // batches_per_epoch)]
        sns.lineplot(np.array(range(len(loss_flat) // batches_per_epoch)) * batches_per_epoch + batches_per_epoch / 2,
                     loss_avg_in_epochs, label=None, ax=ax)

        if mark_epoch:
            x = []
            y = []
            for i in range(epoch):
                x = list(np.array([1, 1]) * (i + 1) * batches_per_epoch)
                y = [0, 0.2]
                plt.plot(x, y, color='r', label='Epoch' if i == 0 else None)
                plt.text(x[0] * .95, y[1] * 1.2, f'Epoch #{i + 1}', rotation=90)
        Plots.plot_tick_formatter(axis='x')
        plt.xlabel('Number of batches')
        plt.ylabel('Loss function')
        # plt.title(f'{filepath}')
        # plt.legend()
        plt.grid()
        # plt.savefig(run.last_run + 'Loss-batch.png')
        # plt.savefig(run.last_run + 'Loss-batch.svg')
    # ax[4, 2].set_visible(False)
    gl_run.save_plot('cae_loss')
    plt.show()

    globals().update(locals())
    print('End fn')


def plot_cae_acc_comparision(mark_epoch=False):
    global gl_run
    fig, ax, fs = plot_format(180, nrows=2, ncols=2)
    cae_num = [
        43, 38, 39, 40,
    ]
    cae_batch_number = []
    names = ['Neural network', 'Random forest', 'SVM']
    for cae_n in range(len(cae_num)):
        path = f'results/CAE/Classification_Comp/run_{cae_num[cae_n]:03d}/'
        grid_search = None
        results = load_var(f'{path}results_pca_c.pkl', verbose=True)
        if not 'param_pca__n_components' in results.columns:
            results['param_pca__n_components'] = results['comp_val']
            results['mean_test_score'] = results['test']
            results['std_test_score'] = results['test_std']
        # grid_search = load_var(path + 'best_clfs_list.pkl')

        axc = ax[np.unravel_index(cae_n, ax.shape)]
        for c in range(3):
            if grid_search is not None:
                best_clfs = grid_search[c]
            else:
                best_clfs = results[results['name'].str.lower() == names[c].lower()]
            best_clfs[:16].plot(x='param_pca__n_components', y='mean_test_score', yerr='std_test_score',
                                label=names[c], ax=axc)

        axc.grid()
        # axc.set_ylim([.8, 1.])
        axc.set_xlabel('')
        axc.set_ylabel('')
        # axc.set_title(f'CAE #{cae_n + 1}')
        axc.text(800, .82, f'CAE #{cae_n + 1}')
        Plots.plot_tick_formatter(ax=axc, axis='x')

    Plots.remove_ticks(ax[0, 0], axis='x', keep_ticks=True)
    Plots.remove_ticks(ax[0, 1], axis='xy', keep_ticks=True)
    Plots.remove_ticks(ax[1, 1], axis='y', keep_ticks=True)

    ax[0, 0].set_ylabel('Accuracy')
    ax[1, 0].set_ylabel('Accuracy')
    ax[1, 0].set_xlabel('PCA components')
    ax[1, 1].set_xlabel('PCA components')

    # fig.text(0.5, -0.0, 'PCA components', ha='center')
    # fig.text(0.0, 0.5, 'Accuracy', va='center', rotation='vertical')
    gl_run.save_plot('cae_acc_comp')
    plt.show()

    globals().update(locals())
    print('End fn')


def plot_cae_exp_var_comparision(mark_epoch=False):
    global gl_run
    fig, ax, fs = plot_format(88, nrows=2, ncols=2)
    cae_num = [
        43, 38, 39, 40,
    ]
    cae_batch_number = []
    names = ['Neural network', 'Random forest', 'SVM-B']
    for cae_n in range(len(cae_num)):
        path = f'results/CAE/Classification_Comp/run_{cae_num[cae_n]:03d}/'
        axc = ax[np.unravel_index(cae_n, ax.shape)]
        exp_var = load_var(path + 'explained_variance.pkl')

        y = np.real(exp_var['Explained_variance'])
        x = np.real(exp_var['n_comp'])

        axc.plot(x, y)

        axc.grid()
        axc.set_xlim([0, 1500])
        axc.set_xlabel('')
        axc.set_ylabel('')
        # axc.set_title(f'CAE #{cae_n + 1}')
        # axc.text(800, .82, f'CAE #{cae_n + 1}')
        Plots.put_costume_text_instead_of_legends(axc, f'CAE #{cae_n + 1}')
        Plots.plot_tick_formatter(ax=axc, axis='x', style='comma')
        axc.tick_params(axis='x', labelrotation=45)

    Plots.remove_ticks(ax[0, 0], axis='x', keep_ticks=True)
    Plots.remove_ticks(ax[0, 1], axis='xy', keep_ticks=True)
    Plots.remove_ticks(ax[1, 1], axis='y', keep_ticks=True)

    # ax[0, 0].set_ylabel('C.E.V.')
    # ax[1, 0].set_ylabel('Accuracy')
    # ax[1, 0].set_xlabel('PCA components')
    # ax[1, 1].set_xlabel('PCA components')

    fig.text(0.5, -0.0, 'PCA components', ha='center')
    # fig.text(0.0, 0.5, 'Commutative explained variance', va='center', rotation='vertical')
    fig.text(0.0, 0.5, 'C.E.V.', va='center', rotation='vertical')
    gl_run.save_plot('cae_exp_var_comp')
    plt.show()

    globals().update(locals())
    print('End fn')


def plot_cae_roc_dist_comparision(mark_epoch=False):
    global gl_run

    cae_num = [
        43,
        38, 39,
        40,
    ]

    cae_batch_number = []
    clf_nick_names = {'MLPClassifier': 'Neural network',
                      'RandomForestClassifier': 'Random forest',
                      # 'SVC': "SVM"
                      }
    fig, ax, fs = plot_format(88, nrows=len(cae_num), ncols=len(clf_nick_names))
    for cae_n in range(len(cae_num)):
        path = expanduser(f'~/Data/cod/results/CAE/Classification_Comp/run_{cae_num[cae_n]:03d}/')
        preds = load_var(path + 'predictions.pkl', verbose=True)

        col = 0
        for clf in clf_nick_names:
            axc = ax[cae_n, col]
            prob = preds['predictions'][clf]
            y = preds['labels']
            prob_pos = prob[y > 0]
            prob_neg = prob[y < 0]
            bins = 20
            pos_plot = sns.distplot(prob_pos, bins=bins,
                                    label=f'Synthesis: {len(y[y > 0]):,} ({len(y[y > 0]) / len(y) * 100:.0f}%)',
                                    ax=ax[cae_n, col], kde=False, norm_hist=False, color='g',
                                    hist_kws=dict(edgecolor="k", linewidth=1)
                                    )
            neg_plot = sns.distplot(prob_neg, bins=bins,
                                    label=f'Anomaly: {len(y[y < 0]):,} ({len(y[y < 0]) / len(y) * 100:.0f}%)',
                                    ax=ax[cae_n, col], kde=False, norm_hist=False, color='r',
                                    hist_kws=dict(edgecolor="k", linewidth=1)
                                    )

            import scipy.stats as sts

            ylim = axc.get_ylim()
            # axc.plot([.5, .5], [0., ylim[1]], '--', color='r', linewidth=1)
            axc.set_ylim(ylim)

            axc.grid()
            axc.set_xlim(0, 1)
            if cae_n == 0:
                ax[cae_n, col].set_title(clf_nick_names[clf])
            col += 1
            if cae_n + 1 != len(cae_num):
                axc.set_xlabel('')
                Plots.remove_ticks(axc, axis='x', keep_ticks=True)
            Plots.plot_tick_formatter(ax=axc, axis='y', style='sci')
            if col > 0:
                axc.set_ylabel('')
                # Plots.remove_ticks(axc, axis='y', keep_ticks=True)
            if col == len(clf_nick_names):
                Plots.ylabel_right_side(axc, f'CAE #{cae_n + 1}')
        # axc.text(800, .82, f'CAE #{cae_n + 1}')
        # Plots.put_costume_text_instead_of_legends(axc, f'CAE #{cae_n + 1}')

    fig.text(0.5, -0.0, 'Probability of synthesis', ha='center')
    fig.text(0.0, 0.5, 'Number of materials', va='center', rotation='vertical')
    gl_run.save_plot('cae_roc_dist_comp')
    plt.show()

    globals().update(locals())
    print('End fn')


def plot_cae_roc_comparision():
    global gl_run

    cae_num = [
        43,
        38, 39,
        40,
    ]

    cae_batch_number = []
    clf_nick_names = {'MLPClassifier': 'Neural network',
                      'RandomForestClassifier': 'Random forest',
                      # 'SVC': "SVM"
                      }
    fig, ax, fs = plot_format(88, nrows=len(cae_num), ncols=len(clf_nick_names))
    for cae_n in range(len(cae_num)):
        path = expanduser(f'~/Data/cod/results/CAE/Classification_Comp/run_{cae_num[cae_n]:03d}/')
        preds = load_var(path + 'predictions.pkl')

        col = 0
        for clf in clf_nick_names:
            axc = ax[cae_n, col]
            prob = preds['predictions'][clf]
            y = preds['labels']
            prob_pos = prob[y > 0]
            prob_neg = prob[y < 0]
            bins = 20

            fpr, tpr, threshold = roc_curve(y, prob)
            roc_auc = auc(fpr, tpr)
            axc.plot(fpr, tpr, color='darkorange')
            axc.plot([0, 1], [0, 1], color='navy', linestyle='--')

            Plots.put_costume_text_instead_of_legends(axc, f'AUC={roc_auc:.2f}')
            axc.grid()
            axc.set_xlim(0, 1)
            axc.set_ylim(0, 1)
            if cae_n == 0:
                ax[cae_n, col].set_title(clf_nick_names[clf])

            if cae_n + 1 != len(cae_num):
                axc.set_xlabel('')
                Plots.remove_ticks(axc, axis='x', keep_ticks=True)
            Plots.plot_tick_formatter(ax=axc, axis='y', style='sci')
            if col > 0:
                axc.set_ylabel('')
                Plots.remove_ticks(axc, axis='y', keep_ticks=True)
            if col == len(clf_nick_names) - 1:
                Plots.ylabel_right_side(axc, f'CAE #{cae_n + 1}')
            col += 1
        # axc.text(800, .82, f'CAE #{cae_n + 1}')
        # Plots.put_costume_text_instead_of_legends(axc, f'CAE #{cae_n + 1}')

    fig.text(0.5, -0.0, 'False Positive Rate', ha='center')
    fig.text(0.0, 0.5, 'True Positive Rate', va='center', rotation='vertical')
    gl_run.save_plot('cae_roc_comp')
    plt.show()

    globals().update(locals())
    print('End fn')


def plot_sky_line_detailed_evaluation(plot_evaluation=True, include_group_3=True, plot_skyline_anomalies=True,
                                      save_fig=False, plot_dist=False, classifier=None):
    sky_line_path = data_path + 'cod/data_sets/skyline/'
    groups = ['group_1', 'group_2', 'group_3']
    # all_gr_prob = load_var('results/Skyline/run_001/all_gr_prob.pkl')
    # prob_g1 = load_var('results/Skyline/run_001/prob_group_1.pkl')

    all_gr_prob = load_var(f'{sky_line_path}all_gr_prob.pkl')
    skyline_vars = load_var(sky_line_path + 'skyline.pkl')

    compounds = skyline_vars['compounds']
    min_eng = skyline_vars['min_eng']
    eng = skyline_vars['eng']

    group_1 = load_var(sky_line_path + 'group_1.pkl')
    group_2 = load_var(sky_line_path + 'group_2.pkl')
    group_3 = load_var(sky_line_path + 'group_3.pkl')

    # merging prob. with eng. data
    for gr in groups:
        print(f'Merging {gr}')
        group = all_gr_prob[gr]
        # gr = prob_g1
        material_id = [re.findall('m[pvc]+-[0-9]*', group['filename'][i])[0] for i in range(len(group))]
        group['material_id'] = material_id
        all_gr_prob[gr] = group.merge(locals()[gr], on='material_id', how='inner')

    classifiers = ['MLPClassifier', 'RandomForestClassifier']
    if classifier is None:
        classifier = classifiers[0]

    threshold = all_gr_prob['group_1'][f'{classifier}_best_threshold'][0]

    n = 2
    if include_group_3:
        n = 3
    gr_1_2 = pd.DataFrame(columns=all_gr_prob['group_1'].columns)
    for gr in groups[:n]:
        gr_1_2 = gr_1_2.append(all_gr_prob[gr], ignore_index=True)
    gr_3 = all_gr_prob['group_3']

    # Plotting
    print('Plotting')
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    x3 = []
    y3 = []

    x_pos = []
    y_pos = []
    x_neg = []
    y_neg = []
    x_hyp_syn = []
    y_hyp_syn = []
    x_hyp_ano = []
    y_hyp_ano = []

    min_convex_hull = []

    for i in range(len(compounds)):
        df1 = group_1[group_1['pretty_formula'] == compounds[i]]
        y1 += list(df1['e_above_hull'])
        x1 += [i] * len(list(df1['e_above_hull']))
        df2 = group_2[group_2['pretty_formula'] == compounds[i]]
        y2 += list(df2['e_above_hull'])
        x2 += [i] * len(list(df2['e_above_hull']))
        df3 = group_3[group_3['pretty_formula'] == compounds[i]]
        y3 += list(df3['e_above_hull'])
        x3 += [i] * len(list(df3['e_above_hull']))

        df_pos = gr_1_2[(gr_1_2[classifier] >= threshold) & (gr_1_2['pretty_formula'] == compounds[i])]
        y_pos += list(df_pos['e_above_hull'])
        x_pos += [i] * len(list(df_pos['e_above_hull']))

        df_neg = gr_1_2[(gr_1_2[classifier] < threshold) & (gr_1_2['pretty_formula'] == compounds[i])]
        y_neg += list(df_neg['e_above_hull'])
        x_neg += [i] * len(list(df_neg['e_above_hull']))

        df_hyp_syn = gr_3[(gr_3[classifier] > threshold) & (gr_3['pretty_formula'] == compounds[i])]
        y_hyp_syn += list(df_hyp_syn['e_above_hull'])
        x_hyp_syn += [i] * len(list(df_hyp_syn['e_above_hull']))

        df_hyp_ano = gr_3[(gr_3[classifier] < threshold) & (gr_3['pretty_formula'] == compounds[i])]
        y_hyp_ano += list(df_hyp_ano['e_above_hull'])
        x_hyp_ano += [i] * len(list(df_hyp_ano['e_above_hull']))

        m = np.average(group_3[group_3['pretty_formula'] == compounds[i]]['energy_per_atom'] -
                       group_3[group_3['pretty_formula'] == compounds[i]]['e_above_hull'])
        min_convex_hull.append(m)
        # if compounds[i] == 'SiO2':
        #     print(compounds[i])
    min_eng_above_ground = min_eng - np.array(min_convex_hull)
    for i in range(len(eng)):
        for j in range(len(eng[i])):
            eng[i][j] -= min_convex_hull[i]
    eng_grouped = eng
    eng = [item for sublist in eng for item in sublist]

    f = None

    if plot_evaluation:
        # sns.set_style('darkgrid')
        # f, ax, font_size = plot_format(180 * .785)
        f, ax, font_size = plot_format(180 * .805)

        sns.barplot(list(range(len(min_eng_above_ground))), min_eng_above_ground,
                    # color='lightseagreen', alpha=0.3, edgecolor='black', linewidth=2,
                    color='bisque', alpha=0.3,
                    edgecolor='black', linewidth=1,
                    errwidth=1, capsize=1,
                    orient='v',
                    # label='Below amorphous limits'
                    )
        for i in range(len(eng_grouped)):
            plt.plot([i, i], [min(eng_grouped[i]), max(eng_grouped[i])], color='gray', linewidth=1)
            sns.scatterplot([i] * len(eng_grouped[i]), eng_grouped[i],
                            color='gray',
                            label=None, marker='_', s=60)

        alpha = 0.55
        size = 50
        # Predictions on group 1 & 2
        sns.regplot(x_pos, y_pos, marker='^', fit_reg=False,
                    scatter_kws={"color": Plots.colors_databases['cod'],
                                 "alpha": alpha, "s": size},
                    label=f'Predicted as synthesizable: ({len(x_pos)})'
                    )

        sns.regplot(x_neg, y_neg, marker='1', fit_reg=False,
                    scatter_kws={"color": 'darkred',
                                 "alpha": alpha, "s": size},
                    label=f"Predicted as anomaly: ({len(x_neg)})"
                    )

        # sns.scatterplot(x1, y1,
        #                 color='r',
        #                 label=f'Crystal, Group 1 (No ICSD): {len(x1)}', marker='+', s=70)
        # sns.scatterplot(x2, y2,
        #                 color='k',
        #                 label=f'Crystal, Group 2 (With ICSD): {len(x2)}', marker='^', s=70)

        if plot_skyline_anomalies:
            sns.scatterplot(x1 + x2, y1 + y2,
                            color='b', s=20, alpha=alpha,
                            label=f'Skyline anomalies: ({len(x1 + x2)})', marker='P')

        # if include_group_3:
        #     sns.scatterplot(x3, y3,
        #                     color='navy',
        #                     label=f'Potentially to be synthesizable: {len(x_hyp_syn)}', marker='8', s=5)

        # Amorphous limits
        # sns.scatterplot(list(range(len(min_eng_above_ground))), min_eng_above_ground,
        #                 color='k',
        #                 label=f'Amorphous limits', marker='_', s=80)
        label = 'Amorphous limit'
        for i in range(len(compounds)):
            sns.lineplot([i - 0.35, i + 0.35], [min_eng_above_ground[i]] * 2, label=label, color='k', linewidth=1.5)
            label = None

        # plt.ylabel('Energy above ground state (eV/atom)')
        ax.set_ylabel('Energy above ground state ($eV/atom$)', fontsize=font_size)
        plt.ylim(top=1, bottom=0)
        plt.xlim((0 - 0.5, len(compounds)))
        # plt.grid('on')
        from thermoelectric import chem2latex
        corrected_compounds = [chem2latex(comp) for comp in compounds]
        plt.xticks(range(len(compounds)), corrected_compounds)
        plt.xticks(rotation=90, fontsize=font_size - 1)
        plt.legend(loc='upper right')

        plt.tight_layout()

        if save_fig:
            print('Saving plots')
            Plots.plot_save(f'skyline_{classifier}', path='plots/paper/', formats=['pdf'], transparent=True)
            # plt.savefig('plots/paper/skyline.png', dpi=800)
            # plt.savefig('plots/paper/skyline.svg', dpi=1800)
            # plt.savefig(f'{sky_line_path}skyline.png', dpi=800)
        plt.show()

    if plot_dist:
        # sns.set_style('darkgrid')
        size = (180 * .2, 88.9969)
        if f is not None:
            size = f.get_size_inches() * 25.4
            size[0] = 180 - size[0]
        f, ax, font_size = plot_format(size)

        verical = True
        sns.distplot(eng, bins=20, vertical=verical, color='k', label='Amorphous\nlimit')
        # sns.distplot(min_eng_above_ground, bins=25, vertical=verical, color='k', label='Amorphous\nlimit')
        sns.distplot(y_pos, bins=30, vertical=verical, color='darkgreen', label='Synthesizable')
        sns.distplot(y_neg, bins=30, vertical=verical, color='darkred', label='Anomaly')

        plt.ylim(top=1, bottom=0)
        plt.grid('on')
        plt.xlabel('PDF')
        plt.legend()
        if plot_evaluation:
            Plots.remove_ticks(ax, axis='y', keep_ticks=True)

        if save_fig:
            print('Saving plots')
            Plots.plot_save(f'skyline_dist_{classifier}', path='plots/paper/', formats=['pdf'], transparent=True)
            # plt.savefig('plots/paper/skyline_dist.png', dpi=800)
            # plt.savefig('plots/paper/skyline_dist.svg', dpi=800)

        plt.show()

    print('End fn.')


def plot_roc(path='results/Classification/run_043_pca_400/', save_fig=True):
    classifiers = load_var(path + 'classifiers.pkl')
    classifiers.pop('PCA')
    classifiers.pop('StandardScaler')
    labels = load_var(path + 'predictions.pkl')['labels']
    predictions = load_var(path + 'predictions.pkl')['predictions']
    y = labels
    # prob = predictions

    clf_nick_names = {'MLPClassifier': 'Neural network',
                      'RandomForestClassifier': 'Random forest',
                      # 'SVC': "SVM",
                      # 'BaggingClassifier': 'SVM-B'
                      }

    f, axs, font_size = plot_format(ncols=2, nrows=2, equal_axis=True)
    stat = {'Classifier': [], 'AUC': [], 'Accuracy': [], 'Sensitivity': [], 'Specificity': []}

    for plt_n, clf_name in enumerate(clf_nick_names):
        sub_plt_pos = np.unravel_index(plt_n * 2, axs.shape)
        ax = axs[sub_plt_pos]
        prob = predictions[clf_name]
        prob_pos = prob[y > 0]
        prob_neg = prob[y < 0]
        fpr, tpr, threshold = roc_curve(y, prob)
        roc_auc = auc(fpr, tpr)

        best_threshold = classifiers[f'{clf_name}_best_threshold']

        # from train_CAE_binary_clf import get_metric_and_best_threshold_from_roc_curve
        from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve

        # best_acc, best_threshold, ind = get_metric_and_best_threshold_from_roc_curve(tpr, fpr, threshold,
        #                                                                              len(prob_pos), len(prob_neg))
        # precision, recall, thresholds = precision_recall_curve(y, prob)
        # F1 = 2 * (precision * recall) / (precision + recall)

        # best_threshold = 0.5

        # best_threshold = thresholds[np.argmax(F1)]

        y_pred = np.sign(np.sign(prob - best_threshold) + .5)
        acc = accuracy_score(y, y_pred)

        # gmeans = np.sqrt(tpr * (1 - fpr))
        # ix = np.argmax(gmeans)
        # print('Best Threshold=%f, G-Mean=%.3f' % (threshold[ix], gmeans[ix]))

        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)

        stat['Classifier'].append(clf_nick_names[clf_name])
        stat['AUC'].append(roc_auc)
        stat['Accuracy'].append(acc)
        stat['Sensitivity'].append(sensitivity)
        stat['Specificity'].append(specificity)

        print(classification_report(y, y_pred))
        print(f'Best threshold = {best_threshold:.5f}')

        lw = 2
        sns.set_color_codes('colorblind')
        ax.plot(fpr, tpr,
                # color='darkorange',
                lw=lw,
                # label='ROC (AUC = %0.2f)' % roc_auc,
                label='ROC',
                )

        ax.text(x=.25, y=0.05, s=f'AUC = {roc_auc:0.2f}', fontsize=font_size)
        ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--',
                label='Dumb classifier', )

        ax.scatter(x=fp / (fp + tn), y=tp / (tp + fn), marker='o', color='orchid', label='Best', s=30)
        # box
        # ax[sub_plt_pos].plot([0, 1], [0, 0], color='k', lw=0.75, linestyle='-')
        # ax[sub_plt_pos].plot([0, 1], [1, 1], color='k', lw=0.75, linestyle='-')
        # ax[sub_plt_pos].plot([1, 1], [0, 1], color='k', lw=0.75, linestyle='-')
        # ax[sub_plt_pos].plot([0, 0], [0, 1], color='k', lw=0.75, linestyle='-')

        if sub_plt_pos[0] + 1 == len(clf_nick_names):
            ax.set_xlabel('False positive', fontsize=font_size)
            ax.legend(loc='upper center')
        else:
            Plots.remove_ticks(ax, keep_ticks=True)
        ax.set_ylabel('True positive', fontsize=font_size)
        n = 0.0
        ax.set_ylim(0 - n, 1 + n)
        ax.set_xlim(0 - n, 1 + n)
        Plots.change_number_of_ticks(ax, 3, dtype=float)
        ax.grid()

        sub_plt_pos = np.unravel_index(plt_n * 2 + 1, axs.shape)
        ax = axs[sub_plt_pos]
        sns.set_color_codes('colorblind')
        bins = 15
        pos_plot = sns.distplot(prob_pos, bins=bins,
                                label=f'Synthesizables', hist_kws=dict(edgecolor="k", linewidth=1),
                                ax=ax, kde=False, norm_hist=True, color='g')
        neg_plot = sns.distplot(prob_neg, bins=bins,
                                label=f'Anomalies', hist_kws=dict(edgecolor="k", linewidth=1),
                                ax=ax, kde=False, norm_hist=True, color='r')

        ax.plot([best_threshold, best_threshold], [0, ax.get_ylim()[1] / 2], color='orchid',
                lw=lw, linestyle='--', label='Best')

        # box
        # ymax = max(neg_plot.dataLim.max[1], pos_plot.dataLim.max[1])
        # ax.plot([0, 1], [0, 0], color='k', lw=0.75, linestyle='-')
        # ax.plot([0, 1], [1.05 * ymax, 1.05 * ymax], color='k', lw=0.75, linestyle='-')
        # ax.plot([1, 1], [0, 1.05 * ymax], color='k', lw=0.75, linestyle='-')
        # ax.plot([0, 0], [0, 1.05 * ymax], color='k', lw=0.75, linestyle='-')
        if sub_plt_pos[0] + 1 == len(clf_nick_names):
            ax.set_xlabel('Predictions', fontsize=font_size)
            ax.legend(loc='upper center')
        else:
            Plots.remove_ticks(ax, keep_ticks=True)
        ax.text(x=.5, y=0.85, s=f'Acc.={acc * 100:0.1f}%', fontsize=font_size, transform=ax.transAxes,
                ha='center')

        print(f'{clf_nick_names[clf_name]} accuracy: {acc * 100:0.1f}%')

        ax.yaxis.set_ticks_position('right')
        import string
        abc = string.ascii_lowercase[sub_plt_pos[0]]
        # abc = r'\textbf{}'.format(abc)
        ax.set_ylabel(f'$\\bf{abc})$ {clf_nick_names[clf_name]}', fontsize=font_size, )
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()

        Plots.change_number_of_ticks(ax, 4)

        # from matplotlib.ticker import MaxNLocator
        # ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        # Plots.ylabel_right_side(ax, f'{clf_nick_names[clf_name]}')
        n = 0.0
        # ax.set_ylim(0 - n, 1 + n)
        ax.set_xlim(0 - n, 1 + n)
        ax.grid()

    stat = pd.DataFrame(stat)
    stat.to_csv('plots/paper/roc-stats.csv')

    if save_fig:
        print('Saving plots')
        # plt.savefig(f'plots/paper/roc{clf_name}.png', dpi=800)
        # plt.savefig(f'plots/paper/roc_{clf_name}.svg', dpi=800)
        plot_save('roc', save_to_papers=True)
    plt.show()
    print('End fn')
    return

    for clf_name in classifiers:
        f, ax, font_size = plot_format((79 / 2, 90 / 2))
        prob = predictions[clf_name]
        prob_pos = prob[y > 0]
        prob_neg = prob[y < 0]

        # prob_neg = prob_neg[(prob_neg < 0.95) & (prob_neg > 0.05)]
        # prob_pos = prob_pos[(prob_pos < 0.95) & (prob_pos > 0.05)]

        sns.set_color_codes('colorblind')
        bins = 20
        pos_plot = sns.distplot(prob_pos, bins=bins,
                                label=f'Synthesizables',
                                ax=ax, kde=False, norm_hist=True, color='g')
        neg_plot = sns.distplot(prob_neg, bins=bins,
                                label=f'Anomalies',
                                ax=ax, kde=False, norm_hist=True, color='r')

        # box
        ymax = max(neg_plot.dataLim.max[1], pos_plot.dataLim.max[1])
        plt.plot([0, 1], [0, 0], color='k', lw=0.75, linestyle='-')
        plt.plot([0, 1], [1.05 * ymax, 1.05 * ymax], color='k', lw=0.75, linestyle='-')
        plt.plot([1, 1], [0, 1.05 * ymax], color='k', lw=0.75, linestyle='-')
        plt.plot([0, 0], [0, 1.05 * ymax], color='k', lw=0.75, linestyle='-')

        ax.set_xlabel('Synthesizability likelihood', fontsize=font_size)
        plt.legend(loc='upper center')
        plt.title(f'{clf_nick_names[clf_name]}')

        ax.yaxis.set_ticks_position('right')
        ax.set_ylabel('PDF', fontsize=font_size)
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        # ax.set_aspect('equal', adjustable='datalim')

        if save_fig:
            print('Saving plots')
            # plt.savefig(f'plots/paper/prediction_syn_dist_{clf_name}.png', dpi=800)
            plt.savefig(f'plots/paper/roc_prediction_syn_dist_{clf_name}.svg', dpi=800)

        ax.set_ylim(0 - n)
        plt.show()

    print('End fn.')


def plot_roc_comparision(save_fig=False):
    for i in [
        38, 39,
        40, 41, 43, 45, 49, 50, 51, 52, 53, 56, 57
    ]:
        path = f'results/CAE/run_{i:03d}/'
        classifiers = load_var(path + 'run.pkl')
        classifiers.pop('PCA')
        classifiers.pop('StandardScaler')
        labels = load_var(path + 'predictions.pkl')['labels']
        predictions = load_var(path + 'predictions.pkl')['predictions']
        y = labels
        # prob = predictions

    clf_nick_names = {'MLPClassifier': 'Neural network', 'RandomForestClassifier': 'Random forest', 'SVC': "SVM"}

    for clf_name in classifiers:
        f, ax, font_size = plot_format((79 / 2, 90 / 2))
        prob = predictions[clf_name]
        fpr, tpr, threshold = roc_curve(y, prob)
        roc_auc = auc(fpr, tpr)

        lw = 2
        sns.set_color_codes('colorblind')
        plt.plot(fpr, tpr,
                 # color='darkorange',
                 lw=lw,
                 # label='ROC (AUC = %0.2f)' % roc_auc,
                 label='ROC',
                 )

        plt.text(x=.25, y=0.05, s=f'AUC={roc_auc:0.2f}', fontsize=font_size)

        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--',
                 label='Dumb Classifier'
                 )
        # box
        plt.plot([0, 1], [0, 0], color='k', lw=0.75, linestyle='-')
        plt.plot([0, 1], [1, 1], color='k', lw=0.75, linestyle='-')
        plt.plot([1, 1], [0, 1], color='k', lw=0.75, linestyle='-')
        plt.plot([0, 0], [0, 1], color='k', lw=0.75, linestyle='-')

        ax.set_xlabel('False Positive Rate', fontsize=font_size)
        ax.set_ylabel('True Positive Rate', fontsize=font_size)
        n = 0.05
        ax.set_ylim(0 - n, 1 + n)
        ax.set_xlim(0 - n, 1 + n)
        plt.legend(loc='lower center')
        plt.title(f'{clf_nick_names[clf_name]}')
        # ax.set_aspect('equal', adjustable='datalim')
        if save_fig:
            print('Saving plots')
            # plt.savefig(f'plots/paper/roc{clf_name}.png', dpi=800)
            plt.savefig(f'plots/paper/roc_{clf_name}.svg', dpi=800)
        plt.show()


def plot_feature_space(clf_path='results/Classification/run_045/classifiers.pkl', cae_path='results/CAE/run_045/'):
    run = RunSet(ini_from_path=cae_path, new_result_path=True,
                 # params={'run_id': 'run_001'}
                 )
    classifiers = load_var(clf_path)
    rf = classifiers['RandomForestClassifier']
    clf = rf
    from imblearn.over_sampling import RandomOverSampler
    from data_preprocess import load_data
    from train_CAE_binary_clf import StandardScaler, PCA
    X_train, y_train, X_test, y_test = load_data(run=run)
    ros = RandomOverSampler(random_state=0)
    X_train, y_train = ros.fit_resample(X_train, y_train)
    X_test, y_test = ros.fit_resample(X_test, y_test)
    ss = StandardScaler()
    ss.fit(X_train)
    X_train = ss.transform(X_train)
    X_test = ss.transform(X_test)
    pca = PCA(n_components=1000, whiten=True, random_state=0)

    pca = pca.fit(X_train)
    print('Explained variance percentage = %0.2f' % sum(pca.explained_variance_ratio_))
    X_train_org = pca.transform(X_train)
    X_test_org = pca.transform(X_test)

    y_test_org = y_test.copy()

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_moons, make_circles, make_classification
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    names = ["SVM",
             "Decision\nTree", "Random\nForest", "Neural\nNet."]

    classifiers = [
        SVC(gamma=2, C=1, random_state=0),
        DecisionTreeClassifier(max_depth=5, random_state=0),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=0),
        MLPClassifier(alpha=1, max_iter=1000, random_state=0)
    ]
    h = .02  # step size in the mesh
    N = 600
    plt.figure(figsize=(8, 10))
    datasets = [[0, 6], [0, 1], [6, 5], [5, 1], [4, 121]]
    i = 1
    for d in range(len(datasets)):

        X = X_test_org[:N, np.array(datasets[d])].copy()
        y = y_test_org[:N].copy()

        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.4, random_state=42)

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # just plot the dataset first
        # cm = plt.cm.RdBu
        # cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        cm = plt.cm.RdYlGn
        cm_bright = ListedColormap(['#FF0000', '#00FF00'])

        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        if d == 0:
            ax.set_title("Input data")
        ax.set_ylabel(f'Feat. {datasets[d][1]}')
        ax.set_xlabel(f'Feat. {datasets[d][0]}')

        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
                   edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())

        i += 1
        for name, clf in zip(names, classifiers):
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            # clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

            # Plot the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                       edgecolors='k')
            # Plot the testing points
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                       edgecolors='k', alpha=0.5)

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())

            if d == 0:
                ax.set_title(name)

            ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                    size=15, horizontalalignment='right')
            i += 1

    plt.show()

    import shap
    # explainer = shap.TreeExplainer(rf)
    # shap_values = explainer.shap_values(X_test)
    # # shap.summary_plot(shap_values, X_test, plot_type="bar")
    # shap.summary_plot(shap_values, X_test)

    # fi = np.argsort(-rf.feature_importances_)

    print('End fn.')


def plot_atomic_number_dist(save_fig=False, show_plot=True, data=None, ax=None):
    if ax is None:
        f, ax, fs = plot_format(180)
    else:
        show_plot = False

    par = 'formula'
    if 'symbols' in data:
        par = 'symbols'
    if par == 'formula':
        formula = [np.unique(i) for i in data[par]]
    else:
        formula = [list(i) for i in data['symbols']]
    formula = np.concatenate(formula)
    ele, rep = np.unique(formula, return_counts=True)
    pd_table = PeriodicTable().table
    pd_table['repeats'] = pd.Series([0] * len(pd_table))
    for e, n in zip(ele, rep):
        # pd_table['repeats'][pd_table['symbol'] == e] = n / sum(rep)
        pd_table.loc[pd_table['symbol'] == e, 'repeats'] = n / sum(rep)

    # f, ax, fs = plot_format(180)
    sns.barplot(x='symbol', y='repeats', data=pd_table[pd_table['repeats'] > 0], ax=ax)
    ax.grid()
    ax.set_xlabel('Symbols')
    ax.set_ylabel('Probability Distribution')
    # f.autofmt_xdate()
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_y(label.get_position()[1] - (i % 2) * 0.05)

    if show_plot:
        plt.show()

    return ax


def plot_acc_vs_threshold(save_fig=True):
    print('plot_acc_vs_threshold')
    path = data_path + 'cod/results/Classification/run_043_pca_400/'
    classifiers = load_var(path + 'classifiers.pkl')
    # classifiers.pop('PCA')
    # classifiers.pop('StandardScaler')
    labels = load_var(path + 'predictions.pkl')['labels']
    predictions = load_var(path + 'predictions.pkl')['predictions']
    y = labels

    # from data_preprocess import data_preparation
    # X_train, y_train, X_test, y_test = data_preparation(use_all_data=False, split_data_frac=0.5, standard_scaler=False,
    #                                                     apply_pca=False)
    # pca = classifiers.pop('PCA')
    # ss = classifiers.pop('StandardScaler')
    # X_train = ss.transform(X_train)
    # X_train = pca.transform(X_train)

    clf_nick_names = {'MLPClassifier': 'Neural network',
                      'RandomForestClassifier': 'Random forest',
                      # 'SVC': "SVM",
                      # 'BaggingClassifier': 'SVM-B'
                      }

    f, axs, font_size = plot_format(ncols=1, nrows=2)
    for plt_n, clf_name in enumerate(clf_nick_names):
        sub_plt_pos = np.unravel_index(plt_n, axs.shape)
        ax = axs[sub_plt_pos]
        prob = predictions[clf_name]
        prob_pos = prob[y > 0]
        prob_neg = prob[y < 0]
        fpr, tpr, threshold = roc_curve(y, prob)
        # roc_auc = auc(fpr, tpr)

        # # calculate the g-mean for each threshold
        # gmeans = np.sqrt(tpr * (1 - fpr))
        # # locate the index of the largest g-mean
        # ix = np.argmax(gmeans)
        # print('Best Threshold=%f, G-Mean=%.3f' % (threshold[ix], gmeans[ix]))

        # prob_train = classifiers[clf_name].predict_proba(X_train)[:, 1]

        acc = []
        acc_train = []
        # threshold = np.linspace(0, 1, 100)
        print('Calc. threshold')
        # for t in threshold:
        #     acc.append(accuracy_score(y, np.sign(np.sign(prob - t) + .5)))
        # acc_train.append(accuracy_score(y_train, np.sign(np.sign(prob_train - t) + .5)))

        num_pos_class = len(prob_pos)
        num_neg_class = len(prob_neg)
        tp = tpr * num_pos_class
        tn = (1 - fpr) * num_neg_class
        acc = (tp + tn) / (num_pos_class + num_neg_class)
        best_threshold_ind = np.argmax(acc)
        best_threshold = threshold[best_threshold_ind]
        # best_threshold = np.floor(100 * best_threshold) / 100

        # from train_CAE_binary_clf import get_metric_and_best_threshold_from_roc_curve
        from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(y, prob)
        F1 = 2 * (precision * recall) / (precision + recall)
        # best_threshold = 0.5
        best_threshold_ind = np.argmax(F1)
        best_threshold = thresholds[best_threshold_ind]

        print(f'{clf_name}, best threshold = {best_threshold}')
        y_pred = np.sign(np.sign(prob - best_threshold) + .5)
        acc_val = accuracy_score(y, y_pred)

        classifiers.update({f'{clf_name}_best_threshold': best_threshold})

        print('Plotting')
        lw = 2
        sns.set_color_codes('colorblind')
        ax.plot(threshold, acc,
                # color='darkorange',
                lw=lw,
                # label='Test',
                # label='ROC',
                )

        # ax.plot(threshold, acc_train,
        #         # color='darkorange',
        #         lw=lw,
        #         # label='Train',
        #         # label='ROC',
        #         )
        ind = np.argmax(acc)
        # ind_train = np.argmax(acc_train)
        ax.legend([f'Test (Max. Acc. = {acc_val:.3f} \n@ Thr. = {best_threshold})',
                   # f'Train (Max. Acc. = {acc_train[ind_train]:.2f} \n@ Thr. = {threshold[ind_train]:.2f})'
                   ])
        # max_acc = max(acc)
        # Plots.put_costume_text_instead_of_legends(ax, f'Max. Acc. = {max_acc:.3f}')

        ax.grid()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Accuracy')
        Plots.ylabel_right_side(ax, clf_nick_names[clf_name])

    ax.set_xlabel('Threshold')
    plt.show()

    save_var(classifiers, path + 'classifiers.pkl')


def plot_electrode_thermo_preds():
    path_thermoelectric = f'{data_path}cod/thermoelectric/'
    preds_thermo = load_var(f'{path_thermoelectric}/pred_list_top_10.pkl')
    preds_bat_mp = load_var(data_path + f'cod/battery/battery-predictions.pkl')
    preds_bat_cod = load_var(data_path + f'cod/battery/preds_bat-cod_all_data.pkl')
    preds_bat_mp['Data set'] = 'MP'
    preds_bat_cod['Data set'] = 'COD'
    preds_bat = pd.concat([preds_bat_mp, preds_bat_cod]).reset_index()
    preds_thermo['Data set'] = preds_thermo['data_set'].str.upper()

    clf_labels = {
        'MLPClassifier': 'Neural network',
        'RandomForestClassifier': 'Random forest',
        # 'BaggingClassifier': 'SVM-Bagging'
    }
    f, axs, font_size = plot_format()
    joint_plot = False
    if joint_plot:

        for tp in ['bat', 'thermo']:
            preds = locals()[f'preds_{tp}']
            for d in ['MP', 'COD']:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df = preds[preds['Data set'] == d]
                    df['MLP'] = df['MLPClassifier']
                    df['RF'] = df['RandomForestClassifier']
                sns.jointplot(data=df, x="MLP", y="RF",
                              xlim=(0, 1), ylim=(0, 1), height=88 / 100 * 3.93701,
                              ratio=2, marginal_ticks=True,
                              # palette=sns.palplot(sns.color_palette("RdBu", 10)),
                              color=Plots.colors_databases[d.lower()],
                              marginal_kws=dict(bins=15, fill=True),
                              joint_kws=dict(gridsize=9),
                              kind='hex')

                clf_comp = np.count_nonzero(df['RandomForestClassifier_label'] *
                                            df['MLPClassifier_label'] > 0) / len(df) * 100
                print(f'Set={d}, sample type={tp} -> agreement = {clf_comp:.1f}%')
                gl_run.save_plot(f'{tp}_{d}')
                plt.show()

    clf_comp_bat = np.count_nonzero(preds_bat['RandomForestClassifier_label'] *
                                    preds_bat['MLPClassifier_label'] > 0) / len(preds_bat) * 100
    clf_comp_thermo = np.count_nonzero(preds_thermo['RandomForestClassifier_label'] *
                                       preds_thermo['MLPClassifier_label'] > 0) / len(preds_thermo) * 100
    print(f'Classifiers agreement for battery materials is: {clf_comp_bat:.1f}')
    print(f'Classifiers agreement for thermoelectric materials is: {clf_comp_thermo:1f}')

    predictions = load_var(data_path + 'cod/results/Classification/run_043_pca_400/predictions.pkl')
    clf = load_var(data_path + 'cod/results/Classification/run_043_pca_400/classifiers.pkl')

    best_threshold = clf['RandomForestClassifier_best_threshold']
    rf = predictions['predictions']['RandomForestClassifier']
    rf = np.sign(np.sign(rf - best_threshold) + .5)

    best_threshold = clf['MLPClassifier_best_threshold']
    mlp = predictions['predictions']['MLPClassifier']
    mlp = np.sign(np.sign(mlp - best_threshold) + .5)

    y = predictions['labels']
    clf_comp = np.count_nonzero(mlp * rf > 0) / len(mlp) * 100
    print(f'The average COD agreement = {clf_comp:.1f}')

    # preds_test = {'MLP': predictions['predictions']['MLPClassifier'],
    #               'RF': predictions['predictions']['RandomForestClassifier']}
    # preds_test = pd.DataFrame(preds_test)
    # sns.jointplot(data=preds_test, x="MLP", y="RF",
    #               xlim=(0, 1), ylim=(0, 1), height=88 / 100 * 3.93701,
    #               ratio=2, marginal_ticks=True,
    #               # palette=sns.palplot(sns.color_palette("RdBu", 10)),
    #               # color=Plots.colors_databases[d.lower()],
    #               marginal_kws=dict(bins=15, fill=True),
    #               joint_kws=dict(gridsize=9),
    #               kind='hex')
    # plt.show()

    kde_plot = False
    df = {'Synthesizability likelihood': [], 'Set': []}
    if kde_plot:
        f, axs, font_size = plot_format()
        print('kde')
        df['Synthesizability likelihood'] += list(preds_bat['MLPClassifier'][preds_bat['Data set'] == 'MP'])
        df['Set'] += ['MP - NN'] * np.count_nonzero(preds_bat['Data set'] == 'MP')
        df['Synthesizability likelihood'] += list(preds_bat['RandomForestClassifier'][preds_bat['Data set'] == 'MP'])
        df['Set'] += ['MP - RF'] * np.count_nonzero(preds_bat['Data set'] == 'MP')
        df['Synthesizability likelihood'] += list(preds_bat['MLPClassifier'][preds_bat['Data set'] == 'COD'])
        df['Set'] += ['COD - NN'] * np.count_nonzero(preds_bat['Data set'] == 'COD')
        df['Synthesizability likelihood'] += list(preds_bat['RandomForestClassifier'][preds_bat['Data set'] == 'COD'])
        df['Set'] += ['COD - RF'] * np.count_nonzero(preds_bat['Data set'] == 'COD')
        df = pd.DataFrame(df)
        # sns.displot(df, x="Synthesizability likelihood", hue="Set", ax=axs, norm_hist=True)
        sns.distplot(preds_bat['MLPClassifier'][preds_bat['Data set'] == 'MP'],
                     label='MP - NN', norm_hist=True, ax=axs)
        sns.distplot(preds_bat['RandomForestClassifier'][preds_bat['Data set'] == 'MP'],
                     label='MP - RF', norm_hist=True, ax=axs)
        sns.distplot(preds_bat['MLPClassifier'][preds_bat['Data set'] == 'COD'],
                     label='COD - NN', norm_hist=True, ax=axs)
        sns.distplot(preds_bat['RandomForestClassifier'][preds_bat['Data set'] == 'COD'],
                     label='COD - RF', norm_hist=True, ax=axs)
        axs.set_xlim((0, 1))
        axs.set_xlabel('Synthesizability likelihood')
        axs.legend()
        plt.show()

    globals().update(locals())


if __name__ == '__main__':
    # import plotsettings
    # publishable = plotsettings.Set('Nature')

    gl_run = RunSet()
    # atom_3d_visualizations()
    # atom_3d_visualizations_2()
    # plot_literature()
    # plot_lattice_constants(save_fig=False)
    # plot_min_dist(save_fig=False)
    # plot_lattice_cnt_and_min_dist(save_fig=True)

    # plot_pca()

    # plot_chemical_elements_distribution(save_fig=False)

    # Loss function plots
    compare_cae = False
    if compare_cae:
        # plot_loss_fn_cae_comparision()
        plot_loss_fn()
        # plot_cae_acc_comparision()
        # plot_cae_exp_var_comparision()
        # plot_cae_roc_dist_comparision()
        # plot_cae_roc_comparision()

    # plot_sky_line_detailed_evaluation(include_group_3=True, save_fig=True,
    #                                   plot_dist=True, classifier='RandomForestClassifier')
    # plot_sky_line_detailed_evaluation(include_group_3=True, save_fig=True, plot_evaluation=True,
    #                                   plot_dist=True, classifier='MLPClassifier', plot_skyline_anomalies=False)
    # plot_sky_line_detailed_evaluation(include_group_3=True, save_fig=True,
    #                                   plot_dist=True, plot_evaluation=False,
    #                                   )

    # plot_roc(save_fig=True)
    # plot_roc_comparision(save_fig=False)
    # plot_acc_vs_threshold(save_fig=True)

    # plot_electrode_thermo_preds()

    # plot_feature_space()

    print('end')
