from xiespp.synthesizability_1.utility import util_plot
from xiespp.synthesizability_1.utility import util_plot as Plots
from ase.formula import Formula

from matplotlib.ticker import StrMethodFormatter


def literature_plot(save_plot=True):
    n_top_comp = 15

    data = pd.read_csv(f'{local_data_path}/data_bases/cspd/summary.csv',
                       index_col=0)
    natural_formula = [
        'TiO2', 'ZnO', 'CO2', 'SiO2', 'Al2O3', 'CSi', 'GaAs', 'GaN', 'ZrO2',
        'SnO2', 'MgO', 'CdS', 'CeO2', 'H2O', 'Fe3O4'
    ]

    data = data.head(n_top_comp)
    data['natural formula'] = natural_formula[:n_top_comp]
    data['natural formula'] = [Formula(i).format('latex') for i in data['natural formula']]

    df1 = data[['natural formula', 'anomaly selected']].rename(columns={'anomaly selected': 'n'})
    df2 = data[['natural formula', 'cod counts']].rename(columns={'cod counts': 'n'})
    df1['type'] = 'Anomaly'
    df2['type'] = 'COD'
    df = pd.concat([df1, df2])

    f, axs, fs = util_plot.plot_format((88, 70), ncols=2)

    ################# Plot b)
    ax = axs[0]
    sns.barplot(
        x="literature",
        y="natural formula",
        data=data,
        label="Study intensity",
        color='slateblue',
        # palette="GnBu_d",
        # palette="pastel",
        ax=ax,
    )

    ax.set_ylabel('')
    ax.set_xlabel('Absolute repetition')
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    # ax.xaxis.set_ticks([0, 1, 2, 3, 4, 5])

    # ax.yaxis.grid(False)
    ax.yaxis.grid(True)
    # ax.grid()
    lines, labels = ax.get_legend_handles_labels()
    # ax.legend(lines + lines2, labels + labels2, loc='lower right')

    ################# Plot c)
    ax = axs[1]

    # sns.barplot(data=data, x='anomaly selected', y='natural formula', color='darkred', ax=ax)
    sns.barplot(data=df, x='n', y='natural formula', ax=ax, hue='type', palette=sns.color_palette("dark", 8))
    ax.set_ylabel('')
    ax.set_xlabel('Number of selected\npolymorphs', fontsize=fs)
    # ax.xaxis.set_ticks([0, 20, 40])

    ax.yaxis.grid(True)
    ax.xaxis.grid(True)

    # ax.grid('on')
    # ax.grid()

    lines2, labels2 = ax.get_legend_handles_labels()
    # ax.get_yaxis().set_visible(False)
    ax.legend(loc='lower right')

    ###############
    Plots.annotate_subplots_with_abc(axs, x=0, y=1.05, start_from=1)
    if save_plot:
        Plots.plot_save('literature_b_c_', save_to_papers=True)
    plt.show()
    pass

def roc_plot(save_plot=True):
    # ###################### Preparation

    from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc

    cnn_dir = 'finalized_results/cnn-3-13-7-over'
    mlp_dir = 'finalized_results/cae-mlp-3-13-7-over'

    yp_test_cnn = pd.read_csv(cnn_dir + '/yp_test.csv').rename(columns={
        'y': 'y_true', 'yp': 'yp_proba', 'ypl': 'yp_label'})
    mlp_clf = load_var(mlp_dir + f'/mlp_clf_pca10/classifier_class.pkl')
    yp_test_mlp = pd.DataFrame(mlp_clf.predictions['test'])

    # Checking if the test set of the classifiers are the same set
    assert (yp_test_cnn['y_true'] == yp_test_mlp['y_true']).all()
    y = yp_test_cnn['y_true']

    predictions = {'CNN-C': yp_test_cnn, 'CAE-MLP-C': yp_test_mlp}
    clf_nick_names = {
        'CNN-C': 'CNN-C',
        'CAE-MLP-C': 'CAE-MLP-C',
    }

    f, axs, font_size = Plots.plot_format(ncols=2, nrows=2, equal_axis=True)
    stat = {'Classifier': [], 'AUC': [], 'Accuracy': [], 'Sensitivity': [], 'Specificity': []}

    for plt_n, clf_name in enumerate(predictions):
        sub_plt_pos = np.unravel_index(plt_n * 2, axs.shape)
        ax = axs[sub_plt_pos]
        prob = predictions[clf_name]['yp_proba']
        prob_pos = prob[y > 0]
        prob_neg = prob[y < 0]
        fpr, tpr, threshold = roc_curve(y, prob)
        roc_auc = auc(fpr, tpr)

        best_threshold = 0.5

        from sklearn.metrics import confusion_matrix
        y_pred = predictions[clf_name]['yp_label']
        acc = accuracy_score(y, y_pred)

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

        # ###################### Plot ROC

        lw = 2
        #     sns.set_color_codes('colorblind')
        sns.set_color_codes('dark')
        ax.plot(fpr, tpr,
                # color='darkorange',
                lw=lw,
                # label='ROC (AUC = %0.2f)' % roc_auc,
                label='ROC',
                )

        ax.text(x=.25, y=0.05, s=f'AUC = {roc_auc:0.3f}', fontsize=font_size)
        ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--',
                label='Dumb classifier', )

        ax.scatter(x=fp / (fp + tn), y=tp / (tp + fn), marker='o', color='k', label='Decision point', s=30)
        #     # box
        #     # ax[sub_plt_pos].plot([0, 1], [0, 0], color='k', lw=0.75, linestyle='-')
        #     # ax[sub_plt_pos].plot([0, 1], [1, 1], color='k', lw=0.75, linestyle='-')
        #     # ax[sub_plt_pos].plot([1, 1], [0, 1], color='k', lw=0.75, linestyle='-')
        #     # ax[sub_plt_pos].plot([0, 0], [0, 1], color='k', lw=0.75, linestyle='-')

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
        #     ax.grid()

        sub_plt_pos = np.unravel_index(plt_n * 2 + 1, axs.shape)

        # ###################### Plot Distribution
        ax = axs[sub_plt_pos]
        bins = 15
        pos_plot = sns.distplot(prob_pos, bins=bins,
                                label=f'Synthesizables', hist_kws=dict(edgecolor="k", linewidth=0, alpha=.65),
                                ax=ax, kde=False, norm_hist=True,
                                color='darkgreen'
                                )
        neg_plot = sns.distplot(prob_neg, bins=bins,
                                label=f'Anomalies', hist_kws=dict(edgecolor="k", linewidth=0, alpha=.65),
                                ax=ax, kde=False, norm_hist=True,
                                color='peru'
                                )

        #     ax.plot([best_threshold, best_threshold], [0, ax.get_ylim()[1] / 2], color='orchid',
        #             lw=lw, linestyle='--', label='Best')

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
            ax.set_xlabel('', fontsize=font_size)
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
        n = -0.0
        # ax.set_ylim(0 - n, 1 + n)
        ax.set_xlim(0 - n, 1 + n)
    #     ax.grid()
    # ax.margins()

    stat = pd.DataFrame(stat)
    stat.to_csv('plots/paper/roc-stats.csv')

    if save_plot:
        print('Saving plots')
        # plt.savefig(f'plots/paper/roc{clf_name}.png', dpi=800)
        # plt.savefig(f'plots/paper/roc_{clf_name}.svg', dpi=800)
        Plots.plot_save('roc', save_to_papers=True)
    plt.show()


if __name__ == '__main__':
    print('Running plots')
    literature_plot()
