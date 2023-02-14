import xiespp.utility.util_plot as Plots
from xiespp.utility.utility_general import *
import seaborn as sns
from sklearn.metrics import accuracy_score


def plot_electrode_thermoelectric_distribution(df_inp0, title='MLP'):
    f, ax, fs = Plots.plot_format(nrows=1, ncols=1, size=150, font_size=12)
    for db in [
        'cod',
        'mp'
    ]:
        df_inp = df_inp0[df_inp0['db'] == db]
        y = df_inp['y'].to_numpy()
        y_pred = df_inp['ypl']
        acc = 100 * accuracy_score(y, y_pred)
        print(f'Data set {db} size = {len(y_pred):,}')

        label = None
        if db == 'cod':
            label = f'COD' + '(Sensitivity={:.1f}%)'.format(acc)
        if db == 'mp':
            label = 'MP' + '(Synthesizability={:.1f}%)'.format(acc)

        sns.distplot(df_inp['yp'], kde=None, label=label, ax=ax, bins=25,
                     color=Plots.colors_databases[db],
                     hist_kws=dict(edgecolor="k", linewidth=1, alpha=0.65),
                     norm_hist=True, )
        ylim = ax.get_ylim()
        #         ax.plot([best_threshold, best_threshold], [0., ylim[1]], '--', color='r', linewidth=2)
        ax.set_ylim(ylim)
        ax.set_ylabel('PDF')
        ax.set_xlabel('Synthesizability likelihood')
        ax.set_xlim([-0.0, 1.0])
        ax.legend(loc='best', prop={'size': 12})
        ax.set_title(title)
    plt.show()
