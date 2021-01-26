from pymatgen import MPRester
from collections import Counter
from predict_synthesis import *
# from plots import *
import requests
from time import sleep

api_key = "MCpqDZgh29W9r6X4Ru"


def get_battery_data(self, formula_or_batt_id):
    """Returns batteries from a batt id or formula.

    Examples:
        get_battery("mp-300585433")
        get_battery("LiFePO4")
    """
    return self._make_request('/battery/%s' % formula_or_batt_id)


def is_bat_mat(bat_1, ele_1, mat_1):
    if mat_1['spacegroup']['hall'] == bat_1['spacegroup']['hall']:
        if '-'.join(sorted(mat_1['elements'])) == ele_1:
            match = True
            for e in re.findall(r'[A-Z][a-z]?[0-9]*', mat_1['pretty_formula']):
                el = re.findall(r'[A-Z][a-z]?', e)[0]
                if el == bat_1['working_ion']:
                    continue
                el_n = re.findall(r'[0-9]+', e)
                if len(el_n) == 0:
                    el_n = 1
                else:
                    el_n = int(el_n[0])
                if bat_1['reduced_cell_composition'][el] != el_n:
                    match = False
            if match:
                return True
    return False


def find_battery_material_match(battery, method='hyb'):
    if isinstance(battery, str):
        battery = mpr.get_battery_data(battery)[0]
    bat_id = battery['battid']
    mp_id = bat_id.split('_')[0]
    mat = None

    if method == 'direct' or method == 'hyb':
        mat = mpr.get_data(mp_id)
        if len(mat) == 1:
            mat = mat[0]
            return mat, 'direct'
        else:
            mat = None
    if (mat is None) and (method == 'hyb'):
        method = 'search'
    if method == 'search':

        elements = '-'.join(sorted([battery['working_ion']] + battery['framework']['elements']))
        materials = mpr.get_data(elements)
        mat_equ = []
        for mat in materials:
            if is_bat_mat(battery, elements, mat):
                mat_equ.append(mat.copy())
        if len(mat_equ) == 0:
            red_print(f'Skipping {bat_id}, no equivalent cif found in the materials database')
            return None, None
        m = mat_equ[0]
        for m in mat_equ:
            if m['icsd_id'] is not None:
                break
        return m, method
    return None, None


MPRester.get_battery_data = get_battery_data

mpr: MPRester = MPRester(api_key)


def get_all_the_battery_data(test=False):
    battery_ids = mpr._make_request('/battery/all_ids')
    battery_data = []
    materials_data = []
    methods_data = []

    # results_battid = mpr.get_battery_data("mp-300585433")
    # results_formula = mpr.get_battery_data("LiFePO4")

    N = len(battery_ids)
    if test:
        N = 10
    for b in range(N):
        print(f'{b}/{N}: {battery_ids[b]}')
        time_out = 0
        time_out_trh = 5
        match, method, battery = None, None, None
        while True:
            try:
                battery = mpr.get_battery_data(battery_ids[b])
                if len(battery) != 1:
                    red_print(f'Skipping {battery_ids[b]}, number of found results: {len(battery)}')
                    battery = None
                    match = None
                    method = None
                    break
                battery = battery[0]
                match, method = find_battery_material_match(battery, method='hyb')
                break
            except Exception as e:
                time_out += 1
                red_print(f'Error {str(e)}')
                print(f'({time_out}/{time_out_trh}) Waiting for 5 seconds and re-trying...')
                sleep(5)
                if time_out > 3:
                    break
        battery_data.append(battery)
        materials_data.append(match)
        methods_data.append(method)
        str_1 = method if match is not None else 'No results'
        print(str_1)
        # print('found')
    save_var(locals(), 'sessions/battery.pkl')
    data = pd.DataFrame({'batteries': battery_data, 'materials': materials_data, 'methods': methods_data})
    for i in ['working_ion', 'battid']:
        data[i] = pd.Series([j[i] if j is not None else None for j in data['batteries']], index=data.index)
    for i in ['pretty_formula', 'icsd_id', 'material_id']:
        data[i] = pd.Series([j[i] if j is not None else None for j in data['materials']], index=data.index)

    save_var(data, data_path + 'cod/battery/battery.pkl')
    return data


def battery_systems():
    battery_ids = mpr._make_request('/battery/all_ids')
    # bid = battery_ids[10]
    working_ion = [i.split('_')[-1] for i in battery_ids]
    print(Counter(working_ion))
    print('End df')


def bat2cif():
    data = load_var(f'{data_path}cod/battery/battery.pkl')
    found_bat = np.array([i for i in range(len(data['material_id'])) if data['material_id'][i] is not None])
    print(Counter(data['working_ion'][found_bat]))
    found_bat = np.array_split(found_bat, len(found_bat) // 100)
    for fol in range(len(found_bat)):
        for atom in data['materials'][found_bat[fol]]:
            mid = atom['material_id']
            write_text(data_path + f'cod/battery/cif/{fol:03d}/{mid}.cif', atom['cif'], makedir=True)


def predict_bat(test=False, classifier=None, fraction_of_data=1.):
    files = list_all_files(data_path + 'cod/battery/cif/', pattern=r'[0-9]*/*.cif')
    if fraction_of_data < 1:
        red_print(f'part of data is being used, fraction: {fraction_of_data}')
        files = files[:int(len(files) * fraction_of_data)]
    N = len(files)
    if test:
        N = 5
    pred = predict_crystal_synthesis(files[:N], classifiers=classifier, save_tmp_encoded_atoms=True,
                                     redo_calc=False, skip_svm=True)

    save_var(pred, data_path + 'cod/battery/battery-predictions.pkl')
    save_df(pred, data_path + 'cod/battery/battery-predictions.txt')

    print('End fn')


def direct_cif_download(test=False):
    battery_ids = mpr._make_request('/battery/all_ids')
    N = len(battery_ids)

    if test:
        N = 10
    for b in range(N):
        print(f'{b}/{N}: {battery_ids[b]}')
        mp_id = battery_ids[b].split('_')[0]
        url = f'https://materialsproject.org/materials/{mp_id}/cif?type=computed&download=true'
        r = requests.get(url)
        txt = str(r.content)
        write_text(data_path + f'cod/battery/cif_direct/{mp_id}', txt)


def check_direct_method():
    data = load_var(f'{data_path}cod/battery/battery.pkl')
    found_bat = np.array([i for i in range(len(data['material_id'])) if data['material_id'][i] is not None])

    for i in found_bat:
        battery = data['batteries'][i]
        bat_id = battery['battid']
        elements = '-'.join(sorted([battery['working_ion']] + battery['framework']['elements']))
        material = data['materials'][i]
        # if not is_bat_mat(battery, elements, material):
        #     red_print('Error')
        print(f'{bat_id}: matched!') if is_bat_mat(battery, elements, material) else red_print(
            f'{bat_id}: did not match')

    print('End fn')


def plot_preds(pred, title=None, save_plot=True, **kwargs):
    classifiers = clf_labels
    # fig, ax = plt.subplots(nrows=len(classifiers), ncols=1, figsize=(4, 6. / 3. * len(classifiers)))
    f, ax, fs = Plots.plot_format(nrows=len(classifiers), ncols=1)
    i = 0

    data_set = kwargs['data_set']
    if 'bat' in kwargs['data_set']:
        data_set = 'cod'

    for c in classifiers:

        sns.distplot(pred[c], kde=None, label=c, ax=ax[i], bins=10,
                     color=Plots.colors_databases[data_set], hist_kws=dict(edgecolor="k", linewidth=1)
                     )

        # plt.legend()
        # ax[i].set_title(clf_labels[c], loc='right', fontsize=fs + 1)

        if i == 0:
            ax[i].text(0.55, ax[i].get_ylim()[1] * .15, 'Synthesis threshold', color='r', rotation=90)
            ax[i].title.set_text(title)

        # if data_set == 'cspd':
        #     Plots.add_hash_pattern(ax[i])

        ylim = ax[i].get_ylim()
        ax[i].plot([.5, .5], [0., ylim[1]], '--', color='r', linewidth=1)
        ax[i].set_ylim(ylim)

        Plots.ylabel_right_side(ax[i], clf_labels[c])
        ax[i].set_ylabel('Number of materials', fontsize=fs)
        ax[i].set_xlabel('')
        ax[i].set_xlim([-0.0, 1.0])
        ax[i].grid(True)
        if not i == len(classifiers) - 1:
            Plots.remove_ticks(ax[i], keep_ticks=True)

        Plots.put_costume_text_instead_of_legends(ax=ax[i], labels='Accuracy = {:.2f}%'.format(pred[c + '_acc'][0]))

        print(f'Classifier: {c}, Predicted positive percentage:'
              f' {np.round(np.count_nonzero(pred[c] > 0.5) / len(pred) * 100, 2)}%')
        i += 1
    plt.xlabel('Probability of Synthesis', fontsize=fs)
    if save_plot:
        run.save_plot(filename=title)
    plt.show()


def plot_clf_compare_points(pred, apply_label_points=False, title=None, save_plot=True, **kwargs):
    def label_point(x, y, val, axp):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for j, point in a.iterrows():
            axp.text(point['x'] + .02, point['y'], str(int(point['val'])))

    clf_n = [
        # 'BaggingClassifier',
        'MLPClassifier',
        'RandomForestClassifier'
    ]
    # clf_n = [c for c in pred.columns if 'Classifier' in c or 'SVC' in c]

    # data_set = kwargs['data_set']
    # if 'bat' in kwargs['data_set']:
    #     data_set = 'cod'

    # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
    f, ax, fs = Plots.plot_format((185, 70), nrows=1, ncols=3, font_size=9)
    sns.scatterplot(x=clf_n[0],
                    y=clf_n[2],
                    ax=ax[0],
                    data=pred)
    ax[0].set_xlabel(clf_labels[clf_n[0]], fontsize=fs)
    ax[0].set_ylabel(clf_labels[clf_n[2]], fontsize=fs)

    sns.scatterplot(x='BaggingClassifier',
                    y='MLPClassifier',
                    ax=ax[1],
                    data=pred)
    ax[1].set_xlabel(clf_labels[clf_n[0]], fontsize=fs)
    ax[1].set_ylabel(clf_labels[clf_n[1]], fontsize=fs)
    labels = [item.get_text() for item in ax[1].get_yticklabels()]
    ax[1].set_yticklabels([''] * len(labels), fontsize=fs)

    sns.scatterplot(x='RandomForestClassifier',
                    y='MLPClassifier',
                    ax=ax[2],
                    data=pred)
    labels = [item.get_text() for item in ax[2].get_yticklabels()]
    ax[2].set_yticklabels([''] * len(labels), fontsize=fs)
    ax[2].set_xlabel(clf_labels[clf_n[2]], fontsize=fs)
    ax[2].set_ylabel(clf_labels[clf_n[1]], fontsize=fs)

    Plots.ylabel_right_side(ax[2], title)

    if apply_label_points:
        label_point(pred[clf_n[0]], pred[clf_n[2]], pd.Series(range(len(pred))), ax[0])
        label_point(pred[clf_n[0]], pred[clf_n[1]], pd.Series(range(len(pred))), ax[1])
        label_point(pred[clf_n[2]], pred[clf_n[1]], pd.Series(range(len(pred))), ax[2])

    import matplotlib.patches as patches
    alpha = 0.25
    for i in range(len(ax)):
        ax[i].add_patch(patches.Rectangle((0, 0), 0.5, 0.5, alpha=alpha, color='g'))
        ax[i].add_patch(patches.Rectangle((0, 0.5), 0.5, 0.5, alpha=alpha, color='r'))
        ax[i].add_patch(patches.Rectangle((0.5, 0), 0.5, 0.5, alpha=alpha, color='r'))
        ax[i].add_patch(patches.Rectangle((0.5, 0.5), 0.5, 0.5, alpha=alpha, color='g'))
        ax[i].set_xlim([-0.05, 1.05])
        ax[i].set_ylim([-0.05, 1.05])

    plt.suptitle(f'Synthesis probability comparision - Data. ')
    if save_plot:
        run.save_plot(filename=title)


def cod_batteries_data():
    bat_data = load_var(f'{data_path}cod/battery/battery.pkl')
    cod_data = run_query('select file, formula, sgHall from cod_data', make_table=True)
    bat_cod_ids = []
    print('Looking for battery materials in COD', flush=True)
    for i in range(len(bat_data)):
        # print(f'({i})/({len(bat_data)})')
        formula = bat_data['pretty_formula'][i]
        if formula is None:
            continue
        formula = '- ' + ' '.join(sorted(re.findall(r'[A-Z][a-z]?[0-9]*', formula))) + ' -'
        sg_hall = bat_data['batteries'][i]['spacegroup']['hall']
        cond_1 = cod_data['formula'] == formula
        cond_2 = cod_data['sgHall'] == sg_hall
        cond = cond_1 & cond_2
        # print(f'{np.count_nonzero(cond)} crystals found in COD for {formula} ,'
        #       f'cond1: {np.count_nonzero(cond_1)}, cond2: {np.count_nonzero(cond_2)}')
        if np.count_nonzero(cond) > 0:
            for j in cod_data['file'][cond]:
                bat_cod_ids.append(j)
    print(f'{len(bat_cod_ids)} battery materials found on COD database.', flush=True)
    save_var(bat_cod_ids, data_path + 'cod/battery/cod_ids.pkl')
    write_text(data_path + 'cod/battery/cod_ids.txt', '\n'.join([str(i) for i in bat_cod_ids]))
    return bat_cod_ids


def cod_batteries_predictions(data_set='cod', predictions=False, save_plot=True, all_data=True):
    files = None
    title = None

    if data_set == 'bat-cod':
        print('Examining the COD battery files.')
        title = 'Observed (+) battery materials'
        bat_cod_ids = load_var(data_path + 'cod/battery/cod_ids.pkl')
        cod_files = []
        for j in bat_cod_ids:
            cod_files.append(data_path + f'cod/cif/{str(j)[0]}/{str(j)[1:3]}/{str(j)[3:5]}/{j}.cif')
        files = cod_files

    if data_set == 'cod':
        red_print('Test run, for checking other COD data points')
        title = '100 random observed (+) materials'
        files = list_all_files(data_path + 'cod/cif/', pattern='**/*.cif', shuffle=True,
                               random_seed=1)
        random.Random(2).shuffle(files)
        files = files[:100]

    if data_set == 'cspd':
        red_print('Test run, for checking other CSPD data points')
        title = '100 random anomaly (-) structures'
        cspd_files = list_all_files(data_path + 'cod/anomaly_cspd/cspd_cif_top_108/', pattern='**/*.cif', shuffle=True,
                                    random_seed=1)
        random.Random(2).shuffle(cspd_files)
        files = cspd_files[:100]

    if all_data:
        all_data = '_all_data'
    else:
        all_data = ''

    if predictions:
        pred = predict_crystal_synthesis(files, auto_encoder=f'results/CAE/run_043/',
                                         # classifiers=f'results/Classification/run_043{all_data}/',
                                         # classifiers=f'results/Classification/run_043/',
                                         # classifiers=f'results/run_265/',
                                         verbose=True, pad_len=70, n_bins=128, bulk_calculations=False,
                                         save_tmp_encoded_atoms=True, redo_calc=False, skip_svm=True)
        save_var(pred, data_path + f'cod/battery/preds_{data_set}{all_data}.pkl')
        save_df(pred, data_path + f'cod/battery/preds_{data_set}{all_data}.txt')

    pred = load_var(data_path + f'cod/battery/preds_{data_set}{all_data}.pkl')
    predY_prob = pred
    y = [1] * len(pred) if not ('cspd' in data_set) else [-1] * len(pred)

    classifiers = clf_labels
    classifiers.pop('BaggingClassifier', None)
    classifiers.pop('SVC', None)

    for c in classifiers:
        print(c)
        ind = predY_prob[c] > -1
        acc = 100 * accuracy_score(np.array(y)[ind], np.sign(np.sign(predY_prob[c][ind] - 0.5) + .5))
        predY_prob[c + '_acc'] = pd.Series([acc] * len(predY_prob))
        print('accuracy = {:.2f}%'.format(acc))

    plot_preds(predY_prob, title=title, save_plot=save_plot, data_set=data_set)
    # plot_clf_compare_points(predY_prob, apply_label_points=True, title=title, save_plot=save_plot, data_set=data_set)

    globals().update(locals())
    print('End df')


def capacity_voltage(save_plot=True):
    pred = load_var(data_path + f'cod/battery/battery-predictions.pkl')
    bat = load_var(data_path + f'cod/battery/battery.pkl')

    pred['material_id'] = pd.Series([i.split('/')[-1].split('.cif')[0] for i in pred['filename']])
    df = pred.merge(bat, on='material_id')
    df = df.sample(frac=1)
    df.reset_index(inplace=True)
    df['Working ion'] = df['working_ion']

    bat_dic = {
        'capacity_vol': 'Tot. Vol. Cap. ($AhI^{-1}$)',
        'average_voltage': 'Avg. Voltage ($V$)',
    }

    # IMPORTANT ****** The next two lines are required to initialize all the font sizes
    # I embedded them into plot format function
    # f, ax, fs = plot_format((85, 85))
    # plt.show()
    f, axs, fs = Plots.plot_format(88, equal_axis=True, ncols=2, nrows=3)

    gs = axs[2, 0].get_gridspec()
    # remove the underlying axes
    for ax in axs[2, :]:
        ax.remove()
    axbig = f.add_subplot(gs[2:, :])

    plt_n = -1
    clfs = ['MLPClassifier', 'RandomForestClassifier']
    for p in bat_dic:
        df[p] = pd.Series([i[p] for i in df['batteries']])
        for c in clfs:
            plt_n += 1
            sub_plt_pos = np.unravel_index(plt_n, axs.shape)
            ax = axs[sub_plt_pos]
            # sns.scatterplot(data=df, x=c,
            #                 y=p,
            #                 hue='working_ion',
            #                 ax=ax,
            #                 # palette='Blues'
            #                 facecolors='none',
            #                 alpha=.8, s=10,
            #                 )
            # sns.stripplot(x=c, y=p, hue="working_ion", data=df,
            #               jitter=True, edgecolor=sns.color_palette("hls", np.unique(df['working_ion']).size),
            #               facecolors="none", split=False, alpha=0.8)

            # c = clfs[1]
            # df_test = df
            # df_test = df_test[df_test[p] > 1500]
            # df_test = df_test[np.logical_not(df_test['Working ion'] == 'Li')]
            # sns.jointplot(data=df_test, x=c, y=p, hue="Working ion", height=3.46457, xlim=(0, 1), ylim=(1000, 6000))

            # pal = sns.color_palette("hls", np.unique(df['working_ion']).size)
            pal = sns.color_palette("muted", np.unique(df['working_ion']).size)
            wi_order = np.unique(df['working_ion'])
            for wi in wi_order:
                ax.scatter(df[df['working_ion'] == wi][c],
                           df[df['working_ion'] == wi][p], s=25, facecolors='none', edgecolors=pal.pop(0),
                           linewidth=1., label=wi)

            # for i in range(len(df)):
            #     ax.scatter(df[c][i],
            #                df[p][i], s=25, facecolors='none',
            #                edgecolors=pal[np.where(wi_order == df['working_ion'][i])[0][0]],
            #                linewidth=1.)

            # sns.jointplot(data=df, x=c, y=p, hue="Working ion", height=2)

            # L = plt.legend()
            # L.get_texts()[0].set_text('Working Ion')

            # if sub_plt_pos[0] + 1 == 2:
            #     ax.set_xlabel(f'{clf_abr[c]}')
            #     # ax.legend(loc='upper center')
            # else:
            #     Plots.remove_ticks(ax, keep_ticks=True, axis='x')
            #
            # if plt_n == 3:
            #     ax.legend(ncol=4)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            # ax.set_xlabel(f'{clf_abr[c]}')
            if sub_plt_pos[1] == 0:
                ax.set_ylabel(bat_dic[p])
            else:
                Plots.remove_ticks(ax, axis='y', keep_ticks=True)

            # if sub_plt_pos[0] == 0:
            #     ax.set_title(f'Synthesizability likelihood')
            if sub_plt_pos[0] == 1:
                # ax.set_xlabel(f'{clf_abr[c]}')
                ax.set_xlabel(f'{clf_labels[c]}')
            else:
                Plots.remove_ticks(ax, axis='x', keep_ticks=True)
            ax.grid()
            # plt.xlabel(f'Synthesis Probability\n(by {clfs[c]})', fontsize=fs)
            # plt.xlabel(f'{clf_abr[c]}')
            # plt.xlabel(f'{clfs[c]}')
            # plt.ylabel(bat_dic[p], fontsize=fs)
            # plt.grid(True)

    Plots.annotate_subplots_with_abc(axs[:2, 0])
    h, l = ax.get_legend_handles_labels()
    axbig.legend(h, l, loc='best', ncol=4)
    axbig.axis('off')

    if save_plot:
        run.save_plot()
    plt.show()

    p = 'working_ion'
    df[p] = pd.Series([i[p] for i in df['batteries']])
    df['Working ion'] = df[p]
    p = 'Working ion'
    clfs = {'MLPClassifier': 'Neural Network', 'RandomForestClassifier': 'Random Forest'}
    f, axs, fs = Plots.plot_format(88, nrows=2)
    for plt_n, c in enumerate(clfs):
        ax = axs[plt_n]
        pal = sns.color_palette("muted", np.unique(df['working_ion']).size)
        sns.scatterplot(data=df, x=c, y=p, hue=p, ax=ax, palette=pal)
        # plt.title(p + f'\n{clfs[c]}')

        # L = ax.legend()
        # L.get_texts()[0].set_text('Working Ion')
        if plt_n == 0:
            Plots.remove_ticks(ax, axis='x', keep_ticks=True)
            ax.set_xlabel('')

        ax.set_ylabel('Working ion')
        # ax.xlabel(f'{clf_abr[c]} predictions')
        Plots.ylabel_right_side(ax, f'{clf_labels[c]}')
        ax.grid(True)
        ax.get_legend().remove()
    ax.set_xlabel('Synthesizability likelihood')

    Plots.annotate_subplots_with_abc(axs)
    # h, l = ax.get_legend_handles_labels()
    # ax = axs[0]
    # ax.legend(h, l, loc='best', ncol=4)
    # ax.axis('off')

    # axs[0].get_legend().remove()
    # axs[1].get_legend().remove()

    if save_plot:
        run.save_plot()
    plt.show()

    globals().update(locals())


def plot_battery_predictions(save_plot=True, skip_svm=True):
    pred = load_var(data_path + f'cod/battery/battery-predictions.pkl')
    clf_thresholds = load_var(data_path + 'cod/results/Classification/run_043_pca_400/' + 'classifiers.pkl')
    classifiers = clf_labels
    # [classifiers.pop(c) for c in classifiers if not c in pred.columns]
    if skip_svm:
        classifiers.pop('BaggingClassifier', None)
        classifiers.pop('SVC', None)
    f, ax, fs = Plots.plot_format(nrows=len(classifiers), ncols=1)

    all_data = '_all_data'

    for data_set in [
        'bat-cod',
        'mp'
    ]:
        i = 0
        title = None

        if data_set == 'bat-cod':
            pred = load_var(data_path + f'cod/battery/preds_{data_set}{all_data}.pkl')
            data_set = 'cod'
            pred = pred[~np.isnan(pred['MLPClassifier'])]

        if data_set == 'mp':
            pred = load_var(data_path + f'cod/battery/battery-predictions.pkl')

        for c in classifiers:

            best_threshold = clf_thresholds[f'{c}_best_threshold']
            y = [1] * len(pred) if not ('cspd' in data_set) else [-1] * len(pred)
            y_pred = np.sign(np.sign(pred[c] - best_threshold) + .5)
            acc = 100 * accuracy_score(y, y_pred)
            pred[c + '_acc'] = pd.Series([acc] * len(pred))
            print(f'Data set {data_set} size = {len(y_pred)}')

            if data_set == 'cod':
                # title = f'Observed' + '(Acc.={:.2f}%)'.format(pred[c + '_acc'][0])
                title = f'COD' + '(Sensitivity={:.1f}%)'.format(pred[c + '_acc'][0])
            if data_set == 'mp':
                title = 'MP' + '(Synthesizability={:.1f}%)'.format(pred[c + '_acc'][0])
                # title = 'Synthesizability = {:.2f}%'.format(pred[c + '_acc'][0])

            sns.distplot(pred[c], kde=None, label=title, ax=ax[i], bins=15,
                         color=Plots.colors_databases[data_set], hist_kws=dict(edgecolor="k", linewidth=1),
                         norm_hist=True,)

            # plt.legend()
            # ax[i].set_title(clf_labels[c], loc='right', fontsize=fs + 1)

            # if i == 0:
            #     ax[i].text(0.55, ax[i].get_ylim()[1] * .15, 'Synthesis threshold', color='r', rotation=90)
            #     ax[i].title.set_text('Battery materials evaluations')

            # if data_set == 'cspd':
            #     Plots.add_hash_pattern(ax[i])

            ylim = ax[i].get_ylim()
            # ax[i].plot([.5, .5], [0., ylim[1]], '--', color='r', linewidth=1)
            ax[i].plot([best_threshold, best_threshold], [0., ylim[1]], '--', color='r', linewidth=2)
            ax[i].set_ylim(ylim)

            Plots.ylabel_right_side(ax[i], clf_labels[c])
            ax[i].set_ylabel('PDF', fontsize=fs)
            # ax[i].set_ylabel('Electrodes', fontsize=fs)
            ax[i].set_xlabel('')
            ax[i].set_xlim([-0.0, 1.0])
            ax[i].grid(True)
            ax[i].legend(loc='best')
            ax[i].get_yaxis().set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
            if not i == len(classifiers) - 1:
                Plots.remove_ticks(ax[i], keep_ticks=True)
            else:
                ax[i].set_xlabel('Synthesizability likelihood', fontsize=fs)

            # Plots.put_costume_text_instead_of_legends(ax=ax[i],
            # labels='Accuracy = {:.2f}%'.format(pred[c + '_acc'][0]))
            tmp = pred[c + '_acc'][0]
            print(f'Classifier: {c}, Predicted positive percentage:'
                  f' {tmp:.3f}%')
            i += 1
    Plots.annotate_subplots_with_abc(ax)
    if save_plot:
        run.save_plot(filename='battery_materials_preds')
    plt.show()

    globals().update(locals())


def battery_table():
    pred_mp = load_var(data_path + f'cod/battery/battery-predictions.pkl')
    mp = load_var(data_path + f'cod/battery/battery.pkl')

    pred_cod = load_var(data_path + f'cod/battery/cod_preds.pkl')

    print('fixing mp')
    from thermoelectric import chem2latex

    new = {'id': [], 'sg': [], 'formula': []}
    for i in range(len(pred_mp)):
        mat_id = pred_mp['filename'][i].split('/')[-1].split('.')[0]
        new['sg'].append(list(mp[mp['material_id'] == mat_id]['batteries'])[0]['spacegroup']['hall'])
        new['id'].append(mat_id)
        formula = list(mp[mp['material_id'] == mat_id]['pretty_formula'])[0]
        new['formula'].append(chem2latex(formula))

    pred_mp['sg'] = pd.Series(new['sg'])
    pred_mp['id'] = pd.Series(new['id'])
    pred_mp['formula'] = pd.Series(new['formula'])

    print('fixing cod')

    cod_data = run_query('select file, formula, sgHall from cod_data', make_table=True)
    new = {'id': [], 'sg': [], 'formula': []}
    for i in range(len(pred_cod)):
        mat_id = pred_cod['filename'][i].split('/')[-1].split('.')[0]
        new['sg'].append(list(cod_data[cod_data['file'] == int(mat_id)]['sgHall'])[0])
        new['id'].append(list(cod_data[cod_data['file'] == int(mat_id)]['file'])[0])
        formula = list(cod_data[cod_data['file'] == int(mat_id)]['formula'])[0]
        # new['formula'].append(''.join(sorted(formula.split(' ')[1:-1])))
        new['formula'].append(chem2latex(''.join(sorted(formula.split(' ')[1:-1]))))
    pred_cod['sg'] = pd.Series(new['sg'])
    pred_cod['id'] = pd.Series(new['id'])
    pred_cod['formula'] = pd.Series(new['formula'])

    pred_cod.sort_values('formula', inplace=True)

    pred_cod.to_csv(data_path + f'cod/battery/cod-preds-mod.csv')
    pred_mp.to_csv(data_path + f'cod/battery/mp-preds-mod.csv')

    ind = np.array(random.Random(4).sample(range(len(pred_mp)), 10))
    selected_mp = pred_mp.iloc[ind, :]
    selected_mp.to_csv(data_path + f'cod/battery/selected_pred_paper.csv')

    globals().update(locals())


def bat_lattice_param_dist_and_atomic_heatmap():
    pred_mp = load_var(data_path + f'cod/battery/battery-predictions.pkl')
    pred_cod = load_var(data_path + f'cod/battery/cod_preds.pkl')

    all_cod_symbols = distribution(
        data_sets=['all/cif_chunks/'],
        prop='symbols',
        previous_run=True,
        return_data=True,
    )
    all_cod_min_dist = distribution(
        data_sets=['all/cif_chunks/'],
        property_fn=atom2prop,
        prop='min_nearest_neighbor',
        previous_run=True,
        bins=[40, 10],
        x_lim_lo=0,
        x_lim_hi=200,
        return_data=True,
    )
    lattice_parameters = load_var('tmp/lattice_parameters.pkl')
    del all_cod_symbols['all/cif_chunks/']['z_score'], all_cod_symbols['all/cif_chunks/']['iqr'], \
        all_cod_symbols['all/cif_chunks/']['outlier'], all_cod_symbols['all/cif_chunks/']['chunk']
    del all_cod_min_dist['all/cif_chunks/']['z_score'], all_cod_min_dist['all/cif_chunks/']['iqr'], \
        all_cod_min_dist['all/cif_chunks/']['outlier'], all_cod_min_dist['all/cif_chunks/']['chunk'], \
        all_cod_min_dist['all/cif_chunks/']['filename']
    all_cod_symbols = pd.DataFrame(all_cod_symbols['all/cif_chunks/'])
    all_cod_min_dist = pd.DataFrame(all_cod_min_dist['all/cif_chunks/'])
    all_cod = all_cod_symbols.merge(all_cod_min_dist, how='inner', on='id')
    non_outliers_ind = all_cod['min_nearest_neighbor'] > 0.5

    if not len(lattice_parameters) == len(all_cod):
        raise ValueError

    print('preparing battery data')

    if not exists(f'{data_path}cod/battery/lattice-param.pkl'):
        data = {'filename': [], 'database': [], 'a': [], 'b': [], 'c': [], 'formula': []}
        for i, d in enumerate([pred_mp, pred_cod]):
            db = 'mp'
            if i == 1:
                db = 'cod'
            for j in range(len(d)):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    a = cif_parser(d['filename'][j])
                data['filename'].append(d['filename'][j])
                data['database'].append(db)
                cell = np.sum(a.get_cell() ** 2, axis=1) ** .5
                data['a'].append(cell[0])
                data['b'].append(cell[1])
                data['c'].append(cell[2])
                data['formula'].append(a.symbols)
        data = pd.DataFrame(data)
        save_var(data, f'{data_path}cod/battery/lattice-param.pkl')
        data.to_csv(f'{data_path}cod/battery/lattice-param.csv')
    data = load_var(f'{data_path}cod/battery/lattice-param.pkl')
    from plots import plot_lattice_constants

    f, axs, fs = Plots.plot_format(88, nrows=3)
    n_samples = []
    xlim_max = 30
    ax = plot_lattice_constants(save_fig=False, show_plot=False, ax=axs[2], xlim_max=xlim_max,
                                lattice_parameters=data[data['database'] == 'mp'])

    # ax.set_title('Materials Project Electrodes')

    # plt.show()

    ax = plot_lattice_constants(save_fig=False, show_plot=False, ax=axs[1], xlim_max=xlim_max,
                                lattice_parameters=data[data['database'] == 'cod'])

    # ax.set_title('COD Electrodes')
    # plt.show()

    ax = plot_lattice_constants(save_fig=False, show_plot=False, ax=axs[0], xlim_max=xlim_max,
                                lattice_parameters=lattice_parameters[non_outliers_ind])

    n_samples.append(len(lattice_parameters[non_outliers_ind]))
    n_samples.append(len(data[data['database'] == 'cod']))
    n_samples.append(len(data[data['database'] == 'mp']))

    # ax.set_title('All COD structures')
    axs[0].set_xlabel("")
    axs[1].set_xlabel("")
    Plots.remove_ticks(axs[0], keep_ticks=True)
    Plots.remove_ticks(axs[1], keep_ticks=True)
    titles = [
        'All COD structures',
        'COD electrodes',
        'MP electrodes',
    ]
    for n, ax in enumerate(axs):
        ax.set_xlim([0, xlim_max])
        ax.plot([], [], ' ', label=f"Crystals: {n_samples[n]:,}")
        import string
        abc = string.ascii_lowercase[n]
        # abc = r'\textbf{}'.format(abc)
        ax.set_ylabel(f'$\\bf{abc})$ {titles[n]}')
        ax.legend()
        if n == 0:
            Plots.put_costume_text_instead_of_legends(ax, f"Crystals: {n_samples[n]:,}")
    run.save_plot('bat_lattice_dist_comp')
    plt.show()

    plot_dist = False
    if plot_dist:
        from plots import plot_atomic_number_dist
        ax = plot_atomic_number_dist(save_fig=False, show_plot=False, data=data[data['database'] == 'mp'])
        ax.set_title('Material Projects Electrodes')
        # plt.title('Material Projects Electrodes')
        # for i, line in enumerate(ax.xaxis.get_ticklines()):
        #     line.set_markersize(line.get_markersize() * (1 + (i % 2) * 2))
        #     line.set_markeredgewidth(line.get_markeredgewidth() * (1 + (i % 2) * 2))
        #     line.set_linewidth(line.get_linewidth() * (1 + (i % 2) * 2))
        run.save_plot()
        plt.show()

        ax = plot_atomic_number_dist(save_fig=False, show_plot=False, data=data[data['database'] == 'cod'])
        ax.set_title('COD Electrodes')
        run.save_plot()
        plt.show()

        ax = plot_atomic_number_dist(save_fig=False, show_plot=False, data=all_cod[non_outliers_ind])
        ax.set_title('All used COD Crystals')
        run.save_plot()
        plt.show()

    # heat maps
    top_elements = None
    data_sets = {
        'all_COD': all_cod[non_outliers_ind],
        'electrodes_MP': data[data['database'] == 'mp'],
        'electrodes_COD': data[data['database'] == 'cod'],
    }
    data_labels = {
        'all_COD': 'COD database',
        'electrodes_MP': 'Electrodes from MP database',
        'electrodes_COD': 'Electrodes from COD database',
    }
    # for elemental_data in [all_cod[non_outliers_ind],
    # data[data['database'] == 'mp'], data[data['database'] == 'cod']]:
    for ds in data_sets:
        elemental_data = data_sets[ds]
        print(f'{data_labels[ds]} #Samples: {len(elemental_data):,}')
        par = 'formula'
        if 'symbols' in elemental_data:
            par = 'symbols'
        if par == 'formula':
            formula = [np.unique(i) for i in elemental_data[par]]
        else:
            formula = [list(i) for i in elemental_data['symbols']]
        formula = np.concatenate(formula)
        ele, rep = np.unique(formula, return_counts=True)
        ind = np.flip(np.argsort(rep))
        top_n = 10
        if top_elements is None:
            top_elements = ele[ind[:top_n]]
        # ind_1 = ind[:top_n]
        # ind_2 = ind[top_n:]
        f, ax, fs = Plots.plot_format((180, 80))
        ax2 = ax.twinx()
        for i in ['top', 'bottom']:
            ind = np.isin(ele, top_elements)
            d = {}
            if i == 'bottom':
                ind = np.logical_not(ind)
            for A, B in zip(ele[ind], rep[ind]):
                d[A] = B
            if i == 'top':
                Plots.periodic_table_heatmap(d, cbar_label="",
                                             show_plot=False, cmap="winter", cmap_range=None, blank_color="grey",
                                             value_format=None, max_row=9,
                                             cbar_label_size=9,
                                             ax=ax, fig=f, not_show_blank_value=False,
                                             )
            if i == 'bottom':
                Plots.periodic_table_heatmap(d, cbar_label="",
                                             show_plot=False, cmap="YlOrRd", cmap_range=None, blank_color="grey",
                                             value_format=None, max_row=9,
                                             cbar_label_size=9,
                                             ax=ax2, fig=f, not_show_blank_value=True,
                                             )
        ax.set_title(data_labels[ds])
        run.save_plot(f'{ds}_elemental_heat-map')
        plt.show()

    globals().update(locals())


def battery_space_group_dist(save_fig=True):
    pred_mp = load_var(data_path + f'cod/battery/battery-predictions.pkl')
    pred_cod = load_var(data_path + f'cod/battery/cod_preds.pkl')
    bat = load_var(data_path + f'cod/battery/battery.pkl')
    pred_mp['material_id'] = pd.Series([f.split('.cif')[0].split('/')[-1] for f in pred_mp['filename']])
    pred_mp = pred_mp.merge(bat, how='inner')

    print('preparing battery data')

    if not exists(f'{data_path}cod/battery/space-group.pkl'):
        data = {'filename': [], 'database': [], 'sg': [], 'formula': []}
        for i, d in enumerate([pred_mp, pred_cod]):
            db = 'mp'
            if i == 1:
                db = 'cod'
            for j in range(len(d)):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    a = cif_parser(d['filename'][j])
                data['filename'].append(d['filename'][j])
                data['database'].append(db)
                sg = None
                if db == 'cod':
                    sg = a.info['spacegroup'].no
                if db == 'mp':
                    sg = d['batteries'][j]['spacegroup']['number']
                data['spacegroup'].append(sg)
                data['formula'].append(a.symbols)
        data = pd.DataFrame(data)
        save_var(data, f'{data_path}cod/battery/space-group.pkl')
        data.to_csv(f'{data_path}cod/battery/space-group.csv')
    data = load_var(f'{data_path}cod/battery/space-group.pkl')

    all_cod_sg = distribution(
        data_sets=['all/cif_chunks/'],
        prop='spacegroup',
        previous_run=True,
        return_data=True,
    )
    all_cod_min_dist = distribution(
        data_sets=['all/cif_chunks/'],
        property_fn=atom2prop,
        prop='min_nearest_neighbor',
        previous_run=True,
        bins=[40, 10],
        x_lim_lo=0,
        x_lim_hi=200,
        return_data=True,
    )
    del all_cod_sg['all/cif_chunks/']['z_score'], all_cod_sg['all/cif_chunks/']['iqr'], \
        all_cod_sg['all/cif_chunks/']['outlier'], all_cod_sg['all/cif_chunks/']['chunk']
    del all_cod_min_dist['all/cif_chunks/']['z_score'], all_cod_min_dist['all/cif_chunks/']['iqr'], \
        all_cod_min_dist['all/cif_chunks/']['outlier'], all_cod_min_dist['all/cif_chunks/']['chunk'], \
        all_cod_min_dist['all/cif_chunks/']['filename']
    all_cod_sg = pd.DataFrame(all_cod_sg['all/cif_chunks/'])
    all_cod_min_dist = pd.DataFrame(all_cod_min_dist['all/cif_chunks/'])
    all_cod = all_cod_sg.merge(all_cod_min_dist, how='inner', on='id')
    non_outliers_ind = all_cod['min_nearest_neighbor'] > 0.5

    data_sets = {
        'all_COD': all_cod[non_outliers_ind],
        'electrodes_MP': data[data['database'] == 'mp'],
        # 'electrodes_COD': data[data['database'] == 'cod'],
    }
    data_labels = {
        'all_COD': 'COD database',
        'electrodes_MP': 'Electrodes from MP database',
        'electrodes_COD': 'Electrodes from COD database',
    }

    # sg_helper = load_var('sg.pkl')
    import gemmi
    # gemmi.find_spacegroup_by_name('I2')

    f, ax, fs = Plots.plot_format(88, nrows=2)
    i = 0
    for ds in data_sets:
        elemental_data = data_sets[ds]
        # elemental_data['sg_hm'] = pd.Series([gemmi.find_spacegroup_by_number(int(i)).hm
        # for i in elemental_data['spacegroup']])
        sg_n, sg_count = np.unique(elemental_data['spacegroup'], return_counts=True)
        ind = np.flip(np.argsort(sg_count))
        sg_n = sg_n[ind]
        sg_count = sg_count[ind]

        N = 9
        labels = [gemmi.find_spacegroup_by_number(int(i)).hm for i in sg_n[:N]] + ['Other']
        sizes = np.array(list(sg_count[:N]) + [np.sum(sg_count[N:])])
        # sizes = sizes / np.sum(sizes) * 100

        ax[i].pie(sizes, labels=labels, autopct='%1.1f%%',
                  shadow=True, startangle=90)
        # ax[i].axis('equal')
        ax[i].set_xlabel(data_labels[ds])
        # Plots.ylabel_right_side(ax[i], data_labels[ds])
        i += 1
    Plots.annotate_subplots_with_abc(ax)
    run.save_plot('battery_spacegroup_comp')
    plt.show()

    globals().update(locals())


if __name__ == '__main__':
    clf_labels = {
        'MLPClassifier': 'MLP',
        'RandomForestClassifier': 'RF',
        'BaggingClassifier': 'SVM-Bagging'
    }
    clf_abr = {
        'MLPClassifier': 'MLP',
        'RandomForestClassifier': 'RF',
        'BaggingClassifier': 'SVM-B',
    }

    run = RunSet()

    # check_direct_method()
    battery_systems()
    run.timer_lap()
    run.data = get_all_the_battery_data()
    run.timer_lap()
    direct_cif_download(test=True)
    bat2cif()
    cod_batteries_data()

    clf_pred_comparision_tests = False
    if clf_pred_comparision_tests:
        for ds in [
            'bat-cod',
            'cod',
            'cspd',
        ]:
            cod_batteries_predictions(data_set=ds, predictions=True, save_plot=True, all_data=True)
            run.timer_lap()

    predict_bat_materials_project = True
    if predict_bat_materials_project:
        predict_bat()
    # predict_bat(classifier='results/Classification/run_043/', fraction_of_data=0.05)

    plot_capacity_voltage = True
    if plot_capacity_voltage:
        capacity_voltage(save_plot=True)

    run.timer_lap()

    plt_bat_preds = True
    if plt_bat_preds:
        plot_battery_predictions(save_plot=True)

    make_table = True
    if make_table:
        battery_table()

    make_lattice_params_plot = True
    if make_lattice_params_plot:
        bat_lattice_param_dist_and_atomic_heatmap()

    make_space_group_dist = True
    if make_space_group_dist:
        battery_space_group_dist()

    run.end()
