from utility.util_crystal import *
from predict_synthesis import *
from time import sleep
from pymatgen import MPRester
from crystal_tools import cod_id_2_path

clf_labels = {
    # 'MLPClassifier': 'Neural network',
    'MLPClassifier': 'MLP',
    'RandomForestClassifier': 'Random forest',
    # 'BaggingClassifier': 'SVM-Bagging'
}
path_thermoelectric = f'{data_path}cod/thermoelectric/'


def extract_top_10():
    if not exists(f'{path_thermoelectric}sup.txt'):
        from tika import parser
        raw = parser.from_file(f'{path_thermoelectric}sup.txt')
        # If the pdf doesn't exist:
        # https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-019-1335-8/
        # MediaObjects/41586_2019_1335_MOESM1_ESM.pdf
    txt = read_text(f'{path_thermoelectric}sup.txt')
    p1 = txt.find('Top 10 thermoelectric predictions')
    p2 = txt.find('Table S3:')
    txt = txt[p1:p2]
    txt = re.split('20[01][0-9][^0-9]', txt)[1:]
    formulas = {}
    year = 2000
    for y in txt:
        year += 1
        # y = re.findall('(([A-Z][a-z]?[0-9. ]*)+)', y)
        y = re.findall("((\(?[A-Z][a-z]?[0-9. ]*\)?[0-9. ]*)+)", y)
        y = [i[0] for i in y if len(i[0]) > 2]

        if not len(y) == 10:
            # y += ['-'] * (12 - len(y))
            print('Error in getting the formulas')
        formulas.update({f'{year}': y})
    formulas = pd.DataFrame(formulas)
    formulas.to_csv(path_thermoelectric + 'formulas.csv')

    api_key = "MCpqDZgh29W9r6X4Ru"
    mpr: MPRester = MPRester(api_key)

    atoms_list = {'year': [], 'formula': [], 'data_set': [], 'atoms': []}
    for y in formulas.columns:
        # break
        atoms_list_mp = []
        atoms_list_cod = []
        for f_mp in formulas[y]:
            el = sorted(re.findall("[A-Z][a-z]?[0-9.]*", f_mp))
            f_mp = ''.join(el)
            f_cod = '- ' + ' '.join(el) + ' -'
            print(f'Getting the results for: {f_mp}')
            atoms = []
            atoms_cod = run_query(f"select file from cod_data where formula == \'{f_cod}\'")
            if len(atoms_cod) > 0:
                atoms_cod = [int(i[0]) for i in atoms_cod]
            for _ in range(3):
                try:
                    atoms = mpr.get_data(f_mp)
                    break
                except Exception as e:
                    red_print(e)
                    print('Trying again')
                    sleep(1)
            print(f'Found results: {len(atoms)}')
            atoms_list_mp.append(atoms)
            atoms_list_cod.append(atoms_cod)
            for a in atoms:
                atoms_list['year'].append(int(y))
                atoms_list['formula'].append(f_mp)
                atoms_list['atoms'].append(a)
                atoms_list['data_set'].append('mp')
            for a in atoms_cod:
                atoms_list['year'].append(int(y))
                atoms_list['formula'].append(f_mp)
                atoms_list['atoms'].append(a)
                atoms_list['data_set'].append('cod')
        formulas[f'{y}_atoms_mp'] = pd.Series([len(i) for i in atoms_list_mp])
        formulas[f'{y}_atoms_cod'] = pd.Series([len(i) for i in atoms_list_cod])

    atoms_list = pd.DataFrame(atoms_list)
    # save_var(atoms_list, f'{path_thermoelectric}/atoms_list_top_10.pkl')
    atoms_list = load_var(f'{path_thermoelectric}/atoms_list_top_10.pkl')
    formulas.to_csv(path_thermoelectric + 'formulas_top_10.csv')

    makedirs(path_thermoelectric + 'cif-top-10s/', exist_ok=True)
    atoms_list['filename'] = pd.Series([None] * len(atoms_list))
    for r in range(len(atoms_list)):
        if atoms_list['data_set'][r] == 'mp':
            file = path_thermoelectric + f'cif-top-10s/' + atoms_list['atoms'][r]['material_id'] + '.cif'
            write_text(file, atoms_list['atoms'][r]['cif'])
            atoms_list['filename'][r] = file

        else:
            # if atoms_list['atoms'][r] == 8103559:
            #     print('Stop')
            shutil.copyfile(cod_id_2_path(atoms_list['atoms'][r]),
                            path_thermoelectric + 'cif-top-10s/' + str(atoms_list['atoms'][r]) + '.cif')
            # a = cif_parser(cod_id_2_path(atoms_list['atoms'][r]))

            atoms_list['filename'][r] = path_thermoelectric + 'cif-top-10s/' + str(atoms_list['atoms'][r]) + '.cif'
            # write(atoms_list['filename'][r], a)
            if not exists(atoms_list['filename'][r]):
                red_print('Error copying')

    save_var(atoms_list, f'{path_thermoelectric}/atoms_list_top_10.pkl')
    globals().update(locals())


def predict_top_10():
    atoms_list = load_var(f'{path_thermoelectric}/atoms_list_top_10.pkl')
    files = atoms_list['filename']

    pred = predict_crystal_synthesis(files, classifiers=None, save_tmp_encoded_atoms=True, redo_calc=False)

    tmp = atoms_list.drop(axis='columns', columns=['atoms'])
    del tmp['filename']
    if not len(tmp) == len(pred):
        raise ValueError('Mismatch between predictions and inputs')
    pred = pd.concat([pred, tmp], axis=1, join='inner')
    pred['id'] = pd.Series([i.split('/')[-1].split('.')[0] for i in pred['filename']])
    save_var(pred, f'{path_thermoelectric}/pred_list_top_10.pkl')
    pred.to_csv(f'{path_thermoelectric}/pred_list_top_10.csv', float_format='%g')

    globals().update(locals())


def plot_pred():
    atoms_list = load_var(f'{path_thermoelectric}/pred_list_top_10.pkl')
    # preds = load_var(f'{path_thermoelectric}/pred_list_top_10.pkl')

    # cod_stat = {'Classifier': [], 'AUC': [], 'Accuracy': [], 'Sensitivity': [], 'Specificity': []}

    f, ax, fs = Plots.plot_format(nrows=len(clf_labels))
    for i in range(len(clf_labels)):
        c = list(clf_labels.keys())[i]

        cod = atoms_list[atoms_list['data_set'] == 'cod'][c]
        cod = cod[np.logical_not(np.isnan(cod))]
        mp = atoms_list[atoms_list['data_set'] == 'mp'][c]
        mp = mp[np.logical_not(np.isnan(mp))]

        from sklearn.metrics import confusion_matrix
        y = [1] * len(cod)
        best_threshold = atoms_list[f'{c}_best_threshold'][0]
        prob = cod
        y_pred = np.sign(np.sign(prob - best_threshold) + .5)
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        specificity = tn / (tn + fp)
        cod_sensitivity = tp / (tp + fn)
        fpr, tpr, threshold = roc_curve(y, prob)
        roc_auc = auc(fpr, tpr)
        cod_acc = accuracy_score(y, y_pred)

        y_pred_mp = np.sign(np.sign(mp - best_threshold) + .5)
        mp_pos_ratio = accuracy_score([1] * len(mp), y_pred_mp)

        # cod_stat['Classifier'].append(clf_labels[c])
        # cod_stat['AUC'].append(roc_auc)
        # cod_stat['Accuracy'].append(acc)
        # cod_stat['Sensitivity'].append(sensitivity)
        # cod_stat['Specificity'].append(specificity)

        # ax[i].grid(color='silver', linestyle='--', linewidth=1, alpha=0.25)
        ax_2 = ax[i].twinx()

        sns.distplot(mp,
                     color=Plots.colors_databases['mp'], hist_kws=dict(edgecolor="k", linewidth=1),
                     bins=15, kde=None, norm_hist=True,
                     label=f'MP (Synthesizability={mp_pos_ratio * 100:.1f}%)',
                     ax=ax[i])

        sns.distplot(cod,
                     color=Plots.colors_databases['cod'], hist_kws=dict(edgecolor="k", linewidth=1),
                     bins=15, kde=None, norm_hist=True,
                     label=f'COD (Sensitivity={cod_sensitivity:.3f})',
                     ax=ax[i])
        ax[i].set_xlim([-0.0, 1.0])
        ax[i].grid(True)
        # ax[i].set_ylabel('Number of materials')
        ax[i].set_ylabel('Number of crystals')
        # ax[i].set_xlabel(clf_labels[c] + ('\nProbability of synthesis' if i == len(clf_labels) - 1 else ''))
        ax_2.set_ylabel(clf_labels[c])
        ax_2.set_yticks([])

        ylim = ax[i].get_ylim()
        # ax[i].plot([.5, .5], [0., ylim[1]], '--', color='r', linewidth=1)
        ax[i].plot([atoms_list[f'{c}_best_threshold'][0]] * 2, [0., ylim[1]], '--', color='r', linewidth=2)
        ax[i].set_ylim(ylim)
        ax[i].legend(loc='upper left')

    # ax[0].text(0.005, ax[0].get_ylim()[1] * .45, 'Synthesis threshold', color='r',
    #            # rotation=90
    #            )

    Plots.remove_ticks(ax[0], keep_ticks=True)
    # Plots.remove_ticks(ax[1], keep_ticks=True)
    ax[0].set_xlabel('')
    # ax[1].set_xlabel('')
    ax[1].set_xlabel('Synthesizability likelihood')
    Plots.annotate_subplots_with_abc(ax)

    Plots.plot_save('thermo_dist', path_thermoelectric, formats=['png', 'pdf'])
    plt.show()

    globals().update(locals())


def plot_pred_fancy():
    atoms_list = load_var(f'{path_thermoelectric}/pred_list_top_10.pkl')
    # preds = load_var(f'{path_thermoelectric}/pred_list_top_10.pkl')

    f, ax, fs = Plots.plot_format((88, 77), ncols=len(clf_labels))
    leg = []
    for i in range(len(clf_labels)):
        c = list(clf_labels.keys())[i]

        cod = atoms_list[atoms_list['data_set'] == 'cod'][c]
        cod = cod[np.logical_not(np.isnan(cod))]
        mp = atoms_list[atoms_list['data_set'] == 'mp'][c]
        mp = mp[np.logical_not(np.isnan(mp))]

        from sklearn.metrics import confusion_matrix
        y = [1] * len(cod)
        best_threshold = atoms_list[f'{c}_best_threshold'][0]
        prob = cod
        y_pred = np.sign(np.sign(prob - best_threshold) + .5)
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        specificity = tn / (tn + fp)
        cod_sensitivity = tp / (tp + fn)
        fpr, tpr, threshold = roc_curve(y, prob)
        roc_auc = auc(fpr, tpr)
        cod_acc = accuracy_score(y, y_pred)

        y_pred_mp = np.sign(np.sign(mp - best_threshold) + .5)
        mp_pos_ratio = accuracy_score([1] * len(mp), y_pred_mp)

        # ax[i].grid(color='silver', linestyle='--', linewidth=1, alpha=0.25)
        ax_2 = ax[i].twinx()

        ax_act = ax[0]
        if i == 0:
            ax_act = ax[0]
            ax_act.invert_xaxis()
        if i == 1:
            ax_act = ax[1]

        sns.distplot(mp,
                     color=Plots.colors_databases['mp'], hist_kws=dict(edgecolor="k", linewidth=1),
                     bins=10, vertical=True, norm_hist=True,
                     label=f'MP (Synthesizability={mp_pos_ratio * 100:.1f}%)',
                     ax=ax_act)

        sns.distplot(cod,
                     color=Plots.colors_databases['cod'], hist_kws=dict(edgecolor="k", linewidth=1),
                     bins=10, vertical=True, norm_hist=True,
                     label=f'COD (Sensitivity={cod_sensitivity:.3f})',
                     ax=ax_act)

        import string
        abc = string.ascii_lowercase[i]

        ax_act.plot([], [], ' ', label=f'$\\bf{abc})$ {clf_labels[c]}')
        ax_act.set_ylim([-0.0, 1.0])
        ax_act.set_ylabel('')
        ax[i].grid(True)
        # ax[i].set_ylabel('Number of materials')
        # ax[i].set_ylabel('Number of crystals')
        # ax[i].set_xlabel(clf_labels[c] + ('\nProbability of synthesis' if i == len(clf_labels) - 1 else ''))
        # ax_2.set_ylabel(clf_labels[c])
        ax_2.set_yticks([])
        leg.append(ax_act.get_legend_handles_labels())

        # ylim = ax[i].get_ylim()
        # ax[i].plot([.5, .5], [0., ylim[1]], '--', color='r', linewidth=1)
        # ax[i].plot([atoms_list[f'{c}_best_threshold'][0]] * 2, [0., ylim[1]], '--', color='r', linewidth=2)
        # ax[i].set_ylim(ylim)
        # ax_act.legend(loc='best')

    # ax[0].text(0.005, ax[0].get_ylim()[1] * .45, 'Synthesis threshold', color='r',
    #            # rotation=90
    #            )

    Plots.remove_ticks(ax[0], keep_ticks=True, axis='y')
    # Plots.remove_ticks(ax[1], keep_ticks=True)
    ax[0].set_xlabel('')
    ax[1].set_xlabel('')
    # ax[1].set_xlabel('Synthesizability likelihood')
    # Plots.annotate_subplots_with_abc(ax)

    Plots.plot_save('thermo_dist', path_thermoelectric, formats=['png', 'pdf'])
    plt.show()

    f, ax, fs = Plots.plot_format(nrows=len(clf_labels))
    ax[0].legend(leg[0][0], leg[0][1])
    ax[1].legend(leg[1][0], leg[1][1])
    Plots.plot_save('thermo_dist_legends', path_thermoelectric, formats=['png', 'pdf'])

    plt.show()
    globals().update(locals())


def prepare_thermo_word_embed_table():
    # table_pkl = load_var(path_thermoelectric + 'formulas_top_10.pkl')
    table = pd.read_csv(path_thermoelectric + 'formulas_top_10.csv')
    years = list(table.columns)
    # all_comp = []
    for y in years:
        comp = []
        for c in table[y]:
            c = chem2latex(c)
            comp.append(c)
        for c in range(len(table[y])):
            table[y][c] = comp[c]
        # all_comp.append('\'' + ' - '.join(comp) + '\'')
    table.to_csv(path_thermoelectric + 'formulas_top_10_table.csv')

    print('E')
    # new_table = {
    #     'Year': list(table.columns),
    #     'Top Anticipations': all_comp
    # }
    # new_table = pd.DataFrame(new_table)
    # new_table.to_csv(path_thermoelectric + 'formulas_top_10_table.csv')


def chem2latex(c):
    n = find_all_numbers(c)

    if len(n) > 0:
        # nn = []
        # if len(n) > 1:
        #     for i in range(len(n)):
        #         if n[i][1] == n[i + 1][0]:
        #             continue
        #         if n[i - 1][1] == n[i][0]:
        #             nn.append((n[i - 1][0], n[i][2]))
        #             continue
        #         nn.append(n[i])
        #     n = nn
        n.reverse()
        for i in n:
            # if c[i[0]-1] == '.':
            #     continue
            c = c[:i[0]] + '_' + c[i[1] - 1:]
    c = '$' + c + '$'
    c.replace(' ', '')
    return c


def top_preds_table():
    atoms_list_2 = load_var(f'{path_thermoelectric}/atoms_list_top_10.pkl')
    atoms_list = load_var(f'{path_thermoelectric}/pred_list_top_10.pkl')

    precision = 8

    atoms_list['atoms'] = atoms_list_2['atoms']

    summation = np.array([0] * len(atoms_list))
    for c in clf_labels:
        summation += atoms_list[c]
    atoms_list['sum'] = pd.Series(summation)
    atoms_list.sort_values(by='sum', ascending=False, inplace=True)
    atoms_list.reset_index(inplace=True)

    sg = []
    # N = 40
    N = len(atoms_list)
    for i in range(N):
        if isinstance(atoms_list['atoms'][i], int):
            file_id = atoms_list['atoms'][i]
            sg.append(run_query(f'select sgHall from cod_data where file == {file_id}')[0][0])
        else:
            sg.append(atoms_list['atoms'][i]['spacegroup']['hall'])

    table = {
        'Material': [chem2latex(f) for f in atoms_list['formula'][:N]],
        'Year': atoms_list['year'][:N],
        'Dataset': atoms_list['data_set'][:N],
        'Material ID': [f.split('/')[-1].split('.')[0] for f in atoms_list['filename'][:N]],
        'Space group': sg,
        'Neural network': np.round(atoms_list['MLPClassifier'][:N], precision),
        'Random forest': np.round(atoms_list['RandomForestClassifier'][:N], precision),
        # 'SVM-Bagging': np.round(atoms_list['BaggingClassifier'][:N], 2),
        'Sum': np.round(atoms_list['sum'][:N], precision),
    }
    table = pd.DataFrame(table)

    table.drop_duplicates(subset=['Material', 'Space group', 'Dataset'], inplace=True)
    table.sort_values(by='Sum', ascending=False, inplace=True)
    table.reset_index(inplace=True)

    table.to_csv(path_thermoelectric + 'top_thermo_preds.csv', index=False, float_format='%g')

    atoms_list.sort_values(by='year', ascending=False, inplace=True)
    atoms_list.reset_index(inplace=True)
    ind = ((atoms_list['year'] == 2018) | (atoms_list['year'] == 2002)) & (atoms_list['data_set'] == 'mp')
    table = {
        'Material': [chem2latex(f) for f in atoms_list['formula'][ind]],
        'Year': atoms_list['year'][ind],
        'Dataset': atoms_list['data_set'][ind],
        'Space group': [f['spacegroup']['symbol'] for f in atoms_list['atoms'][ind]],
        'Material ID': [f.split('/')[-1].split('.')[0] for f in atoms_list['filename'][ind]],
        'Neural network': np.round(atoms_list['MLPClassifier'][ind], precision),
        'Random forest': np.round(atoms_list['RandomForestClassifier'][ind], precision),
        # 'SVM-Bagging': np.round(atoms_list['BaggingClassifier'][ind], 2),
        'Sum': np.round(atoms_list['sum'][ind], precision),
        'Average prediction': np.round(atoms_list['sum'][ind] / 3, precision),
    }
    table = pd.DataFrame(table)

    table.drop_duplicates(subset=['Material', 'Space group', 'Dataset'], inplace=True)
    table.sort_values(by='Sum', ascending=False, inplace=True)
    table.reset_index(inplace=True)

    table.to_csv(path_thermoelectric + '2018vs2002_thermo_preds.csv', index=False, float_format='%g')

    globals().update(locals())


def yearly_preds_table():
    # atoms_list = load_var(f'{path_thermoelectric}/atoms_list_top_10.pkl')
    atoms_list = load_var(f'{path_thermoelectric}/pred_list_top_10.pkl')
    atoms_list_2 = load_var(f'{path_thermoelectric}/atoms_list_top_10.pkl')
    atoms_list['atoms'] = atoms_list_2['atoms']

    precision = 8

    summation = np.array([0] * len(atoms_list))
    for c in clf_labels:
        if 'Bagging' in c:
            continue
        summation += atoms_list[c]
    atoms_list['sum'] = pd.Series(summation)
    atoms_list.sort_values(by='sum', ascending=False, inplace=True)
    atoms_list.reset_index(inplace=True)

    sg = []
    N = 40
    for i in range(len(atoms_list)):
        if isinstance(atoms_list['atoms'][i], int):
            file_id = atoms_list['atoms'][i]
            sg.append(run_query(f'select sgHall from cod_data where file == {file_id}')[0][0])
        else:
            sg.append(atoms_list['atoms'][i]['spacegroup']['hall'])
    atoms_list['sg'] = pd.Series(sg)

    for y in range(2002, 2018 + 1):
        # atoms_list.sort_values(by='year', ascending=False, inplace=True)
        # atoms_list.reset_index(inplace=True)
        ind = atoms_list['year'] == y
        table = {
            'Material': [chem2latex(f) for f in atoms_list['formula'][ind]],
            'Space group': atoms_list['sg'][ind],
            'Material ID': [f.split('/')[-1].split('.')[0] for f in atoms_list['filename'][ind]],
            'Neural network': np.round(atoms_list['MLPClassifier'][ind], precision),
            'Random forest': np.round(atoms_list['RandomForestClassifier'][ind], precision),

            'Year': atoms_list['year'][ind],
            'Dataset': atoms_list['data_set'][ind],
            # 'SVM-Bagging': np.round(atoms_list['BaggingClassifier'][ind], 2),
            'Sum': np.round(atoms_list['sum'][ind], precision),
            'Average prediction': np.round(atoms_list['sum'][ind] / 2, precision),
        }
        table = pd.DataFrame(table)
        table.drop_duplicates(subset=['Material', 'Space group', 'Dataset'], inplace=True)
        table.sort_values(by='Sum', ascending=False, inplace=True)
        table.reset_index(inplace=True)
        table.to_csv(path_thermoelectric + f'yearly/{y}_thermo_preds.csv', index=False)

    globals().update(locals())


def plot_therm_atoms():
    atoms_list = load_var(f'{path_thermoelectric}/atoms_list_top_10.pkl')
    summation = np.array([0] * len(atoms_list))
    for c in clf_labels:
        summation += atoms_list[c]
    atoms_list['sum'] = pd.Series(summation)
    atoms_list.sort_values(by='sum', ascending=False, inplace=True)
    atoms_list.reset_index(inplace=True)

    filename = atoms_list['filename'][0]
    # from ase.visualize import view
    from ase.visualize.plot import plot_atoms
    for i in range(40):
        filename = atoms_list['filename'][i]
        atoms = cif_parser(filename=filename)
        f, ax, fs = Plots.plot_format(size=(35, 35))
        atoms = cif_parser(filename=filename)
        atoms = atoms.repeat((2, 2, 2))
        plot_atoms(atoms, ax, radii=0.3, rotation='0x, 0y, 0z', show_unit_cell=False)
        Plots.remove_ticks(ax=ax, axis='x')
        Plots.remove_ticks(ax=ax, axis='y')
        run.save_plot(formats=['svg'], transparent=True)
        plt.show()

    globals().update(locals())


if __name__ == '__main__':
    run = RunSet()

    extract_top_10()
    predict_top_10()

    plot_pred()
    plot_pred_fancy()
    top_preds_table()  # 2018vs2002
    yearly_preds_table()

    prepare_thermo_word_embed_table()

    plot_therm_atoms()

    run.end()
