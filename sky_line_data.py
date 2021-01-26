from utility.util_crystal import *
from utility.util_plot import *
from data_preprocess_positive import cif2chunk, cif_chunk2mat
from predict_synthesis import predict_crystal_synthesis

compounds = [None]
min_eng = [None]


def data_collection():
    global compounds, min_eng, group_1, group_2, group_3
    # Converting the sup. inf. to text file
    if not exists(sky_line_path + 'skyline_sup.txt'):
        # If the pdf doesn't exist:
        # https://advances.sciencemag.org/content/advances/suppl/2018/04/16/4.4.eaaq0148.DC1/aaq0148_SM.pdf
        if not exists(sky_line_path + 'skyline_sup.pdf'):
            import urllib.request
            print('the supporting info does not exist, downloading the info from the web')
            url = 'https://advances.sciencemag.org/content/advances/suppl/2018/04/16/4.4.eaaq0148.DC1/aaq0148_SM.pdf'
            urllib.request.urlretrieve(url, sky_line_path + 'skyline_sup.pdf')

        from tika import parser
        raw = parser.from_file(sky_line_path + 'skyline_sup.pdf')

        write_text(sky_line_path + 'skyline_sup.pdf', raw['content'])
    txt = read_text(sky_line_path + 'skyline_sup.txt')

    # Group 1: having no corresponding ICSD entry, as high-pressure structure, or as hypothetical
    # structure are listed below
    if not exists(sky_line_path + 'group_1.pkl'):
        p1 = txt.find('Al2O3 mp-684677')
        p2 = txt.find('Below we list the polymorphs that are above their respective amorphous limits and have a')
        group_1_txt = txt[p1:p2].split('\n')
        group_1_txt = [i for i in group_1_txt if 'mp-' in i]
        group_1 = [re.findall('mp-[0-9]*', i)[0] for i in group_1_txt]
        group_1 = materials_project_cif(group_1)
        save_var(group_1, sky_line_path + 'group_1.pkl')

    # Group 2:  have a
    # corresponding ICSD entry, but have no further information in database to describe why
    # structure is above the amorphous limit. Therefore, these structures are inspected
    # manually.
    if not exists(sky_line_path + 'group_2.pkl'):
        p1 = txt.find('Al2O3 mp-638765')
        p2 = txt.find('Accuracy of density functional theory in predicting the amorphous limit')
        group_2_txt = txt[p1:p2].split('\n')
        group_2_txt = [i for i in group_2_txt if 'mp-' in i]
        group_2 = [re.findall('mp-[0-9]*', i)[0] for i in group_2_txt]
        group_2 = materials_project_cif(group_2)
        save_var(group_2, sky_line_path + 'group_2.pkl')

    # Group 3: Synthesizable structure
    # Energies of amorphous configurations database
    # https://advances.sciencemag.org/highwire/filestream/203374/field_highwire_adjunct_files/0/aaq0148_DatabaseS1.zip
    amorphous = read_text(sky_line_path + 'aaq0148_Database S1.json')
    amorphous = re.findall("\"[A-Z][A-Za-z0-9]*\": \[[\-0-9., ]+\]", amorphous)
    compounds = [re.findall("\".*\"", i)[0][1:-1] for i in amorphous]
    energies = np.array([[float(e) for e in re.findall("[\-0-9.]+", d.split(':')[1])] for d in amorphous])
    min_eng = [np.min(i) for i in energies]
    ind = np.argsort(compounds)
    compounds = np.array(compounds)[ind]
    min_eng = np.array(min_eng)[ind]
    energies = energies[ind]
    # del energies
    if not exists(sky_line_path + 'group_3.pkl'):
        group_3 = materials_project_cif(compounds)
        for i in range(len(compounds)):
            indexes = (group_3['pretty_formula'] == compounds[i]) & (group_3['energy_per_atom'] > min_eng[i])
            group_3.drop(group_3.index[indexes], inplace=True)
        save_var(group_3, sky_line_path + 'group_3.pkl')
    save_var({'compounds': compounds,
              'eng': energies,
              'min_eng': min_eng,
              'groups': groups,
              'group_1': group_1,
              'group_2': group_2,
              'group_3': group_3},
             sky_line_path + 'skyline.pkl')


def plot_skyline():
    # Plotting
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    x3 = []
    y3 = []
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
        m = np.average(group_3[group_3['pretty_formula'] == compounds[i]]['energy_per_atom'] -
                       group_3[group_3['pretty_formula'] == compounds[i]]['e_above_hull'])
        min_convex_hull.append(m)
    avg_eng_above_ground = min_eng - np.array(min_convex_hull)
    # plt.scatter(x1, y1)
    sns.set()
    sns.set_style('darkgrid')
    Plots.plot_format((95 * 2.5, 100))
    sns.barplot(list(range(len(avg_eng_above_ground))), avg_eng_above_ground, color='gray', alpha=0.4)
    sns.scatterplot(x1, y1, color='r', label=f'Group 1: {len(x1)} structures')
    sns.scatterplot(x2, y2, color='k', label=f'Group 2: {len(x2)} structures')
    sns.scatterplot(x3, y3, color='g', label=f'Group 3: {len(x3)} structures')
    plt.ylabel('Energy above ground state (eV/atom)')
    plt.ylim(top=1, bottom=0)
    plt.grid()
    plt.xticks(range(len(compounds)), compounds)
    plt.xticks(rotation=90)
    plt.savefig('plots/skyline.png', dpi=800)
    plt.savefig(f'{sky_line_path}skyline.png', dpi=800)
    plt.show()


# witting the cif files
def skyline_save_cif_chunks():
    global compounds, min_eng, group_1, group_2, group_3
    groups = ['group_1', 'group_2', 'group_3']
    for g in groups:
        df = globals()[g]
        print(f'Saving cifs, {g}: {len(df)} cif files')
        for index, row in df.iterrows():
            write_text(f'{sky_line_path}cif/{g}/{row.material_id}.cif', row['cif'], makedir=True)
        cif2chunk(total_sections=max(1, len(df) // 50 + 1), data_set=f'cod/data_sets/skyline/cif/{g}/',
                  output_path=f'cod/data_sets/skyline/cif_chunks/{g}/', shuffle=True)
    print('Saving CIF files finished.')


def skyline_2_mat():
    groups = ['group_1', 'group_2', 'group_3']
    for i in [4, 2, 1]:
        for g in groups:
            cif_chunk2mat(target_data=f'cod/data_sets/skyline/cif_chunks/{g}/',
                          output_dataset=f'cod/data_sets/skyline/mat_set_{32 * i}/{g}/',
                          n_bins=32 * i, pad_len=17.5 * i, parser='ase', n_cpu=1,
                          check_for_outliers=False,
                          )


def sky_line_evaluation(cae_path='results/CAE/run_043/',
                        clf_path='results/Classification/run_045_all_data/classifiers.pkl'):
    run = RunSet(ini_from_path=cae_path, new_result_path=True)

    all_gr_prob = {}
    for gr in groups:
        print(f'Analyzing {gr}')
        data_set = f'cod/data_sets/skyline/cif/{gr}/'
        all_files = list_all_files(data_path + data_set, pattern='**/*.cif', shuffle=False)
        print(f'{len(all_files)} samples.')
        prob = predict_crystal_synthesis(all_files,)
        save_var(prob, run.results_path + f'prob_{gr}.pkl')
        save_df(prob, run.results_path + f'prob_{gr}.txt')
        all_gr_prob.update({gr: prob})
    save_var(all_gr_prob, run.results_path + 'all_gr_prob.pkl')
    save_var(all_gr_prob, f'{data_path}data_sets/skyline' + 'all_gr_prob.pkl')


def plot_prob_skyline():
    all_gr_prob = load_var('results/Skyline/run_001/all_gr_prob.pkl')

    # Plotting
    print('Plotting')
    sns.set_style('darkgrid')
    classifiers = ['RandomForestClassifier', 'MLPClassifier']
    for clf in classifiers:
        f, ax, font_size = Plots.plot_format((90 * 2, 100))
        for gr in groups:
            if gr == 'group_3':
                plt.show()
                f, ax, font_size = Plots.plot_format((90 * 2, 100))
            sns.set_style('darkgrid')
            label = f'Anomaly {gr}'
            if gr == 'group_3':
                label = f'Hypothetical + Anomalies + Experimental Structures'
            sns.distplot(all_gr_prob[gr][clf], bins=10, kde=None, label=label)
            plt.title(clf)
            plt.legend(loc='best')
            plt.xlabel('Probability of Synthesizability')
            plt.ylabel('Number of Samples')
        plt.show()

    print('End fn.')


if __name__ == '__main__':
    sky_line_path = data_path + 'cod/data_sets/skyline/'
    groups = ['group_1', 'group_2', 'group_3']
    group_1 = load_var(sky_line_path + 'group_1.pkl')
    group_2 = load_var(sky_line_path + 'group_2.pkl')
    group_3 = load_var(sky_line_path + 'group_3.pkl')
    data_collection()

    plot_skyline()
    skyline_save_cif_chunks()
    skyline_2_mat()
    sky_line_evaluation()
    plot_prob_skyline()

    print("The End")
