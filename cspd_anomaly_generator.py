from cspd.cspd import *
from cod import cod_list
from mat2vec.processing import MaterialsTextProcessor
from gensim.models import Word2Vec
from utility.util_crystal import *
import seaborn as sns
import matplotlib.pyplot as plt
import pandas

# https://github.com/materialsintelligence/mat2vec


def list_all_the_formulas():
    """
    Creating a table of all the compositions available in the Literature and their number of repetitions
    Literature Dataset borrowed from:

    # https://github.com/materialsintelligence/mat2vec
    To able to run this function follow mat2vec installation guidelines from the above link

    :return: None
    """
    text_processor = MaterialsTextProcessor()
    w2v_model = Word2Vec.load("mat2vec/training/models/pretrained_embeddings")

    # w2v_model.wv.most_similar("thermoelectric")

    # for i in run_query("select distinct formula from cod_data "):
    #     formula = text_processor.normalized_formula(i[0][1:-1].replace(' ', ''))
    #     count =
    #     print(i)

    formula_table = {'formula': [], 'literature': []}
    cod_count = []
    # j = 0
    for i in w2v_model.wv.vocab.keys():
        # noinspection PyBroadException
        try:
            cond = text_processor.is_simple_formula(i)
        except:
            continue
        if cond:

            cod_filter = False
            if cod_filter:
                f = '\'- ' + ' '.join(re.findall('[A-Z][0-9a-z]*', i)) + ' -\''
                f = len(run_query('select formula from cod_data where formula == {}'.format(f)))
                if f == 0:
                    continue
                cod_count.append(f)

            # noinspection PyTypeChecker
            formula_table['formula'].append(f"{i:20s}")
            formula_table['literature'].append(w2v_model.wv.vocab[i].count)
            # j += 1
            # if j > 500:
            #     break

    if len(cod_count) > 0:
        formula_table['cod'] = cod_count

    formula_table = pd.DataFrame(formula_table).sort_values('literature', ascending=False)
    formula_table.reset_index(inplace=True, drop=True)
    formula_table.to_excel('formula.xlsx')
    formula_table.to_csv('formula.txt', header=True, index=True, sep='\t', mode='w')
    print(formula_table)


def cspd_hypotheticals(elements, verbose=False, cod=None, multi_outputs=False):
    """
    Generate all the possible hypothetical crystal structures given the input elements.
    :param elements:
    :param verbose:
    :param cod:
    :param multi_outputs:
    :return:
    """
    if verbose:
        print('\nConnecting to CSPD and searching for ', elements)
    # if len(re.findall(".", ''.join(elements))) > 0:  # empty output
    #     return pandas.DataFrame(columns=['oid', 'symbols', 'sgn', 'formation_energy_per_atom'])
    cspd_tot = atomic_structure_generator(symbols=''.join(elements), verbose=False)
    cspd_Stot = pandas.DataFrame(
        [[cspd_tot[i], cspd_tot[i].oid, cspd_tot[i].sgn, str(cspd_tot[i].symbols)] for i in range(len(cspd_tot))],
        columns=['atom', 'oid', 'sgn', 'symbols'])
    if cod is None:
        cod = cod_list(elements, verbose=verbose)
    cond = cspd_tot['sgn'].isin(cod['sgn'])
    cspd_cod = cspd_tot[cond]
    cspd_cod.reset_index(inplace=True, drop=True)
    cspd_hyp = cspd_tot[~cond]
    cspd_hyp.reset_index(inplace=True, drop=True)
    if verbose:
        print('Displaying all the CSPD structures found in COD:')
        with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
            print(cspd_cod[['oid', 'symbols', 'sgn']])
        print('\nDisplaying all hypothetical generated structures:')
        with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
            print(cspd_hyp[['oid', 'symbols', 'sgn']])
        print('\n')
        print(len(cspd_cod), ' elements of CSPD found in COD and deleted')
        print(len(cspd_tot), 'CSPD hypothetical structures generated')
    if multi_outputs:
        return cspd_hyp, cspd_tot, cspd_cod
    return cspd_hyp


def anomaly_gen_lit_based():
    """
    Creates crystal anomaly by generating hypothetical structures of well studied crystals and removing known structure
    from the COD

    To use this function download the following package into the same directory and un-tar the zip file.
    https://github.com/SUNCAT-Center/AtomicStructureGenerator
    :return:
    """
    formula_table = pandas.read_csv('mat2vec-master/formula no filter.txt', sep='\t', index_col=0)
    formula_table['lit_per'] = formula_table['literature'] / np.sum(formula_table['literature']) * 100
    stats = formula_table.describe(percentiles=[.25, .5, .8, .85, .9, .95, .99, .995, .998, .999])
    # stats.to_excel('stats.xlsx')
    print(stats)
    hyp_table = formula_table[:int((1 - .999) * len(formula_table))]
    # hyp_table = formula_table[:5]
    hyp_table['lit_per'] = hyp_table['lit_per'].round(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hyp_table['hyp'], hyp_table['tot_cspd'], hyp_table['cod'] = [None] * 3
    print(str(hyp_table[:1]).split('\n')[0])

    for i in range(10, len(hyp_table)):
        formula = formula_table['formula'][i].strip()
        elements = re.findall('[A-Z][0-9a-z]*', formula)
        cspd_hyp, cspd_tot, cspd_cod = cspd_hypotheticals(elements=elements, verbose=False, multi_outputs=True)

        hyp_table['hyp'][i] = len(cspd_hyp)
        hyp_table['tot_cspd'][i] = len(cspd_tot)
        hyp_table['cod'][i] = len(cspd_cod)
        # hyp_table['atom'][i] = list(cspd_hyp['atom'])

        for j in range(len(cspd_hyp['atom'])):
            path = data_path + f'anomaly/cspd_cif_top_{len(hyp_table)}/{formula}/'
            makedirs(path, exist_ok=True)
            write(path + f'{i:03}{j:04}.cif', cspd_hyp['atom'][j])
            write_text(path + 'atoms.txt', str(hyp_table[i:i + 1]))
            save_var(list(cspd_hyp['atom']), path + 'atoms.pkl')

        print(str(hyp_table[:i + 1]).split('\n')[-1])

    print('-' * 10)
    hyp_table['tot_hyp'] = hyp_table['hyp'].cumsum()

    path = data_path + f'cod/anomaly_cspd/cspd_cif_top_{len(hyp_table)}/'
    save_var(hyp_table, path + 'info.pkl')
    hyp_table.to_excel(path + 'info.xlsx')
    hyp_table.to_csv(path + 'info.txt', header=True, index=True, sep='\t', mode='w')

    print(hyp_table)


def plot_results():
    path = data_path + f'cod/anomaly_cspd/cspd_cif_top_{108}/'
    n_top_comp = 15
    data = load_var(path + 'info.pkl')[:n_top_comp]
    data['formula'] = data['formula'].str.strip()
    data['hyp_cod'] = data['hyp'] + data['cod']

    for i in range(len(data)):
        formula = data['formula'][i]
        formula = '$' + ''.join([f'_{j}' if j.isdigit() else j for j in formula]) + '$'
        data['formula'][i] = formula
    plt.close('all')
    f, ax = plt.subplots(1, 2, figsize=(88 / 100 * 3.93701, 100 / 100 * 3.93701))
    f.tight_layout()
    f.subplots_adjust(right=10, left=5)
    # f, ax = plt.subplots(figsize=(88 / 100 * 3.93701, 88 / 100 * 3.93701))
    # sns.set(style="whitegrid")
    sns.set_style("darkgrid")
    sns.barplot(x="lit_per", y="formula", data=data,
                label="Study intensity",
                palette="GnBu_d",
                # palette="pastel",
                ax=ax[0],
                )
    sns.barplot(x="hyp_cod", y="formula", data=data,
                label="Selected Anomalies",
                # palette="YlOrRd",
                palette="Reds",
                ax=ax[1],
                )
    sns.barplot(x="cod", y="formula", data=data,
                label="Observed structures",
                palette="BuGn_r",
                # palette="pastel",
                ax=ax[1],
                )
    ax[0].set_ylabel('')
    ax[1].set_ylabel('')
    ax[1].set_yticklabels([])
    ax[0].set_xlabel('Frequency%\n in Literature')  # fontsize=20
    ax[1].set_xlabel('Number of\n Instances')
    ax[1].xaxis.set_ticks([0, 100, 200, 300])
    ax[0].xaxis.set_ticks([0, 1, 2, 3, 4, 5])
    lines, labels = ax[0].get_legend_handles_labels()
    lines2, labels2 = ax[1].get_legend_handles_labels()
    # f.legend([lines, lines2], labels=['labels', 'labels2'])
    ax[1].legend(lines + lines2, labels + labels2, loc='lower right')
    plt.savefig(path + 'fig-2.png')
    plt.savefig(path + 'fig-2.svg')
    plt.show()
    print('End plot')


if __name__ == '__main__':
    list_all_the_formulas()
    # A = cod_l0ist(elements=['O2', 'Ti'])
    anomaly_gen_lit_based()

    plot_results()

    # path = data_path + f'cod/anomaly_cspd/cspd_cif_top_108/'
    # A = load_var(path + 'info.pkl')
    # A['formula'] = [' '.join(chem2latex(c).split()) for c in A['formula']]  # Thermoelectric
    # A.to_csv(path + 'info_latex_formula.csv')
    print('End')
