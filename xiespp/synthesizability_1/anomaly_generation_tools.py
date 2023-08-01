from AtomicStructureGenerator.cspd import atomic_structure_generator
from xiespp.synthesizability_1.utility import run_query_on_cod, SpaceGroups
from config import cspd_file


def generate_anomaly_structures(elements, verbose=False, cod=None, multi_outputs=False):
    if verbose:
        print('\nConnecting to CSPD and searching for ', elements)
    if isinstance(elements, str):
        elements = re.findall('[A-Z][0-9a-z]*', elements)
    cspd_output = atomic_structure_generator(symbols=''.join(elements),
                                             cspd_file=cspd_file)
    cspd_tot = pd.DataFrame(columns=['atom', 'oid', 'sgn', 'symbols'])
    if len(cspd_output) > 0:
        cspd_tot = pd.DataFrame(
            [[cspd_output[i], cspd_output[i].oid, cspd_output[i].sgn,
              str(cspd_output[i].symbols)] for i in range(len(cspd_output))],
            columns=['atom', 'oid', 'sgn', 'symbols'])
    if cod is None:
        cod = cod_list(elements, verbose=verbose)

    # Calculating Hall Space Group
    # from ase.spacegroup import get_spacegroup
    # get_spacegroup(cspd_output[0])

    cond = cspd_tot['sgn'].isin(cod['sgn'])
    cspd_cod = cspd_tot[cond]
    cspd_cod.reset_index(inplace=True, drop=True)
    cspd_hyp = cspd_tot[~cond]
    cspd_hyp.reset_index(inplace=True, drop=True)
    if verbose:
        print('Displaying all the CSPD structures found in COD:')
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(cspd_cod[['oid', 'symbols', 'sgn']])
        print('\nDisplaying all hypothetical generated structures:')
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(cspd_hyp[['oid', 'symbols', 'sgn']])
        print('\n')
        print(len(cspd_cod), ' elements of CSPD found in COD and deleted')
        print(len(cspd_tot), 'CSPD hypothetical structures generated')
    if multi_outputs:
        return {'df': cspd_hyp, 'total cspd': len(cspd_tot), 'total cod': len(cod), 'total cspd cod': len(cspd_cod),
                'cod': cod}
    return cspd_hyp


def cod_list(elements, verbose=False, stoichiometry=True):
    if isinstance(elements, str):
        elements = re.findall('[A-Z][0-9a-z]*', elements)
    elements.sort()
    if verbose:
        print('\nCOD: Looking for the elements {} in COD, {} '
              'considering the stoichiometry:'.format(str(elements), 'with' if stoichiometry else 'without'))

    if stoichiometry is False:
        raise ValueError('This func can not handle search without stoichiometry right now')
    columns = ['file', 'cellformula', 'formula', 'sg', 'sgHall']
    query = f'select {", ".join(columns)} from data' \
            f' where nel ={len(elements)} and formula = \'- {" ".join(elements)} -\''
    cod = run_query_on_cod(query)

    sg = SpaceGroups()
    cod['sgn'] = pd.Series([sg.convert(i, 'HMu', 'sgn') for i in cod['sg']])
    if verbose:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(cod)
    if verbose:
        print('{} COD structures were found'.format(len(cod)))
    return cod


if __name__ == '__main__':
    SpaceGroups().convert('F d -3 m', 'HMu', 'sgn')
    cod_list(['Al2', 'O3'])
    generate_anomaly_structures(['Al2', 'O3'])
