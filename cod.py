import sqlite3
import re
import pandas
from os.path import expanduser
from util import *

sg = pandas.read_pickle(expanduser('~') + '/Documents/GitHub/synthesizability-predictor/ver1/sg.pkl')


def regexp(expr, item):
    reg = re.compile(expr)
    return reg.search(item) is not None


def sg_conversion(Hall, HMu):
    def search(col, str1):
        for i in range(len(sg)):
            if sg[col][i].strip() == str1.strip():
                return int(sg['ITCn'][i])
        return 0

    p = Hall.find('(')
    if p > 0:
        Hall = Hall[:p - 1]
    sgn_hmu = search('HMu', HMu)
    sgn_hall = search('Hall', Hall)
    # returning -1 means there is an error in the sg of the entry
    if max(sgn_hmu, sgn_hall) == 0:
        return -1
    elif min(sgn_hmu, sgn_hall) == 0:
        return max(sgn_hmu, sgn_hall)
    elif sgn_hall == sgn_hmu:
        return sgn_hmu
    else:
        return -1


def cod_list(elements, verbose=False, stoichiometry=True):
    elements.sort()
    if verbose:
        print('\nCOD: Looking for the lementis {} in COD, {} considering the stoichiometry:'.format(str(elements),
                                                                                                    'with' if stoichiometry else 'without'))
    # stich = elements.copy()
    # str2_regexp = ''
    # for i in range(len(elements)):
    #     str2_regexp += ' ({})+{}'.format(stich[i], '' if stoichiometry else '([.0-9])*')
    # columns = ['file', 'cellformula', 'formula', 'sg', 'sgHall']
    # query = 'select {} from cod_data where nel ={} and formula regexp \'-{} -\''.format(
    #     ''.join(x + ', ' for x in columns)[:-2], len(elements), str2_regexp)
    if stoichiometry is False:
        raise ValueError('This func can not handle search without stoichiometry right now')
    columns = ['file', 'cellformula', 'formula', 'sg', 'sgHall']
    query = f'select {", ".join(columns)} from cod_data' \
            f' where nel ={len(elements)} and formula = \'- {" ".join(elements)} -\''
    rows = runquery(query)
    cod = pandas.DataFrame(list(rows), columns=columns)
    cod.insert(cod.columns.get_loc('sgHall'), 'sgn', int(0))
    pandas.options.mode.chained_assignment = None  # default='warn'
    for index, row in cod.iterrows():
        cod['sgn'][index] = sg_conversion(row['sgHall'], row['sg'])
    if verbose:
        with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
            print(cod)
    if verbose:
        print('{} COD structures were found'.format(len(cod)))
    return cod

# sg_conversion('-P 1 (-x,-1/2*y+1/2*z,1/2*y+1/2*z)', 'A -1')
# print(sg_conversion(Hall=' -I 2b 2c 3', HMu=''))
# cod_list(['Ca','Mg','Si','O'], verbose=True, stoichiometry=False)
# cod_list(['Ti', 'O'], verbose=True, stoichiometry=True)
# cod_list(['C','O','Sr'], verbose=True, stoichiometry=False)
# cod_list(['Cu','Fe','O','P','Xe'], verbose=True, stoichiometry=False)
