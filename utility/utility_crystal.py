from utility.utility_general import *


class PeriodicTable:
    def __init__(self):
        import bokeh.sampledata.periodic_table as pt
        import re

        self.table = pt.elements.copy()
        self.table['atomic mass'] = [float(re.findall('[0-9.]+', i)[0]) for i in self.table['atomic mass']]


class SpaceGroups:
    def __init__(self):
        import pandas as pd
        from ase.spacegroup import Spacegroup as aseSg
        i = 0
        sg = []
        while True:
            i += 1
            try:
                sg.append(aseSg(i))
            except:
                break
        sg = pd.DataFrame({'sgn': [i.no for i in sg],
                           'HM symbol': [i.symbol for i in sg]})
        self.table_ase = sg
        self.table_cod = pd.DataFrame()
        path = f'{local_data_path}/data_bases/cod/mysql/spacegroups.txt'
        if not exists(path):
            path = f'{data_path}cod/mysql/spacegroups.txt'
        if exists(path):
            fp = open(path, 'r')
            data = []
            for i in fp:
                st = i.split('\t')
                # st = re.split('(.)+\t',i)
                data.append(st)
            header = ['id', 'sgn', 'Hall', 'Schoenflies', 'HM', 'HMu', 'class', 'Nau']
            data = np.asarray(data)
            sg = pd.DataFrame(data[0:, 1:], index=data[0:, 0], columns=header[1:])
            sg['sgn'] = np.array(sg['sgn'], dtype=int)
            sg['Hall'] = sg['Hall'].str.strip()
            # sg.to_csv('sg.csv')
            fp.close()
            self.table_cod = sg
        else:
            raise FileNotFoundError(path)

    def convert(self, spacegroup_input, initial_sg, final_sg, pick_one=False):
        df = self.table_cod
        spacegroup = df[df[initial_sg] == spacegroup_input][final_sg].tolist()

        if len(spacegroup) == 0 & (initial_sg == 'HMu'):
            spacegroup = df[df['HM'] == spacegroup_input][final_sg].tolist()

        if len(spacegroup) == 0:
            # ValueError('Could not convert the spacegroup')
            return None
        if len(spacegroup) > 1:
            if len(np.unique(spacegroup)) == 1:
                spacegroup = [spacegroup[0]]
        if (len(spacegroup) == 1) or pick_one:
            spacegroup = spacegroup[0]

        return spacegroup

    def save_as_csv(self, path=''):
        if path == '':
            path = 'spacegroups.csv'
        self.table_cod.to_csv(path)


def run_query_on_cod(query, db_path=cod_sql_file):
    '''
    Examples:


    :param query:
    :param db_path:
    :return:
    '''
    # table_name = run_query(, db_path)['name'].tolist()[0]
    # input_name = re.findall('from [A-Za-z_]*', "SELECT name FROM "
    #                                            "sqlite_master WHERE type='table';".lower())[0].split(' ')[1]
    # query.replace
    return run_query(query, db_path)


def chemical_formula_2_elements_coef(formula, sort=False):
    formula = chemical_formula_2_list(formula, sort=sort)
    ele = [re.findall('[A-Z][a-z]?', i)[0] for i in formula]
    num = [int(re.findall('[0-9]+', '1' + i)[-1]) for i in formula]
    return ele, num


def chemical_formula_2_list(formula, sort=False):
    if len(re.findall('\.', formula)) > 0:
        raise Exception('This can not handle compositions with fractional elements')
    formula = re.findall('[A-Z][0-9a-z]*', formula)
    if sort:
        formula.sort()
    return formula


def full_formula_2_simple_formula(formula, sort=False):
    import math, functools

    ele, num = chemical_formula_2_elements_coef(formula, sort=sort)
    gcd = functools.reduce(math.gcd, num)
    f1 = lambda n, g: str(n // g) if n // g > 1 else ''
    formula = ''.join([e + f1(n, gcd) for e, n in zip(ele, num)])
    return formula


if __name__ == '__main__':
    # run_query('select count(*) from data')
    full_formula_2_simple_formula('Sc2Y2O6')
    a = SpaceGroups()
    SpaceGroups().table_cod.to_csv('/home/ali/Downloads/spacegroups.csv')
    a.save_as_csv()
    a.convert(3, 'sgn', 'HM')
    a.convert('I 41/a m d :1', 'HMu', 'sgn')


if __name__ == '__main__':
    sg = SpaceGroups()
    sg.convert(2, 'sgn', 'HM')