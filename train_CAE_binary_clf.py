from utility.util_crystal import *
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, \
    roc_curve, auc
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from utility.util_plot import *
from sklearn.model_selection import GridSearchCV
from joblib import Parallel, delayed
from data_preprocess import data_preparation

run_1 = None

start_time = datetime.now()


def plot_roc(clf, x, y, f1=True, tie_line=True, title_show=True, grid_on=True, prob=None, run=None):
    if run is None:
        run = run_1
    TN_FP = len(y[y < 0])  # Total Neg
    TP_FN = len(y[y > 0])  # Total Pos
    if prob is None:
        prob = clf.predict_proba(x)
        prob = prob[:, 1]
    prob_pos = prob[y > 0]
    prob_neg = prob[y < 0]
    fpr, tpr, threshold = roc_curve(y, prob)
    TP = tpr * TP_FN
    FP = fpr * TN_FP
    recall = tpr
    precision = TP / (TP + FP)
    F1 = 2 * (recall * precision) / (recall + precision)
    roc_auc = auc(fpr, tpr)  # The same results as roc_auc_score(y, prob)
    plt.figure()
    lw = 2
    plt.close('all')
    fig, ax = plt.subplots(figsize=(11, 10))
    ax_1 = ax
    ax_1_2 = ax_1.twinx()

    ax_1.tick_params(labelsize=20)
    ax_1_2.tick_params(labelsize=20)

    ax_1.plot(fpr, tpr, color='darkorange',
              lw=lw, label='ROC curve (AUC = %0.3f)' % roc_auc)
    # print('ROC curve (AUC = %0.2f)' % roc_auc)
    if tie_line:
        ax_1.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    threshold[threshold > 1] = 1
    ax_1.plot(fpr, threshold, color='green', lw=lw, linestyle='--', label='Threshold for classification')
    if f1:
        ax_1.plot(fpr, F1, color='black', lw=lw, linestyle='-.', label='F1 Score')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    if title_show:
        plt.title('Receiver Operating Characteristic\nClassifier: ' + str(clf)[:str(clf).find('(')])
    if grid_on:
        ax_1.grid()
    # plt.xlim([0.025, 1.0])
    # plt.ylim([0.0, 1.0])
    ax_1.set_ylim(-0.1, 1.15)
    ax_1.set_xlim(0 - 0.035, 1 + 0.035)
    bins = 20
    sns.set_color_codes('colorblind')
    pos_plot = sns.distplot(prob_pos, bins=bins,
                            label=f'Synthesis: {len(y[y > 0]):,} ({len(y[y > 0]) / len(y) * 100:.0f}%)',
                            ax=ax_1_2, kde=False, norm_hist=False, color='g')
    neg_plot = sns.distplot(prob_neg, bins=bins,
                            label=f'Anomaly: {len(y[y < 0]):,} ({len(y[y < 0]) / len(y) * 100:.0f}%)',
                            ax=ax_1_2, kde=False, norm_hist=False, color='r')
    ymax = max(neg_plot.dataLim.max[1], pos_plot.dataLim.max[1])
    ax_1_2.set_ylim(0, ymax * 1.1)
    # ax.axis('square')
    # ax.legend(loc="best")
    # ax2.legend(loc="upper right")
    # ax.legend(bbox_to_anchor=(.55, 1.0), loc='upper left')
    # ax2.legend(bbox_to_anchor=(.75, .8), loc='upper left')
    lines, labels = ax_1.get_legend_handles_labels()
    lines2, labels2 = ax_1_2.get_legend_handles_labels()
    ax_1_2.legend(lines + lines2, labels + labels2, loc="center right", fontsize=26)

    ax_1.set_xlabel('False Positive Rate', fontsize=26)
    ax_1.set_ylabel('True Positive Rate', fontsize=26)
    ax_1_2.set_ylabel('Count', fontsize=26)
    ax_1.set_aspect('equal', adjustable='datalim')
    # plt.legend()

    if run is not None:
        with pandas.ExcelWriter(run_1.results_path + 'out_' + str(clf)[:str(clf).find('(')] + '.xlsx') as writer:
            rates = pandas.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': threshold, 'f1': F1})
            rates.to_excel(writer, sheet_name='rates')
            pandas.DataFrame({'probability positives': prob_pos}).to_excel(writer, sheet_name='probability positives')
            pandas.DataFrame({'probability negatives': prob_neg}).to_excel(writer, sheet_name='probability negatives')

        # ax_2 = ax[1]
        plt.savefig(run_1.results_path + str(clf)[:str(clf).find('(')] + '.svg')
        plt.savefig(run_1.results_path + str(clf)[:str(clf).find('(')] + '.png', dpi=600)
        # plt.savefig(run.results_path + str(clf)[:str(clf).find('(')] + '.eps', format='eps', dpi=1200)
    plt.show()

    plot_meta_data = False
    if plot_meta_data:
        fig, ax = plt.subplots(figsize=(10, 10 / 3))
        # ax_2 = plt.subplot(gs[1])
        ax_2 = ax
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # place a text box in upper left in axes coords
        txt = '\n'.join(meta_data)
        txt = re.sub('\t', '|', txt)
        ax_2.text(0.05, 0.95, txt, transform=ax_2.transAxes, fontsize=20,
                  verticalalignment='top', bbox=props)
        ax_2.axis('off')

        if run is not None:
            plt.savefig(run_1.results_path + str(clf)[:str(clf).find('(')] + '_meta_.svg')
            plt.savefig(run_1.results_path + str(clf)[:str(clf).find('(')] + '_meta_.png', dpi=600)

        plt.show()


def binary_train(use_all_data=True, split_data_frac=1.):
    out = data_preparation(use_all_data=use_all_data, split_data_frac=split_data_frac,
                           pca_n_comp=get_arg_terminal('pca_n_comp', default=1000),
                           return_pca_ss=True)
    X_train, y_train, X_test, y_test = out.pop('data')
    ss = out.pop('ss')
    pca = out.pop('pca')
    del out

    cpu = get_arg_terminal('cpu', default=2)
    print('Fitting the model', flush=True)

    nn_clf = MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500, random_state=0)
    rf_clf = RandomForestClassifier(max_depth=60, random_state=0, n_estimators=50,
                                    n_jobs=get_arg_terminal('cpu', default=4))

    classifiers = {}
    predictions = {}
    pre_load_classifiers = False
    if pre_load_classifiers:
        if exists(run_1.results_path + 'classifiers.pkl'):
            classifiers = load_var(run_1.results_path + 'classifiers.pkl')
        else:
            warnings.warn('')

    clf_list = [
        rf_clf,
        nn_clf,
    ]
    if get_arg_terminal('classifier') is not None:
        clf_list = [locals()[get_arg_terminal('classifier')]]

    print(f'List of classifiers for training:\n{clf_list}')
    metrics = {'Classifier': [], 'Set': [], 'comp': [], 'val': []}
    clf_dic = {
        'MLPClassifier': 'Neural Network',
        'RandomForestClassifier': 'Random Forest',
    }

    for c in clf_list:
        print('-' * 50)
        print(c)
        clf_name = str(c)[:str(c).find('(')]
        print('Classifier name: ', re.sub(r"(\w)([A-Z])", r"\1 \2", str(c)[:str(c).find('(')]), flush=True)
        t1 = datetime.now()
        if not str(c)[:str(c).find('(')] in classifiers.keys():
            c.fit(X_train, y_train)
        else:
            c = classifiers[str(c)[:str(c).find('(')]]

        predY_prob = c.predict_proba(X_test)[:, 1]
        ind = np.array(random.choices(range(len(X_train)), k=len(X_test)))
        predY_prob_train = c.predict_proba(X_train[ind])[:, 1]
        # predY = c.predict(X_test)
        predY = np.sign(np.sign(predY_prob - 0.5) + .5)

        fpr, tpr, threshold = roc_curve(y_test, predY_prob)
        roc_auc_test = auc(fpr, tpr)
        fpr, tpr, threshold = roc_curve(y_train[ind], predY_prob_train)
        roc_auc_train = auc(fpr, tpr)

        metrics['Classifier'].append(clf_dic[clf_name])
        metrics['Set'].append('Test')
        metrics['comp'].append(X_test.shape[1])
        metrics['val'].append(roc_auc_test)
        metrics['Classifier'].append(clf_dic[clf_name])
        metrics['Set'].append('Train')
        metrics['comp'].append(X_train.shape[1])
        metrics['val'].append(roc_auc_train)

        print(f'Train auc = {roc_auc_train}')
        print(f'Test auc = {roc_auc_test}')

        plot_roc(c, X_test, y_test, tie_line=True, f1=True, title_show=True)
        if sky_roc:
            plot_roc(c, X_sky_g1, y_sky_g1, tie_line=True, f1=True, title_show=True)
            plot_roc(c, X_sky_g2, y_sky_g2, tie_line=True, f1=True, title_show=True)
        dt = datetime.now() - t1
        print('Fitting and predicting time: {}'.format(str(dt).split('.')[0]))
        print('accuracy = {:.2f}%'.format(100 * accuracy_score(y_test, predY)))
        print(classification_report(y_test, predY), flush=True)
        print(classification_report(y_test, np.round(predY_prob) * 2 - 1), flush=True)
        classifiers.update({str(c)[:str(c).find('(')]: c})
        predictions.update({str(c)[:str(c).find('(')]: predY_prob})
    labels = y_test
    classifiers.update({'PCA': pca})
    classifiers.update({'StandardScaler': ss})
    metrics = pd.DataFrame(metrics)

    print('Training Voting Classifier')

    save_var(classifiers, run_1.results_path + 'classifiers.pkl')
    save_var({'predictions': predictions, 'labels': labels}, run_1.results_path + 'predictions.pkl')
    save_var(metrics, run_1.results_path + 'metrics.pkl')
    metrics.to_csv(run_1.results_path + 'metrics.csv')


def over_fitting_features_parallel(study='features', clf_path=None, battery_results=True):
    # global X_bat_cod
    # global X_bat
    data = data_preparation(over_sampling=True, standard_scaler=False, random_state=1,
                            apply_pca=False, pca_n_comp=4048, split_data_frac=get_arg_terminal('frac', default=1.),
                            pos_frac=1.0, make_dev_data=True, use_all_data=get_arg_terminal('all', default=False),
                            )
    X_train, y_train, X_test, y_test, X_dev, y_dev = data
    global xt_global, yt_global
    xt_global = X_train
    yt_global = y_train
    X_bat = None
    if battery_results:
        bat_files = list_all_files(data_path + 'cod/battery/cif/', pattern=r'[0-9]*/CAE/run_043/*.pkl')
        # if no bat file found run predict_bat() from battery_mp.py
        X_bat = []
        for i in range(len(bat_files)):
            X_bat.append(load_var(bat_files[i]))
        X_bat = np.vstack(X_bat)

        cod_files = list_all_files(data_path + f'cod/battery/battery_cod/', pattern='[0-9]*.cif.pkl')
        X_bat_cod = np.vstack([load_var(j) for j in cod_files if exists(j)])

    tasks = max(get_arg_terminal('tasks', default=1), 1)
    cpu_per_task = max(1, tot_cpu // tasks)
    print(f'Tot. CPU cores: {tot_cpu}\nTot. tasks: {tasks}\nCPU cores per task: {cpu_per_task}')

    clf_labels = [
        'Random Forest',
        'Neural Network',
    ]

    pca_1 = PCA(n_components=1000)
    clf_dic = {'Neural Network': MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500, random_state=0),
               'Random Forest': RandomForestClassifier(max_depth=60, random_state=0, n_estimators=50,
                                                       n_jobs=cpu_per_task),
               }

    print('Calculations for ', clf_labels)

    components = None
    comp_label = 'features'

    study = get_arg_terminal('study', default='features-rf')
    cv = get_arg_terminal('cv', default=None)

    if study == 'features':
        comp_label = 'features'
        n_features = [
            1, 2,
            4, 8, 16,
            32, 50,
            64, 128, 256,
            512, 1024,
            1500, 2000, 2500, 3000, 4096
        ]
        # n_features += list(np.array(range(1, 14)) * 75)
        # n_features += list(np.array(range(1, 8)) * 150)
        # n_features = list(np.array(np.linspace(1, 100, 25), dtype=int))
        n_features = [1000]
        components = n_features

    if study == 'features-rf':
        clf_labels = ['Random Forest']
        comp_label = 'features'
        n_features = list(np.array(np.linspace(1, 130, 2), dtype=int))
        components = n_features

    if study == 'rf__max_depth':
        clf_labels = ['Random Forest']
        comp_label = 'max depth'
        components = list(np.array(np.linspace(1, 80, 15), dtype=int))

    if study == 'rf__n_estimators':
        clf_labels = ['Random Forest']
        comp_label = 'number of estimators'
        components = list(np.array(np.linspace(1, 100, 10), dtype=int))

    if study == 'nn__n_layers':
        comp_label = 'number of layers'
        clf_labels = ['Neural Network']
        n_layers = [
            1, 2, 3, 4,
            5, 6, 7,
        ]
        components = n_layers

    if study == 'nn__layers_size':
        comp_label = 'layers\' size'
        clf_labels = ['Neural Network']
        components = list(range(4, 26, 3))

    if study == 'svm__c':
        comp_label = 'C parameter'
        clf_labels = ['SVM-Bagging']
        components = list(np.linspace(.025, .15, 4))
        # components += list(np.linspace(2, 100, 5))

    # n_features = np.array(np.linspace(2, 1000, 30), dtype=int)
    # data_points = np.array(np.linspace(2000, 19000, 10), dtype=int)

    print(f'Studying: {study}')
    print('Components = ', components)

    clfs = []
    for k in clf_labels:
        for c in components:
            if study == 'features' or study == 'features-rf':
                pca_1 = PCA(n_components=int(c))
                # issue possibility ????????????????? where did I use this var
            if study == 'nn__n_layers':
                clf_dic = {
                    'Neural Network': MLPClassifier(hidden_layer_sizes=tuple([13] * c), max_iter=500, random_state=0,
                                                    # verbose=True
                                                    )
                }
            if study == 'nn_layers_size':
                clf_dic = {
                    'Neural Network': MLPClassifier(hidden_layer_sizes=tuple([c] * 3), max_iter=500, random_state=0,
                                                    # verbose=1
                                                    )
                }

            if study == 'rf__max_depth':
                clf_dic = {
                    'Random Forest': RandomForestClassifier(max_depth=c, random_state=0, n_estimators=50,
                                                            n_jobs=cpu_per_task),
                }
            if study == 'rf__n_estimators':
                clf_dic = {
                    'Random Forest': RandomForestClassifier(max_depth=20, random_state=0, n_estimators=c,
                                                            n_jobs=cpu_per_task),
                }
            if study == 'svm__c':
                clf_dic = {
                    'SVM-Bagging': BaggingClassifier(base_estimator=svm.SVC(gamma='auto', probability=True, C=c),
                                                     n_estimators=8, random_state=0,
                                                     verbose=0,
                                                     max_samples=min(15000, len(X_train) // cv * (cv - 1)),
                                                     n_jobs=cpu_per_task),
                }

            clfs.append({
                'name': k,
                'comp_val': c,
                'comp_name': comp_label,
                'cv': cv,
                'clf': Pipeline(steps=[
                    ('ss', StandardScaler()),
                    ('pca', pca_1),
                    ('clf', clf_dic[k]),
                ]),
                # 'xt': X_train.copy(),
                # 'yt': y_train.copy(),
            })

    loading_clfs = False
    save_results = True
    if loading_clfs:
        print('Loading clfs from: ')
        clfs = load_var(f'{clf_path}clfs.pkl', verbose=True)
    else:
        # if tasks > 0:
        clfs = Parallel(n_jobs=tasks)(delayed(fit_classifiers)(c) for c in clfs)
        # else:
        #     clfs = [fit_classifiers(c) for c in clfs]
        for r in clfs:
            r.pop('xt', None)
            r.pop('yt', None)
    if save_results:
        save_var(clfs, run_1.results_path + 'clfs.pkl')
    run_1.timer_lap()

    print('Calculating the scores', flush=True)
    calc_scores = True
    if not calc_scores:
        print('End fn')
        return locals()

    results = pd.DataFrame()
    loading_results = False
    if loading_results:
        results = load_var(clf_path + 'results.pkl')
    else:
        for c in clfs:
            c[f'X_dev'] = locals()[f'X_dev']
            c[f'y_dev'] = locals()[f'y_dev']

            # c['X_train'] = X_train.copy()
            # c['y_train'] = y_train.copy()
            if battery_results:
                for d in ['bat', 'bat_cod']:
                    c[f'X_{d}'] = locals()[f'X_{d}']
                c[f'y_bat'] = [1] * len(c['X_bat'])
                c[f'y_bat_cod'] = [1] * len(c['X_bat_cod'])
            else:
                c['data_set'] = {'X_dev': 'dev', 'X_train': 'train'}

        # results = run_in_parallel(clf_score_p, clfs, n_jobs=tasks, method=2)
        print('Parallel scoring', flush=True)
        results = None
        # tasks = 1
        if tasks > 1:
            results = Parallel(n_jobs=tasks)(delayed(clf_score_p)(c) for c in clfs)
        else:
            results = [clf_score_p(c) for c in clfs]
        # results = Parallel(n_jobs=tasks, verbose=1, backend="threading")(
        #     map(delayed(clf_score_p), clfs))

        [r.pop('clf') for r in results]
        for r in results:
            r.pop('X_train')
            r.pop('y_train')
            r.pop('X_dev')
            r.pop('y_dev')
            if battery_results:
                r.pop('X_bat')
                r.pop('y_bat')
                r.pop('X_bat_cod')
                r.pop('y_bat_cod')
        results = pd.DataFrame(results)
        save_var(results, run_1.results_path + 'results.pkl')
        results.to_csv(run_1.results_path + 'results.csv')
        save_df(results, run_1.results_path + 'results.txt')
        run_1.timer_lap()

    # print('Plotting', flush=True)
    # plot_overfitting(results)

    print('End fn')
    return locals()


def plot_overfitting(results, battery_results=True):
    # results = load_var(path + 'results.pkl')
    clf_labels = list(np.unique(results['name']))
    comp_label = list(np.unique(results['comp_name']))[0]
    if 'pca__n_comp' in comp_label:
        comp_label = 'Number of features'

    over_plotting = False
    if over_plotting:
        plt.figure(figsize=(5, 5))
        ax1 = plt.gca()
        for key in clf_labels:
            best_clfs = results[pd.DataFrame(results)['name'] == key]
            best_clfs.plot(x='comp', y='dev', yerr='dev_std',
                           label=f'{key} (dev)', ax=ax1)
            best_clfs.plot(x='comp', y='train', yerr='train_std',
                           label=f'{key} (train)', ax=ax1)
            plt.legend(loc='best')
            ax1.set_ylabel('Accuracy')
            ax1.set_xlabel(f'{comp_label.capitalize()}')
            plt.title('Overfitting Test')

        plt.savefig(run_1.results_path + 'Overfitting' + '.svg')
        plt.savefig(run_1.results_path + 'Overfitting' + '.png', dpi=600)
        for key in clf_labels:
            best_clfs = results[pd.DataFrame(results)['name'] == key]
            best_clfs.plot(x='comp', y='bat', yerr='bat_std',
                           label=f'{key} (battery-MP POS ratio)', ax=ax1)
            best_clfs.plot(x='comp', y='bat_cod', yerr='bat_cod_std',
                           label=f'{key} (battery-COD POS ratio)', ax=ax1)
            plt.legend(loc='best')

            ax1.set_xlabel(f'{comp_label.capitalize()}')
        plt.savefig(run_1.results_path + 'Overfitting-battery' + '.svg')
        plt.savefig(run_1.results_path + 'Overfitting-battery' + '.png', dpi=600)
        plt.show()

    for key in clf_labels:
        f, ax, fs = Plots.plot_format()
        ax1 = ax

        best_clfs = results[pd.DataFrame(results)['name'] == key]
        best_clfs.plot(x='comp', y='dev', yerr='dev_std',
                       label=f'Development', ax=ax1)
        best_clfs.plot(x='comp', y='train', yerr='train_std',
                       label=f'Train', ax=ax1)
        # best_clfs.plot(x='comp', y='bat', yerr='bat_std',
        #                label=f'{key} (battery-MP POS ratio)', ax=ax1)
        best_clfs.plot(x='comp', y='bat_cod', yerr='bat_cod_std',
                       label=f'Observed battery', ax=ax1)
        plt.legend(loc='best')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel(f'{comp_label.capitalize()}')
        ax1.grid()
        plt.title(f'{key}')
        run_1.save_plot(f'{comp_label}-{key}')
        # plt.savefig(run_1.results_path + f'{comp_label}-{key}' + '.svg')
        # plt.savefig(run_1.results_path + f'{comp_label}-{key}' + '.png', dpi=600)
        plt.show()


def fit_classifiers(inp):
    xt = yt = None
    if 'xt' in inp:
        xt = inp['xt']
        yt = inp['yt']
    else:
        global xt_global, yt_global
        xt = xt_global
        yt = yt_global
    train_score = inp.get('train_score', False)
    out = inp.copy()
    n = len(xt)
    cv = None
    random_state = 0
    name = None
    if isinstance(inp, dict):
        if inp.get('comp_name', None) == 'n':
            n = int(inp.get('comp_val', len(xt)))
        cv = inp.get('cv', None)
        if cv == 1:
            cv = None
        random_state = inp.get('random_state', 0)
        name = inp.get('name', None)
        clf_in = inp['clf']
    else:
        clf_in = inp
    kf = [(np.array(range(n)), 0)]
    if cv:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=cv, random_state=random_state, shuffle=True)
        kf = [(i, j) for i, j in kf.split(range(n))]
    clf_out = []
    t_0 = datetime.now()

    c_val = inp['comp_val']
    c_name = inp['comp_name']

    for cv in range(len(kf)):
        print(f'Training {name} {cv + 1}/{len(kf)} with {c_name} = {c_val} on {n:,} data points.', flush=True)
        x = xt[kf[cv][0]]
        y = yt[kf[cv][0]]

        from sklearn.base import clone
        clf_tmp = clone(clf_in)
        ss = clf_tmp['ss']
        pca = clf_tmp['pca']
        clf = clf_tmp['clf']

        x = ss.fit_transform(x)
        x = pca.fit_transform(x)

        clf.fit(x, y)

        score = None
        if train_score:
            score = clf.score(x, y)

        clf_out.append({
            'name': name,
            'ss': ss,
            'pca': pca,
            'clf': clf,
            'train_score': score,
        })

    out['clf'] = clf_out

    t_1 = datetime.now()

    print('*' * 10)
    print('Finished: ', flush=True, end='')
    print(f'the training {name} {cv + 1}/{len(kf)} with {c_name} = {c_val} on {n:,} data points.')
    print('Run Time: ', str(t_1 - t_0).split('.')[0], flush=True)
    print('*' * 10)
    return out


def clf_score(results, key, clf_l, x, y, data_set):
    print(f'Calc. score {key}: ', data_set)
    results[key][f'{data_set}_l'].append([clf['pipe'].score(x, y) for clf in clf_l])
    results[key][f'{data_set}'].append(np.mean(results[key][f'{data_set}_l']))
    results[key][f'{data_set}_std'].append(np.std(results[key][f'{data_set}_l']))


def clf_score_p(clf):
    name = clf['name']
    c_val = clf['comp_val']
    c_name = clf['comp_name']
    clf['comp'] = clf['comp_val']
    data_set = clf.pop('data_set', None)
    xt = yt = None
    if not 'xt' in clf:
        global xt_global, yt_global
        clf['X_train'] = xt_global[:len(xt_global) // 10]
        clf['y_train'] = yt_global[:len(xt_global) // 10]
    t_0 = datetime.now()

    print(f'Calc. scores for {name} with {c_name} = {c_val}', flush=True)

    # data_set = ['train', 'dev', 'bat', 'bat_cod']
    if data_set is None:
        data_set = {'xt': 'train', 'X_dev': 'dev', 'X_bat': 'bat', 'X_bat_cod': 'bat_cod'}
    for d in data_set:
        r = []
        p_list = []
        y_list = []
        for cl in clf['clf']:
            x = clf[d]
            y = clf['y' + d[1:]]
            x = cl['ss'].transform(x)
            x = cl['pca'].transform(x)
            p = cl['clf'].predict_proba(x)[:, 1]

            p_list.append(p)
            y_list.append(y)
            fpr, tpr, threshold = roc_curve(y, p)
            roc_auc = auc(fpr, tpr)
            r.append(roc_auc)

        clf[f'{data_set[d]}_prob'] = p_list
        clf[f'{data_set[d]}_true_y'] = y_list
        clf[f'{data_set[d]}_all'] = r
        clf[f'{data_set[d]}'] = np.mean(r)
        clf[f'{data_set[d]}_std'] = np.std(r)

    t_1 = datetime.now()
    run_time = str(t_1 - t_0).split('.')[0]
    print(f'Finished scoring for {name} with {c_name} = {c_val} (Run time: {run_time})', flush=True)
    return clf


if __name__ == '__main__':
    run_1 = RunSet()

    # X_dev, y_dev = [None] * 2
    # X_bat, X_bat_cod = None, None
    meta_data = None
    ss, pca = None, None
    X_sky_g1, y_sky_g1, X_sky_g2, y_sky_g2 = [None] * 4
    xt_global = yt_global = None

    do_overfitting_analysis = False
    if do_overfitting_analysis:
        out = over_fitting_features_parallel(battery_results=False)
        locals().update(out)

    sky_roc = False
    train_classifiers = True
    if train_classifiers:
        print('Training classifiers', flush=True)
        binary_train(use_all_data=get_arg_terminal('all', default=True),
                     split_data_frac=get_arg_terminal('frac', default=1.))

    run_1.end()
