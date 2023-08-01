from xiespp.synthesizability_1.utility.utility_general import *
# import utility.util_plot


def plot_roc_curve(y, yp, ax=None, show_plot=False, plot_roc=True, plot_dist=True):
    from sklearn.metrics import roc_curve, auc
    import seaborn as sns

    lw = 2
    # TN_FP = len(y[y < 0])  # Total Neg
    # TP_FN = len(y[y > 0])  # Total Pos
    prob_pos = yp[y > 0]
    prob_neg = yp[y < 0]
    fpr, tpr, threshold = roc_curve(y, yp)
    # TP = tpr * TP_FN
    # FP = fpr * TN_FP
    # recall = tpr
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     precision = TP / (TP + FP)
    # F1 = 2 * (recall * precision) / (recall + precision)
    roc_auc = auc(fpr, tpr)  # The same results as roc_auc_score(y, prob)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    ax_1 = ax
    ax_1_2 = ax_1.twinx()

    if plot_roc:
        ax_1.plot(fpr, tpr, color='darkorange',
                  lw=lw, label='ROC curve (AUC = %0.3f)' % roc_auc)
        ax_1.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        # ax_1.plot(fpr, threshold, color='green', lw=lw, linestyle='--', label='Threshold for classification')

    bins = 20
    sns.set_color_codes('colorblind')
    if plot_dist:
        sns.distplot(prob_pos, bins=bins,
                     label=f'Synthesis: {len(y[y > 0]):,} ({len(y[y > 0]) / len(y) * 100:.0f}%)',
                     ax=ax_1_2, kde=False, norm_hist=False, color='g')
        sns.distplot(prob_neg, bins=bins,
                     label=f'Anomaly: {len(y[y < 0]):,} ({len(y[y < 0]) / len(y) * 100:.0f}%)',
                     ax=ax_1_2, kde=False, norm_hist=False, color='r')

    lines, labels = ax_1.get_legend_handles_labels()
    lines2, labels2 = ax_1_2.get_legend_handles_labels()
    ax_1_2.legend(lines + lines2, labels + labels2, loc="center right")
    ax_1.set_xlabel('False Positive Rate')
    ax_1.set_ylabel('True Positive Rate')
    ax_1_2.set_ylabel('Count')
    ax_1.set_aspect('equal', adjustable='datalim')
    if show_plot:
        plt.show()


class ClassifierType1:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier

    def __init__(self, clf: MLPClassifier, result_dir=None, ss: StandardScaler = None, pca: PCA = None,
                 name=None, label='None'):
        self.clf = clf
        self.result_dir = result_dir
        makedirs(result_dir, exist_ok=True)
        self.ss = ss
        self.pca = pca
        self.stats = {}
        self.predictions = {}
        self.threshold = 0.5
        if name is None:
            name = str(self.clf)[:str(self.clf).find('(')]
            if label == 'None':
                label = ''.join([char for char in name.split('Classifier')[0] if char.isupper()])
        self.name = name
        self.label = label

    def fit(self, X_train, y_train):
        X = X_train.copy()
        if self.ss is not None:
            self.ss.fit(X)
            X = self.ss.transform(X)
        if self.pca is not None:
            self.pca.fit(X)
            X = self.pca.transform(X)
        self.clf.fit(X, y_train)

        _, stats = self.predict_proba(X_train, y_train, set_name='train')

        if self.result_dir is not None:
            save_var(self.clf, self.result_dir + '/clf.pkl')
            save_var(self.ss, self.result_dir + '/ss.pkl')
            save_var(self.pca, self.result_dir + '/pca.pkl')
        return stats

    def set_threshold(self, y_true, yp_prob=None, X=None):
        if yp_prob is None:
            yp_prob = self.predict_proba(X)
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(y_true, yp_prob)
        F1 = 2 * (precision * recall) / (precision + recall)
        best_threshold_ind = np.argmax(F1)
        best_threshold = thresholds[best_threshold_ind]
        self.threshold = best_threshold
        return self.threshold

    def predict_proba(self, X, y_true=None, set_name=None, set_threshold=False, threshold=None):
        from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score
        X_trans = X.copy()
        if self.ss is not None:
            X_trans = self.ss.transform(X_trans)
        if self.pca is not None:
            X_trans = self.pca.transform(X_trans)
        yp_proba = self.clf.predict_proba(X_trans)[:, 1]

        # print('self.clf.predict_proba(X_trans)[:, 1]')
        # print(self.clf.predict_proba(X_trans)[:, 1])
        # print('self.clf.predict_proba(X_trans)[:, 0]')
        # print(self.clf.predict_proba(X_trans)[:, 0])
        # print(self.clf)

        if y_true is None:
            return yp_proba

        if set_threshold is True:
            self.set_threshold(y_true, yp_proba)
        if threshold is None:
            threshold = self.threshold

        yp_label = np.sign(np.sign(yp_proba - threshold) + .5)
        yp_label05 = np.sign(np.sign(yp_proba - 0.5) + .5)
        roc_auc = np.nan
        precision = np.nan
        recall = np.nan
        acc_pos = np.nan
        acc_neg = np.nan
        if ~np.all(y_true == 1):
            fpr, tpr, _ = roc_curve(y_true, yp_proba)
            roc_auc = auc(fpr, tpr)
            precision = precision_score(y_true, yp_label)
            recall = recall_score(y_true, yp_label)
            acc_pos = 100 * accuracy_score(y_true[y_true > 0], yp_label[y_true > 0])
            acc_neg = 100 * accuracy_score(y_true[y_true < 0], yp_label[y_true < 0])

        acc = 100 * accuracy_score(y_true, yp_label)
        acc05 = 100 * accuracy_score(y_true, yp_label05)

        stats = {
            'roc_auc': roc_auc,
            'acc': acc,
            'acc_pos': acc_pos,
            'acc_neg': acc_neg,
            'acc_05': acc05,  # Accuracy when the threshold is 0.5
            'precision': precision,
            'recall': recall,
            'threshold': threshold,
        }
        predictions = {
            'yp_proba': yp_proba,
            'yp_label': yp_label,
            'y_true': y_true,
        }
        # predictions = pd.DataFrame(predictions)
        if set_name is not None:
            self.stats[set_name] = stats
            self.predictions[set_name] = predictions
            pd.DataFrame(predictions).to_csv(self.result_dir + f'/{set_name}.csv')
            self.get_stats().to_csv(self.result_dir + f'/stats.csv')
            save_var(self, self.result_dir + '/classifier_class_autosave.pkl')
        return yp_proba, stats

    def __str__(self):
        string = []
        string += [self.name]
        string += ['Standard Scalar: ' + str(self.ss)]
        string += ['PCA: ' + str(self.pca)]
        string += [str(self.clf)]
        string += [f'Best threshold = {self.threshold:.3f}']
        string += [f'{k}: {v}' for k, v in self.stats.items()]
        return '\n'.join(string)

    def get_stats(self):
        stats = self.stats.copy()
        for k, v in stats.items():
            v['set'] = k
        stats = [v for _, v in stats.items()]
        stats = pd.DataFrame(stats)
        return stats

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, X_dev=None, y_dev=None):
        self.fit(X_train, y_train)
        # self.set_threshold(y_test, X=X_test)
        self.predict_proba(X=X_test, y_true=y_test, set_name='test')
        if X_dev is not None:
            self.predict_proba(X=X_dev, y_true=y_dev, set_name='dev')
        return str(self)
