"""
Questo programma valuta vari modelli sul dataset 'Cards Image Dataset-Classification'
contenente immagini divise di carte da gioco divise in 53 classi (7624 train, 265 test, 265 validation)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV


class EstimatorSelectionHelper:

    def __init__(self, models, params):
        """
        Se vengono dati iperparametri sbagliati al costruttore viene sollevato un errore
        """
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Parametri mancanti: %s" % missing_params)

        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv=3, n_jobs=3, verbose=1, scoring=None, refit=False):
        """
        per ogni modello definito viene fatta la GridSearchCV
          
        :param X: X_train
        :param y: y_train
        :param cv: 
        :param n_jobs: 
        :param verbose: 
        :param scoring: 
        :param refit: 
        :return: 
        """
        for key in self.keys:
            print("GridSearchCV per %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit,
                              return_train_score=True)
            gs.fit(X, y)
            self.grid_searches[key] = gs

    def score_summary(self, sort_by='mean_score'):
        """
        Salva i risultati di ogni search coi relativi dati, li ordina in base all'accuracy media e li salva in un csv
        :param sort_by:
        :return:
        """

        def row(key, scores, params):
            d = {
                'estimator': key,
                'min_score': min(scores),
                'max_score': max(scores),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
            }
            return pd.Series({**params, **d})

        rows = []
        for k in self.grid_searches:
            print(k)
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            for i in range(self.grid_searches[k].cv):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]
                scores.append(r.reshape(len(params), 1))

            all_scores = np.hstack(scores)
            for p, s in zip(params, all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]
        df.to_csv('results.csv', sep=',')
        return df[columns]


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC

models = {
    'ExtraTreesClassifier': ExtraTreesClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'SVC': SVC()
}

params = {
    'ExtraTreesClassifier': {'n_estimators': [20, 50, 100]},
    'RandomForestClassifier': {'bootstrap': [True, False],
                               'n_estimators': [20, 50, 100]},
    'AdaBoostClassifier': {'n_estimators': [20, 50, 100]},
    'GradientBoostingClassifier': {'n_estimators': [20, 50, 100], 'learning_rate': [0.8, 1.0]},
    'SVC': [
        {'kernel': ['linear'], 'C': [1, 10]},
        {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001]},
    ]
}

"""
Inizialmente i dati venivano importati con la libreria glob poi è stata sostituita con la funzione 
  'tf.keras.utils.image_dataset_from_directory' 
"""
import tensorflow as tf
import os

train_path = os.path.abspath('data/cards-image-datasetclassification/train')

train = tf.keras.utils.image_dataset_from_directory(train_path, label_mode='int',
                                                    batch_size=100, image_size=(180, 180), seed=42,
                                                    color_mode='grayscale')

for dict_slice, target in train.take(1):
    features = np.array(dict_slice).reshape(100, -1)
    labels = np.array(target, dtype=np.uint32)

print(labels.shape)
features = features / 255.0

helper = EstimatorSelectionHelper(models, params)
helper.fit(features, labels, scoring='accuracy', n_jobs=2)
helper.score_summary(sort_by='max_score')
"""
Il programma impiega circa 15-20 minuti i risultati sono in un file allegato ai programmi, in generale i modelli
 performano male, il migliore ha un'accuracy massima di circa il 24%, quindi ho deciso di usare tensorflow e keras. 
"""
