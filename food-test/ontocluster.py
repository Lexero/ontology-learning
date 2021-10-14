from time import time
from hdbscan.hdbscan_ import HDBSCAN
from matplotlib.pyplot import grid
import owlready2 as owl
import types
from scipy import rand
from numpy import sqrt, array, sum, mean, abs, array

import pandas as pd
from sklearn.cluster import AffinityPropagation, MeanShift
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from pyclustering.cluster.xmeans import xmeans

from itertools import product


AVAILABLE_ALGORITHMS = ['xmeans', 'hdbscan', 'affinitypropagation', 'meanshift']

def delete_operators(columns):
    ''' Deletes operators from DataFrame columns '''
    translation_table = dict.fromkeys(map(ord, ['+', '-', '*', '/', '%', '=', '!', '<', '>', '?', '\\', '|', '&', '@', '\"', '\'', '#', '$', '^', '(', ')', ',', '.']), None)
    return [column.translate(translation_table) for column in columns]

def find_nearest_idx(a, value):
    arr = array(a)
    return abs(arr - value).argmin()

def parse_itemset(itemset):
    ''' Parse itemset and return cluster label, feature name and value of feature '''
    if type(itemset[0]) == frozenset:
        itemset = [list(item)[0] for item in itemset]

    if itemset[0].split('_', 1)[0] == 'label':
        cluster = int(itemset[0].split('_')[1])
        feature, value = itemset[1].split('_', 1)
    else:
        cluster = int(itemset[1].split('_', 1)[1])
        feature, value = itemset[0].split('_', 1)

    return cluster, feature, value

def clusters_to_labels(clusters):
    ''' Convert xmeans clusters to labels of clusters '''
    count = len([j for sub in clusters for j in sub])
    labels = [0 for i in range(count)]
    for i, cluster in enumerate(clusters):
        for sample in cluster:
            labels[sample] = i
    return labels

def get_indicies(a, value):
    ''' Returns indicies of value in list a '''
    indicies = []
    for i in range(len(a)):
        if a[i] == value:
            indicies.append(i)
    return indicies




class GridSearch:
    def __init__(self, model, scoring, grid_params : dict):
        self.model = model
        if type(scoring) == str:
            self.scoring = {
                'silhouette_score':silhouette_score,
                'calinski_harabasz_score':calinski_harabasz_score,
                'davies_bouldin_score':davies_bouldin_score
            }[scoring]
        else:
            self.scoring = scoring
        self.grid_params = grid_params
        self.grid = [{k:v for k,v in zip(grid_params.keys(), values)} for values in product(*grid_params.values())]

    def fit(self, X):
        self.result_grid = []
        if self.model == MeanShift or self.model == AffinityPropagation or self.model == HDBSCAN:
            for params in self.grid:
                model = self.model(**params)
                model.fit(X)
                try:    
                    score = self.scoring(X, model.labels_)
                    self.result_grid.append([params, score])
                except:
                    continue
        elif self.model == xmeans:
            for params in self.grid:
                model = self.model(X, **params)
                model.process()
                try:    
                    score = self.scoring(X, clusters_to_labels(model.get_clusters()))
                    self.result_grid.append([params, score])
                except:
                    continue

        if len(self.result_grid) == 0:
            print('Warning: default estimator!')
            self.best_estimator = self.model()
            self.best_params = {}
            return None

        if self.scoring == silhouette_score:
            self.best_params = self.result_grid[find_nearest_idx(array(self.result_grid)[:, 1], 1)][0]
        elif self.scoring == calinski_harabasz_score:
            self.best_params = self.result_grid[array(self.result_grid)[:, 1].argmax()][0]
        elif self.scoring == davies_bouldin_score:
            self.best_params = self.result_grid[find_nearest_idx(array(self.result_grid)[:, 1], 0)][0]

        if self.model == MeanShift or self.model == AffinityPropagation or self.model == HDBSCAN:
            self.best_estimator = self.model(**self.best_params)
        else:
             self.best_estimator = self.model(X, **self.best_params)



class OntoClusterer:
    def __init__(self, algorithm : str) -> None:
        assert algorithm in AVAILABLE_ALGORITHMS, f"No algorithm {algorithm}\nAvailable algorithms: {AVAILABLE_ALGORITHMS}"
        self.algorithm = algorithm
        

    def fit(self, X, data : pd.DataFrame, grid_params = None, scoring = None, **kwargs) -> owl.Ontology:
        ''' Fits clusterization algorithm on matrix X and converts to ontology'''
        if grid_params: assert scoring, "Choose scoring"
        if scoring: assert grid_params, "Choose grid_params"
        data.columns = delete_operators(data.columns)

        if self.algorithm == 'xmeans':
            if grid_params == None:        
                self.model = xmeans(X, **kwargs)
            else:
                gs = GridSearch(xmeans, scoring, grid_params)
                gs.fit(X)
                self.model = gs.best_estimator

            self.model.process()
            self.clusters = self.model.get_clusters()
            self.labels = clusters_to_labels(self.clusters)
            self.centers = self.model.get_centers()

        elif self.algorithm == 'hdbscan':
            if grid_params == None:
                self.model = HDBSCAN(**kwargs)
            else:
                gs = GridSearch(HDBSCAN, scoring, grid_params)
                gs.fit(X)
                self.model = gs.best_estimator
            
            self.model.fit(X)
            self.labels = self.model.labels_ + 1
            self.clusters = [get_indicies(self.model.labels_, label) for label in set(self.model.labels_)]
            
            self.centers = []
            for label in set(self.model.labels_):
                if label != -1:
                    self.centers.append(list(self.model.weighted_cluster_centroid(label)))
                else:
                    if X.__class__ == pd.DataFrame: 
                        self.centers.append(mean(X.loc[get_indicies(self.model.labels_, -1), :], axis = 0))
                    else:
                        self.centers.append(mean(X[get_indicies(self.model.labels_, -1)], axis = 0))
                    
        elif self.algorithm == 'affinitypropagation':
            if grid_params == None:
                self.model = AffinityPropagation(random_state = 42, **kwargs)
            else:
                gs = GridSearch(AffinityPropagation, scoring, grid_params)
                gs.fit(X)
                self.model = gs.best_estimator
            
            self.model.fit(X)
            self.labels = self.model.labels_
            self.clusters = [get_indicies(self.model.labels_, label) for label in set(self.model.labels_)]
            self.centers = self.model.cluster_centers_

        elif self.algorithm == 'meanshift':
            if grid_params == None:
                self.model = MeanShift(**kwargs)
            else:
                gs = GridSearch(MeanShift, scoring, grid_params)
                gs.fit(X)
                self.model = gs.best_estimator
            
            self.model.fit(X)
            self.labels = self.model.labels_
            self.clusters = [get_indicies(self.model.labels_, label) for label in set(self.model.labels_)]
            self.centers = self.model.cluster_centers_
            
        onto = owl.get_ontology("http://test.org/onto.owl")

        with onto:
            class Data(owl.Thing): pass
            class Cluster(Data): pass
            class Feature(Data): pass
            
            for i, samples in enumerate(self.clusters):
                new_cluster = types.new_class(f'Cluster{i}', (Cluster, ))
                for sample in samples:
                    new_cluster(str(sample))

            for feature_name in data.select_dtypes(['object']).columns:
                new_feature = types.new_class(f'feature_{feature_name}', (Feature, ), )
                new_property_feature = types.new_class(f'has_{feature_name}', (owl.ObjectProperty, Cluster >> new_feature, ))

            for feature_name in data.select_dtypes(['object']).columns:
                temp = pd.DataFrame(data[feature_name])
                temp['label'] = self.labels
                temp = pd.get_dummies(temp, columns = temp.columns)
                
                apriori_data = apriori(temp, min_support = .1, use_colnames=True)
                frequent_itemsets = apriori_data
                frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)
                frequent_itemsets = frequent_itemsets[frequent_itemsets['length'] == 2]

                for pair in frequent_itemsets['itemsets']:
                    cluster, feature, value = parse_itemset(list(pair))
                    exec(f"new_value = types.new_class(value, (onto.feature_{feature}, ))")
                    exec(f"onto.Cluster{cluster}.is_a.append(onto.has_{feature}.some(new_value))")
                    
                if len(apriori_data) > 0:
                    ar = association_rules(apriori_data, min_threshold = 1)
                    for pair in ar[['antecedents', 'consequents']].values:
                        cluster, feature, value = parse_itemset(pair)
                        exec(f"new_value = types.new_class(value, (onto.feature_{feature}, ))")
                        exec(f"onto.Cluster{cluster}.is_a.append(onto.has_{feature}.some(new_value))")
                    
            for feature_name in data.select_dtypes(exclude = ['object']).columns:
                new_property_feature = types.new_class(f'has_{str(feature_name)}', (owl.DataProperty, Cluster >> float, ))        

        return onto
    
    def score(self, X):
        from time import time_ns
        scores = pd.DataFrame()
        model = HDBSCAN()
        start_time = time_ns()
        model.fit(X)
        estimation_time = (time_ns() - start_time)//1e6
        scores.at['HDBSCAN', 'Time'] = str(estimation_time)
        scores.at['HDBSCAN', 'Silhouette score'] = str(silhouette_score(X, model.labels_))
        scores.at['HDBSCAN', 'Calinski-Harabasz score'] = str(calinski_harabasz_score(X, model.labels_))
        scores.at['HDBSCAN', 'Davies-Bouldin score'] = str(davies_bouldin_score(X, model.labels_))
        

        model = xmeans(X)
        start_time = time_ns()
        model.process()
        estimation_time = (time_ns() - start_time)//1e6
        scores.at['XMeans', 'Time'] = str(estimation_time)
        scores.at['XMeans', 'Silhouette score'] = str(silhouette_score(X, clusters_to_labels(model.get_clusters())))
        scores.at['XMeans', 'Calinski-Harabasz score'] = str(calinski_harabasz_score(X, clusters_to_labels(model.get_clusters())))
        scores.at['XMeans', 'Davies-Bouldin score'] = str(davies_bouldin_score(X, clusters_to_labels(model.get_clusters())))

        model = AffinityPropagation(random_state=42)
        start_time = time_ns()
        model.fit(X)
        estimation_time = (time_ns() - start_time)//1e6
        scores.at['Affinity Propagation', 'Time'] = str(estimation_time)
        try:
            scores.at['Affinity Propagation', 'Silhouette score'] = str(silhouette_score(X, model.labels_))
            scores.at['Affinity Propagation', 'Calinski-Harabasz score'] = str(calinski_harabasz_score(X, model.labels_))
            scores.at['Affinity Propagation', 'Davies-Bouldin score'] = str(davies_bouldin_score(X, model.labels_))
        except:
            scores.at['Affinity Propagation', ['Silhouette score', 'Calinski-Harabasz score', 'Davies-Bouldin score']] = ['-', '-', '-']
        

        model = MeanShift()
        start_time = time_ns()
        model.fit(X)
        estimation_time = (time_ns() - start_time)//1e6
        scores.at['Mean Shift', 'Time'] = str(estimation_time)
        scores.at['Mean Shift', 'Silhouette score'] = str(silhouette_score(X, model.labels_))
        scores.at['Mean Shift', 'Calinski-Harabasz score'] = str(calinski_harabasz_score(X, model.labels_))
        scores.at['Mean Shift', 'Davies-Bouldin score'] = str(davies_bouldin_score(X, model.labels_))
        
        scores.at['Best', 'Time'] = 'Lower'
        scores.at['Best', 'Silhouette score'] = '1'
        scores.at['Best', 'Calinski-Harabasz score'] = 'Higher'
        scores.at['Best', 'Davies-Bouldin score'] = '0'

        return scores

        
        
