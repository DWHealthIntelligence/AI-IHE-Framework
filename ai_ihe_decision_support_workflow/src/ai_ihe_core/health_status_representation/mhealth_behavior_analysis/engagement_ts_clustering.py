"""
HealthBehavior Core: Engagement Time Series Clustering

@author: DanWu and Agreewithu (Ruixin Dai)
"""

import math
import random
import numpy as np
from sklearn import metrics

class TimeSeriesClusterer:
    """
    Time Series Clustering: Based on DTW (Dynamic Time Warping) and K-Means
    """
    def __init__(self, n_clusters: int = 4, max_iter: int = 100, tol: float = 1e-5):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = []
        self.cluster_dict = {}

    @staticmethod
    def _dtw_distance(s1: list, s2: list) -> float:
        """DWT Distance"""
        DTW = {}
        for i in range(len(s1)):
            DTW[(i, -1)] = float('inf')
        for i in range(len(s2)):
            DTW[(-1, i)] = float('inf')
        DTW[(-1, -1)] = 0

        for i in range(len(s1)):
            for j in range(len(s2)):
                dist = (s1[i] - s2[j]) ** 2
                DTW[(i, j)] = dist + min(DTW[(i-1, j)], DTW[(i, j-1)], DTW[(i-1, j-1)])

        return math.sqrt(DTW[len(s1)-1, len(s2)-1])

    def _init_centroids(self, data_set: np.ndarray) -> list:
        """Randomly Initialize Cluster Centers"""
        data_list = list(data_set)
        return random.sample(data_list, self.n_clusters)

    def _assign_clusters(self, data_set: np.ndarray, centroids: list) -> dict:
        """Assign the closest cluster center"""
        cluster_dict = {}
        for item in data_set:
            min_dis = float("inf")
            flag = -1
            for i in range(self.n_clusters):
                distance = self._dtw_distance(item, centroids[i])
                if distance < min_dis:
                    min_dis = distance
                    flag = i
            if flag not in cluster_dict:
                cluster_dict.setdefault(flag, [])
            cluster_dict[flag].append(item)
        return cluster_dict

    def _calculate_compactness(self, centroids: list, cluster_dict: dict) -> float:
        """Compactness (Sum of squared errors / distances)"""
        sum_cp = 0.0
        for key in cluster_dict.keys():
            vec1 = centroids[key]
            distance = 0.0
            for item in cluster_dict[key]:
                distance += self._dtw_distance(vec1, item)
            sum_cp += distance
        return sum_cp

    def _update_centroids(self, cluster_dict: dict) -> list:
        """Update Cluster Centroids"""
        centroid_list = []
        for key in cluster_dict.keys():
            centroid = np.mean(cluster_dict[key], axis=0)
            centroid_list.append(centroid)
        return centroid_list

    def fit(self, data_set: np.ndarray):
        """Fit Cluster Model"""
        self.centroids = self._init_centroids(data_set)
        self.cluster_dict = self._assign_clusters(data_set, self.centroids)
        
        new_var = self._calculate_compactness(self.centroids, self.cluster_dict)
        old_var = 1.0
        times = 1

        while abs(new_var - old_var) >= self.tol and times <= self.max_iter:
            self.centroids = self._update_centroids(self.cluster_dict)
            self.cluster_dict = self._assign_clusters(data_set, self.centroids)
            old_var = new_var
            new_var = self._calculate_compactness(self.centroids, self.cluster_dict)
            times += 1

    def evaluate_silhouette_score(self) -> float:
        """Silhouette Score"""
        if not self.cluster_dict:
            raise ValueError("Model must be fitted before evaluation.")
            
        labels = []
        X = self.cluster_dict[0]
        for i in range(self.n_clusters):
            temp = np.zeros(len(self.cluster_dict[i]))
            temp[:] = i
            if i == 0:
                labels = temp
            else:
                labels = np.append(labels, temp)
        
        for i in range(1, self.n_clusters):
            X = np.append(X, self.cluster_dict[i], axis=0)

        score = metrics.silhouette_score(X, labels)
        return score

    def cluster_engagement(self, engagement_ts: list) -> str:
        """Obtain the Nearest Cluster Branch as the Engagement Category"""
        if not self.cluster_dict:
            raise ValueError("Model must be fitted before calculating engagement category.")
        
        # engagement time series distance with cluster centroids
        distance_dict = {key: self._dtw_distance(engagement_ts, cluster_centroid) for key, cluster_centroid in self.cluster_dict.items()}
        
        engagement_category = str(min(distance_dict, key=distance_dict.get))

        return engagement_category