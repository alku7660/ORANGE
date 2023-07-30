"""
Instance of Interest constructor
"""

"""
Imports
"""
import numpy as np
import pandas as pd
from true_explanations_search import minimize_closest_point_circle, expf_circle, plot_values
from true_explanations_search import minimize_closest_point_f2, expf_2, minimize_closest_point_f3, expf_3
from true_explanations_search import minimize_closest_point_diagonal_0, minimize_closest_point_diagonal_1_8, expf_diagonal
from true_explanations_search import minimize_closest_point_diagonal_plane, expf_diagonal_plane
from true_explanations_search import minimize_closest_point_cubic_0, minimize_closest_point_cubic_1_8, expf_cubic_0, expf_cubic_1_8

class IOI:
    """
    Class corresponding to the Instance of Interest
    """
    def __init__(self,idx, x, normal_x, data_obj, k) -> None:
        
        self.idx = idx
        self.x = x
        self.normal_x = normal_x
        self.k = k
        self.x_kNN = self.kNN(data_obj)
        self.LID_x = self.lid_calculation()
        self.true_expl_x, self.closest_x, self.x1_x2 = self.true_explanations(self.x, data_obj.name)

    def kNN(self, data):
        """
        Method that returns the k-NN from the training dataset
        Input data: Data object
        """
        distance, counter_zero = [], 0
        for i in range(data.processed_train_pd.shape[0]):
            dist = np.sqrt(np.sum((data.processed_train_pd.iloc[i,:]-self.normal_x)**2))
            if dist == 0:
                counter_zero += 1
                if counter_zero > int(self.k*0.05):
                    continue
            if not isinstance(data.train_target.iloc[i], pd.Series):
                distance.append((data.processed_train_pd.iloc[i,:].values,data.train_target.iloc[i],dist))
            else:
                distance.append((data.processed_train_pd.iloc[i,:].values,data.train_target.iloc[i].values,dist))
        distance.sort(key=lambda x: x[-1])
        x_kNN = distance[:self.k]
        return x_kNN

    def lid_calculation(self):
        """
        Method that calculates the LID of the IOI
        """
        laplace_add = 0.00001
        sum_val = 0
        distance_x_kNN = [i[-1] for i in self.x_kNN]
        distance_x_kNN_max = distance_x_kNN[-1]
        for i in distance_x_kNN:
            sum_val += np.log((i+laplace_add)/distance_x_kNN_max)
        LID_x = (1-len(distance_x_kNN))/sum_val
        return LID_x

    def true_explanations(self,x,data_str):
        """
        Method that obtains the true explanation for a set of datasets
        Input x: Instance of Interest
        Input data_str: Dataset name
        """
        closest_x, true_expl_x, x1_x2 = None, None, None
        if data_str == '3-cubic':
            closest_x = minimize_closest_point_f2(x)
            closest_x = np.array(closest_x).reshape(1,-1)
            true_expl_x = expf_2(closest_x)
            true_expl_x = true_expl_x.reshape(1, -1)
        elif data_str == 'sinusoid':
            closest_x = minimize_closest_point_f3(x)
            closest_x = np.array(closest_x).reshape(1,-1)
            x1_x2 = plot_values(data_str)
            true_expl_x = expf_3(closest_x)
            true_expl_x = true_expl_x.reshape(1, -1)
        elif data_str == 'synthetic_circle':
            closest_x = minimize_closest_point_circle(x)
            closest_x = np.array(closest_x).reshape(1,-1)
            x1_x2 = plot_values(data_str)
            true_expl_x = expf_circle(closest_x)
            true_expl_x = true_expl_x.reshape(1, -1)
        elif data_str == 'synthetic_diagonal_0':
            closest_x = minimize_closest_point_diagonal_0(x)
            closest_x = np.array(closest_x).reshape(1,-1)
            x1_x2 = plot_values(data_str)
            true_expl_x = expf_diagonal()
            true_expl_x = true_expl_x.reshape(1, -1)
        elif data_str == 'synthetic_diagonal_1_8':
            closest_x = minimize_closest_point_diagonal_1_8(x)
            closest_x = np.array(closest_x).reshape(1,-1)
            x1_x2 = plot_values(data_str)
            true_expl_x = expf_diagonal()
            true_expl_x = true_expl_x.reshape(1, -1)
        elif 'synthetic_diagonal_plane' in data_str:
            closest_x = minimize_closest_point_diagonal_plane(x)
            closest_x = np.array(closest_x).reshape(1,-1)
            x1_x2 = plot_values(data_str)
            true_expl_x = expf_diagonal_plane()
            true_expl_x = true_expl_x.reshape(1, -1)
        elif data_str == 'synthetic_cubic_0':
            closest_x = minimize_closest_point_cubic_0(x)
            closest_x = np.array(closest_x).reshape(1,-1)
            x1_x2 = plot_values(data_str)
            true_expl_x = expf_cubic_0(closest_x)
            true_expl_x = true_expl_x.reshape(1, -1)
        elif data_str == 'synthetic_cubic_1_8':
            closest_x = minimize_closest_point_cubic_1_8(x)
            closest_x = np.array(closest_x).reshape(1,-1)
            x1_x2 = plot_values(data_str)
            true_expl_x = expf_cubic_1_8(closest_x)
            true_expl_x = true_expl_x.reshape(1, -1)
        return true_expl_x, closest_x, x1_x2