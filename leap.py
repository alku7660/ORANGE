"""
The code presented here has mostly been shared by the authors of the paper "Improving the Quality of Explanations with Local Embedding Perturbations"

The full publication is cited:

From Jia, Y., Bailey, J., Ramamohanarao, K., Leckie, C., & Houle, M. E. (2019). 
Improving the Quality of Explanations with Local Embedding Perturbations. 
Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining

We thank the authors for their collaboration sharing the implementation
"""

import numpy as np
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances

def leap_method(perturbator, data_obj, ioi, ioi_label, regul = 0.1, seed_sel = None, robustness_call = False):
    """
    Local Embedding Aided Perturbation: Generates a neighborhood around a prediction.
    """
    def calculate_weights(processed_pert):
        """
        Method to calculate weights (based on LIME)
        """
        kernel_width = np.sqrt(len(perturbator.features))*0.25 # Originally 0.75 * sqrt(number_features)
        d = euclidean_distances(processed_pert, ioi.normal_x.values.reshape(1,-1))
        weights = np.sqrt(np.exp(-(d**2)/kernel_width**2))
        return weights

    if seed_sel is None:
        np.random.seed(perturbator.seed_int)
    else:
        np.random.seed(seed_sel)

    start_time = time.time()
    LID = int(ioi.LID_x)+1 if int(ioi.LID_x)+1 <= len(perturbator.processed_features) else len(perturbator.processed_features)
    neis = np.array([i[0] for i in ioi.x_kNN])
    neis = np.concatenate((ioi.normal_x.values.reshape(1,-1),neis),axis=0)
    pca = PCA(n_components = LID)
    pca_neis = pca.fit_transform(neis)
    pca_x = pca_neis[0]
    pca_scaler = StandardScaler(with_mean=False)
    pca_scaler.fit(pca_neis)
    data = np.random.normal(0,1,perturbator.N*LID).reshape(perturbator.N, LID)
    data = data * pca_scaler.scale_ + pca_x
    data = pca.inverse_transform(data)
    labelized_data = {}
    labelized_neigh = {}
    training_data = data_obj.processed_train_pd.values
    unique_labels = np.unique(data_obj.train_target)
    for label in unique_labels:
        labelized_data[label]=[]
    for ins,label in zip(training_data, data_obj.train_target.values):
        try:
            labelized_data[label[0]].append(ins)
        except:
            labelized_data[label].append(ins)
    for label in unique_labels:
        labelized_neigh[label]=NearestNeighbors(n_neighbors=5)
        labelized_neigh[label].fit(labelized_data[label])
    op_data = []
    for c in unique_labels:
        if c == perturbator.global_model.call_predict(ioi.normal_x.values.reshape(1,-1)):
            continue
        dist, op_nei = labelized_neigh[c].kneighbors(ioi.normal_x.values.reshape(1,-1))           
        for ins_ind in op_nei[0]:
            #print(self.labelized_data[c][0])
            op_data.append(labelized_data[c][ins_ind])
        #op_data.append((self.labelized_data[c])[op_nei[0,0]])
    new_size = len(op_data)+perturbator.N
    op_data = np.array(op_data)
    new_data = np.zeros((new_size, data[0].size))
    new_data[:perturbator.N,:] = data
    new_data[perturbator.N:,:] = op_data
    processed_pert = new_data
    generate_neighbors_time = time.time()
    weights = calculate_weights(processed_pert)
    weighting_time = time.time()
    total_time = weighting_time - start_time

    # print(f'LEAP  Step 1: Neighbors generation         : {np.round(generate_neighbors_time - start_time, 4)} (s) / ({np.round((generate_neighbors_time - start_time)*100/total_time, 2)}%)')
    # print(f'LEAP  Step 2: Weighting                    : {np.round(weighting_time - generate_neighbors_time, 4)} (s) / ({np.round((weighting_time - generate_neighbors_time)*100/total_time, 2)}%)')
    # print(f'LEAP  Total                                : {np.round(total_time, 4)} (s)')

    return processed_pert, processed_pert, weights, total_time