"""
The code presented here is mostly extracted from the GitHub repository: https://github.com/

The full publication is cited:


"""
import numpy as np
import time
from sklearn.metrics.pairwise import euclidean_distances

def lime_method(perturbator, ioi, ioi_label, regul = 0.1, seed_sel = None, robustness_call = False):
    """
    Method that creates random perturbations (base LIME method)
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
    pert = np.random.uniform(size=(perturbator.N,len(perturbator.processed_features)))
    generate_neighbors_time = time.time()
    processed_pert = pert
    weights = calculate_weights(processed_pert)
    weighting_time = time.time()
    total_time = weighting_time - start_time

    # print(f'LIME  Step 1: Neighbors generation         : {np.round(generate_neighbors_time - start_time, 4)} (s) / ({np.round((generate_neighbors_time - start_time)*100/total_time,2)}%)')
    # print(f'LIME  Step 2: Weighting                    : {np.round(weighting_time - generate_neighbors_time, 4)} (s) / ({np.round((weighting_time - generate_neighbors_time)*100/total_time,2)}%)')
    # print(f'LIME  Total                                : {np.round(total_time, 4)} (s)')

    return pert, processed_pert, weights, total_time