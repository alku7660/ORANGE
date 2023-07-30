from growingspheres import GrowingSpheres
import numpy as np
import time
from sklearn.linear_model import RidgeClassifier
from itertools import product
from itertools import chain
from scipy.spatial import distance_matrix
import copy

def orange_method(perturbator, data, ioi, ioi_label, regul = 0.1, seed_sel = None, opposite_weighting=True, boost_weighting=True, robustness_call = False):
    """
    Method that creates perturbations based on the orange perturbation method: (1) Feasible close-to-boundary search 
                                                                              (2) Opposite label weighting 
                                                                              (3) Disagreement boost
    """        
    def preliminary_adjust(points):
        """
        Method that verifies whether points are close to a valid value
        """
        points[np.where(points > 1)] = 1
        points[np.where(points < 0)] = 0
        return points

    def recursive_combinations_feat(all_comb, all_feat_possible_value, feat_pointer):
        """
        Method that creates all combinations for each pair of possible arrays in the found possible values for an instance
        """
        if len(all_comb) == 0:
            first_group = all_feat_possible_value[0]
        else:
            first_group = all_comb
        second_group = all_feat_possible_value[feat_pointer]
        list_feat_pointer = list(product(first_group, second_group))
        for i in range(len(list_feat_pointer)):
            list_feat_pointer[i] = list(list_feat_pointer[i])
            for j in range(len(list_feat_pointer[i])):
                if type(list_feat_pointer[i][j]) != list:
                    list_feat_pointer[i][j] = [float(list_feat_pointer[i][j])]
            list_feat_pointer[i] = list(chain(*list_feat_pointer[i]))
        all_comb = list_feat_pointer.copy()
        feat_pointer += 1
        if feat_pointer == len(all_feat_possible_value):
            return all_comb
        else:                   
            all_comb = recursive_combinations_feat(all_comb,all_feat_possible_value,feat_pointer)
        return all_comb

    def generate_instances_around_cf(closest_cf, data):
        """
        Method that generates the neighbors around the closest of the found instances on the adjusted polyhedron 
        """
        covar = np.cov(data.processed_train_pd.values, rowvar=False)
        factor = 0.01
        found_all_classes = False
        while not found_all_classes:
            all_pert_processed = np.random.multivariate_normal(mean = closest_cf, cov = covar*factor, size = perturbator.N)
            target_pert_processed = perturbator.global_model.call_predict(all_pert_processed)
            found_all_classes = len(list(np.unique(target_pert_processed))) == 2
            factor += 0.01
            # print(f'Factor: {np.round(factor,2)}')
        return all_pert_processed, target_pert_processed

    def opposite_class_nn_weight_calculation(all_pert_processed, perturbations_predictions):
        """
        Method that weighs the perturbations based on their closeness to an opposite class labeled synthetic instance
        """
        perturbations_with_pred_0_index = np.where(perturbations_predictions == 0)[0]
        perturbations_with_pred_1_index = np.where(perturbations_predictions == 1)[0]
        perturbations_with_pred_0 = all_pert_processed[perturbations_with_pred_0_index]
        perturbations_with_pred_1 = all_pert_processed[perturbations_with_pred_1_index]
        distance_mat = 1/distance_matrix(perturbations_with_pred_0, perturbations_with_pred_1, p = 1)
        distance_mat = (distance_mat - np.min(distance_mat))/(np.max(distance_mat) - np.min(distance_mat))
        pred_0_weight = np.sum(distance_mat, axis=1)
        pred_1_weight = np.sum(distance_mat, axis=0)
        tuple_pred_0_weight = list(zip(perturbations_with_pred_0_index, pred_0_weight))
        tuple_pred_1_weight = list(zip(perturbations_with_pred_1_index, pred_1_weight))
        tuple_pred_all_weight = tuple_pred_0_weight+tuple_pred_1_weight
        tuple_pred_all_weight.sort(key=lambda x: x[0])
        weights = [x[1] for x in tuple_pred_all_weight]
        weights = np.array(weights)
        if np.sum(weights) == 0:
            print(f'Warning: all weights equal to zero in opposite class nn. Setting equal weight to all.')
            weights = np.ones(weights.shape)
        return weights

    def pert_disagreement_calculation(all_pert_processed, target_pert_processed, weights):
        """
        Method that calculates the prediction probabilities disagreement between linear and local model
        """
        lin_model = RidgeClassifier(alpha=regul)
        lin_model.fit(all_pert_processed, target_pert_processed, sample_weight=weights)
        pert_dist_lin_decision = lin_model.decision_function(all_pert_processed)
        pert_dist_lin_decision = [[0.5*(1-i),0.5*(1+i)] for i in pert_dist_lin_decision]
        pert_lin_proba = np.array([np.exp(i) / np.sum(np.exp(i)) for i in pert_dist_lin_decision])
        if perturbator.global_model_type == 'nonlinear':
            pert_global_proba = perturbator.global_model.call_predict_proba(all_pert_processed)
        else:
            pert_dist_global_decision = perturbator.global_model.call_decision_function(all_pert_processed)
            pert_dist_global_decision = [[0.5*(1-i), 0.5*(1+i)] for i in pert_dist_global_decision]
            pert_global_proba = np.array([np.exp(i) / np.sum(np.exp(i)) for i in pert_dist_global_decision])
        fidelity = lin_model.predict(all_pert_processed) == perturbator.global_model.call_predict(all_pert_processed)
        pert_disagreement = np.sqrt(np.sum((pert_lin_proba-pert_global_proba)**2,axis=1))
        return pert_disagreement, fidelity

    def find_close_cfs(first_rad = 0.1, decrease_rad = 10, amount_cfs = 10):
        """
        Method to find a valid CF
        """
        center = ioi.normal_x.values.reshape(1, -1)
        sphere_cf_bool = False
        i = 0
        while not sphere_cf_bool:
            print(f'orange Iteration {i}: Trying to find CF through GS')
            growing_sphere = GrowingSpheres(center, perturbator.global_model.call_predict, target_class = 1 - ioi_label, n_in_layer = int(1000*np.sqrt(len(perturbator.features))), layer_shape = 'ring',
                                            first_radius = first_rad, decrease_radius = decrease_rad, several = True, verbose = True, amount = amount_cfs)
            raw_cfs = np.array(growing_sphere.find_counterfactual())
            pre_cfs = preliminary_adjust(raw_cfs)
            if pre_cfs.shape[0] > 0:
                sphere_cf_bool = True
            i += 1
            random_seed_int = np.random.randint(10000,99999)
            np.random.seed(random_seed_int)
        return pre_cfs, sphere_cf_bool

    def sphere():
        """
        Method that outputs the array of all instances on the optimal polyhedron found
        """
        sphere_cfs_bool = False
        counts = 0
        first_rad, decrease_rad,  = 0.1, 10
        if perturbator.name in ['sinusoid','synthetic_circle','synthetic_cubic_0','synthetic_cubic_1_8','synthetic_diagonal_0','synthetic_diagonal_1_8','synthetic_diagonal_plane']:
            amount_cfs = int(np.round(np.log(len(perturbator.processed_features))))*10
        else:
            amount_cfs = int(np.round(np.log(len(perturbator.processed_features))))
        while not sphere_cfs_bool:
            if counts >= 3:
                print(f'Increasing amound required by {10} to {amount_cfs + 10}')
                amount_cfs += 10
            sphere_cfs, sphere_cfs_bool = find_close_cfs(first_rad, decrease_rad, amount_cfs)
            counts += 1
        return sphere_cfs

    def generate_neighbors(sphere_cf_array, data):
        """
        Method that generates the neighbors around the closest of the found instances on the adjusted polyhedron 
        """
        if sphere_cf_array.shape[0] > 0:
            min_distance_to_cf_array_idx = np.argmin(np.sum((sphere_cf_array - ioi.normal_x.values)**2,axis=1))
            nearest_polyhedron_cf_to_x = sphere_cf_array[min_distance_to_cf_array_idx]
            all_pert_processed, target_pert_processed = generate_instances_around_cf(nearest_polyhedron_cf_to_x, data)
        else:
            print(f'No CF instance found! Process failed.')
        return all_pert_processed, target_pert_processed
    
    def disagreement_boost_weight_calculation(all_pert_processed, target_pert_processed, opposite_class_nn_weight):
        """
        Method that implements an additional weighting process based on boosting
        """
        critical_change = 0.01
        marginal_weight_change = perturbator.N
        old_weights_sum = perturbator.N
        best_fidelity_ratio = 0
        no_change_counter = 0
        weights = np.ones((all_pert_processed.shape[0],))
        iter = 0
        iter_list = []
        weights_list = []
        fidelity_ratio_list = []
        while marginal_weight_change >= critical_change and no_change_counter < 5 and iter < 100: 
            disagreement, fidelity = pert_disagreement_calculation(all_pert_processed,target_pert_processed,weights)
            fidelity_ratio = np.sum(fidelity)/len(fidelity)
            if fidelity_ratio > best_fidelity_ratio:
                best_weights = weights
                best_fidelity_ratio = fidelity_ratio
            weights = opposite_class_nn_weight + disagreement
            weights_sum = np.sum(weights)
            old_marginal_weight_change = np.copy(marginal_weight_change)
            marginal_weight_change = np.abs(weights_sum - old_weights_sum)
            if np.abs(marginal_weight_change - old_marginal_weight_change) < 0.1:
                no_change_counter += 1
            old_weights_sum = weights_sum
            iter_list.append(iter)
            weights_list.append(best_weights)
            fidelity_ratio_list.append(best_fidelity_ratio)
            if best_fidelity_ratio == 1:
                break
            iter += 1
        return best_weights
    
    def bin_feat_range(processed_feat_j):
        """
        Obtains the range of values for binary features
        """
        copy_normal_x = copy.deepcopy(ioi.normal_x)
        value = copy_normal_x[processed_feat_j]
        list_values = [value]
        new_value = 1 if value == 0 else 0
        copy_normal_x[processed_feat_j] = new_value
        if perturbator.global_model.call_predict(copy_normal_x.values.reshape(1,-1)) != ioi_label:
            list_values.append(new_value)
        return list_values

    def cat_feat_range(processed_feat_j):
        """
        Obtains the range of values for categorical features
        """        
        copy_normal_x = copy.deepcopy(ioi.normal_x)
        cat_feat_j_idx = perturbator.idx_cat_cols_dict[processed_feat_j[:-2]]
        list_j = [0]*len(cat_feat_j_idx)
        list_draw_j = [list(copy_normal_x.iloc[cat_feat_j_idx].values)]
        for feat_idx in range(len(cat_feat_j_idx)):
            list_j_comb = list_j.copy()
            list_j_comb[feat_idx] = 1
            copy_normal_x.iloc[cat_feat_j_idx] = list_j_comb
            if perturbator.global_model.call_predict(copy_normal_x.values.reshape(1,-1)) != ioi_label:
                list_draw_j.append(list_j_comb)
        return list_draw_j, cat_feat_j_idx

    def ord_cont_feat_range(processed_feat_j):
        """
        Obtains the range of values for continuous features
        """
        copy_normal_x = copy.deepcopy(ioi.normal_x)
        feat_step_value = data.feat_step[processed_feat_j]
        upper_value, lower_value = ioi.normal_x[processed_feat_j], ioi.normal_x[processed_feat_j]
        update_upper_value, update_lower_value = ioi.normal_x[processed_feat_j], ioi.normal_x[processed_feat_j]
        lower_found_j, upper_found_j = False, False
        while update_lower_value >= 0 and not lower_found_j:
            update_lower_value = update_lower_value - feat_step_value
            copy_normal_x[processed_feat_j] = update_lower_value
            if perturbator.global_model.call_predict(copy_normal_x.values.reshape(1,-1)) != ioi_label:
                lower_value = update_lower_value
                lower_found_j = True
        while update_upper_value <= 1 and not upper_found_j:
            update_upper_value = update_upper_value + feat_step_value
            copy_normal_x[processed_feat_j] = update_upper_value
            if perturbator.global_model.call_predict(copy_normal_x.values.reshape(1,-1)) != ioi_label:
                upper_value = update_upper_value
                upper_found_j = True
        list_feat_j = list(np.linspace(start=lower_value, stop=upper_value, num=int((upper_value - lower_value)/feat_step_value + 1)))
        return list_feat_j

    def define_features_search_ranges():
        """
        Obtains the feature search ranges as list of lists
        """
        feat_index = range(len(perturbator.processed_features))
        ranges_list = []
        feat_checked_list = []
        for j in feat_index:
            processed_feat_j = perturbator.processed_features[j]
            if j not in feat_checked_list:
                if data.feat_type[processed_feat_j] == 'bin':
                    list_feat_j = bin_feat_range(processed_feat_j)
                    checked_feat = [j]
                elif data.feat_type[processed_feat_j] == 'cat':
                    list_feat_j, checked_feat = cat_feat_range(processed_feat_j)
                elif data.feat_type[processed_feat_j] == 'ord' or data.feat_type[processed_feat_j] == 'cont':
                    list_feat_j = ord_cont_feat_range(processed_feat_j)
                    checked_feat = [j]
                feat_checked_list.extend(checked_feat)
                ranges_list.append(list_feat_j)
        return ranges_list
    
    def find_all_instances_polyhedron(features_search_ranges):
        """
        Obtains the instances based on the feature search ranges
        """
        polyhedron = []
        feat_pointer = 1
        polyhedron = recursive_combinations_feat(polyhedron, features_search_ranges, feat_pointer)
        polyhedron_array = np.array(polyhedron)
        polyhedron_labels = perturbator.global_model.call_predict(polyhedron_array)
        polyhedron_array = polyhedron_array[polyhedron_labels != ioi_label]
        if len(polyhedron_array) == 0:
            polyhedron_array = np.array(polyhedron)
        return polyhedron_array

    if seed_sel is None:
        np.random.seed(perturbator.seed_int)
    else:
        np.random.seed(seed_sel)

    start_time = time.time()
    if len(np.unique(data.feat_type.values)) == 1 and data.feat_type.values[0] == 'cont':
        sphere_cfs = sphere()
        sphere_time = time.time()
        all_pert_processed, target_pert_processed = generate_neighbors(sphere_cfs, data)
    else:
        start_time = time.time()
        features_search_ranges = define_features_search_ranges()
        polyhedron_instances = find_all_instances_polyhedron(features_search_ranges)
        sphere_time = time.time()
        all_pert_processed, target_pert_processed = generate_neighbors(polyhedron_instances, data)
    generate_neighbors_time = time.time()
    if opposite_weighting:
        opposite_class_nn_weight = opposite_class_nn_weight_calculation(all_pert_processed, target_pert_processed)
    else:
        opposite_class_nn_weight = np.ones((all_pert_processed.shape[0],))
    disagreement, fidelity_normal = pert_disagreement_calculation(all_pert_processed,target_pert_processed,np.ones((all_pert_processed.shape[0],)))
    fidelity_normal_ratio = np.sum(fidelity_normal)/len(fidelity_normal)
    disagreement, fidelity_opposite = pert_disagreement_calculation(all_pert_processed,target_pert_processed,opposite_class_nn_weight)
    fidelity_opposite_ratio = np.sum(fidelity_opposite)/len(fidelity_opposite)
    if fidelity_normal_ratio < fidelity_opposite_ratio:
        best_weights = opposite_class_nn_weight
        best_fidelity_ratio = fidelity_opposite_ratio
    else:
        best_weights = np.ones((all_pert_processed.shape[0],))
        best_fidelity_ratio = fidelity_normal_ratio
    opposite_class_nn_weight_time = time.time()
    if boost_weighting:
        disagreement_weights = disagreement_boost_weight_calculation(all_pert_processed, target_pert_processed, best_weights)
    else:
        best_weights = opposite_class_nn_weight
    disagreement, fidelity_boosting = pert_disagreement_calculation(all_pert_processed, target_pert_processed, disagreement_weights)
    fidelity_boosting_ratio = np.sum(fidelity_boosting)/len(fidelity_boosting)
    if best_fidelity_ratio < fidelity_boosting_ratio:
        best_weights = disagreement_weights
    else:
        best_weights = best_weights
    if not opposite_weighting and not boost_weighting:
        best_weights = np.ones((all_pert_processed.shape[0],))
    else:
        best_weights = best_weights
    disagreement_boost_weight_calculation_time = time.time()
    total_time = disagreement_boost_weight_calculation_time - start_time

    return all_pert_processed, all_pert_processed, best_weights, total_time