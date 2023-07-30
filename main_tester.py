"""
Main tester script
"""
"""
Imports
"""
import numpy as np
import time
from data_constructor import Dataset
from ioi_constructor import IOI
from global_model_constructor import Global_model
from perturbator_constructor import Perturbator
from evaluator_constructor import Evaluator
from address import results_cf_obj_dir, save_obj

# datasets = ['synthetic_circle','synthetic_diagonal_0','synthetic_diagonal_1_8','synthetic_cubic_0','synthetic_cubic_1_8','3-cubic','sinusoid','synthetic_diagonal_plane_1ord_2con','synthetic_diagonal_plane_2ord_1con','synthetic_diagonal_plane_1bin_1ord_1con','synthetic_diagonal_plane_2bin_1con',
#            'adult','bank','compass','credit','dutch','german','ionosphere','kdd_census','law','oulad','student']
# perturbators =['lime','leap','rls','ls','only_poly_bosse','only_oppo_bosse','only_boost_bosse','no_poly_bosse','no_oppo_bosse','no_boost_bosse','bosse']

seed_int = 54321
k = 50
N = 300
regul = 0.1
x_iterations = 10
step = 0.01
number_tests_include = 30
global_model_type = 'nonlinear'                               
datasets = ['synthetic_diagonal_plane_1ord_2con','synthetic_diagonal_plane_2ord_1con','synthetic_diagonal_plane_1bin_1ord_1con','synthetic_diagonal_plane_2bin_1con'] # 'adult','bank','compass','credit','dutch','german','ionosphere','kdd_census','law','oulad','student'
perturbators = ['orange']  #'lime','leap','rls','ls','lemon','bosse','new_bosse','new_bosse_no_oppo','new_bosse_no_boost','new_bosse_no_oppo_no_boost','new_bosse_new_boost'                               

if __name__ == "__main__":
    for i in datasets:
        data_i = Dataset(i, 0.7, seed_int, step)
        global_model = Global_model(global_model_type,data_i)
        data_test_pd_range = range(number_tests_include)
        for j in perturbators:
            perturbator_start_time = time.time()
            perturbator_j = Perturbator(j, N, data_i, global_model, global_model_type, seed_int, regul)
            for idx in data_test_pd_range:
                # idx = 23
                idx_start_time = time.time()
                x = data_i.test_pd.iloc[idx,:]
                x_normal = data_i.processed_test_pd.iloc[idx,:]
                ioi = IOI(idx, x, x_normal, data_i, k)
                perturbator_j.add_ioi(ioi, data_i)
                idx_end_time = time.time()
                # print('----------------------------------')
                print(f'Dataset: {i}, Perturbator: {j}, Index: {idx} executed: {np.round(idx_end_time - idx_start_time,2)} (s)')
            perturbator_end_time = time.time()
            print(f'Dataset: {i}, Perturbator: {j} *All instances* executed: {np.round(perturbator_end_time - perturbator_start_time,2)} (s)')
            eval = Evaluator(data_i, perturbator_j, x_iterations)
            save_obj(eval, results_cf_obj_dir, f'{i}_{j}_eval.pkl')