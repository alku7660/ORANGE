"""
Perturbator constructor
"""

"""
Imports
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import lime
import leap
import ls
import lemon
import orange

class Perturbator:
    """
    Class that creates a perturbator module
    """ 

    def __init__(self, pert_type, N, data_obj,
                 global_model, model_type, seed, regul = 0.1) -> None:

        self.name = data_obj.name
        self.seed_int = seed
        self.type = pert_type
        self.N = N
        self.feature_dist, self.processed_feat_dist = data_obj.feature_dist, data_obj.processed_feat_dist
        self.class_values = np.unique(data_obj.train_target)
        self.features = data_obj.train_pd.columns.to_list()
        self.binary, self.categorical, self.ordinal, self.continuous = data_obj.binary, data_obj.categorical, data_obj.ordinal, data_obj.continuous
        self.bin_encoder, self.cat_encoder, self.ord_scaler, self.con_scaler = data_obj.bin_encoder, data_obj.cat_encoder, data_obj.ord_scaler, data_obj.con_scaler
        self.global_model = global_model
        self.global_model_type = model_type
        self.regul = regul
        self.bin_enc_cols = data_obj.bin_enc_cols
        self.cat_enc_cols = data_obj.cat_enc_cols
        self.features = data_obj.features
        self.processed_features = data_obj.processed_features
        self.processed_feat_dist = data_obj.processed_feat_dist
        self.idx_cat_cols_dict = self.idx_cat_columns()
        self.idx_cont_cols = self.idx_cont_columns()
        self.ioi_dict = {} 
    
    def add_ioi(self, ioi_obj, data):
        processed_x1_x2 = self.perturbation_processing(ioi_obj.x1_x2) if ioi_obj.x1_x2 is not None else None
        processed_closest_x = self.perturbation_processing(ioi_obj.closest_x) if ioi_obj.closest_x is not None else None
        ioi_label = self.global_model.call_predict(ioi_obj.normal_x.values.reshape(1,-1))
        perturbations, processed_perturbations, pert_weights, exec_time = self.perturbation_function(ioi_obj, data, ioi_label)
        perturbations_label = self.perturbation_target(processed_perturbations)
        self.ioi_dict[ioi_obj.idx] = {'ioi':ioi_obj, 'proc_x1_x2':processed_x1_x2, 'proc_closest_x':processed_closest_x, 
                                      'label':ioi_label, 'perturbations':perturbations, 'proc_perturbations':processed_perturbations,
                                      'perturbations_weights':pert_weights, 'perturbations_label':perturbations_label, 'time': exec_time}

    def idx_cat_columns(self):
        """
        Method that obtains the indices of the columns of the categorical variables 
        """
        feat_index = range(len(self.processed_features))
        dict_idx_cat = {}
        for i in self.cat_enc_cols:
            if i[:-2] not in list(dict_idx_cat.keys()): 
                cat_cols_idx = [s for s in feat_index if i[:-3] in self.processed_features[s]]
                dict_idx_cat[i[:-2]] = cat_cols_idx
        return dict_idx_cat

    def idx_cont_columns(self):
        """
        Method that obtains the indices of the columns of the continuous variables 
        """
        feat_index = range(len(self.processed_features))
        idx_cont_list = []
        for i in feat_index:
            processed_feat_i = self.processed_features[i]
            if processed_feat_i in self.continuous:
                idx_cont_list.append(i)
        return idx_cont_list

    def perturbation_function(self, ioi, data, ioi_label, seed_sel = None, robustness_call = False):
        """ ['lime','leap','rls','ls','lemon','bosse']
        Method to call the different perturbation methods depending on the type
        """
        if self.type == 'lime':
            pert, processed_pert, weights, exec_time = lime.lime_method(self, ioi, ioi_label, seed_sel = seed_sel, robustness_call = robustness_call)
        elif self.type == 'leap':
            pert, processed_pert, weights, exec_time = leap.leap_method(self, data, ioi, ioi_label, seed_sel = seed_sel, robustness_call = robustness_call)
        elif self.type == 'ls':
            pert, processed_pert, weights, exec_time = ls.ls_method(self, data, ioi, ioi_label, seed_sel = seed_sel, robustness_call = robustness_call)
        elif self.type == 'lemon':
            pert, processed_pert, weights, exec_time = lemon.lemon_method(self, data, ioi)
        elif self.type == 'orange':
            pert, processed_pert, weights, exec_time = orange.orange_method(self, data, ioi, ioi_label, seed_sel = seed_sel, robustness_call = robustness_call)
        return pert, processed_pert, weights, exec_time

    def perturbation_processing(self,perturbations):
        """
        Method that encodes and scales the perturbations obtained
        """
        perturbations_pd = pd.DataFrame(perturbations,columns=self.features)
        processed_perturbations_pd = pd.DataFrame()
        bin_perturbations, cat_perturbations, ord_perturbations, con_perturbations = perturbations_pd[self.binary], perturbations_pd[self.categorical], perturbations_pd[self.ordinal], perturbations_pd[self.continuous]
        if bin_perturbations.shape[1] > 0:
            bin_enc_perturbations = self.bin_encoder.transform(bin_perturbations).toarray()
            bin_enc_cols = self.bin_encoder.get_feature_names_out(self.binary)
            bin_enc_perturbations_pd = pd.DataFrame(bin_enc_perturbations,index=bin_perturbations.index,columns=bin_enc_cols)
            processed_perturbations_pd = pd.concat((processed_perturbations_pd,bin_enc_perturbations_pd),axis=1)
        if cat_perturbations.shape[1] > 0:
            cat_enc_perturbations = self.cat_encoder.transform(cat_perturbations).toarray()
            cat_enc_cols = self.cat_encoder.get_feature_names_out(self.categorical)
            cat_enc_perturbations_pd = pd.DataFrame(cat_enc_perturbations,index=cat_perturbations.index,columns=cat_enc_cols)
            processed_perturbations_pd = pd.concat((processed_perturbations_pd,cat_enc_perturbations_pd),axis=1)
        if ord_perturbations.shape[1] > 0:    
            ord_scaled_perturbations = self.ord_scaler.transform(ord_perturbations)
            ord_scaled_perturbations_pd = pd.DataFrame(ord_scaled_perturbations,index=ord_perturbations.index,columns=self.ordinal)
            processed_perturbations_pd = pd.concat((processed_perturbations_pd,ord_scaled_perturbations_pd),axis=1)
        if con_perturbations.shape[1] > 0:
            con_scaled_perturbations = self.con_scaler.transform(con_perturbations)
            con_scaled_perturbations_pd = pd.DataFrame(con_scaled_perturbations,index=con_perturbations.index,columns=self.continuous)
            processed_perturbations_pd = pd.concat((processed_perturbations_pd,con_scaled_perturbations_pd),axis=1)
        processed_pert = processed_perturbations_pd.values
        return processed_pert

    def perturbation_target(self, processed_perturbations):
        """
        Method that labels all the obtained perturbations
        """
        return self.global_model.call_predict(processed_perturbations)