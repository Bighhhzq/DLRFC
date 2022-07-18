import numpy as np
import pandas as pd
import torch

def calculate_channel_l2_norm(input):
    l2_norm = torch.norm(input, dim=(2,3)).cpu().detach().tolist()
    return l2_norm

def calculate_all_layer_filter_index(list_all_P , list_target, cfg = None):
    all_layer_mixent_index = []
    for i in range(49):
        if i in [9,21,39,48]:
            if i == 9:
                each_column_ent1=calculate_single_layer_aal_ent(list_all_P[i] , list_target)
                each_column_ent2=calculate_single_layer_aal_ent(list_all_P[i-3] , list_target)
                each_column_ent3=calculate_single_layer_aal_ent(list_all_P[i-6] , list_target)
                each_column_ent4=calculate_single_layer_aal_ent(list_all_P[-4] , list_target)
                each_column_ent = np.array(each_column_ent4)+np.array(each_column_ent3)+np.array(each_column_ent2)+np.array(each_column_ent1)
            if i == 21:
                each_column_ent1=calculate_single_layer_aal_ent(list_all_P[i] , list_target)
                each_column_ent2=calculate_single_layer_aal_ent(list_all_P[i-3] , list_target)
                each_column_ent3=calculate_single_layer_aal_ent(list_all_P[i-6] , list_target)
                each_column_ent4=calculate_single_layer_aal_ent(list_all_P[i-9] , list_target)
                each_column_ent5=calculate_single_layer_aal_ent(list_all_P[-3] , list_target)
                each_column_ent = np.array(each_column_ent5) + np.array(each_column_ent4)+np.array(each_column_ent3)+np.array(each_column_ent2)+np.array(each_column_ent1)
            if i == 39:
                each_column_ent1=calculate_single_layer_aal_ent(list_all_P[i] , list_target)
                each_column_ent2=calculate_single_layer_aal_ent(list_all_P[i-3] , list_target)
                each_column_ent3=calculate_single_layer_aal_ent(list_all_P[i-6] , list_target)
                each_column_ent4=calculate_single_layer_aal_ent(list_all_P[i-9] , list_target)
                each_column_ent5=calculate_single_layer_aal_ent(list_all_P[i-12] , list_target)
                each_column_ent6=calculate_single_layer_aal_ent(list_all_P[i-15] , list_target)
                each_column_ent7=calculate_single_layer_aal_ent(list_all_P[-2] , list_target)
                each_column_ent = np.array(each_column_ent7) + np.array(each_column_ent6) + np.array(each_column_ent5) + np.array(each_column_ent4)+np.array(each_column_ent3)+np.array(each_column_ent2)+np.array(each_column_ent1)
            if i == 48:
                each_column_ent1 = calculate_single_layer_aal_ent(list_all_P[i], list_target)
                each_column_ent2 = calculate_single_layer_aal_ent(list_all_P[i - 3], list_target)
                each_column_ent3 = calculate_single_layer_aal_ent(list_all_P[i - 6], list_target)
                each_column_ent4 = calculate_single_layer_aal_ent(list_all_P[-1], list_target)
                each_column_ent =  np.array(each_column_ent4) + np.array(each_column_ent3) + np.array(each_column_ent2) + np.array(each_column_ent1)
            each_column_mixent_index = calc_filter_index(each_column_ent.tolist())[:cfg[i]]
            all_layer_mixent_index.append([each_column_mixent_index])
        else:
            each_column_ent=calculate_single_layer_aal_ent(list_all_P[i] , list_target)
            each_column_mixent_index = calc_filter_index(each_column_ent)[:cfg[i]]
            all_layer_mixent_index.append([each_column_mixent_index])
    return all_layer_mixent_index

def calculate_single_layer_aal_ent(P , list_target):
    each_column_ent = []
    for i in range(len(P[0])):
        each_column = [j[i] for j in P]
        max_640_index = np.argsort(each_column)[::-1][:640]
        max_640_label = list(list_target[x] for x in max_640_index)
        each_column_ent.append(calc_ent(max_640_label))
    return each_column_ent

def calc_ent(data):
    prob1 = pd.value_counts(data) / len(data)
    return sum(np.log2(prob1) * prob1 * (-1))

def calc_filter_index(each_column_ent):
    each_column_mixent_index = np.argsort(each_column_ent)[::]
    return each_column_mixent_index