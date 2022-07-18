import torch.nn as nn
import numpy as np
import torch

#prune direct according to the index without makeing a mask
def pruning_modelx(model , newmodel, fliter_index):
    start_mask = torch.ones(3)
    layer_id_in_cfg = 0
    end_mask = fliter_index[layer_id_in_cfg]
    start_mask = np.argwhere(np.asarray(start_mask.cpu().numpy()))
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(end_mask)
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            m1.weight.data = m0.weight.data[idx1.tolist()]
            m1.bias.data = m0.bias.data[idx1.tolist()]
            m1.running_mean = m0.running_mean[idx1.tolist()]
            m1.running_var = m0.running_var[idx1.tolist()]
            layer_id_in_cfg += 1
            start_mask = end_mask
            if layer_id_in_cfg < len(fliter_index):  # do not change in Final FC
                end_mask = fliter_index[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            idx0 = np.squeeze(start_mask)
            idx1 = np.squeeze(end_mask)
            #print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[:, idx0.tolist(), :, :] # pruning depth for match up-layer channles
            w1 = w1[idx1.tolist(), :, :, :]  # pruning filter index
            m1.weight.data = w1
        elif isinstance(m0, nn.Linear):
            if layer_id_in_cfg == len(fliter_index):
                idx0 = np.squeeze(fliter_index[-1])
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                m1.weight.data = m0.weight.data[:, idx0]
                m1.bias.data = m0.bias.data
                layer_id_in_cfg += 1
                continue
        elif isinstance(m0, nn.BatchNorm1d):
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()
    return newmodel
#because all_layer_mixent_index no 'M' layer index so delete 'M'



#cfg = [64, 64,'M',64, 64,'M',128, 128, 128, 'M', 128, 128, 128,'M',128, 128, 424]

skip = [3,6,12,15,18,24,27,30,33,36,42,45]
skip_reduial = [4,13,25,43]


def make_cfg_mask(model , all_layer_mixent_index , cfg = None):
    layer_id = 0
    cfg_mask = []
    redual = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            out_channels = m.weight.data.shape[0]
            if redual in [1,3,5,7,9]:
                if layer_id in skip:
                    cfg_mask.append(torch.ones(out_channels))
                    layer_id += 1
                    redual += 1
                    continue
                if out_channels == cfg[layer_id]:
                    cfg_mask.append(torch.ones(out_channels))
                    layer_id += 1
                    redual += 1
                    continue
                #            weight_copy = m.weight.data.abs().clone()
                #            weight_copy = weight_copy.cpu().numpy()
                #            L1_norm = np.sum(weight_copy, axis=(1, 2, 3))
                #            arg_max = np.argsort(L1_norm)
                arg_max_rev = all_layer_mixent_index[layer_id][0][:cfg[layer_id]]
                #arg_max_rev = np.squeeze(arg_max_rev)
                assert arg_max_rev.size == cfg[layer_id], "size of arg_max_rev not correct"
                mask = torch.zeros(out_channels)
                mask[arg_max_rev.tolist()] = 1
                cfg_mask.append(mask)
                layer_id += 1
                redual += 1
                continue

            if layer_id in skip_reduial:
                redual += 1
                continue

            elif layer_id in skip:
                cfg_mask.append(torch.ones(out_channels))
                layer_id += 1
                continue

            if out_channels == cfg[layer_id]:
                cfg_mask.append(torch.ones(out_channels))
                layer_id += 1
                continue
#            weight_copy = m.weight.data.abs().clone()
#            weight_copy = weight_copy.cpu().numpy()
#            L1_norm = np.sum(weight_copy, axis=(1, 2, 3))
#            arg_max = np.argsort(L1_norm)
            arg_max_rev = all_layer_mixent_index[layer_id][0][:cfg[layer_id]]
            #arg_max_rev = np.squeeze(arg_max_rev)
            assert arg_max_rev.size == cfg[layer_id], "size of arg_max_rev not correct"
            mask = torch.zeros(out_channels)
            mask[arg_max_rev.tolist()] = 1
            cfg_mask.append(mask)
            layer_id += 1
    return cfg_mask


def make_newmodel(resnet56 , newmodel , cfg_mask):
    start_mask = torch.ones(3)
    conv_count = 0
    layer_id_in_cfg = 0
    end_mask = cfg_mask[layer_id_in_cfg]
    redual=0
    for [m0, m1] in zip(resnet56.modules(), newmodel.modules()):
        if isinstance(m0, nn.BatchNorm2d):
            if conv_count in skip_reduial and redual in [1,3,5,7,9]:
                if conv_count in [4]:
                    end_mask_skip_reduail = cfg_mask[9]
                elif conv_count in [13]:
                    end_mask_skip_reduail = cfg_mask[21]
                elif conv_count in [25]:
                    end_mask_skip_reduail = cfg_mask[39]
                elif conv_count in [43]:
                    end_mask_skip_reduail = cfg_mask[48]
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask_skip_reduail.cpu().numpy())))
            else:
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            #print(idx1)
            #print(layer_id_in_cfg)
            #print(m0)
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            if conv_count in skip_reduial and redual in [1,3,5,7,9]:
                layer_id_in_cfg = layer_id_in_cfg
                continue
            else:
                layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                if conv_count in [3,6]:
                    end_mask = cfg_mask[9]
                    continue
                elif conv_count in [12,15,18]:
                    end_mask = cfg_mask[21]
                    continue
                elif conv_count in [24,27,30,33,36]:
                    end_mask = cfg_mask[39]
                    continue
                elif conv_count in [42,45]:
                    end_mask = cfg_mask[48]
                    continue
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]

        elif isinstance(m0, nn.Conv2d):
            if conv_count == 0:
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                w1 = m0.weight.data[idx1.tolist(), :, :, :].clone()  # pruning filter index
                m1.weight.data = w1.clone()
                conv_count += 1
                continue
            if redual in [1,3,5,7,9]:
                conv_count += 1
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                #print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))

                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()  # pruning depth for match up-layer channles
                w1 = w1[idx1.tolist(), :, :, :].clone()  # pruning filter index
                m1.weight.data = w1.clone()
                redual += 1
                continue

            if conv_count in skip_reduial:
                start_mask_skip_reduail = cfg_mask[conv_count-4]
                if conv_count in [4]:
                    end_mask_skip_reduail = cfg_mask[9]
                elif conv_count in [13]:
                    end_mask_skip_reduail = cfg_mask[21]
                elif conv_count in [25]:
                    end_mask_skip_reduail = cfg_mask[39]
                elif conv_count in [43]:
                    end_mask_skip_reduail = cfg_mask[48]
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask_skip_reduail.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask_skip_reduail.cpu().numpy())))
                #print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))

                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()  # pruning depth for match up-layer channles
                w1 = w1[idx1.tolist(), :, :, :].clone()  # pruning filter index
                m1.weight.data = w1.clone()
                redual += 1
                continue

            conv_count += 1
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            #print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))

            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()  # pruning depth for match up-layer channles
            w1 = w1[idx1.tolist(), :, :, :].clone()  # pruning filter index
            m1.weight.data = w1.clone()


        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))

            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()

