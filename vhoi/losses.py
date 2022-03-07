from functools import partial

import torch.nn.functional as F

from pyrutils.torch.losses import multi_task_loss, binary_cross_entropy_loss, budget_loss


def select_loss(model_name: str, model_input_type: str, dataset_name: str, cfg):
    if model_name == '2G-GCN':
        misc = cfg.get('misc', default_value={})
        # Budget
        hb_weight = ob_weight = 0.0
        add_budget_loss = misc.get('budget_loss', {}).get('add', False)
        if add_budget_loss:
            hb_weight = misc.get('budget_loss', {}).get('human_weight', 1.0)
            ob_weight = misc.get('budget_loss', {}).get('object_weight', 1.0)
        if dataset_name == 'cad120':
            weight = [hb_weight, ob_weight]
        else:
            weight = [hb_weight]
        # Segmentation
        hs_weight = os_weight = 0.0
        s_weight = misc.get('segmentation_loss', {}).get('weight', 1.0)
        add_segmentation_loss = misc.get('segmentation_loss', {}).get('add', False)
        input_human_segmentation = misc.get('input_human_segmentation', False)
        if add_segmentation_loss and not input_human_segmentation:
            hs_weight = s_weight
        input_object_segmentation = misc.get('input_object_segmentation', False)
        if add_segmentation_loss and not input_object_segmentation:
            os_weight = s_weight
        if dataset_name == 'cad120':
            weight += [hs_weight, os_weight]
        else:
            weight += [hs_weight]
        if add_segmentation_loss and misc.get('segmentation_loss', {}).get('pretrain', False):
            weight_val = 0.0
        else:
            weight_val = 1.0
        anticipation_loss_weight = misc.get('anticipation_loss_weight', 1.0)
        fl_loss_weight = misc.get('first_level_loss_weight', 0.0)
        if dataset_name == 'cad120':
            weight += [fl_loss_weight] * 4
            weight += [weight_val, anticipation_loss_weight, weight_val, anticipation_loss_weight]
            criterion = partial(multi_task_loss,
                                loss_functions=(budget_loss, budget_loss,
                                                binary_cross_entropy_loss, binary_cross_entropy_loss,
                                                F.nll_loss, F.nll_loss, F.nll_loss, F.nll_loss,
                                                F.nll_loss, F.nll_loss, F.nll_loss, F.nll_loss),
                                weight=weight)
            loss_names = ['B_HS', 'B_OS', 'BCE_HS', 'BCE_OS',
                          'NLL_SAR_F', 'NLL_SAP_F', 'NLL_OAR_F', 'NLL_OAP_F',
                          'NLL_SAR', 'NLL_SAP', 'NLL_OAR', 'NLL_OAP']
        else:
            weight += [fl_loss_weight] * 2
            weight += [weight_val, anticipation_loss_weight]
            criterion = partial(multi_task_loss,
                                loss_functions=(budget_loss, binary_cross_entropy_loss,
                                                F.nll_loss, F.nll_loss,
                                                F.nll_loss, F.nll_loss),
                                weight=weight)
            loss_names = ['B_HS', 'BCE_HS', 'NLL_SAR_F', 'NLL_SAP_F', 'NLL_SAR', 'NLL_SAP']
    elif model_name == 'bimanual_baseline':
        criterion = partial(multi_task_loss, loss_functions=(F.nll_loss,))
        loss_names = ['NLL_SAR']
    elif model_name == 'cad120_baseline':
        criterion = partial(multi_task_loss, loss_functions=(F.nll_loss, F.nll_loss))
        loss_names = ['NLL_SAR', 'NLL_OAR']
    else:
        raise ValueError(f'Unknown model {model_name}')
    return criterion, loss_names


def select_loss_types(model_name: str, dataset_name: str, cfg):
    if model_name == '2G-GCN':
        if dataset_name == 'cad120':
            loss_types = ['budget'] * 2 + ['bce'] * 2 + ['softmax'] * 8
        else:
            loss_types = ['budget', 'bce'] + ['softmax'] * 4
    else:
        raise ValueError(f'Multi-task learning option not implemented for {model_name}')
    return loss_types


def select_loss_learning_mask(model_name: str, dataset_name: str, cfg):
    if model_name == '2G-GCN':
        if dataset_name == 'cad120':
            mask = [False] * 4 + [True] * 8
        else:
            mask = [False] * 2 + [True] * 4
    else:
        raise ValueError(f'Multi-task learning option not implemented for {model_name}')
    return mask


def extract_value(cfg, group, key, default=False):
    try:
        value = cfg[group][key]
    except (KeyError, TypeError):
        value = default
    return value


def decide_num_main_losses(model_name: str, dataset_name: str, misc_dict: dict):
    num_main_losses = None
    if model_name == '2G-GCN':
        add_segmentation_loss = misc_dict.get('segmentation_loss', {}).get('add', False)
        pretrain_segmentation = misc_dict.get('segmentation_loss', {}).get('pretrain', False)
        if add_segmentation_loss and pretrain_segmentation:
            num_main_losses = 10 if dataset_name == 'cad120' else 5
        else:
            num_main_losses = 4 if dataset_name == 'cad120' else 2
    return num_main_losses
