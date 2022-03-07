import argparse
from collections import defaultdict
import json
import os

import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import classification_report, precision_recall_fscore_support
import torch

from pyrutils.metrics import f1_at_k, f1_at_k_single_example
from pyrutils.utils import read_dictionary, cleanup_directory
from vhoi.data_loading import load_testing_data, select_model_data_feeder, select_model_data_fetcher
from vhoi.data_loading import determine_num_classes
from vhoi.losses import extract_value, decide_num_main_losses
from vhoi.models import select_model
from vhoi.visualisation import plot_segmentation


def predict(model_dir, frame_to_segment_level=False, inspect_model=False):
    torch.manual_seed(42)
    hydra_configs_dir = os.path.join(model_dir, '.hydra')
    cfg = OmegaConf.load(os.path.join(hydra_configs_dir, 'config.yaml'))
    num_threads = cfg.get('resources', default_value={}).get('num_threads', 4)
    torch.set_num_threads(num_threads)
    # Data
    model_name, model_input_type = cfg.metadata.model_name, cfg.metadata.input_type
    checkpoint_file = os.path.join(model_dir, os.path.basename(model_dir) + '.tar')
    use_gpu = extract_value(cfg, group='resources', key='use_gpu', default=True)
    device = 'cuda' if torch.cuda.is_available() and use_gpu else 'cpu'
    checkpoint = torch.load(checkpoint_file, map_location=device)
    scalers = checkpoint.get('scalers', None)
    test_loader, data_info, segmentations, test_ids = load_testing_data(cfg.data, model_name, model_input_type,
                                                                        batch_size=128, scalers=scalers)
    # Load model
    Model = select_model(model_name)
    model_creation_args = cfg.parameters
    model_creation_args = {**data_info, **model_creation_args}
    dataset_name = cfg.data.get('name', default_value='cad120')
    num_classes = determine_num_classes(model_name, model_input_type, dataset_name)
    model_creation_args['num_classes'] = num_classes
    model = Model(**model_creation_args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    # Predict
    misc_dict = cfg.get('misc', default_value={})
    fetch_model_data = select_model_data_fetcher(model_name, model_input_type,
                                                 dataset_name=dataset_name, **{**misc_dict, **cfg.parameters})
    feed_model_data = select_model_data_feeder(model_name, model_input_type, dataset_name=dataset_name,
                                               **{**misc_dict, 'inspect_model': inspect_model})
    num_main_losses = decide_num_main_losses(model_name, dataset_name, {**misc_dict, **cfg.parameters})
    downsampling = cfg.data.get('downsampling', default_value=1)
    outputs, targets, attentions = [], [], []
    with torch.no_grad():
        for dataset in test_loader:
            data, target = fetch_model_data(dataset, device=device)
            output = feed_model_data(model, data)
            if inspect_model:
                output, attention_scores = output
                attention_scores = [att_score[:, 0] for att_score in attention_scores]
            if num_main_losses is not None:
                output = output[-num_main_losses:]
                target = target[-num_main_losses:]
            if downsampling > 1:
                for i, (out, tgt) in enumerate(zip(output, target)):
                    if out.ndim != 4:
                        raise RuntimeError(f'Number of dimensions for output is {out.ndim}')
                    out = torch.repeat_interleave(out, repeats=downsampling, dim=-2)
                    out = match_shape(out, tgt)
                    output[i] = out
                if inspect_model:
                    a_target = target[0]
                    attention_scores = [torch.repeat_interleave(att_score, repeats=downsampling, dim=-2)
                                        for att_score in attention_scores]
                    attention_scores = [match_att_shape(att_score, a_target) for att_score in attention_scores]
                    attentions.append(attention_scores)
            outputs.append(output)
            targets.append(target)
    if dataset_name == 'bimanual':
        with open(cfg.data.video_id_to_video_fps, mode='r') as f:
            video_id_to_video_fps = json.load(f)
        outputs, targets = downsample_bad_bimanual_videos(outputs, targets, test_ids, video_id_to_video_fps)
    index_to_name = select_index_to_name_mapping(model_name, model_input_type, dataset_name)
    is_safe_to_summarize_frames_into_segments = \
        misc_dict.get('input_human_segmentation', False) and misc_dict.get('input_object_segmentation', False)
    if frame_to_segment_level and is_safe_to_summarize_frames_into_segments:
        outputs = summarize_frames_into_segments(outputs, segmentations, is_ground_truth=False)
        targets = summarize_frames_into_segments(targets, segmentations, is_ground_truth=True)
    outputs = process_output(outputs, is_ground_truth=False, index_to_name=index_to_name)
    targets = process_output(targets, is_ground_truth=True, index_to_name=index_to_name)
    attentions = process_attentions(attentions)
    return outputs, targets, test_ids


def match_shape(out, tgt):
    out_shape, tgt_shape = out.size(), tgt.size()
    out_dim = out.ndim
    if out_dim == 3:
        out_steps, tgt_steps = out_shape[-1], tgt_shape[-1]
        if out_steps >= tgt_steps:
            return out[..., :tgt_steps]
        else:
            diff_steps = tgt_steps - out_steps
            padding = out[..., -1:]
            out = torch.cat([out, torch.repeat_interleave(padding, diff_steps, dim=-1)], dim=-1)
            return out
    elif out_dim == 4:
        out_steps, tgt_steps = out_shape[-2], tgt_shape[-2]
        if out_steps >= tgt_steps:
            return out[:, :, :tgt_steps]
        else:
            diff_steps = tgt_steps - out_steps
            padding = out[:, :, -1:]
            out = torch.cat([out, torch.repeat_interleave(padding, diff_steps, dim=-2)], dim=-2)
            return out
    return out


def match_att_shape(att_score, a_target):
    att_steps, tgt_steps = att_score.size(1), a_target.size(1)
    if att_steps >= tgt_steps:
        return att_score[..., :tgt_steps]
    else:
        diff_steps = tgt_steps - att_steps
        padding = att_score[:, -1:, :]
        out = torch.cat([att_score, torch.repeat_interleave(padding, diff_steps, dim=1)], dim=1)
        return out


def process_attentions(attentions):
    if attentions:
        return attentions[0]
    return attentions  # Quick hack for now


def downsample_bad_bimanual_videos(outputs, targets, test_ids, video_id_to_video_fps):
    for video_index, video_id in enumerate(test_ids):
        video_fps = video_id_to_video_fps[video_id]
        if video_fps != 15:
            continue
        for output, target in zip(outputs, targets):
            for out, tar in zip(output, target):
                y_pred, y_true = out[video_index], tar[video_index]
                original_len = len(y_true)
                y_pred, y_true = y_pred[:, 1::2, :], y_true[1::2, :]
                new_len = len(y_true)
                diff_len = original_len - new_len
                rubbish_values = torch.full([y_pred.size(0), diff_len, y_pred.size(2)], fill_value=-100.0,
                                            dtype=y_pred.dtype, device=y_pred.device)
                y_pred = torch.cat([y_pred, rubbish_values], dim=1)
                out[video_index] = y_pred
                negative_ones = torch.full([diff_len, y_true.shape[1]], fill_value=-1, dtype=y_true.dtype,
                                           device=y_true.device)
                y_true = torch.cat([y_true, negative_ones], dim=0)
                tar[video_index] = y_true
    return outputs, targets


def summarize_frames_into_segments(labels, segmentations, is_ground_truth):
    max_pad_length = max(len(segmentation) for segmentation in segmentations)
    device = labels[0][0].device
    segmentations = [torch.tensor([segment[0] for segment in segmentation], device=device)
                     for segmentation in segmentations]
    dim_offset = 1 if is_ground_truth else 0
    summarized_labels = []
    for label in labels:
        summarized_labels.append([])
        for tensor in label:
            summarized_tensors = [torch.index_select(tensor_slice, dim=1 - dim_offset, index=slice_segmentation)
                                  for tensor_slice, slice_segmentation in zip(tensor, segmentations)]
            summarized_tensors_ = []
            for summarized_tensor in summarized_tensors:
                pad_length = max_pad_length - summarized_tensor.size(1 - dim_offset)
                if summarized_tensor.ndim > 2 - dim_offset:
                    summarized_tensor = torch.transpose(summarized_tensor, 1 - dim_offset, 2 - dim_offset)
                summarized_tensor = torch.nn.functional.pad(summarized_tensor, [0, pad_length],
                                                            mode='constant', value=-1.0)
                if summarized_tensor.ndim > 2 - dim_offset:
                    summarized_tensor = torch.transpose(summarized_tensor, 1 - dim_offset, 2 - dim_offset)
                summarized_tensors_.append(summarized_tensor)
            summarized_tensors = torch.stack(summarized_tensors_, dim=0)
            summarized_labels[-1].append(summarized_tensors)
    return summarized_labels


def process_output(outputs, is_ground_truth=False, index_to_name=None):
    outputs = [[tensor.cpu().numpy() for tensor in output] for output in outputs]
    index_to_tensors = defaultdict(list)
    for output in outputs:
        for i, tensor in enumerate(output):
            if index_to_name is not None:
                index = index_to_name[i]
            else:
                index = i
            index_to_tensors[index].append(tensor)
    index_to_processed_tensors = {}
    for index, tensors in index_to_tensors.items():
        tensors = np.concatenate(tensors, axis=0)
        if not is_ground_truth:
            tensors = np.argmax(tensors, axis=1)
        index_to_processed_tensors[index] = tensors
    return index_to_processed_tensors


def evaluate_predictions(targets, outputs, print_report=True, subactivity_names=None, affordance_names=None):
    results = {}
    for index, target in sorted(targets.items()):
        output = outputs[index].reshape(-1)
        target = target.reshape(-1)
        output = output[target != -1]
        target = target[target != -1]
        if print_report:
            problem_type = 'Recognition' if 'recognition' in index else 'Prediction'
            if 'affordance' in index:
                problem_class = 'Affordance'
                target_names = affordance_names
            else:
                problem_class = 'Sub-activity'
                target_names = subactivity_names
            labels = range(len(target_names))
            print(f'{problem_class} {problem_type}')
            print(classification_report(target, output, labels=labels, target_names=target_names, digits=4))
        for average in ['micro', 'macro']:
            precision, recall, f1, _ = precision_recall_fscore_support(target, output, average=average)
            results[str(index) + '-' + average] = {'precision': precision, 'recall': recall, 'f1': f1}
    return results


def evaluate_f1_at_k(targets, outputs, num_subactivites, num_affordances, overlap: float = 0.25):
    results = {}
    print(f'\n\nF1@{overlap} metric.')
    for index, target in sorted(targets.items()):
        output = outputs[index]
        if target.ndim == 3:
            target = np.swapaxes(target, 1, 2)
            output = np.swapaxes(output, 1, 2)
        num_steps = output.shape[-1]
        output, target = output.reshape(-1, num_steps), target.reshape(-1, num_steps)
        problem_type = 'Recognition' if 'recognition' in index else 'Prediction'
        problem_class = 'Affordance' if 'affordance' in index else 'Sub-activity'
        num_classes = num_affordances if problem_class == 'Affordance' else num_subactivites
        f1 = f1_at_k(target, output, num_classes, overlap=overlap, ignore_value=-1.0)
        print(f'{problem_class} {problem_type}')
        print(f'F1@{overlap}: {f1:.4f}')
        results[index] = f1
    return results


def select_index_to_name_mapping(model_name, model_input_type, dataset_name):
    if model_name == '2G-GCN':
        if dataset_name == 'cad120':
            return {0: 'sub-activity_recognition', 1: 'sub-activity_prediction',
                    2: 'affordance_recognition', 3: 'affordance_prediction'}
        else:
            return {0: 'sub-activity_recognition', 1: 'sub-activity_prediction'}
    elif model_name in {'bimanual_baseline'}:
        return {0: 'sub-activity_recognition'}
    elif model_name in {'cad120_baseline'}:
        return {0: 'sub-activity_recognition', 1: 'affordance_recognition'}
    if model_input_type == 'human':
        return {0: 'sub-activity_recognition', 1: 'sub-activity_prediction'}
    else:
        return {0: 'affordance_recognition', 1: 'affordance_prediction'}


def maybe_load_class_dictionaries(model_dir):
    hydra_configs_dir = os.path.join(model_dir, '.hydra')
    cfg = OmegaConf.load(os.path.join(hydra_configs_dir, 'config.yaml'))
    # Data
    subactivity_id_to_name = affordance_id_to_name = None
    dataset_name = cfg.data.get('name', default_value='cad120')
    if dataset_name == 'cad120':
        dictionaries_dir = os.path.dirname(cfg.data.video_id_to_subject_id)
        subactivity_path = os.path.join(dictionaries_dir, 'subactivity-id_to_subactivity-name.txt')
        try:
            subactivity_id_to_name = read_dictionary(subactivity_path)
        except FileNotFoundError:
            pass
        else:
            subactivity_id_to_name = {int(k) - 1: v for k, v in subactivity_id_to_name.items()}
        affordance_path = os.path.join(dictionaries_dir, 'affordance-id_to_affordance-name.txt')
        try:
            affordance_id_to_name = read_dictionary(affordance_path)
        except FileNotFoundError:
            pass
        else:
            affordance_id_to_name = {int(k) - 1: v for k, v in affordance_id_to_name.items()}
    elif dataset_name == 'bimanual':
        dictionaries_dir = os.path.dirname(cfg.data.video_id_to_video_fps)
        dictionary_path = os.path.join(dictionaries_dir, 'bimacs_action_id_to_action_name.json')
        with open(dictionary_path, mode='r') as f:
            subactivity_id_to_name = json.load(f)
        subactivity_id_to_name = {int(k): v for k, v in subactivity_id_to_name.items()}
    else:
        dictionaries_dir = os.path.dirname(cfg.data.path)
        dictionary_path = os.path.join(dictionaries_dir, 'mphoi_action_id_to_action_name.json')
        with open(dictionary_path, mode='r') as f:
            subactivity_id_to_name = json.load(f)
        subactivity_id_to_name = {int(k): v for k, v in subactivity_id_to_name.items()}
    return subactivity_id_to_name, affordance_id_to_name


def maybe_get_class_names(id_to_name: dict = None):
    names = None
    if id_to_name is not None:
        names = [v for _, v in sorted(id_to_name.items(), key=lambda x: x[0])]
    return names


def fetch_dataset_name(model_dir):
    hydra_configs_dir = os.path.join(model_dir, '.hydra')
    cfg = OmegaConf.load(os.path.join(hydra_configs_dir, 'config.yaml'))
    dataset_name = cfg.data.get('name', default_value='cad120')
    return dataset_name


def predict_all(args):
    pretrained_model_dir = args.pretrained_model_dir
    cross_validate = args.cross_validate
    convert_frame_to_segment_level = args.convert_frame_to_segment_level
    save_visualisations_dir = args.save_visualisations_dir
    inspect_model = args.inspect_model

    subactivity_id_to_name, affordance_id_to_name = maybe_load_class_dictionaries(pretrained_model_dir)
    subactivity_names = maybe_get_class_names(subactivity_id_to_name)
    affordance_names = maybe_get_class_names(affordance_id_to_name)
    overlaps = [0.10, 0.25, 0.50]
    if cross_validate:
        basename = os.path.basename(pretrained_model_dir)
        model_id_parts = basename.split(sep='_')
        model_id = '_'.join(model_id_parts[:-1])
        dirname = os.path.dirname(pretrained_model_dir)
        outputs_per_subject = {}
        dataset_name = fetch_dataset_name(pretrained_model_dir)
        if dataset_name == 'cad120':
            test_subject_ids = ['Subject1', 'Subject3', 'Subject4', 'Subject5']
        elif dataset_name == 'bimanual':
            test_subject_ids = list('123456')
        else:
            test_subject_ids = ['Subject45', 'Subject25', 'Subject14']
        for subject_id in test_subject_ids:
            current_model_dir = os.path.join(dirname, model_id + '_' + subject_id)
            try:
                outputs, targets, test_ids = predict(current_model_dir, convert_frame_to_segment_level, inspect_model)
            except FileNotFoundError:
                continue
            else:
                outputs_per_subject[subject_id] = outputs, targets, test_ids
        results_per_subject = {}
        f1_results_per_subject = {}
        for subject_id, (output, target, _) in sorted(outputs_per_subject.items()):
            print(f'\n{subject_id}')
            results = evaluate_predictions(target, output, subactivity_names=subactivity_names,
                                           affordance_names=affordance_names)
            results_per_subject[subject_id] = results
            num_subactivities = len(subactivity_names) if subactivity_names is not None else None
            num_affordances = len(affordance_names) if affordance_names is not None else None
            for overlap in overlaps:
                results_f1 = evaluate_f1_at_k(target, output, num_subactivities, num_affordances, overlap=overlap)
                f1_results_per_subject.setdefault(subject_id, {}).setdefault(overlap, results_f1)
        # Micro and macro P/R/F1 results.
        final_results = defaultdict(list)
        for subject_id, results_per_label in sorted(results_per_subject.items()):
            for label_id, results_per_metric in results_per_label.items():
                for metric_name, result in results_per_metric.items():
                    final_results[label_id + '_' + metric_name].append(result)
        print('\n\nSummary Performance for Cross-validation.')
        for result_id, result_values in final_results.items():
            print(f'{result_id}\n\tValues: {[round(result, 4) for result in result_values]}')
            print(f'\tMean: {np.mean(result_values):.4f}\tStd: {np.std(result_values):.4f}')
        # F1@k results.
        final_f1_results = {}
        for subject_id, f1_results_per_overlap in sorted(f1_results_per_subject.items()):
            for overlap, f1_results_per_label in sorted(f1_results_per_overlap.items()):
                for label_id, f1_per_label in f1_results_per_label.items():
                    final_f1_results.setdefault(label_id, {}).setdefault(overlap, []).append(f1_per_label)
        print('\nSummary F1@k results.')
        for label_id, f1s_per_overlap in final_f1_results.items():
            print(f'{label_id}')
            for overlap, f1s_per_label in f1s_per_overlap.items():
                print(f'\tOverlap: {overlap}')
                print(f'\tValues: {[round(f1, 4) for f1 in f1s_per_label]}')
                print(f'\tMean: {np.mean(f1s_per_label):.4f}\tStd: {np.std(f1s_per_label):.4f}\n')
        if save_visualisations_dir is not None and os.path.isdir(save_visualisations_dir):
            for subject_id, (output, target, test_ids) in outputs_per_subject.items():
                test_ids = [f'{subject_id}_{test_id}' for test_id in test_ids]
                save_visualisations_subject_dir = os.path.join(save_visualisations_dir, subject_id)
                os.makedirs(save_visualisations_subject_dir, exist_ok=True)
                cleanup_directory(save_visualisations_subject_dir)
                dump_visualisations(save_visualisations_subject_dir, output, target, test_ids,
                                    subactivity_id_to_name, affordance_id_to_name)
                for overlap in overlaps:
                    dump_f1_scores_per_example(save_visualisations_subject_dir, output, target, test_ids,
                                               subactivity_id_to_name, affordance_id_to_name, overlap)
            id_to_gt_sa = {}
            id_to_pd_sa = {}
            id_to_gt_af = {}
            id_to_pd_af = {}
            for output, target, test_ids in outputs_per_subject.values():
                id_to_gt_sa = {**id_to_gt_sa, **to_dict(target['sub-activity_recognition'], test_ids)}
                id_to_pd_sa = {**id_to_pd_sa, **to_dict(output['sub-activity_recognition'], test_ids)}
                try:
                    id_to_gt_af = {**id_to_gt_af, **to_dict(target['affordance_recognition'], test_ids)}
                    id_to_pd_af = {**id_to_pd_af, **to_dict(output['affordance_recognition'], test_ids)}
                except KeyError:
                    pass
            id_to_gt_sa, id_to_pd_sa = cleanup_padding_values(id_to_gt_sa, id_to_pd_sa)
            id_to_gt_af, id_to_pd_af = cleanup_padding_values(id_to_gt_af, id_to_pd_af)
            save_output_dir = os.path.join(save_visualisations_dir, 'outputs')
            os.makedirs(save_output_dir, exist_ok=True)
            cleanup_directory(save_output_dir)
            filenames = ['gt_sa.json', 'our_sa.json', 'gt_af.json', 'our_af.json']
            dicts = [id_to_gt_sa, id_to_pd_sa, id_to_gt_af, id_to_pd_af]
            for filename, d in zip(filenames, dicts):
                if not d:
                    continue
                save_file = os.path.join(save_output_dir, filename)
                with open(save_file, mode='w') as f:
                    json.dump(d, f)
    else:
        outputs, targets, test_ids = predict(pretrained_model_dir, convert_frame_to_segment_level, inspect_model)
        evaluate_predictions(targets, outputs, subactivity_names=subactivity_names, affordance_names=affordance_names)
        num_subactivities = len(subactivity_names) if subactivity_names is not None else None
        num_affordances = len(affordance_names) if affordance_names is not None else None
        for overlap in overlaps:
            evaluate_f1_at_k(targets, outputs, num_subactivities, num_affordances, overlap=overlap)
        if save_visualisations_dir is not None and os.path.isdir(save_visualisations_dir):
            subject_id = pretrained_model_dir.split(sep='_')[-1]
            save_visualisations_subject_dir = os.path.join(save_visualisations_dir, subject_id)
            os.makedirs(save_visualisations_subject_dir, exist_ok=True)
            cleanup_directory(save_visualisations_subject_dir)
            dump_visualisations(save_visualisations_subject_dir, outputs, targets, test_ids,
                                subactivity_id_to_name, affordance_id_to_name)
            for overlap in overlaps:
                dump_f1_scores_per_example(save_visualisations_subject_dir, outputs, targets, test_ids,
                                           subactivity_id_to_name, affordance_id_to_name, overlap)


def dump_visualisations(save_visualisations_dir, outputs, targets, test_ids,
                        subactivity_id_to_name, affordance_id_to_name):
    problem_types = list(outputs.keys())
    for problem_type in problem_types:
        class_id_to_label = subactivity_id_to_name if 'sub-activity' in problem_type else affordance_id_to_name
        output, target = outputs[problem_type], targets[problem_type]
        for out, tar, test_id in zip(output, target, test_ids):
            for ent_id in range(out.shape[1]):
                save_file = os.path.join(save_visualisations_dir, f'{test_id}_{problem_type}_{ent_id}.png')
                tar_ent, out_ent = tar[:, ent_id], out[:, ent_id]
                out_ent = out_ent[tar_ent != -1]
                tar_ent = tar_ent[tar_ent != -1]
                if tar_ent.size:
                    plot_segmentation(tar_ent, out_ent,
                                      class_id_to_label=class_id_to_label, save_file=save_file, xlabels_type='id')


def dump_f1_scores_per_example(save_visualisations_dir, outputs, targets, test_ids,
                               subactivity_id_to_name, affordance_id_to_name, overlap):
    problem_types = list(outputs.keys())
    save_file = os.path.join(save_visualisations_dir, f'f1_scores_{overlap:.2f}.txt')
    with open(save_file, mode='w') as f:
        for problem_type in problem_types:
            class_id_to_label = subactivity_id_to_name if 'sub-activity' in problem_type else affordance_id_to_name
            output, target = outputs[problem_type], targets[problem_type]
            for out, tar, test_id in zip(output, target, test_ids):
                for ent_id in range(out.shape[1]):
                    tar_ent, out_ent = tar[:, ent_id], out[:, ent_id]
                    out_ent = out_ent[tar_ent != -1]
                    tar_ent = tar_ent[tar_ent != -1]
                    if tar_ent.size:
                        f1 = f1_at_k_single_example(tar_ent, out_ent, len(class_id_to_label), overlap=overlap)
                        f.write(f'{problem_type}_{test_id}_{ent_id}: {f1:.4f}\n')
            f.write('\n')


def to_dict(output, video_ids):
    """Convert ndarray to dict.

    Arguments:
        output - A tensor of shape (num_videos, max_num_steps, max_num_entities).
        video_ids - A list containing the ID of each video.
    Returns:
        A dictionary mapping each video id to a dictionary mapping each entity to a list containing the
        frame-wise labels.
    """
    video_id_to_labeling = {}
    for video_id, per_entity_labeling in zip(video_ids, output):
        per_entity_labeling = np.transpose(per_entity_labeling)
        for ent_id, labeling in enumerate(per_entity_labeling, 1):
            labeling = labeling.tolist()
            video_id_to_labeling.setdefault(video_id, {})[ent_id] = labeling
    return video_id_to_labeling


def cleanup_padding_values(id_to_gt, id_to_pd):
    video_ids = set(id_to_gt.keys())
    for video_id in video_ids:
        gt, pd = id_to_gt[video_id], id_to_pd[video_id]
        ent_ids = set(gt.keys())
        for ent_id in ent_ids:
            gt_e, pd_e = gt[ent_id], pd[ent_id]
            gt_e, pd_e = np.array(gt_e), np.array(pd_e)
            pd_e = pd_e[gt_e != -1.0]
            gt_e = gt_e[gt_e != -1.0]
            gt_e, pd_e = gt_e.tolist(), pd_e.tolist()
            if gt_e:
                id_to_gt[video_id][ent_id] = gt_e
                id_to_pd[video_id][ent_id] = pd_e
            else:
                del id_to_gt[video_id][ent_id]
                del id_to_pd[video_id][ent_id]
    return id_to_gt, id_to_pd


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Predict Module.')

    parser.add_argument('--pretrained_model_dir', type=str, required=True,
                        help='Path to directory containing the pre-trained model information.')
    parser.add_argument('--cross_validate', action='store_true',
                        help='If specified, run prediction for the specified model on all splits present in the same '
                             'directory of pretrained_model_dir. For instance, if pretrained_model_dir is '
                             '.../my_model_Subject5, then we look for all other .../my_model_SubjectS in .../, run all '
                             'of them, and average their results to obtain the cross-validated performance. If '
                             'not specified, then the code only generates predictions and results for the '
                             'model specified in pretrained_model_dir.')
    parser.add_argument('--convert_frame_to_segment_level', action='store_true',
                        help='If specified, convert frame-level predictions into segment-level predictions '
                             'for model evaluation. Only meaningful if the model being tested was trained with '
                             'ground-truth segmentation and is a frame-level model.')
    parser.add_argument('--save_visualisations_dir', type=str,
                        help='If specified, save ground-truth/predicted segmentations to that dir. The specified '
                             'dir must already exist.')
    parser.add_argument('--inspect_model', action='store_true',
                        help='Not fully functional yet. If specified fetch attention scores for the model.')
    parser.set_defaults(func=predict_all)
    return parser


def main():
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
