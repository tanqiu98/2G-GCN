import argparse
import json
import os

import matplotlib.pyplot as plt
import torch

from pyrutils.utils import read_dictionary
from vhoi.visualisation import plot_segmentation


def analyse_two_files_diff(filepath_1, filepath_2, save_file):
    with open(filepath_1, mode='r') as f1, open(filepath_2, mode='r') as f2, open(save_file, mode='w') as f3:
        for line1, line2 in zip(f1, f2):
            line1, line2 = line1.strip(), line2.strip()
            if line1 and line2:
                line_id, f1_1 = line1.split(sep=':')
                f1_1 = float(f1_1.strip())
                _, f1_2 = line2.split(sep=':')
                f1_2 = float(f1_2.strip())
                f1_d = f1_1 - f1_2
                f3.write(f'{line_id}: {f1_d:7.4f}\n')
            else:
                f3.write('\n')


def analyse_two_dirs_diff(dirpath_1, dirpath_2, save_dir):
    filenames_1 = {filename for filename in os.listdir(dirpath_1) if filename.endswith('.txt')}
    filenames_2 = {filename for filename in os.listdir(dirpath_2) if filename.endswith('.txt')}
    filenames = filenames_1 & filenames_2
    for filename in filenames:
        filepath_1 = os.path.join(dirpath_1, filename)
        filepath_2 = os.path.join(dirpath_2, filename)
        save_file = os.path.join(save_dir, filename)
        analyse_two_files_diff(filepath_1, filepath_2, save_file)


def analyse_diff(args):
    model_1_dir = args.model_1_dir
    model_2_dir = args.model_2_dir
    save_dir = args.save_dir
    analyse_two_dirs_diff(model_1_dir, model_2_dir, save_dir)


def plot_comparisons(args):
    ground_truth_json = args.ground_truth
    with open(ground_truth_json, mode='r') as f:
        ground_truth = json.load(f)
    predictions_json = args.predictions
    predictions = []
    for prediction_json in predictions_json:
        with open(prediction_json, mode='r') as f:
            predictions.append(json.load(f))
    class_id_to_label = args.class_id_to_label
    if class_id_to_label.endswith('.txt'):
        class_id_to_label = read_dictionary(class_id_to_label)
        class_id_to_label = {int(k) - 1: v for k, v in class_id_to_label.items()}
    else:
        with open(class_id_to_label, mode='r') as f:
            class_id_to_label = json.load(f)
        class_id_to_label = {int(k): v for k, v in class_id_to_label.items()}
    save_dir = args.save_dir
    bar_height, bar_width = args.bar_height, args.bar_width
    video_ids = set(ground_truth.keys())
    for video_id in video_ids:
        gt = ground_truth[video_id]
        pds = [prediction[video_id] for prediction in predictions]
        ent_ids = set(gt.keys())
        for ent_id in ent_ids:
            gt_e = gt[ent_id]
            pds_e = [pd[ent_id] for pd in pds]
            filename = f'{video_id}_{int(ent_id) - 1}.png'
            save_file = os.path.join(save_dir, filename)
            plot_segmentation(gt_e, *pds_e, class_id_to_label=class_id_to_label, save_file=save_file,
                              bar_height=bar_height, bar_width=bar_width, xlabels_type='None')


def plot_training_curves(args):
    model_dirs = args.model_dirs
    names = args.names
    for name, model_dir in zip(names, model_dirs):
        checkpoint_file = os.path.join(model_dir, os.path.basename(model_dir) + '.tar')
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        train_losses = [sum(epoch_losses) for _, epoch_losses in checkpoint['train_losses']]
        plt.plot(range(1, len(train_losses) + 1), train_losses, label=name)
    plt.legend()
    if (save_file := args.save_file) is not None:
        plt.savefig(save_file)
    else:
        plt.show()


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Analyse results.')
    subparsers = parser.add_subparsers(title='sub-commands', description='Valid sub-commands.')
    # Performance Diff
    parser_perf_diff = subparsers.add_parser('performance_diff',
                                             help='Extract F1 performance difference between two models.')

    parser_perf_diff.add_argument('--model_1_dir', type=str, required=True,
                                  help='Path to directory containing the result files of first model.')
    parser_perf_diff.add_argument('--model_2_dir', type=str, required=True,
                                  help='Path to directory containing the results files of second model.')
    parser_perf_diff.add_argument('--save_dir', type=str, required=True,
                                  help='Directory to save diff performance files.')
    parser_perf_diff.set_defaults(func=analyse_diff)
    # Plot segmentation between ground-truth and multiple predictions
    parser_plot_comparison = subparsers.add_parser('plot_comparison',
                                                   help='Plot a ground-truth segmentation and multiple output ones.')

    parser_plot_comparison.add_argument('--ground_truth', type=str, required=True,
                                        help='json file containing the ground-truth segmentation.')
    parser_plot_comparison.add_argument('--predictions', nargs='+',
                                        help='json files containing the predictions of other methods.')
    parser_plot_comparison.add_argument('--class_id_to_label', type=str,
                                        help='File mapping ids to label.')
    parser_plot_comparison.add_argument('--save_dir', type=str,
                                        help='Directory to save plot comparisons.')
    parser_plot_comparison.add_argument('--bar_height', default=30, type=int, help='Bar height.')
    parser_plot_comparison.add_argument('--bar_width', default=2000, type=int, help='Bar width.')
    parser_plot_comparison.set_defaults(func=plot_comparisons)
    # Plot training curves
    parser_plot_curves = subparsers.add_parser('plot_curves',
                                               help='Plot training curves for input models.')

    parser_plot_curves.add_argument('--model_dirs', nargs='+',
                                    help='Directories containing the model checkpoints.')
    parser_plot_curves.add_argument('--names', nargs='+',
                                    help='Name for the model plots.')
    parser_plot_curves.add_argument('--save_file', type=str,
                                    help='png file to save plot.')
    parser_plot_curves.set_defaults(func=plot_training_curves)
    return parser


def main():
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
