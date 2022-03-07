import hydra
from omegaconf import DictConfig
import torch

from pyrutils.torch.train_utils import train, save_checkpoint
from pyrutils.torch.multi_task import MultiTaskLossLearner
from vhoi.data_loading import load_training_data, select_model_data_feeder, select_model_data_fetcher
from vhoi.data_loading import determine_num_classes
from vhoi.losses import select_loss, decide_num_main_losses, select_loss_types, select_loss_learning_mask
from vhoi.models import select_model, load_model_weights


@hydra.main(config_path='conf/config.yaml')
def main(cfg: DictConfig):
    seed = 42
    torch.set_num_threads(cfg.resources.num_threads)
    # Data
    model_name, model_input_type = cfg.metadata.model_name, cfg.metadata.input_type
    batch_size, val_fraction = cfg.optimization.batch_size, cfg.optimization.val_fraction
    misc_dict = cfg.get('misc', default_value={})
    sigma = misc_dict.get('segmentation_loss', {}).get('sigma', 0.0)
    train_loader, val_loader, data_info, scalers = load_training_data(cfg.data, model_name, model_input_type,
                                                                      batch_size=batch_size,
                                                                      val_fraction=val_fraction,
                                                                      seed=seed, debug=False, sigma=sigma)
    # Model
    Model = select_model(model_name)
    model_creation_args = cfg.parameters
    model_creation_args = {**data_info, **model_creation_args}
    dataset_name = cfg.data.name
    num_classes = determine_num_classes(model_name, model_input_type, dataset_name)
    model_creation_args['num_classes'] = num_classes
    device = 'cuda' if torch.cuda.is_available() and cfg.resources.use_gpu else 'cpu'
    model = Model(**model_creation_args).to(device)
    if misc_dict.get('pretrained', False) and misc_dict.get('pretrained_path') is not None:
        state_dict = load_model_weights(misc_dict['pretrained_path'])
        model.load_state_dict(state_dict, strict=False)
    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=cfg.optimization.learning_rate)
    criterion, loss_names = select_loss(model_name, model_input_type, dataset_name, cfg=cfg)
    mtll_model = None
    if misc_dict.get('multi_task_loss_learner', False):
        loss_types = select_loss_types(model_name, dataset_name, cfg=cfg)
        mask = select_loss_learning_mask(model_name, dataset_name, cfg=cfg)
        mtll_model = MultiTaskLossLearner(loss_types=loss_types, mask=mask).to(device)
        optimizer.add_param_group({'params': mtll_model.parameters()})
    # Some config + model training
    tensorboard_log_dir = cfg.logging.root_log_dir
    checkpoint_name = cfg.logging.checkpoint_name
    fetch_model_data = select_model_data_fetcher(model_name, model_input_type,
                                                 dataset_name=dataset_name, **{**misc_dict, **cfg.parameters})
    feed_model_data = select_model_data_feeder(model_name, model_input_type, dataset_name=dataset_name, **misc_dict)
    num_main_losses = decide_num_main_losses(model_name, dataset_name, {**misc_dict, **cfg.parameters})
    checkpoint = train(model, train_loader, optimizer, criterion, cfg.optimization.epochs, device, loss_names,
                       clip_gradient_at=cfg.optimization.clip_gradient_at,
                       fetch_model_data=fetch_model_data, feed_model_data=feed_model_data,
                       val_loader=val_loader, mtll_model=mtll_model, num_main_losses=num_main_losses,
                       tensorboard_log_dir=tensorboard_log_dir, checkpoint_name=checkpoint_name)
    # Logging
    if (log_dir := cfg.logging.log_dir) is not None:
        checkpoint['scalers'] = scalers
        save_checkpoint(log_dir, checkpoint, checkpoint_name=checkpoint_name, include_timestamp=False)


if __name__ == '__main__':
    main()
