from datetime import datetime
import os
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from pyrutils.torch.fetchers import single_input_single_output
from pyrutils.torch.forwarders import basic_forward


def train(model, train_loader, optimizer, criterion, epochs, device, loss_names, clip_gradient_at=0.0,
          fetch_model_data=single_input_single_output, feed_model_data=basic_forward,
          val_loader=None, initial_epoch=1, mtll_model=None, print_raw_losses=False,
          num_main_losses=None, **kwargs):
    """General training function to train a PyTorch model.

    If validation data is not given, the returned checkpoint is the one obtained after training the model for the
    specified number of epochs, regardless of the final training loss. If validation data is given, the checkpoint
    returned is the one with the lowest validation loss, which could have been obtained in some epoch before the
    last one.
    Arg(s):
        model - PyTorch model.
        train_loader - Batch generator for model training.
        optimizer - Model optimizer.
        criterion - Specific loss function for the given model. This function receives as input the output of
            model and the ground-truth target, and returns a list of batch losses (for multi-loss models). Even if
            the model has a single loss, the return value of criterion must be a list containing this single loss.
        epochs - Maximum number of epochs for model training.
        device - Which device to use for model training. Either cuda or cpu.
        loss_names - Names for the individual losses output by criterion.
        clip_gradient_at - If nonzero clips the norm of the gradient vector at the specified value. The gradient
            vector is a vector obtained by concatenating all parameters of the model.
        fetch_model_data - Function to fetch the input and output tensors for the model.
        feed_model_data - Function to feed the input tensors to the model.
        val_loader - Batch generator for model validation.
        **kwargs - Any extra parameters to be passed during training.
    Returns:
        A dictionary containing the history of train losses, the model's weights and associated epoch, and if
        val_loader is specified, the history of validation losses as well.
    """
    checkpoint_name = kwargs.get('checkpoint_name', None)
    tensorboard_log_dir = kwargs.get('tensorboard_log_dir', None)
    writer = None
    if tensorboard_log_dir is not None and checkpoint_name is not None:
        writer = SummaryWriter(os.path.join(tensorboard_log_dir, 'runs', checkpoint_name))
    checkpoint = {}
    train_losses, val_losses, train_raw_losses, val_raw_losses = [], [], [], []
    val_loss = float('Inf')
    for epoch in range(initial_epoch, epochs + initial_epoch):
        # Train
        print(f'\nEpoch: [{epoch:4d}/{epochs + initial_epoch - 1:4d}]')
        train_single_epoch(model, data_loader=train_loader, optimizer=optimizer, criterion=criterion,
                           device=device, loss_names=loss_names, clip_gradient_at=clip_gradient_at,
                           fetch_model_data=fetch_model_data, feed_model_data=feed_model_data, log_interval=25,
                           mtll_model=mtll_model, num_main_losses=num_main_losses, **kwargs)
        current_train_loss, current_train_losses, current_train_raw_loss, current_train_raw_losses = \
            test(model, data_loader=train_loader, criterion=criterion,
                 device=device, loss_names=loss_names, fetch_model_data=fetch_model_data,
                 feed_model_data=feed_model_data, test_set_name='Train', mtll_model=mtll_model,
                 print_raw_losses=print_raw_losses, num_main_losses=num_main_losses, **kwargs)
        train_losses.append([current_train_loss, current_train_losses])
        if mtll_model is not None:
            train_raw_losses.append([current_train_raw_loss, current_train_raw_losses])
        if writer is not None:
            base_str = 'Loss/train_mtll/' if mtll_model is not None else 'Loss/train/'
            for loss_name, loss in zip(loss_names, current_train_losses):
                writer.add_scalar(base_str + loss_name, loss, epoch)
            writer.add_scalar(base_str + 'total', current_train_loss, epoch)
            if mtll_model is not None:
                loss_weights = mtll_model.get_weights()
                for loss_name, raw_loss, loss_weight in zip(loss_names,
                                                            current_train_raw_losses, loss_weights):
                    writer.add_scalar(f'Loss/train/{loss_name}', raw_loss, epoch)
                    if loss_weight is not None:
                        writer.add_scalar(f'Loss/mtll_weight/{loss_name}', loss_weight, epoch)
                writer.add_scalar('Loss/train/total', current_train_raw_loss, epoch)
        # Validate
        if val_loader is not None:
            current_val_loss, current_val_losses, current_val_raw_loss, current_val_raw_losses = \
                test(model, data_loader=val_loader, criterion=criterion,
                     device=device, loss_names=loss_names, fetch_model_data=fetch_model_data,
                     feed_model_data=feed_model_data, test_set_name='Validation', mtll_model=mtll_model,
                     print_raw_losses=print_raw_losses, num_main_losses=num_main_losses)
            val_losses.append([current_val_loss, current_val_losses])
            if mtll_model is not None:
                val_raw_losses.append([current_val_raw_loss, current_val_raw_losses])
            if writer is not None:
                base_str = 'Loss/val_mtll/' if mtll_model is not None else 'Loss/val/'
                for loss_name, loss in zip(loss_names, current_val_losses):
                    writer.add_scalar(base_str + loss_name, loss, epoch)
                writer.add_scalar(base_str + 'total', current_val_loss, epoch)
                if mtll_model is not None:
                    for loss_name, raw_loss in zip(loss_names, current_val_raw_losses):
                        writer.add_scalar('Loss/val/' + loss_name, raw_loss, epoch)
                    writer.add_scalar('Loss/val/total', current_val_raw_loss, epoch)
            if current_val_loss < val_loss:
                val_loss = current_val_loss
                checkpoint['epoch'] = epoch
                checkpoint['model_state_dict'] = model.state_dict()
                if mtll_model is not None:
                    checkpoint['mtll_model_state_dict'] = mtll_model.state_dict()
        else:
            checkpoint['epoch'] = epoch
            checkpoint['model_state_dict'] = model.state_dict()
            if mtll_model is not None:
                checkpoint['mtll_model_state_dict'] = mtll_model.state_dict()
    print('Lowest val_loss is', val_loss)
    checkpoint['train_losses'] = train_losses
    checkpoint['val_losses'] = val_losses
    checkpoint['train_raw_losses'] = train_raw_losses
    checkpoint['val_raw_losses'] = val_raw_losses
    if writer is not None:
        writer.close()
    return checkpoint


def train_single_epoch(model, data_loader, optimizer, criterion, device, loss_names, clip_gradient_at=0.0,
                       fetch_model_data=single_input_single_output, feed_model_data=basic_forward,
                       log_interval=25, mtll_model=None, num_main_losses=None, **kwargs):
    """General training function to train a PyTorch model for a single epoch.

    Arg(s):
        model - PyTorch model.
        data_loader - Batch generator for model training.
        optimizer - Model optimizer.
        criterion - Specific loss function for the given model. This function receives as input the output of
            model and the ground-truth target, and returns a list of batch losses (for multi-loss models). Even if
            the model has a single loss, the return value of criterion must be a list containing this single loss.
        device - Which device to use for model training. Either cuda or cpu.
        loss_names - Names for the individual losses output by criterion.
        clip_gradient_at - If nonzero clips the norm of the gradient vector at the specified value. The gradient
            vector is a vector obtained by concatenating all parameters of the model.
        fetch_model_data - Function to fetch the input and output tensors for the model.
        feed_model_data - Function to feed the input tensors to the model.
        log_interval - Print training statistics every log_interval batches.
        **kwargs - Any extra parameter that needs to be passed to the feed_model_data of a model.
    """
    model.train()
    if mtll_model is not None:
        mtll_model.train()
    num_examples = len(data_loader.dataset)
    for batch_idx, dataset in enumerate(data_loader):
        data, target = fetch_model_data(dataset, device=device)
        optimizer.zero_grad()
        output = feed_model_data(model, data, **kwargs)
        losses = criterion(output, target, reduction='mean')
        if mtll_model is not None:
            losses = mtll_model(losses)
        loss = sum(losses)
        loss.backward()
        if clip_gradient_at:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_gradient_at)
        optimizer.step()
        log_now, is_last_batch = (batch_idx % log_interval) == 0, batch_idx == (len(data_loader) - 1)
        if log_now or is_last_batch:
            num_main_losses = num_main_losses if num_main_losses is not None else len(losses)
            loss = sum(losses[-num_main_losses:])
            batch_initial_example_idx = min((batch_idx + 1) * data_loader.batch_size, num_examples)
            epoch_progress = 100 * (batch_idx + 1) / len(data_loader)
            print(f'(Train) Batch [{batch_initial_example_idx:6d}/{num_examples:6d} ({epoch_progress:3.0f}%)] ',
                  f'Loss: {loss.item(): 8.4f}', end='')
            for loss_name, single_loss in zip(loss_names, losses):
                print(f'  {loss_name}: {single_loss: 6.4f}', end='')
            print()


def test(model, data_loader, criterion, device, loss_names, fetch_model_data=single_input_single_output,
         feed_model_data=basic_forward, test_set_name='Test', mtll_model=None,
         print_raw_losses=False, num_main_losses=None, **kwargs):
    """General testing function to test a PyTorch model.

    Arg(s):
        model - PyTorch model.
        data_loader - Batch generator for model testing.
        criterion - Specific loss function for the given model. This function receives as input the output of
            model and the ground-truth target, and returns a list of batch losses (for multi-loss models). Even if
            the model has a single loss, the return value of criterion must be a list containing this single loss.
        device - Which device to use for model testing. Either cuda or cpu.
        loss_names - Names for the individual losses output by criterion.
        fetch_model_data - Function to fetch the input and output tensors for the model.
        feed_model_data - Function to feed the input tensors to the model.
        test_set_name - Optional name given to the set being evaluated. Useful for logging purposes.
        num_main_losses - The final test loss is the sum of all non-auxiliary losses. Auxiliary losses should be in
            the beginning of the output list.
        **kwargs - Any extra parameters that need to be passed to the feed_model_data function.
    Returns:
        The model loss.
    """
    model.eval()
    if mtll_model is not None:
        mtll_model.eval()
    test_raw_losses = None
    test_losses = None
    with torch.no_grad():
        for dataset in data_loader:
            data, target = fetch_model_data(dataset, device=device)
            output = feed_model_data(model, data, **kwargs)
            raw_losses = criterion(output, target, reduction='mean')
            if mtll_model is not None:
                test_raw_losses = _collect_losses(raw_losses, test_raw_losses)
                losses = mtll_model(raw_losses)
            else:
                losses = raw_losses
            test_losses = _collect_losses(losses, test_losses)
    num_main_losses = num_main_losses if num_main_losses is not None else len(test_losses)
    test_losses = [test_loss / len(data_loader) for test_loss in test_losses]
    total_test_loss = sum(test_losses[-num_main_losses:])

    test_set_name = f'({test_set_name})'
    print(f'{test_set_name:>12} Loss: {total_test_loss: 7.4f}', end='')
    for loss_name, loss in zip(loss_names, test_losses):
        print(f'   {loss_name}: {loss: 6.4f}', end='')
    print()
    total_test_raw_loss = None
    if test_raw_losses is not None:
        test_raw_losses = [test_raw_loss / len(data_loader) for test_raw_loss in test_raw_losses]
        total_test_raw_loss = sum(test_raw_losses[-num_main_losses:])
        if print_raw_losses:
            print(f'{test_set_name:>12} Loss: {total_test_raw_loss: 7.4f}', end='')
            for loss_name, raw_loss in zip(loss_names, test_raw_losses):
                print(f'   {loss_name}: {raw_loss: 6.4f}', end='')
            print()
    return total_test_loss, test_losses, total_test_raw_loss, test_raw_losses


def _collect_losses(output_losses: list, losses: Optional[list] = None):
    try:
        losses = [loss + output_loss.item() for loss, output_loss in zip(losses, output_losses)]
    except TypeError:
        losses = [output_loss.item() for output_loss in output_losses]
    return losses


def save_checkpoint(log_dir, checkpoint: dict, checkpoint_name: Optional[str] = None, include_timestamp: bool = False):
    """Save PyTorch model checkpoint.

    Arg(s):
        log_dir - Directory to save checkpoint file. It must already exist.
        checkpoint - A dictionary containing the model checkpoint and other metadata such as data scalers and
            model creation arguments.
        checkpoint_name - If given, use that as the file name to save. Otherwise, the file name is 'checkpoint'. A
            '.tar' is appended to checkpoint_name, so there is no need to include it in the passed checkpoint name.
        include_timestamp - Whether to prepend the timestamp to the checkpoint name or not.
    """
    file_save_name = checkpoint_name if checkpoint_name is not None else 'checkpoint'
    if include_timestamp:
        time_now = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S')
        file_save_name = time_now + '_' + file_save_name
    file_save_name += '.tar'
    file_save_path = os.path.join(log_dir, file_save_name)
    torch.save(checkpoint, file_save_path)
    print(f'log files written to {file_save_path}')


def numpy_to_torch(*arrays, device='cpu'):
    """Convert any number of numpy arrays to PyTorch tensors."""
    return [torch.from_numpy(array).to(device) for array in arrays]
