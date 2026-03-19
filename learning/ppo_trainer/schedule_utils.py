"""
Training schedule helpers.
"""


def update_linear_learning_rate(optimizer, current_epoch, total_epochs, initial_lr):
    """
    Linearly decay the learning rate.
    """
    lr = initial_lr - (initial_lr * (current_epoch / float(total_epochs)))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def flatten_time_batch(time_steps, batch_size, tensor):
    """
    Flatten (T, B, ...) -> (T*B, ...)
    """
    return tensor.view(time_steps * batch_size, *tensor.size()[2:])


def reshape_flatten_time_batch(time_steps, batch_size, tensor):
    """
    Flatten with reshape for non-contiguous tensors.
    """
    return tensor.reshape(time_steps * batch_size, *tensor.size()[2:])