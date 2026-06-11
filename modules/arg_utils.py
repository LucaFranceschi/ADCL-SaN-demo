import argparse
from typing import List, Optional, Union, Tuple


def int_or_int_list_or_none(value: Optional[Union[int, str]]) -> List[Optional[int]]:
    """
    Parse an input value into a list of integers or a single integer, or None.

    Args:
        value (Optional[Union[int, str]]): The input value to parse.

    Returns:
        List[Optional[int]]: A list containing either a single integer, a list of integers,
                             or a single None value.

    Raises:
        argparse.ArgumentTypeError: If the input value cannot be parsed into the specified formats.
    """
    if value in ['None', 'null']:
        return [None]
    try:
        # If the value contains commas, parse it as a comma-separated list of integers
        if ',' in value:
            return [int(x) for x in value.split(',')]
        # If it's a single integer, pack it into a list
        else:
            return [int(value)]
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid format. Use an integer, a comma-separated list of integers, or None.")


def int_or_float(value):
    if '.' in value:
        try:
            return float(value)
        except ValueError:
            raise argparse.ArgumentTypeError("Quality level must be an integer or a float")
    else:
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError("Quality level must be an integer or a float")

def create_param_groups_with_sigmoid_lr(model, base_lr, sigmoid_lr_scale=0.1):
    """
    Create parameter groups where sigmoid parameters get a different learning rate.

    Args:
        model: The model instance
        base_lr: Base learning rate for most parameters
        sigmoid_lr_scale: Multiplier for sigmoid parameter LR (e.g., 0.1 means 10% of base_lr)

    Returns:
        List of parameter groups
    """
    sigmoid_params = []
    other_params = []

    # Identify sigmoid parameters from ADCL model if present
    if hasattr(model, 'm') and hasattr(model.m, 'epsilon'):
        sigmoid_params.extend([model.m.epsilon, model.m.epsilon2, model.m.tau])

    # Collect all other trainable parameters
    param_ids = {id(p) for p in sigmoid_params}
    for param in model.parameters():
        if param.requires_grad and id(param) not in param_ids:
            other_params.append(param)

    # Create parameter groups
    param_groups = [
        {'params': other_params, 'lr': base_lr},
        {'params': sigmoid_params, 'lr': base_lr * sigmoid_lr_scale}
    ]

    return param_groups