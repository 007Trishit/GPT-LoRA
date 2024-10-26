import torch
import numpy as np


class ArgStorage:
    """
    A simple class to store arguments as attributes.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def get_device(id: int = -1):
    """
    Get the device to use for training.
    """
    return torch.device(
        f'cuda:{id}' if torch.cuda.is_available() and id >= 0
        else 'mps' if torch.backends.mps.is_available() else 'cpu')


def get_data_loader(data_path, batch_size, tokenizer, shuffle=True, max_len=20):
    """
    Get a data loader for the training data.
    """
    data = np.loadtxt(data_path, delimiter='\t', dtype=str)
    X, y = data[:, -1], data[:, 1]
    X = tokenizer.batch_encode_plus(
        X.tolist(), max_length=max_len, truncation=True, padding='max_length')
    X, mask = X['input_ids'], X['attention_mask']
    X = torch.tensor(np.array(X))
    mask = torch.tensor(np.array(mask))
    y = torch.tensor(np.array(y, dtype=int))
    data = torch.utils.data.TensorDataset(X, mask, y)
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=shuffle)
    return data_loader
