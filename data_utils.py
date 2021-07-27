import torch
from torch.nn.utils.rnn import pad_sequence


def pad_collate(batch):
    """
    collate function that pads with zeros for variable lenght data-points.
    pass into the dataloader object.
    """
    np_x, np_y = zip(*batch)
    lens = [x.shape[0] for x in np_x]
    np_x = pad_sequence(np_x, batch_first=True, padding_value=0)
    np_y = pad_sequence(np_y, batch_first=True, padding_value=0)

    return np_x, np_y, torch.tensor(lens, dtype=torch.float)

