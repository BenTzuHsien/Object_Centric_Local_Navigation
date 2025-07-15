import torch
from collections import OrderedDict

class BaseModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def load_weight(self, weight_path):
        """
        Load model weights from a file, with support for DataParallel-trained checkpoints.

        Parameters
        ----------
        weight_path : str
            Path to the weight (.pth) file.
        device : str or torch.device
            Target device to map the loaded tensors (e.g., 'cuda:0' or 'cpu').
        """
        state_dict = torch.load(weight_path, map_location=next(self.parameters()).device)
        if any(k.startswith("module.") for k in state_dict.keys()):
            # Trained with DataParallel, strip "module."
            new_state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
        else:
            new_state_dict = state_dict
        self.load_state_dict(new_state_dict, strict=False)