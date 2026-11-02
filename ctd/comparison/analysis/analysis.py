from functools import wraps

import torch


def analysis_inference(module_attribute):
    """Export trajectories without autograd or training-only layers.

    Restore every submodule's previous mode, including mixed train/eval
    states, so later training and gradient-based dynamics analyses are intact.
    Use no_grad rather than inference_mode so exported tensors can still be
    used as inputs to subsequent gradient-based calculations.
    """

    def decorate(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            module = getattr(self, module_attribute)
            modes = [(child, child.training) for child in module.modules()]
            try:
                module.eval()
                with torch.no_grad():
                    return method(self, *args, **kwargs)
            finally:
                for child, training in modes:
                    child.training = training

        return wrapped

    return decorate


class Analysis:
    def __init__(self, run_name, filepath):
        self.run_name = run_name
        self.filepath = filepath

    def load_wrapper(self, filepath):
        # Throw a warning
        return None

    def get_model_output(self):
        return None

    def compute_FPs(self, latents, inputs):
        return None
