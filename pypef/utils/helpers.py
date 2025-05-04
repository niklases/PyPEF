
import torch

from pypef.settings import USE_RAY


class ConditionalDecorator(object):
    def __init__(self, decorator, condition):
        self.decorator = decorator
        self.condition = condition

    def __call__(self, func):
        if not self.condition:
            # Return the function unchanged, not decorated
            return func
        return self.decorator(func)


if USE_RAY:
    import ray
    ray_conditional_decorator = ConditionalDecorator(ray.remote, USE_RAY)
else:
    ray_conditional_decorator = ConditionalDecorator(None, False)


def get_device():
    return (
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
