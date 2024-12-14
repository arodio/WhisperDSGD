import math


def make_dp_optimizer(cls):
    class DPOptimizerClass(cls):
        def __init__(self, params, l2_norm_clip, minibatch_size, *args, **kwargs):
            super(DPOptimizerClass, self).__init__(params, *args, **kwargs)
            self.minibatch_size = minibatch_size
            self.l2_norm_clip = l2_norm_clip

        def step(self, dp_noise=None, *args, **kwargs):
            # Compute total norm of all gradients
            total_norm = 0.
            for group in self.param_groups:
                for param in group['params']:
                    if param.requires_grad and param.grad is not None:
                        param_norm = param.grad.data.norm(2).item()
                        total_norm += param_norm ** 2
            total_norm = math.sqrt(total_norm)

            # Compute clipping factor
            clip_coef = min(self.l2_norm_clip / (total_norm + 1e-6), 1.)

            # Clip, add noise, and update
            noise_offset = 0
            for group in self.param_groups:
                for param in group['params']:
                    if param.requires_grad and param.grad is not None:
                        if dp_noise is not None:
                            # Clip gradient
                            param.grad.data.mul_(clip_coef)

                            # Add DP noise
                            noise_slice = dp_noise[noise_offset:noise_offset + param.numel()]
                            param.grad.add_(noise_slice.view_as(param))
                            noise_offset += param.numel()

            super(DPOptimizerClass, self).step(*args, **kwargs)

    return DPOptimizerClass

# def make_dp_optimizer(cls):
#     class DPOptimizerClass(cls):
#         def __init__(self, params, *args, **kwargs):
#             super(DPOptimizerClass, self).__init__(params, *args, **kwargs)
#
#         def step(self, dp_noise):
#             super(DPOptimizerClass, self).step()
#
#     return DPOptimizerClass
