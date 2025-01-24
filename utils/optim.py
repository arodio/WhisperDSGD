import torch


def make_dp_optimizer(cls):
    class DPOptimizerClass(cls):
        def __init__(self, params, l2_norm_clip, minibatch_size, microbatch_size, *args, **kwargs):
            super(DPOptimizerClass, self).__init__(params, *args, **kwargs)

            self.l2_norm_clip = l2_norm_clip
            self.minibatch_size = minibatch_size
            self.microbatch_size = microbatch_size

            for group in self.param_groups:
                group['accum_grads'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in
                                        group['params']]

        def zero_microbatch_grad(self):
            super(DPOptimizerClass, self).zero_grad()

        def microbatch_step(self):
            # Compute total norm of microbatch gradients
            total_norm = 0.
            for group in self.param_groups:
                for param in group['params']:
                    if param.requires_grad:
                        total_norm += param.grad.data.norm(2).item() ** 2.
            total_norm = total_norm ** .5

            # Compute clipping factor
            clip_coef = min(self.l2_norm_clip / (total_norm + 1e-6), 1.)

            # Clip and accumulate gradient
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        accum_grad.add_(param.grad.data.mul(clip_coef))

        def zero_grad(self):
            for group in self.param_groups:
                for accum_grad in group['accum_grads']:
                    if accum_grad is not None:
                        accum_grad.zero_()

        def step(self, dp_noise=None, *args, **kwargs):
            noise_offset = 0
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad and param.grad is not None:
                        if dp_noise is not None:
                            # Clone accumulated gradients
                            param.grad.data = accum_grad.clone()
                            # Add DP noise
                            noise_slice = dp_noise[noise_offset:noise_offset + param.numel()]
                            param.grad.add_(noise_slice.view_as(param))
                            noise_offset += param.numel()
                            # Normalize by minibatch size
                            param.grad.data.mul_(self.microbatch_size / self.minibatch_size)

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
