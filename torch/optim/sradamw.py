import torch
from .adamw import AdamW


# TODO(crcrpar): Decide whether to override `state_dict` method as
# this optimizer tracks square root of `exp_avg_sq` and `max_exp_avg_sq`
# that contradicts with its parent optimizer :class:`torch.optim.Adam`.
class SRAdamW(AdamW):
    r"""Implements AdamW algorithm with Stochastic Rounding.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.

    With Stochastic Rounding, param, `exp_avg`, `exp_avg_sq`, and optionally `max_exp_avg_sq`
    can be represented with 16 bits.
    This optimizer requires CUDA.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    _step_supports_amp_scaling = True

    @torch.no_grad()
    def step(self, closure=None, grad_scaler=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if grad_scaler is not None:
            found_inf = grad_scaler._check_inf_per_device(
                self)[torch.device(torch.cuda.current_device())]
            scale = grad_scaler._get_scale_async()
            inv_scale = scale.double().reciprocal().float()
        else:
            found_inf = torch.zeros((1,), dtype=torch.float, device=torch.cuda.current_device())
            inv_scale = torch.ones((1,), dtype=torch.float, device=torch.cuda.current_device())


        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError('SRAdamW does not support sparse gradients')

                state = self.state[param]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                beta1, beta2 = group['betas']

                state['step'] += 1

                torch.stochastic_rounding_adam_step(
                    param, grad,
                    state['exp_avg'], state['exp_avg_sq'], state['max_exp_avg_sq'],
                    inv_scale, found_inf,
                    group['lr'], beta1, beta2,
                    group['weight_decay'], group['eps'], state['step'],
                    True, group['amsgrad'])

        return loss
