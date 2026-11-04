"""Training stabilisers shared by the experiments: lr schedule, gradient clipping, EMA of the weights; and a
diagnosis of the gradient spikes."""
import math, statistics
from collections import deque
import torch


def lr_factor(epoch, args):
    """Multiplier of the base lrs at this epoch: 1 for --lr-schedule constant; for cosine, a linear warmup over
    --lr-warmup epochs then a cosine decay to --lr-final-ratio at --epochs. Stateless, so a resumed run (even
    with a new --epochs) picks it up from the epoch alone."""
    if getattr(args, 'lr_schedule', 'constant') == 'constant':
        return 1.0
    warmup = args.lr_warmup
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = min(1.0, (epoch - warmup) / max(1, args.epochs - warmup))
    return args.lr_final_ratio + (1 - args.lr_final_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def set_lr(optimizer, base_lrs, epoch, args):
    for group, base in zip(optimizer.param_groups, base_lrs):
        group['lr'] = base * lr_factor(epoch, args)


class GradClipper:
    """Clips the gradient norm to --grad-clip (absolute) and/or --grad-clip-rel times the median of the last
    `window` unclipped norms; the relative threshold follows the norms as they shrink during training. Clipped
    norms are left out of the median: counting them (even at the bound) lets a burst of large gradients ratchet
    the threshold up by rel every window/2 steps, so it stops clipping exactly when it is needed. If the norms
    grow for good, every step is clipped to the frozen threshold, a rescaling that Adam mostly normalises away.
    A step with a non-finite norm is skipped. Keeps per-log statistics of the norms, and the norm and the
    median of the last step (last_norm, last_median: None before min_history steps) for SpikeDiagnoser."""

    def __init__(self, max_norm=0.0, rel=0.0, window=1000, min_history=10):
        self.max_norm, self.rel, self.min_history = max_norm, rel, min_history
        self.history = deque(maxlen=window)
        self.last_norm, self.last_median = math.nan, None
        self.reset_stats()

    def reset_stats(self):
        self.norms, self.n_clipped, self.n_skipped = [], 0, 0

    def median(self):
        return statistics.median(self.history) if len(self.history) >= self.min_history else None

    def threshold(self, median=None):
        bounds = []
        if self.max_norm > 0:
            bounds.append(self.max_norm)
        if self.rel > 0 and median is not None:
            bounds.append(self.rel * median)
        return min(bounds) if bounds else math.inf

    def __call__(self, params):
        """Clip the gradients of params in place; returns False if the step must be skipped."""
        self.last_median = self.median()
        bound = self.threshold(self.last_median)
        norm = torch.nn.utils.clip_grad_norm_(params, bound).item()
        self.last_norm = norm
        if not math.isfinite(norm):
            self.n_skipped += 1
            return False
        self.norms.append(norm)
        if norm > bound:
            self.n_clipped += 1
        else:
            self.history.append(norm)
        return True

    def summary(self):
        """Gradient norms since the last reset_stats (pre-clip)."""
        if not self.norms:
            return ' -- grad norm: no step'
        msg = ' -- grad norm: median %.3g | max %.3g' % (statistics.median(self.norms), max(self.norms))
        if self.max_norm > 0 or self.rel > 0:
            msg += ' | clipped %i/%i steps' % (self.n_clipped, len(self.norms))
        if self.n_skipped:
            msg += ' | %i non-finite steps skipped' % self.n_skipped
        return msg

    def state_dict(self):
        return {'history': list(self.history)}

    def load_state_dict(self, state):
        self.history.extend(state['history'])


def term_grad_norms(terms_fn, params):
    """Norm of the gradient of each loss term w.r.t. params. terms_fn() recomputes the terms (dict of (S, 1, 1)
    tensors), averaged over the samples as in the loss without saturation; it is called once per term since a
    compiled backward cannot run twice on a retained graph."""
    params = [p for p in params if p.requires_grad]
    names, norms = list(terms_fn().keys()), {}
    for k in names:
        v = terms_fn()[k]
        grads = [] if not v.requires_grad else [
            g for g in torch.autograd.grad(v.mean(), params, allow_unused=True) if g is not None]
        norms[k] = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads])).item() if grads else 0.0
    return norms


class SpikeDiagnoser:
    """Finds the loss term behind a gradient spike: on a step whose norm exceeds `factor` times the running median
    of the GradClipper, the caller recomputes the batch and logs the gradient norm of each term (report). Within a
    log period, a step is diagnosed again only if at least twice as large as the last diagnosed one, so a long burst
    costs a few diagnoses (one forward and backward per term each), not one per step."""

    def __init__(self, factor):
        self.factor = factor
        self.reset_stats()

    def reset_stats(self):
        self.largest = 0.0

    def triggered(self, clipper):
        norm, median = clipper.last_norm, clipper.last_median
        if median is None or not math.isfinite(norm) or norm <= self.factor * median or norm <= 2 * self.largest:
            return False
        self.largest = norm
        return True

    @staticmethod
    def report(epoch, step, clipper, loss, term_norms):
        terms = ' | '.join('%s: %.3g' % kv for kv in sorted(term_norms.items(), key=lambda kv: -kv[1]))
        return 'grad spike at epoch %i, step %i: norm %.3g (%.3g x median %.3g), batch loss %.4g -- per term: %s' % (
            epoch, step, clipper.last_norm, clipper.last_norm / clipper.last_median, clipper.last_median, loss, terms)


class WeightEMA:
    """Exponential moving average of a module's state dict, updated after every optimizer step with
    decay min(decay, (1 + n) / (10 + n)) at step n, so the early, fast-moving weights are not averaged in for long."""

    def __init__(self, module, decay):
        self.decay, self.n_updates = decay, 0
        self.state = {k: v.detach().clone() for k, v in module.state_dict().items()}

    @torch.no_grad()
    def update(self, module):
        decay = min(self.decay, (1 + self.n_updates) / (10 + self.n_updates))
        for k, v in module.state_dict().items():
            if v.is_floating_point():
                self.state[k].lerp_(v, 1 - decay)
            else:
                self.state[k].copy_(v)
        self.n_updates += 1

    def swap(self, module):
        """Load the EMA weights into module; returns the weights it had, to load back afterwards."""
        current = {k: v.detach().clone() for k, v in module.state_dict().items()}
        module.load_state_dict(self.state)
        return current

    def state_dict(self):
        return {'state': self.state, 'n_updates': self.n_updates}

    def load_state_dict(self, state):
        self.state, self.n_updates = state['state'], state['n_updates']
