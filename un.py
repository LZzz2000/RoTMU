"""Core RoTMU losses and feature extraction used by the released scripts."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@torch.no_grad()
def get_feature(model, data, logits=False):
    """Extract logits or penultimate-layer features without shuffling samples."""
    if isinstance(data, Dataset):
        data = DataLoader(data, batch_size=64, shuffle=False)

    device = next(model.parameters()).device
    model.eval()
    outputs = []
    for inputs, _ in data:
        inputs = inputs.to(device)
        outputs.append(model(inputs) if logits else model.get_feature(inputs))
    return torch.cat(outputs, dim=0)


def _preservation_loss(model, inputs, targets, flags, noise_level):
    """Unit-weight CE losses for retaining and noise-perturbed easy samples."""
    terms = []
    retaining = flags == 0
    easy = flags == 1

    if retaining.any():
        terms.append(F.cross_entropy(model(inputs[retaining]), targets[retaining]))
    if easy.any():
        noisy_inputs = torch.clamp(
            inputs[easy] + noise_level * torch.randn_like(inputs[easy]), -1.0, 1.0
        )
        terms.append(F.cross_entropy(model(noisy_inputs), targets[easy]))

    return sum(terms) if terms else None


def noise_train_SAM(model, optimizer, unlearning_dl, noise_level=3.0):
    """Run one RoTMU epoch.

    Flags are retaining=0, easy=1, and hard=2.  The L2-SAM perturbation is
    computed only from ell_f=-CE(D_f^h); CE(noisy D_f^e)+CE(D_r) is evaluated
    at the unperturbed parameters and added with unit weights.
    """
    model.train()
    device = next(model.parameters()).device
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    counts = {"retaining": 0, "easy": 0, "hard": 0}

    for inputs, targets, flags in unlearning_dl:
        inputs = inputs.to(device)
        targets = targets.to(device)
        flags = flags.to(device)
        hard = flags == 2

        counts["retaining"] += int((flags == 0).sum())
        counts["easy"] += int((flags == 1).sum())
        counts["hard"] += int(hard.sum())

        optimizer.zero_grad()

        if hard.any():
            # Standard first-order L2-SAM inner step from ell_f=-CE(D_f^h).
            (-F.cross_entropy(model(inputs[hard]), targets[hard])).backward()
            optimizer.first_step(zero_grad=True)

            (-F.cross_entropy(model(inputs[hard]), targets[hard])).backward()
            hard_gradients = [
                None if parameter.grad is None else parameter.grad.detach().clone()
                for parameter in parameters
            ]

            # Return to theta before evaluating the preservation term.
            with torch.no_grad():
                for parameter in parameters:
                    if "old_p" in optimizer.state[parameter]:
                        parameter.copy_(optimizer.state[parameter]["old_p"])
            optimizer.zero_grad()
        else:
            hard_gradients = [None] * len(parameters)

        preservation = _preservation_loss(model, inputs, targets, flags, noise_level)
        if preservation is not None:
            preservation.backward()

        with torch.no_grad():
            for parameter, hard_gradient in zip(parameters, hard_gradients):
                if hard_gradient is None:
                    continue
                if parameter.grad is None:
                    parameter.grad = hard_gradient
                else:
                    parameter.grad.add_(hard_gradient)

        if any(parameter.grad is not None for parameter in parameters):
            optimizer.base_optimizer.step()
        optimizer.zero_grad()

    print(
        "RoTMU samples - retaining: {retaining}, easy: {easy}, hard: {hard}".format(
            **counts
        )
    )
    return counts
