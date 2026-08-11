"""Inspect the latent space of a trained autoencoder for collapse and related issues.

Reports, per latent stage: how much information the stage carries, how it is spread over
its dimensions, whether the prior scale has run to its bound, and how much of the class
the stage encodes. The last one matters for counterfactuals: whatever the latent already
says about the class, the condition has to overrule.
"""

import logging

import torch
import torch.nn.functional as F

from src.config import AllConfig, Experiment, hydra_main
from src.data.structures import Inputs
from src.module.autoencoder import PRIOR_LOGVAR_BOUND
from src.train.loaders import get_loaders
from src.train.models import load_extract_autoencoder_module, load_extract_classifier_module

N_BATCHES = 125
DEAD_KL = 0.01  # nats per dimension below which a dimension carries essentially nothing


@torch.no_grad()
def collect(module, loader, device, n_batches: int):
    """Accumulate per-dimension KL, prior scales and latents over a few batches."""
    kl_sum: list[torch.Tensor] = []
    p_logvar_sum: list[torch.Tensor] = []
    d_mu_sq_sum: list[torch.Tensor] = []
    roots: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    n_seen = 0
    for i, (inputs, targets) in enumerate(loader):
        if i >= n_batches:
            break

        inputs = Inputs(*(tensor.to(device) for tensor in inputs))
        out = module(inputs)
        batch = inputs.image.size(0)
        for stage, (d_mu, d_logvar, p_logvar) in enumerate(
            zip(out.d_mu, out.d_logvar, out.prior_log_var, strict=True)
        ):
            # per-element KL of the residual parameterisation, summed over the batch
            kl = 0.5 * (-1 - d_logvar + d_logvar.exp() + d_mu**2 * torch.exp(-p_logvar))
            if i == 0:
                kl_sum.append(kl.sum(0))
                p_logvar_sum.append(p_logvar.sum(0))
                d_mu_sq_sum.append((d_mu**2).sum(0))
            else:
                kl_sum[stage] += kl.sum(0)
                p_logvar_sum[stage] += p_logvar.sum(0)
                d_mu_sq_sum[stage] += (d_mu**2).sum(0)

        roots.append(out.d_mu[0].float().cpu())
        labels.append(targets.label.cpu())
        n_seen += batch

    kl = [total / n_seen for total in kl_sum]
    p_logvar = [total / n_seen for total in p_logvar_sum]
    d_mu_sq = [total / n_seen for total in d_mu_sq_sum]
    return kl, p_logvar, d_mu_sq, torch.cat(roots), torch.cat(labels), n_seen


def probe(features: torch.Tensor, labels: torch.Tensor, steps: int = 400) -> float:
    """Held-out accuracy of a linear probe predicting the label from the latent."""
    features = (features - features.mean(0)) / features.std(0).clamp_min(1e-6)
    split = int(0.8 * features.size(0))
    x_train, x_test = features[:split], features[split:]
    y_train, y_test = labels[:split], labels[split:]
    layer = torch.nn.Linear(features.size(1), int(labels.max()) + 1)
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-2, weight_decay=1e-3)
    with torch.enable_grad():
        for _ in range(steps):
            optimizer.zero_grad()
            F.cross_entropy(layer(x_train), y_train).backward()
            optimizer.step()

    return float((layer(x_test).argmax(1) == y_test).float().mean())


@torch.no_grad()
def report() -> None:
    """Log the latent diagnostics for the configured checkpoint."""
    cfg = Experiment.get_config()
    module = load_extract_autoencoder_module()
    device = next(module.parameters()).device
    classifier = load_extract_classifier_module() if cfg.autoencoder.model.name.name == 'VAEX' else None
    _, eval_loader = get_loaders(cfg.autoencoder.train.batch_size_per_device, classifier=classifier)
    kl, p_logvar, d_mu_sq, roots, labels, n_seen = collect(module, eval_loader, device, N_BATCHES)

    logging.info('samples: %d | sigma_2: %s', n_seen, cfg.autoencoder.objective.sigma_2)
    header = f'{"stage":>5} {"dims":>6} {"KL":>9} {"KL/dim":>8} {"dead%":>7} {"KL p90/p50":>11} '
    header += f'{"p_logvar":>9} {"at bound%":>10}'
    logging.info(header)
    for stage, (stage_kl, stage_p, _) in enumerate(zip(kl, p_logvar, d_mu_sq, strict=True)):
        flat_kl = stage_kl.flatten().float()
        flat_p = stage_p.flatten().float()
        dead = float((flat_kl < DEAD_KL).float().mean())
        p50 = float(flat_kl.median())
        p90 = float(flat_kl.quantile(0.9))
        at_bound = float((flat_p.abs() > 0.98 * PRIOR_LOGVAR_BOUND).float().mean())
        logging.info(
            '%5d %6d %9.1f %8.4f %6.1f%% %11.1f %9.3f %9.1f%%',
            stage,
            flat_kl.numel(),
            float(flat_kl.sum()),
            float(flat_kl.mean()),
            100 * dead,
            p90 / max(p50, 1e-9),
            float(flat_p.mean()),
            100 * at_bound,
        )

    # How far the condition can move the root, against how far the encoder already moved it.
    one_hot = torch.eye(module.dim_target, device=device)
    p_mu, _ = module.root_prior(one_hot)
    class_shift = float((p_mu[1] - p_mu[0]).norm())
    encoder_shift = float(roots.norm(dim=1).mean())
    logging.info('')
    logging.info('root class shift ||p_mu(1) - p_mu(0)||: %.3f', class_shift)
    logging.info('root encoder offset  mean ||d_mu||   : %.3f', encoder_shift)
    logging.info('ratio (class / encoder)              : %.3f', class_shift / max(encoder_shift, 1e-9))

    # A counterfactual rewrites p_mu and leaves d_mu alone, so whatever class displacement
    # sits in d_mu is the part of the gap the intervention cannot close.
    delta_p = (p_mu[1] - p_mu[0]).float().cpu()
    delta_d = roots[labels == 1].mean(0) - roots[labels == 0].mean(0)
    total = delta_d + delta_p
    cosine = float(F.cosine_similarity(delta_d, delta_p, dim=0))
    logging.info('')
    logging.info('class displacement in p_mu (rewritten) : %.3f', float(delta_p.norm()))
    logging.info('class displacement in d_mu (left as is): %.3f', float(delta_d.norm()))
    logging.info('cosine between them                    : %.3f', cosine)
    logging.info('total class separation in z_root       : %.3f', float(total.norm()))
    logging.info('fraction the counterfactual supplies   : %.3f', float(delta_p.norm() / total.norm()))

    accuracy = probe(roots, labels)
    logging.info('')
    logging.info('linear probe on root d_mu -> label: %.3f accuracy', accuracy)
    logging.info('label balance: %.3f', float(labels.float().mean()))
    split_class_content(module, eval_loader, device, N_BATCHES)
    per_stage_split_probe(module, eval_loader, device, N_BATCHES)
    live_dimension_probe(module, eval_loader, device, N_BATCHES)
    per_stage_class_content(module, eval_loader, device, N_BATCHES)
    return


def cond_widths(module) -> list[int]:
    """Flat width of the conditioned half at each stage, or 0 where the stage is not split.

    Mirrors the construction in the decoder: a stage of `z_dim` channels gives `z_dim // 2`
    to the conditioned half, and each channel occupies ``((res + 1) // 2) ** 2`` positions
    once the stride-2 projection has been flattened.
    """
    widths = [getattr(module, 'flat_z_cond_dim', 0)]
    z_dims = list(reversed(module.c_z_dims))
    resolutions = list(reversed(module.resolutions))
    for z_dim, res in zip(z_dims, resolutions, strict=True):
        spatial = ((res + 1) // 2) ** 2
        widths.append(max(0, min(module.z_cond_dim, z_dim - 1)) * spatial)

    return widths


@torch.no_grad()
def per_stage_split_probe(module, loader, device, n_batches: int) -> None:
    """Probe the conditioned and free halves of every split stage.

    With the split applied throughout, the adversary is meant to hold the free half at
    chance at every level, not only at the root.
    """
    widths = cond_widths(module)
    if not any(widths):
        return

    means: list[list[torch.Tensor]] = []
    offsets: list[list[torch.Tensor]] = []
    labels: list[torch.Tensor] = []
    for i, (inputs, targets) in enumerate(loader):
        if i >= n_batches:
            break

        inputs = Inputs(*(tensor.to(device) for tensor in inputs))
        out = module(inputs)
        for stage, (p_mu, d_mu) in enumerate(zip(out.prior_mu, out.d_mu, strict=True)):
            if i == 0:
                means.append([])
                offsets.append([])

            means[stage].append((p_mu + d_mu).float().cpu())
            offsets[stage].append(d_mu.float().cpu())

        labels.append(targets.label.cpu())

    y = torch.cat(labels)
    base = float(max(y.float().mean(), 1 - y.float().mean()))
    logging.info('')
    logging.info('mu = what the decoder reads; d = the residual an intervention keeps')
    logging.info(
        '%5s %7s %7s %9s %9s %9s %9s  (base rate %.3f)',
        'stage', 'cond', 'free', 'mu cond', 'mu free', 'd cond', 'd free', base,
    )
    for stage, parts in enumerate(means):
        mu = torch.cat(parts)
        d_mu = torch.cat(offsets[stage])
        split = widths[stage] if stage < len(widths) else 0
        if not split or split >= mu.size(1):
            # Not split: the stage is conditioned throughout, so it is all 'cond'.
            logging.info(
                '%5d %7d %7s %9.3f %9s %9.3f %9s',
                stage, mu.size(1), '-', probe(mu, y), '-', probe(d_mu, y), '-',
            )
            continue

        logging.info(
            '%5d %7d %7d %9.3f %9.3f %9.3f %9.3f',
            stage,
            split,
            mu.size(1) - split,
            probe(mu[:, :split], y),
            probe(mu[:, split:], y),
            probe(d_mu[:, :split], y),
            probe(d_mu[:, split:], y),
        )

    return


@torch.no_grad()
def live_dimension_probe(module, loader, device, n_batches: int) -> None:
    """Probe each stage for the class using only the dimensions that carry information.

    A stage whose dimensions are mostly dead dilutes a probe over all of them, so a
    localised signal - a beard occupying a few positions at a fine scale - can read as
    chance. Restricting to dimensions above a KL floor asks whether the class is in the
    part of the stage that is actually being used.
    """
    stage_mu: list[list[torch.Tensor]] = []
    stage_kl: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    n_seen = 0
    for i, (inputs, targets) in enumerate(loader):
        if i >= n_batches:
            break

        inputs = Inputs(*(tensor.to(device) for tensor in inputs))
        out = module(inputs)
        for stage, (d_mu, d_logvar, p_logvar) in enumerate(
            zip(out.d_mu, out.d_logvar, out.prior_log_var, strict=True)
        ):
            kl = 0.5 * (-1 - d_logvar + d_logvar.exp() + d_mu**2 * torch.exp(-p_logvar))
            if i == 0:
                stage_mu.append([])
                stage_kl.append(kl.sum(0).float().cpu())
            else:
                stage_kl[stage] += kl.sum(0).float().cpu()

            stage_mu[stage].append(d_mu.float().cpu())

        labels.append(targets.label.cpu())
        n_seen += inputs.image.size(0)

    y = torch.cat(labels)
    base = float(max(y.float().mean(), 1 - y.float().mean()))
    logging.info('')
    logging.info('%5s %8s %8s %12s %12s  (base rate %.3f)', 'stage', 'dims', 'live', 'probe(all)', 'probe(live)', base)
    for stage, parts in enumerate(stage_mu):
        mu = torch.cat(parts)
        kl = (stage_kl[stage] / n_seen).flatten()
        live = (kl >= DEAD_KL).nonzero().flatten()
        if live.numel() < 2:
            logging.info('%5d %8d %8d %12.3f %12s', stage, kl.numel(), live.numel(), probe(mu, y), 'n/a')
            continue

        logging.info(
            '%5d %8d %8d %12.3f %12.3f',
            stage, kl.numel(), live.numel(), probe(mu, y), probe(mu[:, live], y)
        )

    return


@torch.no_grad()
def split_class_content(module, loader, device, n_batches: int) -> None:
    """Report the conditioned and unconditioned halves of the root separately.

    The adversary is meant to empty the free half of class information while the
    conditioned half keeps it, so the two have to be read apart. The log-variance of the
    free half is included because it is the one channel a mean-only adversary cannot see.
    """
    split = getattr(module, 'flat_z_cond_dim', 0)
    if not split:
        return

    means: list[torch.Tensor] = []
    offsets: list[torch.Tensor] = []
    logvars: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for i, (inputs, targets) in enumerate(loader):
        if i >= n_batches:
            break

        inputs = Inputs(*(tensor.to(device) for tensor in inputs))
        out = module(inputs)
        means.append((out.prior_mu[0] + out.d_mu[0]).float().cpu())
        offsets.append(out.d_mu[0].float().cpu())
        logvars.append(out.d_logvar[0].float().cpu())
        labels.append(targets.label.cpu())

    mu = torch.cat(means)
    d_mu = torch.cat(offsets)
    logvar = torch.cat(logvars)
    y = torch.cat(labels)
    halves = {'cond': slice(0, split), 'free': slice(split, mu.size(1))}
    logging.info('')
    logging.info('%6s %6s %10s %14s', 'half', 'dims', 'probe(mu)', '||delta d_mu||')
    for name, part in halves.items():
        delta = d_mu[y == 1][:, part].mean(0) - d_mu[y == 0][:, part].mean(0)
        logging.info('%6s %6d %10.3f %14.3f', name, mu[:, part].size(1), probe(mu[:, part], y), float(delta.norm()))

    logging.info('probe on free log-variance: %.3f  (base rate %.3f)',
                 probe(logvar[:, halves['free']], y),
                 float(max(y.float().mean(), 1 - y.float().mean())))
    return


@torch.no_grad()
def confidence_residual(module, loader, device, n_batches: int) -> None:
    """Test whether the class left in the root offset is classifier-confidence in disguise.

    The root prior is affine in the condition, so two images of one class with different
    classifier confidence get different prior means, and the encoder has to cancel the
    difference. If that is what ``d_mu`` is doing, its class content should vanish once
    the confidence-linear component is removed.
    """
    roots: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    confidences: list[torch.Tensor] = []
    for i, (inputs, targets) in enumerate(loader):
        if i >= n_batches:
            break

        inputs = Inputs(*(tensor.to(device) for tensor in inputs))
        out = module(inputs)
        roots.append(out.d_mu[0].float().cpu())
        labels.append(targets.label.cpu())
        confidences.append(module.condition_from(inputs)[:, 1].float().cpu())

    d_mu = torch.cat(roots)
    y = torch.cat(labels)
    p = torch.cat(confidences)

    # Centre within each class so the slope reflects confidence alone, not the class.
    d_centred = d_mu.clone()
    p_centred = p.clone()
    for cls in (0, 1):
        mask = y == cls
        d_centred[mask] -= d_mu[mask].mean(0)
        p_centred[mask] -= p[mask].mean()

    slope = (d_centred * p_centred[:, None]).sum(0) / (p_centred**2).sum()
    residual = d_centred - slope[None, :] * p_centred[:, None]
    r2 = 1 - float(residual.var(0).sum() / d_centred.var(0).sum())

    delta_p = (module.root_prior(torch.eye(module.dim_target, device=device))[0][1]
               - module.root_prior(torch.eye(module.dim_target, device=device))[0][0]).float().cpu()
    logging.info('')
    logging.info('within-class confidence spread (std of p): %.4f', float(p_centred.std()))
    logging.info('variance of d_mu explained by confidence : %.4f', r2)
    logging.info('cosine(slope, delta_p_mu)                : %+.3f  (compensation predicts -1)',
                 float(F.cosine_similarity(slope, delta_p, dim=0)))

    # Strip the confidence-linear part and ask the probe again.
    stripped = d_mu - slope[None, :] * (p - p.mean())[:, None]
    logging.info('probe on d_mu                            : %.3f', probe(d_mu, y))
    logging.info('probe on d_mu minus confidence component : %.3f', probe(stripped, y))
    logging.info('base rate                                : %.3f', float(max(y.float().mean(), 1 - y.float().mean())))
    return


@torch.no_grad()
def per_stage_class_content(module, loader, device, n_batches: int) -> None:
    """Locate the class inside each stage's posterior offset, and see if it can respond.

    A counterfactual rewrites the priors but never the encoder. Whatever class content
    survives in ``d_mu`` is what the intervention has to overpower, so it matters both how
    much each stage holds and whether swapping the condition moves it at all.
    """
    stage_mu: list[list[torch.Tensor]] = []
    stage_mu_cf: list[list[torch.Tensor]] = []
    labels: list[torch.Tensor] = []
    for i, (inputs, targets) in enumerate(loader):
        if i >= n_batches:
            break

        inputs = Inputs(*(tensor.to(device) for tensor in inputs))
        out = module(inputs)
        # Swap every sample to the class it is not, exactly as a counterfactual would.
        flipped = 1 - targets.label.to(device)
        cf_condition = F.one_hot(flipped, module.dim_target).to(inputs.image.dtype)
        out_cf = module.decode(module.encode(inputs.image), cf_condition, module.full_posterior_weights)
        for stage, (mu, mu_cf) in enumerate(zip(out.d_mu, out_cf.d_mu, strict=True)):
            if i == 0:
                stage_mu.append([])
                stage_mu_cf.append([])

            stage_mu[stage].append(mu.float().cpu())
            stage_mu_cf[stage].append(mu_cf.float().cpu())

        labels.append(targets.label.cpu())

    y = torch.cat(labels)
    logging.info('')
    logging.info(
        '%5s %8s %12s %10s %12s', 'stage', 'probe', '||d class||', 'cos w/ cf', 'cf response'
    )
    for stage, (mu_parts, mu_cf_parts) in enumerate(zip(stage_mu, stage_mu_cf, strict=True)):
        mu = torch.cat(mu_parts)
        mu_cf = torch.cat(mu_cf_parts)
        delta = mu[y == 1].mean(0) - mu[y == 0].mean(0)
        # How far the stage's own posterior moves when only the condition changes.
        response = float((mu_cf - mu).norm(dim=1).mean() / mu.norm(dim=1).mean())
        cosine = float(F.cosine_similarity((mu_cf - mu).mean(0), -delta, dim=0))
        logging.info(
            '%5d %8.3f %12.3f %10.3f %11.1f%%',
            stage,
            probe(mu, y),
            float(delta.norm()),
            cosine,
            100 * response,
        )

    return


@hydra_main
def main(cfg: AllConfig) -> None:
    """Set up the experiment and run the latent diagnostics."""
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    with exp.create_run(resume=True):
        report()

    return None


if __name__ == '__main__':
    main()
