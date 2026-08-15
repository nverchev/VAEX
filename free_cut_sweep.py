"""Ask whether cutting the free half removes facial hair, and what identity it costs."""

import logging

import pandas as pd
import torch

from stage_ablation import FACE_AXES, FACE_CENTRE, outside_face
from src.config import AllConfig, Experiment, hydra_main
from src.data.derived_datasets import EvaluatedDataset
from src.data.dataset import get_databuilder, get_datasets
from src.data.structures import Inputs
from src.train.models import load_extract_classifier_module, load_extract_generative_module

N_SAMPLES = 512
"""Bearded men drawn for the sweep."""
BATCH = 16
"""Samples per forward pass."""
CHIN_CENTRE = (0.72, 0.48)
"""Centre of the chin ellipse, as fractions of the image."""
CHIN_AXES = (0.18, 0.26)
"""Semi-axes of the chin ellipse, as fractions of the image."""
FREE_LEVELS = (1.0, 0.75, 0.5, 0.25, 0.0)
"""Weights applied to the free half at every stage."""


def ellipse(size: int, device: torch.device, centre: tuple[float, float], axes: tuple[float, float]) -> torch.Tensor:
    """Boolean mask selecting the inside of an ellipse."""
    rows = torch.linspace(0.0, 1.0, size, device=device)[:, None]
    cols = torch.linspace(0.0, 1.0, size, device=device)[None, :]
    return (((rows - centre[0]) / axes[0]) ** 2 + ((cols - centre[1]) / axes[1]) ** 2) <= 1.0


@torch.inference_mode()
def run() -> None:
    module = load_extract_generative_module()
    device = next(module.parameters()).device
    classifier = load_extract_classifier_module()
    _, eval_dataset = get_datasets()
    evaluated = EvaluatedDataset(eval_dataset, classifier)

    extra = pd.read_csv(get_databuilder().data_dir / 'list_attr_celeba.txt', sep=r'\s+', skiprows=1, usecols=['No_Beard'])
    extra.reset_index(inplace=True)
    extra.columns = ['image_id', 'No_Beard']
    extra['image_id'] = extra['image_id'].astype(str).str.replace('.jpg', '', regex=False).astype(int)
    joined = eval_dataset.metadata.merge(extra, on='image_id', how='left')
    bearded = joined.index[(joined['No_Beard'] == -1) & (joined['Male'] == 1)].to_numpy()[:N_SAMPLES]

    size = module.x_dim
    chin = ellipse(size, device, CHIN_CENTRE, CHIN_AXES)
    face = ellipse(size, device, FACE_CENTRE, FACE_AXES)
    context = outside_face(size, device)
    rest = face & ~chin

    cut = tuple([0.0] * len(module.z_dims))
    logging.info('bearded men: %d | posterior cut on the conditioned half is total', len(bearded))
    logging.info('%9s %9s %11s %11s %11s', 'free', 'flip', 'chin drift', 'face drift', 'context')
    for level in FREE_LEVELS:
        free = tuple([level] * len(module.z_dims))
        flips, chins, rests, contexts = [], [], [], []
        for start in range(0, len(bearded), BATCH):
            index_list = bearded[start: start + BATCH].tolist()
            pairs = evaluated.__getitems__(index_list)
            image = torch.stack([p[0].image for p in pairs]).to(device)
            condition = torch.stack([p[0].condition for p in pairs]).to(device)
            log_probs = torch.stack([p[0].log_probs for p in pairs]).to(device)
            inputs = Inputs(image, condition, log_probs)

            recon = module(inputs).recon
            target = int(1 - log_probs[0].argmax())
            out_cf = module.generate_counterfactual(
                inputs, target_dim=target, posterior_weights=cut, free_weights=free
            )
            delta = (out_cf.recon - recon) ** 2
            probs = torch.softmax(classifier(Inputs(image=out_cf.recon)), dim=1)
            flips.append(probs[:, target].cpu())
            chins.append(delta[:, :, chin].mean((1, 2)).cpu())
            rests.append(delta[:, :, rest].mean((1, 2)).cpu())
            contexts.append(delta[:, :, context].mean((1, 2)).cpu())

        logging.info(
            '%9.2f %9.4f %11.5f %11.5f %11.5f',
            level,
            float(torch.cat(flips).mean()),
            float(torch.cat(chins).mean()),
            float(torch.cat(rests).mean()),
            float(torch.cat(contexts).mean()),
        )

    return None


@hydra_main
def main(cfg: AllConfig) -> None:
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    with exp.create_run(resume=True, record=False):
        run()

    return None


if __name__ == '__main__':
    main()
