"""Ask whether the classifier reads gender off facial hair, by occluding the chin."""

import logging

import pandas as pd
import torch

from free_cut_sweep import BATCH, CHIN_AXES, CHIN_CENTRE, ellipse
from src.config import AllConfig, Experiment, hydra_main
from src.data.dataset import get_databuilder, get_datasets
from src.data.structures import Inputs
from src.train.models import load_extract_classifier_module

N_SAMPLES = 1024
"""Samples drawn per group."""
CONTROL_CENTRE = (0.28, 0.48)
"""Centre of the control ellipse, mirroring the chin across the face."""


@torch.inference_mode()
def run() -> None:
    classifier = load_extract_classifier_module()
    device = next(classifier.parameters()).device
    _, eval_dataset = get_datasets()

    extra = pd.read_csv(get_databuilder().data_dir / 'list_attr_celeba.txt', sep=r'\s+', skiprows=1, usecols=['No_Beard'])
    extra.reset_index(inplace=True)
    extra.columns = ['image_id', 'No_Beard']
    extra['image_id'] = extra['image_id'].astype(str).str.replace('.jpg', '', regex=False).astype(int)
    joined = eval_dataset.metadata.merge(extra, on='image_id', how='left')

    groups = {
        'bearded men': joined.index[(joined['No_Beard'] == -1) & (joined['Male'] == 1)].to_numpy()[:N_SAMPLES],
        'clean-shaven men': joined.index[(joined['No_Beard'] == 1) & (joined['Male'] == 1)].to_numpy()[:N_SAMPLES],
        'women': joined.index[joined['Male'] == -1].to_numpy()[:N_SAMPLES],
    }

    sample = torch.stack([eval_dataset[int(i)][0].image for i in range(0, 512)]).to(device)
    mean_face = sample.mean(0, keepdim=True)
    size = sample.size(-1)
    chin = ellipse(size, device, CHIN_CENTRE, CHIN_AXES)
    control = ellipse(size, device, CONTROL_CENTRE, CHIN_AXES)
    logging.info('chin pixels %d | control pixels %d', int(chin.sum()), int(control.sum()))
    logging.info('')
    logging.info('%-20s %10s %10s %10s %12s', 'group', 'P(male)', 'chin occ', 'ctrl occ', 'chin effect')

    for name, index in groups.items():
        base, occ_chin, occ_ctrl = [], [], []
        for start in range(0, len(index), BATCH):
            image = torch.stack([eval_dataset[int(i)][0].image for i in index[start: start + BATCH]]).to(device)
            variants = (
                image,
                torch.where(chin, mean_face, image),
                torch.where(control, mean_face, image),
            )
            for store, variant in zip((base, occ_chin, occ_ctrl), variants, strict=True):
                probs = torch.softmax(classifier(Inputs(image=variant)), dim=1)
                store.append(probs[:, 1].cpu())

        p_base = float(torch.cat(base).mean())
        p_chin = float(torch.cat(occ_chin).mean())
        p_ctrl = float(torch.cat(occ_ctrl).mean())
        logging.info(
            '%-20s %10.4f %10.4f %10.4f %12.4f',
            name, p_base, p_chin, p_ctrl, (p_base - p_chin) - (p_base - p_ctrl),
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
