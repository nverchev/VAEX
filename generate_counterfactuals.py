"""Generate visual counterfactual images and save image grids for selected test samples."""

import logging
from collections.abc import Sized
import matplotlib.pyplot as plt
import numpy as np
import torch

from drytorch import Model

from src.config import AllConfig, Experiment, hydra_main
from src.data.dataset import get_dataset
from src.data.split import Partitions
from src.data.structures import Inputs
from src.module.classifier import get_classifier
from src.train.models import load_extract_autoencoder_module


def format_probs(probs: np.ndarray, class_names: list[str]) -> str:
    return f'({", ".join(f"{name}: {100 * p:.1f}%" for name, p in zip(class_names, probs, strict=True))})'


def to_numpy_img(tensor: torch.Tensor) -> np.ndarray:
    img_np = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(img_np, 0.0, 1.0)


@torch.inference_mode()
def create_and_render_counterfactuals() -> None:
    cfg = Experiment.get_config()
    cfg_user = cfg.user
    scale_prior = cfg.autoencoder.objective.scale_prior
    save_dir_base = cfg_user.path.version_dir / 'images' / cfg.name
    save_dir_base.mkdir(parents=True, exist_ok=True)
    classifier_module = get_classifier().eval()
    classifier = Model(classifier_module, name=cfg.classification.name, device=cfg_user.device)
    try:
        classifier.load_state()
    except Exception:
        logging.warning('Could not load classifier weights; using initialised parameters.')

    partition = Partitions.test if cfg.final else Partitions.val
    test_dataset = get_dataset(partition)
    assert isinstance(test_dataset, Sized)
    class_names = cfg.data.dataset.classes
    ema_module = load_extract_autoencoder_module()
    sample_indices = cfg_user.plot.sample_indices

    for sample_idx in sample_indices:
        inputs_item, target_item = test_dataset[sample_idx]
        img_tensor = inputs_item.image.unsqueeze(0).to(classifier.device)
        target_label = int(target_item.label.item())
        inputs = Inputs(image=img_tensor, condition=target_item.label.unsqueeze(0).to(classifier.device).float())
        logits_orig = classifier(inputs)
        probs_orig = torch.softmax(logits_orig, dim=1).cpu().numpy()[0]
        inputs = inputs._replace(logits=logits_orig)
        out_recon = ema_module(inputs)
        recon_img = out_recon.recon if out_recon.recon is not None else img_tensor
        logits_recon = classifier(Inputs(image=recon_img))
        probs_recon = torch.softmax(logits_recon, dim=1).cpu().numpy()[0]
        n_delta = cfg_user.plot.n_delta
        cf_grids = []
        for target_cls in range(len(class_names)):
            row_imgs = []
            row_probs = []
            for step in range(n_delta, -1, -1):
                scale_s = step / float(n_delta)
                out_cf = ema_module.generate_counterfactual(
                    inputs,
                    target_dim=target_cls,
                    target_value=scale_prior,
                    posterior_weight=scale_s,
                )
                cf_img = out_cf.recon if out_cf.recon is not None else img_tensor
                l_cf = classifier(Inputs(image=cf_img))
                p_cf = torch.softmax(l_cf, dim=1).cpu().numpy()[0]
                row_imgs.append(to_numpy_img(cf_img[0]))
                row_probs.append(p_cf)

            cf_grids.append((row_imgs, row_probs))

        _fig, axes = plt.subplots(len(class_names) + 1, n_delta + 1, figsize=(18, 4 * (len(class_names) + 1)))

        # Top row: Original & Reconstruction
        axes[0, 0].imshow(to_numpy_img(img_tensor[0]))
        axes[0, 0].set_title(f'Original ({class_names[target_label]}: {probs_orig[target_label]:.2f})')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(to_numpy_img(recon_img[0]))
        axes[0, 1].set_title(f'Recon ({class_names[target_label]}: {probs_recon[target_label]:.2f})')
        axes[0, 1].axis('off')

        for c in range(2, n_delta + 1):
            axes[0, c].axis('off')

        for r, (row_imgs, row_probs) in enumerate(cf_grids):
            for c, (img, probs) in enumerate(zip(row_imgs, row_probs, strict=False)):
                ax = axes[r + 1, c]
                ax.imshow(img)
                target_prob = probs[r]
                ax.set_title(f'{class_names[r]}: {target_prob:.2f}')
                ax.axis('off')

        plt.tight_layout()
        save_path = save_dir_base / f'counterfactual_sample_{sample_idx}.png'
        plt.savefig(save_path)
        plt.close()

    return None


@hydra_main
def main(cfg: AllConfig) -> None:
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.user.path.version_dir, tags=cfg.tags)
    with exp.create_run(resume=True):
        create_and_render_counterfactuals()

    return None


if __name__ == '__main__':
    main()
