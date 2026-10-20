"""Publication of the figures of the samples and of their counterfactuals."""

from src.config import Experiment
from src.train.counterfactuals import CounterfactualSweep
from src.train.events import ImageGridEvent

__all__ = [
    'publish_counterfactual_sweep',
]


def publish_counterfactual_sweep(sweep: CounterfactualSweep, /, *, folder: str) -> None:
    """Publish a sample and its counterfactuals as a grid, with the probabilities of the classifier.

    The first row holds the sample, its reconstruction and its simple cut; each further row turns
    the sample towards one class, as the relaxation goes from one towards zero.

    Args:
        sweep: the sample and its counterfactuals.
        folder: the folder of the sample.
    """
    class_names = Experiment.get_config().data.dataset.classes
    original, recon, cut = sweep.probabilities.tolist()
    images = [[sweep.original, sweep.recon, sweep.cut]]
    captions = [
        [
            f'Original ({class_names[sweep.label]}: {original:.2f})',
            f'Reconstruction ({recon:.2f})',
            f'Simple cut ({cut:.2f})',
        ]
    ]
    for target, (row, probabilities) in enumerate(
        zip(sweep.counterfactuals, sweep.counterfactual_probabilities, strict=True)
    ):
        images.append(list(row.unbind()))
        captions.append(
            [
                f'r = {relaxation:.2f} ({class_names[target]}: {probability:.2f})'
                for relaxation, probability in zip(sweep.relaxations, probabilities.tolist(), strict=True)
            ]
        )

    ImageGridEvent(images=images, captions=captions, title='counterfactuals', folder=folder)
    return
