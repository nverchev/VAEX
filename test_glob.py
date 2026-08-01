import pathlib
from src.config import Experiment

cfg = Experiment.get_config()
c_name = cfg.classification.name
print("c_name:", c_name)
checkpoints_dir = cfg.user.path.version_dir / "checkpoints"
print("checkpoints_dir:", checkpoints_dir)
paths = list(checkpoints_dir.glob(f"*/*/*/{c_name}"))
for p in paths:
    print(p)
