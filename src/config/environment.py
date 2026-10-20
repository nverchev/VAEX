"""Environment-related configuration and utilities."""

import enum
import inspect
import pathlib
import shutil
import subprocess
import sys
import tomllib

from pydantic_settings import BaseSettings, SettingsConfigDict


with open(pathlib.Path(__file__).resolve().parent.parent.parent / 'pyproject.toml', 'rb') as f:
    pyproject = tomllib.load(f)

PROJECT_NAME: str = pyproject['project']['name']
VERSION: str = pyproject['project']['version']


def _runs_under_debugger() -> bool:
    if sys.gettrace() is not None or any(module in sys.modules for module in ('pydevd', 'debugpy')):
        return True

    # pdb and the debuggers built on bdb run the script inside bdb.run, also after removing their trace
    frame = inspect.currentframe()
    while frame is not None:
        if frame.f_globals.get('__name__') == 'bdb':
            return True

        frame = frame.f_back

    return False


# a debugger runs the process: no data of the experiment is written
DEBUG_MODE: bool = _runs_under_debugger()


class EnvSettings(BaseSettings):
    """Settings loaded from environment variables or the ``.env`` file.

    Attributes:
        data_dir: Root directory for the datasets, shared by all versions.
        metadata_dir: Root directory for the dataset metadata.
        root_exp_dir: Root directory for all experiment artefacts.
    """

    data_dir: pathlib.Path = pathlib.Path('./data')
    metadata_dir: pathlib.Path = pathlib.Path('./dataset_metadata')
    root_exp_dir: pathlib.Path = pathlib.Path('./experiments')
    model_config = SettingsConfigDict(env_file='.env')


class ConfigPath(enum.StrEnum):
    """Registered configuration directory names (relative to ``configs/``)."""

    EXPERIMENT = 'experiment'
    TUNING = 'tuning'

    @classmethod
    def get_folder(cls) -> str:
        """Top-level config folder name."""
        return 'configs'

    def get_path(self) -> pathlib.Path:
        """Absolute :class:`~pathlib.Path` to this config group directory."""
        return pathlib.Path(__file__).parent.parent.parent / self.get_folder() / self

    def absolute(self) -> str:
        """Absolute string path (used by :func:`hydra.main`)."""
        return str(self.get_path().absolute().resolve())

    def relative(self) -> str:
        """Relative string path (used by :func:`hydra.initialize`)."""
        return f'../../{self.get_folder()}/{self}'


def get_current_branch() -> str:
    """Return the current git branch name.

    Returns:
        The name of the currently checked-out branch.

    Raises:
        ValueError: If ``git`` is not found in PATH, or HEAD is detached.
    """
    git_command = shutil.which('git')
    if git_command is None:
        raise ValueError('git not found in PATH.')

    try:
        result = subprocess.run(
            [git_command, 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        msg = f'Failed to determine git branch: {e.stderr.strip()}'
        raise ValueError(msg) from e

    branch = result.stdout.strip()
    if branch == 'HEAD':
        msg = 'Detached HEAD state detected. Please checkout a branch.'
        raise ValueError(msg)

    return branch
