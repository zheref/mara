# Copyright © 2024 Apple Inc.

import sys
from pathlib import Path

from setuptools import setup

package_dir = Path(__file__).parent / "mlx_lm"
with open("requirements.txt") as fid:
    requirements = [l.strip() for l in fid.readlines()]

sys.path.append(str(package_dir))
from _version import __version__

setup(
    name="mlx-lm",
    version=__version__,
    description="LLMs with MLX and the Hugging Face Hub",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    author_email="mlx@group.apple.com",
    author="MLX Contributors",
    url="https://github.com/ml-explore/mlx-lm",
    license="MIT",
    install_requires=requirements,
    packages=["mlx_lm", "mlx_lm.models", "mlx_lm.quant", "mlx_lm.tuner"],
    python_requires=">=3.8",
    extras_require={
        "test": ["datasets", "lm-eval"],
        "train": ["datasets", "tqdm"],
        "evaluate": ["lm-eval", "tqdm"],
    },
    entry_points={
        "console_scripts": [
            "mara.awq = mlx_lm.quant.awq:main",
            "mara.dwq = mlx_lm.quant.dwq:main",
            "mara.dynamic_quant = mlx_lm.quant.dynamic_quant:main",
            "mara.gptq = mlx_lm.quant.gptq:main",
            "mara.cache_prompt = mlx_lm.cache_prompt:main",
            "mara.chat = mlx_lm.chat:main",
            "mara.convert = mlx_lm.convert:main",
            "mara.evaluate = mlx_lm.evaluate:main",
            "mara.fuse = mlx_lm.fuse:main",
            "mara.generate = mlx_lm.generate:main",
            "mara.lora = mlx_lm.lora:main",
            "mara.server = mlx_lm.server:main",
            "mara.manage = mlx_lm.manage:main",
            "mara.upload = mlx_lm.upload:main",
        ]
    },
)
