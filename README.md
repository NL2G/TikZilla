# TikZilla: Scaling Text-to-TikZ with High-Quality Data and Reinforcement Learning

<p align="center">
  <img src="TikZilla_Logo.png" width="400">
</p>

<p align="center"> <a href="https://arxiv.org/abs/2603.03072"> <img src="https://img.shields.io/badge/arXiv-2603.03072-B31B1B.svg"> </a> <a href="https://openreview.net/forum?id=rJv2byEWA3"> <img src="https://img.shields.io/badge/ICLR-2026-blue"> </a> <a href="https://huggingface.co/collections/nllg/tikzilla"> <img src="https://img.shields.io/badge/🤗%20HuggingFace-Models%20%26%20Datasets-yellow"> </a> <a href="https://github.com/NL2G/TikZilla"> <img src="https://img.shields.io/badge/GitHub-Code-black?logo=github"> </a> </p>

TikZilla is a family of open-source language models for **Text-to-TikZ generation**, enabling high-quality scientific figures to be synthesized directly from natural language descriptions.

Built on **DaTikZ-V4** and trained with supervised fine-tuning and reinforcement learning, TikZilla produces TikZ figures that are visually coherent, semantically aligned, and executable as LaTeX graphics programs.

## Models & Datasets

All TikZilla models and datasets are available on Hugging Face:

https://huggingface.co/collections/nllg/tikzilla

## Overview

TikZilla addresses key challenges in Text-to-TikZ generation:

- noisy and limited training data
- weak caption-to-figure alignment
- low compilation rates
- poor spatial reasoning
- hallucinated or incomplete figure elements

The project introduces:

1. **DaTikZ-V4**, a large-scale high-quality Text-to-TikZ dataset.
2. **Reinforcement learning for Text-to-TikZ**, using visual-semantic reward signals.
3. **Compact open models** that achieve strong performance against proprietary baselines.

## Citation
If TikZila have been beneficial for your research or applications, we kindly request you to acknowledge this by citing them as follows:

```bibtex
@inproceedings{greisinger2026tikzilla,
    title={TikZilla: Scaling Text-to-TikZ with High-Quality Data and Reinforcement Learning},
    author={Christian Greisinger and Steffen Eger},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=rJv2byEWA3}
}
```