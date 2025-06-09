# Learning contrast and content representations for synthesizing magnetic resonance image of arbitrary contrast (MedIA 2025)

This repository implements a deep learning framework for synthesizing MRI scans with arbitrary contrasts. The method learning contrast and content representation to generate high-fidelity MRI images across different contrast domains.

## 🔍 Highlights

- ✔️ Supports synthesis across multiple MRI contrasts (e.g., T1, T2, FLAIR)
- ✔️ Flexible training on multi-parametric datasets

---

## 🏗️ Usage

###  Many-to-Many MRI Contrast Synthesis

To train the model on the BRATS dataset for general contrast-to-contrast synthesis:

```bash
python train_brats.py --config configs/train_config.yaml
```

###  Multi-Parameter & Zero-Shot Contrast Synthesis

To train with multi-parametric inputs and enable zero-shot synthesis on unseen contrasts:

```bash
python train_multi_param.py --config configs/multi_param_config.yaml
```
