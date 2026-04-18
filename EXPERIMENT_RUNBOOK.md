# Experiment Runbook (HW3: LFW Finetuning)

这个文档用于“真正执行实验并记录结果”。

## 1) 环境检查

```bash
python -V
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
```

## 1.5) 数据检查

确保手动下载好的 LFW 在 `data/lfw-py/` 下，至少包含：

- `lfw-deepfunneled/`
- `pairs.txt`、`pairsDevTrain.txt`、`pairsDevTest.txt`
- `people.txt`、`peopleDevTrain.txt`、`peopleDevTest.txt`

## 2) Frozen baseline

```bash
python src/evaluate_verification.py --model alexnet --checkpoint none --metric cosine --out_dir outputs/alexnet_frozen
python src/evaluate_verification.py --model vgg16   --checkpoint none --metric cosine --out_dir outputs/vgg16_frozen
```

## 3) Finetune

```bash
python src/train_finetune.py --model alexnet --epochs 5 --batch_size 32 --lr 1e-4 --freeze_backbone false --out_dir outputs/alexnet_ft
python src/train_finetune.py --model vgg16   --epochs 5 --batch_size 16 --lr 1e-4 --freeze_backbone false --out_dir outputs/vgg16_ft
```

## 4) Verification after finetune

```bash
python src/evaluate_verification.py --model alexnet --checkpoint outputs/alexnet_ft/best.pt --metric cosine --out_dir outputs/alexnet_ft_eval
python src/evaluate_verification.py --model vgg16   --checkpoint outputs/vgg16_ft/best.pt --metric cosine --out_dir outputs/vgg16_ft_eval
```

## 5) Quick mode (for fast smoke test)

```bash
python src/train_finetune.py --model alexnet --epochs 1 --batch_size 32 --max_train_samples 800 --max_val_samples 400 --out_dir outputs/quick_alexnet_ft
python src/evaluate_verification.py --model alexnet --checkpoint outputs/quick_alexnet_ft/best.pt --max_pairs 1000 --out_dir outputs/quick_alexnet_ft_eval
```

## 6) Result Table

| Setting | Checkpoint | AUC | EER | Best Threshold | Notes |
|---|---|---:|---:|---:|---|
| AlexNet Frozen | `none` | 0.754839 | 0.309316 | 0.533085 | Baseline feature extractor with ImageNet weights. |
| AlexNet Finetuned | `outputs/alexnet_ft/best.pt` | 0.876314 | 0.210945 | 0.609274 | Large gain after finetuning on LFW identities. |
| VGG16 Frozen | `none` | 0.748849 | 0.310358 | 0.539412 | Baseline feature extractor with ImageNet weights. |
| VGG16 Finetuned | `outputs/vgg16_ft/best.pt` | 0.906191 | 0.172978 | 0.741527 | Best overall verification quality among all settings. |

### 6.1) Improvement Summary

| Model | AUC Gain (Finetuned - Frozen) | EER Reduction (Frozen - Finetuned) |
|---|---:|---:|
| AlexNet | +0.121475 | 0.098372 |
| VGG16 | +0.157341 | 0.137379 |

### 6.2) Findings

- Finetuning improves verification quality for both models with clear AUC increase and EER decrease.
- VGG16 shows stronger post-finetune performance than AlexNet on this setup.
- The final best setting is `VGG16 Finetuned` with AUC `0.906191` and EER `0.172978`.

### 6.3) Reproducibility Notes

- Metric: `cosine`
- Data root: `archive` (CSV protocol and LFW image folders)
- Worker setting used for stable execution in current environment: `--num_workers 0`
- Output directories:
  - `outputs/alexnet_frozen`
  - `outputs/alexnet_ft`
  - `outputs/alexnet_ft_eval`
  - `outputs/vgg16_frozen`
  - `outputs/vgg16_ft`
  - `outputs/vgg16_ft_eval`

## 7) Submit to GitHub

```bash
git add .
git commit -m "Add experiment results and ROC figures"
git push
```

