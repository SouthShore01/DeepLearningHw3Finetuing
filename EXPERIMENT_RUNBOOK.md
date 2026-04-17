# Experiment Runbook (HW3: LFW Finetuning)

这个文档用于“真正执行实验并记录结果”。

## 1) 环境检查

```bash
python -V
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
```

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

## 6) Result Table Template

| Setting | AUC | EER | Notes |
|---|---:|---:|---|
| AlexNet Frozen | TBD | TBD | |
| AlexNet Finetuned | TBD | TBD | |
| VGG16 Frozen | TBD | TBD | |
| VGG16 Finetuned | TBD | TBD | |

## 7) Submit to GitHub

```bash
git add .
git commit -m "Add experiment results and ROC figures"
git push
```

