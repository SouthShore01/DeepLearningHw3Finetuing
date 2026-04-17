# LFW Finetuning & Verification

使用 PyTorch 在 LFW 上完成以下流程：

1. 用 ImageNet 预训练的 **AlexNet / VGG16** 做特征提取与验证；
2. 在 LFW identities 上微调分类模型；
3. 在 LFW pairs 协议上评估微调前后效果；
4. 输出 AUC / EER / ROC。

## 环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 目录

- `src/train_finetune.py`：微调分类模型并保存 `best.pt`
- `src/evaluate_verification.py`：pairs 验证，输出 `metrics.json` / `roc.png`
- `src/utils.py`：数据加载、评分、ROC 绘图
- `EXPERIMENT_RUNBOOK.md`：完整命令顺序

## 快速开始

### 1) Frozen baseline

```bash
python src/evaluate_verification.py --model alexnet --checkpoint none --metric cosine --out_dir outputs/alexnet_frozen
python src/evaluate_verification.py --model vgg16   --checkpoint none --metric cosine --out_dir outputs/vgg16_frozen
```

### 2) Finetune

```bash
python src/train_finetune.py --model alexnet --epochs 5 --batch_size 32 --lr 1e-4 --freeze_backbone false --out_dir outputs/alexnet_ft
python src/train_finetune.py --model vgg16   --epochs 5 --batch_size 16 --lr 1e-4 --freeze_backbone false --out_dir outputs/vgg16_ft
```

### 3) Verification after finetune

```bash
python src/evaluate_verification.py --model alexnet --checkpoint outputs/alexnet_ft/best.pt --metric cosine --out_dir outputs/alexnet_ft_eval
python src/evaluate_verification.py --model vgg16   --checkpoint outputs/vgg16_ft/best.pt --metric cosine --out_dir outputs/vgg16_ft_eval
```

## 快速 smoke test（小样本）

```bash
python src/train_finetune.py --model alexnet --epochs 1 --batch_size 32 --max_train_samples 800 --max_val_samples 400 --out_dir outputs/quick_alexnet_ft
python src/evaluate_verification.py --model alexnet --checkpoint outputs/quick_alexnet_ft/best.pt --max_pairs 1000 --out_dir outputs/quick_alexnet_ft_eval
```

## 备注

- 数据由 `torchvision.datasets.LFWPeople/LFWPairs` 自动下载到 `data/`。
- 支持相似度：`cosine` / `euclidean` / `l1`。
