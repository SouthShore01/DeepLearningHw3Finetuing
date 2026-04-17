# DeepLearning HW3 Finetuning (LFW Verification)

这个仓库给你一套**可直接在 GitHub 上提交作业**的模板，并且完整覆盖题目要求：

- 使用 **AlexNet** 和 **VGG16** 预训练模型；
- 在 **LFW** 数据集上做特征提取与验证；
- 比较 **不微调 (frozen)** 与 **微调 (fine-tuned)** 两种设置；
- 计算相似度分数，绘制 **ROC 曲线** 并汇总结论。

---

## 1. 作业目标映射

题目要求拆解：

1. 下载 LFW；
2. 使用 AlexNet / VGG16 提取特征；
3. 在有/无微调两种条件下做验证；
4. 使用匹配算法（cosine / euclidean 等）得到分数；
5. 画 ROC 并总结。

本项目对应脚本：

- `src/train_finetune.py`：在 LFW identities 划分上微调分类头（可选择冻结 backbone）；
- `src/evaluate_verification.py`：在 LFW pairs 上做验证评估，输出 ROC/AUC/EER，并保存图像；
- `src/utils.py`：数据集、特征提取、评分和绘图工具函数。

---

## 2. 环境准备

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> 说明：`torchvision.datasets.LFWPeople / LFWPairs` 会自动下载数据到 `./data`。

---

## 3. 推荐实验流程（按学生作业顺序）

### Step A: 先跑“不微调”基线

AlexNet frozen：

```bash
python src/evaluate_verification.py \
  --model alexnet \
  --checkpoint none \
  --metric cosine \
  --out_dir outputs/alexnet_frozen
```

VGG16 frozen：

```bash
python src/evaluate_verification.py \
  --model vgg16 \
  --checkpoint none \
  --metric cosine \
  --out_dir outputs/vgg16_frozen
```

### Step B: 微调分类器（建议只训练最后全连接层）

AlexNet fine-tune：

```bash
python src/train_finetune.py \
  --model alexnet \
  --epochs 5 \
  --batch_size 32 \
  --lr 1e-4 \
  --freeze_backbone false \
  --out_dir outputs/alexnet_ft
```

VGG16 fine-tune：

```bash
python src/train_finetune.py \
  --model vgg16 \
  --epochs 5 \
  --batch_size 16 \
  --lr 1e-4 \
  --freeze_backbone false \
  --out_dir outputs/vgg16_ft
```

### Step C: 用微调后权重做验证

AlexNet ft checkpoint:

```bash
python src/evaluate_verification.py \
  --model alexnet \
  --checkpoint outputs/alexnet_ft/best.pt \
  --metric cosine \
  --out_dir outputs/alexnet_ft_eval
```

VGG16 ft checkpoint:

```bash
python src/evaluate_verification.py \
  --model vgg16 \
  --checkpoint outputs/vgg16_ft/best.pt \
  --metric cosine \
  --out_dir outputs/vgg16_ft_eval
```

---

## 4. 输出文件说明

每次评估会输出：

- `scores_labels.npz`：保存 pair 分数和标签；
- `metrics.json`：AUC、EER、最佳阈值；
- `roc.png`：ROC 曲线图；
- 终端打印指标，方便写报告。

---

## 5. 报告撰写建议（可直接套用）

你可以在报告里按这个结构写：

1. **实验设置**：输入大小、预训练权重、优化器、epoch、batch size；
2. **对比实验**：
   - AlexNet frozen vs fine-tuned
   - VGG16 frozen vs fine-tuned
3. **评价指标**：AUC、EER、ROC 曲线形状；
4. **现象分析**：
   - 微调后 AUC 是否提升；
   - 哪个模型更鲁棒；
   - 可能原因（网络容量、过拟合、LFW规模等）。

---

## 6. GitHub 提交流程（作业友好）

```bash
git add .
git commit -m "Complete HW3: LFW verification with AlexNet/VGG16 frozen vs fine-tuned"
git push origin <your-branch>
```

建议在仓库 README 放上：

- 实验命令；
- 最终指标表格；
- ROC 图（`outputs/.../roc.png`）；
- 结论。

---

## 7. 常见问题

1. **下载慢**：可以提前手动下载 LFW 到 `data/lfw-py`；
2. **显存不够**：减小 batch size，优先冻结 backbone；
3. **训练太慢**：先跑 1~2 epoch 验证 pipeline；
4. **结果波动**：固定随机种子（脚本已支持）。

祝你作业顺利！
