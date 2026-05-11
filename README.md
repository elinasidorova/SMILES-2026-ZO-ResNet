
# SMILES-2026: Zero-Order Fine-Tuning of ResNet18 on CIFAR100

**Final accuracy: 24.29%**

## Quick start

```bash
git clone https://github.com/elinasidorova/SMILES-2026-ZO-ResNet.git
cd SMILES-2026-ZO-ResNet
pip install -r requirements.txt sentence-transformers
python validate.py --data_dir ./data --batch_size 32 --n_batches 256 --output results.json
```

`sentence-transformers` downloads `all-MiniLM-L6-v2` model on first run (~90 MB)

## Files

|File|What is does|
|:-------:|:-------:|
| `SOLUTION.md` | Full report |
| `zo_optimizer.py` | Guided Random Search optimizer |
| `head_init.py` | Semantic initialization ImageNet -> CIFAR100 |
| `augmentation.py` | Training augmentations |
| `train_data.py` | Data loading |
| `model.py`, `validate.py` | Infrastructure |

## Result

`results.json` — `val_accuracy_top1_finetuned`
