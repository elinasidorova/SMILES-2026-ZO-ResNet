# Zero-Order Fine-Tuning of ResNet18 on CIFAR100 - Solution
**Final Result:** 24.29% Accuracy (Checkpoint 3)

---

## 1. Reproducibility
```bash
git clone clone https://github.com/elinasidorova/SMILES-2026-ZO-ResNet.git
cd SMILES-2026-ZO-ResNet
pip install -r requirements.txt sentence-transformers
```

## 2. Run evaluation

```bash
python validate.py \
    --data_dir ./data \
    --batch_size 32 \
    --n_batches 256 \
    --output results.json
```

## 3. Final Solution
### 3.1 Semantic Initialization (`head_init.py`)

The default ResNet18 has a 1000-class ImageNet head. Replacing it with a randomly initialized 100-class head (Kaiming uniform) gives only 1.21% accuracy.

**Idea:** The pretrained head already contains high-quality visual prototypes and maybe it is possible to transfer them to CIFAR100 by matching classes semantically. I only modify head_init.py, a student-editable file. Don't train anything, don't use gradients, and don't modify fixed infrastructure (model.py, validate.py). The sentence transformer is only used for weight selection, not during training or evaluation.

**What was done:**

- The original ResNet18 (just for its head weights) was loaded: old_weight (1000×512), old_bias (1000,)
- All 1000 ImageNet class names and all 100 CIFAR100 class names were taken
- They were embeded with 384-dim embeddings all-MiniLM-L6-v2 (sentence-transformer, lightweight)
- For each CIFAR100 class, the closest ImageNet class by cosine similarity was found
- New head was built: the corresponding rows from old_weight and old_bias were selected
- Scaling by sqrt(1000/100) to keep logit variance similar

**In code:**

```python

from sentence_transformers import SentenceTransformer

# Get class names
imagenet_classes = ["tench", "goldfish", ...]
cifar100_classes = ["apple", "aquarium_fish", ...]

# Embed class names
st = SentenceTransformer('all-MiniLM-L6-v2')
imagenet_emb = st.encode(imagenet_classes)
cifar100_emb = st.encode(cifar100_classes)

# Cosine similarity
similarity = cifar100_emb @ imagenet_emb.T
best_idx = similarity.argmax(dim=1)

# Copy weights from matched ImageNet classes
old_weight = pretrained_resnet.fc.weight
old_bias = pretrained_resnet.fc.bias
scale = math.sqrt(1000 / 100)

new_weight = old_weight[best_idx] * scale
new_bias = old_bias[best_idx] * scale
```

**Result:** The model starts with 19.13% accuracy.

### 3.2 Guided Random Search Optimizer (`zo_optimizer.py`)

SPSA estimates a gradient from one random direction per step. Here we have 51K parameters and it it too noisy. With 256 steps (full budget), the model has +0.2% accuracy over init.

**Idea:** Try multiple random directions, evaluate each, take the best.

**Final hyperparameters** (Optuna + manual search):

| Parameter	| Value	|
|:------:|:-------------------:|
| lr | 0.0157 |
| eps | 0.0003 |
| n_directions | 11 |
| bias_lr_mult | 4.0 |
| Perturbation | Rademacher (±1) |

**Algorithm per step:**
```python
for step in range(256):
    # Cosine learning rate schedule
    lr = 0.0133 * 0.5 * (1 + cos(π * step / 256))

    loss_before = loss_fn()
    best_loss = float('inf')
    best_delta_w, best_delta_b = None, None
    best_sign = 0

    # Try 11 random directions
    for _ in range(11):
        # Rademacher: each element ±1
        delta_w = torch.where(rand_like(weight) > 0.5, 1.0, -1.0)
        delta_b = torch.where(rand_like(bias) > 0.5, 1.0, -1.0)

        weight += 0.0003 * delta_w
        bias += 0.0003 * delta_b
        loss_plus = loss_fn()
        weight -= 0.0003 * delta_w
        bias -= 0.0003 * delta_b

        if loss_plus < best_loss:
            best_loss = loss_plus
            best_delta_w, best_delta_b = delta_w, delta_b
            best_sign = +1

        weight -= 0.0003 * delta_w
        bias -= 0.0003 * delta_b
        loss_minus = loss_fn()
        weight += 0.0003 * delta_w
        bias += 0.0003 * delta_b

        if loss_minus < best_loss:
            best_loss = loss_minus
            best_delta_w, best_delta_b = delta_w, delta_b
            best_sign = -1

    # Apply best direction with separate learning rates
    weight += lr * best_sign * best_delta_w
    bias += lr * 5.2 * best_sign * best_delta_b
  ```

**Why 11 directions?**
Tested 6-15 via Optuna. Too few don't give enough exploration. With too many the optimizer starts overfitting to the specific batch.

**Why higher bias lr?**
Bias has 100 elements, weight has 51200 so bias can converge faster (4 is found by Optuna).

**Why cosine decay?**
2-phase schedule caused accuracy to drop from 19.3% to 18.9%. Tried cosine and it worked.

**Why no momentum/Adam?**
Here each step uses a different random direction. momentum accumulates randomness, Adam's second moment tracks the variance of random vectors, it stays permanently high. So it has no sense.

## 3.3 Data Augmentation (`augmentation.py`)

**In code:**

```python

T.Compose([
    T.Resize(224),
    T.RandomHorizontalFlip(),
    T.RandomCrop(224, padding=28),
    T.TrivialAugmentWide(),
    T.ToTensor(),
    T.Normalize(cifar100_mean, cifar100_std),
    T.RandomErasing(p=0.2),
])
```
**Why:**
Augmentations don't consume the 8192-sample limit, but create variations. TrivialAugmentWide alone contributed +2-3% accuracy.

## 4. Experiments and Failed Attempts

### 4.1 Initialization Strategies

| Method	| Checkpoint-2	| Comment |
|:------:|:-----:|:-------------------:|
| Kaiming init |	1.21%	| Default init |
| Orthogonal init	| 0.90%	| Too low, discarded |
| Semantic (single best class)	| 19.13%	| Used in final solution |
| Semantic (top-3 weighted average)	| 17.56%	| Averaging dilutes strong signal |
| Semantic + bias=0	| 19.12%	| Bias from ImageNet matters, a little |

### 4.2 Augmentation experiments

Before finding semantic initialization, I tested which augmentations help. Tained using full budget:
```python
    --batch_size 32
    --n_batches 256
```

| Augmentation | Optimizer | Checkpoint-2 | Checkpoint-3 | Comment |
|:-------------:|:----------:|:-------------:|:----------:|:-------------:|
| Resize + HorizontalFlip + Normalize | SPSA: lr=1e-4, eps=1e-4, rademacher | 1.21% | 1.26% | — |
| Resize + RandomCrop(224,p=28) + Normalize | SPSA: lr=1e-4, eps=1e-4, rademacher, gradient clipping |  1.21% | 1.22% | — |
| TrivialAugmentWide + Normalize | SPSA: lr=1e-3, eps=1e-4, rademacher, momentum + clip + SignSGD |  1.21% | 1.54% | Automatic augmentation helps even with weak init |
| TrivialAugmentWide + Normalize | Guided Random Search: lr=1e-2, eps=1e-3, n=10 |  1.21% | 1.80% | The same |
| Resize + HorizontalFlip + RandomCrop + TrivialAugmentWide + RandomErasing(0.2) | GRS(2-stage): only bias -> bias + weights, n=8, lr=1e-2 | 1.21% | 1.15% | Didn't work well, most likely because of the optimizer |

### 4.3 Layer Selection

Only the final classification head was fine-tuned: fc.weight and fc.bias.
I tried adding layer4 parameters with orthogonal init, the accuracy dropped to 0.95% at checkpoint-3.
layer4 has ~2M elements, is is too mush for ZO-optimization.
Single random perturbation vector is applied across all parameters and the information per element decreases as the total number of parameters grows.
I chose to focus the entire optimization budget on the classification head, where each step can make a less noisy update.

### 4.4 Experiments with semantic init (19.13% accuracy on checkpoint-2)

One augmentation strategy was chosen:

`Resize + HorizontalFlip + RandomCrop + TrivialAugmentWide + RandomErasing(0.2)`

| Optimizer | Checkpoint-3 | Comment |
|:--------:|:-----:|:--------------:|
| GRS baseline (lr=1e-2, bias=1, n=10) | 22.80% | First strong result |
| GRS + Stochastic Weight Averaging (last 57 steps) | 22.72% | Averaging doesn't help |
| GRS + bias lr boost and cosine lr decay (lr=1.5e-2, eps=1e-3, n=10, bias_boost=3) | 23.30% | Second strong result  |
| GRS: two-phase LR (80% steps lr=2e-2, 20% steps lr= 5e-3), bias_boost=3| 18.87% | The optimizer switched from large to small steps before the parameters had converged |
| GRS: eps decay, high eps=2e-3 (exploration) low eps=1e-4 (exploitation) | 19.81% | Too slow early exploration |
| Top-3 semantic init & GRS + SVD denoising of weight gradient | 17.53% | SVD kills signal (compared to 19.83% with GRS baseline) |

### 4.5 Manual hyperparameter tuning

| lr | bias_mult | n_directions | Checkpoint-3 |
|:-------------:|:----------:|:-------------:|:----------:|
| 1.5e-2 | 3 | 10 | 23.30% |
| 2e-2 | 3 | 10 | 23.05% |
| 1e-2 | 3 | 10 | 23.19% |
| 1.5e-2 | 5 | 10 | 23.17% |
| 2e-2 | 3 | 12 | 21.63% |
| 1e-2 | 5 | 10 | 22.65% |

Changing one parameter at a time doesn't help. Need joint optimization.

### 4.6 Optuna hyperparameter search

**Broad search**
- Search space:

| Parameter | Range | Sampling | Why this range |
|:-----------:|:-------:|:----------:|:----------------:|
| lr | [1e-3, 3e-2] | log-uniform | Wide enough to cover both conservative and aggressive |
| bias_mult | [1, 8] | uniform | From no boost to aggressive boost |
| n_directions | [6, 14] | integer | < 6 may miss directions, > 14 may overfit |
| eps | [1e-4, 5e-3] | log-uniform | Small enough for precise signal, large enough for measurable loss change |

- 18 trials
- **Best result:** 22.78% at lr=0.01304291113312082, bias_mult=6.533541590269971, n_directions=12, eps=0.00010023799172297744

Maybe, smaller eps gives cleaner gradient estimates and it allows faster bias learning without destabilizing weights.

**Refined search**
- Search space:

| Parameter | Range | Why this range |
|:-----------:|:-------:|:--------------:|
| lr | [0.007, 0.020] | Best trials in broad search were 0.008-0.013 |
| bias_mult | [4, 7] | Best trials in broad search were 5-6.5 |
| n_directions | [11, 15] | Best trials in broad search were 12-14 |
| eps | [5e-5, 5e-4] | Best trial in broad search was 1e-4, explore even smaller |

-  21 trials
- **Best result:** 24.28% at lr=0.015674580289891053, bias_mult=4.006821301975223, n_directions=11, eps=0.0002718046430938339

### 5. Why specific approaches failed

**SVD denoising:**

Idea from a paper (https://arxiv.org/abs/2602.17155): clean noisy zero-order gradients by keeping only top singular components via SVD. The idea is that true gradients are low-rank, noise is high-rank.

```python
U, S, Vh = torch.linalg.svd(grad_weight)
S[20:] = 0
denoised = U @ diag(S) @ Vh
```

It failed because the original ZO-Muon method averages multiple SPSA estimates first (to build higher rank), then applies SVD.
Our SPSA gradient is a scalar times a random matrix. It has rank 1 by construction, there is no low-rank structure to extract.
Truncating to 20 components just changes which random direction we follow.
Moreover, it costs extra forward passes.

**SWA (Stochastic Weight Averaging):**

Idea was to average weights from last N steps instead of using final weights. Sometimes it helps to find a flatter minimum that generalizes better.

It failed here because SWA helps when the model oscillates around a minimum, but with 256 steps we are still descending — there is no oscillation to fix.

**Bias-first curriculum:**

Bias alone (100 params) cannot learn class structure from 512-dim features. Pretrained weights are already good and delaying their optimization wastes budget.

### 6. What Contributed Most

**Semantic initialization**

The largest improvement. Transferring weights from semantically similar ImageNet classes gives the head meaningful prototypes before any training. Without this, no optimizer exceeded 2% accuracy.

**Guided Random Search**

SPSA estimates gradient from one random direction — too noisy for 51K parameters. GRS tests 11 directions, picks the best. Each evaluation uses the same batch, so it costs no extra budget, only compute time.

**Bias LR boost**

Bias has 100 elements, weight has 51200. Bias can learn faster without destabilizing. Final multiplier (4) - found by Optuna (earlier 5.2 and 3 gave slightly worse results).

**Augmentation**

TrivialAugmentWide + RandomCrop + RandomErasing. Augmentations don't consume the 8192-sample limit, but create variations.

**Optuna joint optimization**

Manual tuning changed one parameter at a time and missed the optimal combination. Optuna jointly optimized lr, bias_mult, n_directions, and eps.


