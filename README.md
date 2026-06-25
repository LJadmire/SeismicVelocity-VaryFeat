# SeismicVelocity-VaryFeat
  A triangulation framework for auditing physics-derived input features in deep-learning time-lapse seismic velocity   inversion.

A systematic framework for auditing physics-derived input features in deep-learning
time-lapse velocity inversion. Evaluates whether each of eight physically motivated
attribute maps genuinely improves prediction — and whether gradient-based attribution
methods (Integrated Gradients, SHAP) reliably identify those that do.

## Core idea

Post-hoc attribution measures what a network **uses**, not what **benefits** prediction.
We triangulate three evidence sources:

| Source | Question answered | Cost |
|--------|-------------------|------|
| **Physical Proxy (PP)** | Does the feature carry signal? | 1 pass over dataset, no training |
| **Attribution (IG, SHAP)** | Does the network attend to the feature? | 1 backward pass per sample |
| **Ablation** | Does the feature provide *unique* value? | 1 full training run per config |

## Directory structure
TimeVaryFeat/
│
├── feature_study.py          # 主训练+消融流水线 (20 个实验配置)
├── feature_study_v7/         # 20 组实验输出
│   ├── all_8/                #   全特征基线
│   ├── single_0..7/          #   单特征隔离实验 (8 个)
│   ├── remove_0..7/          #   留一法消融实验 (8 个)
│   ├── top_2..4/             #   前向 Top-K 选择实验 (3 个)
│   ├── physical_proxy/       #   物理代理相关性分析输出
│   └── summary_results.csv   #   全实验汇总表
├── multi_seed_study.py       # 多种子复制实验 (5 seeds × 5 configs)
├── run_shap.py               # SHAP 归因分析 (GradientExplainer)

## Requirements

- Python 3.8+ with PyTorch, NumPy, SciPy, Pandas, Matplotlib, SHAP
- MATLAB (for `main.m`–`main4.m` visualization scripts)
- GPU recommended for training (batch size 8, ~120 epochs per config)

## Usage examples

```bash
# Full ablation study (20 configs)
python feature_study.py --mode ablation

# Integrated Gradients for a trained config
python feature_study.py --mode ig --ig_config feature_study_v7/all_8/

# Physical proxy correlation (all 15,000 samples)
python feature_study.py --mode physical

# SHAP attribution
python run_shap.py --config feature_study_v7/all_8/

# Multi-seed replication
python multi_seed_study.py
```
