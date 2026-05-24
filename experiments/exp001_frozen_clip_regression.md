# Exp001 Frozen CLIP + Regression Head

## Setting
- CLIP: frozen
- Trainable: regression head only
- Epochs: 5
- Loss: MSELoss
- Metrics: SRCC / PLCC / KRCC

## Results

| Epoch | Loss | SRCC | PLCC | KRCC |
|---|---:|---:|---:|---:|
| 1 | 0.0561 | 0.7286 | 0.7787 | 0.5407 |
| 2 | 0.0180 | 0.7958 | 0.8451 | 0.5987 |
| 3 | 0.0138 | 0.8267 | 0.8663 | 0.6335 |
| 4 | 0.0115 | 0.8340 | 0.8703 | 0.6422 |
| 5 | 0.0100 | 0.8568 | 0.8781 | 0.6660 |

## Observation
Loss 持续下降，SRCC/PLCC/KRCC 持续上升，说明 regression head 能从冻结 CLIP 特征中学习到质量分数映射。