# SQL 注入检测实验运行说明

## 功能
- 严格读取固定数据文件：
  - `prepared_sql_dataset/sql_train_augmented.csv`
  - `prepared_sql_dataset/sql_val.csv`
  - `prepared_sql_dataset/sql_test.csv`
- 训练并评估：TF-IDF+LR、TF-IDF+SVM、TF-IDF+RF、RULE 基线、Risk-Gated LSTM 系列消融模型，以及 DQN 决策融合模型。
- 输出 Accuracy / Precision / Recall / F1 / AUC。
- 保存结果与图表。

## 运行方式
在仓库根目录执行：

```bash
python sql_injection_project/train_sql_injection_models.py
```

## 输出文件
- `outputs/sql_injection_detection_results.csv`
- `outputs/model_probabilities.pkl`
- `outputs/f1_comparison.png`
- `outputs/metrics_comparison.png`
- `outputs/roc_curves.png`
- `outputs/pr_curves.png`

## 依赖
建议 Python 3.10+，并安装：

```bash
pip install numpy pandas scikit-learn matplotlib torch
```
