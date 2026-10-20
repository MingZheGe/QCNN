# 在 PyCharm 中运行

1. 在 PyCharm 选择 **File → Open**，打开此 QCNN 文件夹。
2. 如果提示设置 Python Interpreter，选择 **Add Local Interpreter → Existing**，指定项目内的 `.venv\Scripts\python.exe`（Python 3.11）。不需要重新安装依赖。
3. 选择右上角的 **QCNN Quick** 配置并运行。也可以右键根目录 `run_quick.py`，选择 Run。

`run_quick.py` 会遍历 `conv1d / conv2d / fusion` 和 `gasf / gadf / both` 的全部 9 种组合，每组同时训练 CNN 和 QCNN，共执行 18 次模型训练。默认参数为 stride=1、epochs=20、seed=59、patience=5。

结果分别位于 `hybrid_qcnn_small\results\<input-mode>_<gaf-mode>_stride1\horizon_1`，包含模型、预测和评价指标。

每个模型和 seed 都有独立目录，例如 `cnn\seed_59` 或 `qcnn\seed_59`。目录中包括：

- `trained_model.pt`：模型权重以及输入模式、GAF 模式、seed、horizon 等恢复信息。
- `model_state_dict.pth`：可直接传给 PyTorch `load_state_dict` 的纯权重。
- `predictions.csv`：测试集预测，列顺序为 `timestamp,symbol,target_return,actual,prob_down,prob_up,predicted`。
- `metrics.json` 和 `training_history.json`：测试指标和训练历史。

## 正式训练

新建 Python 运行配置：

- Script path：`hybrid_qcnn_small\train.py`
- Working directory：本项目根目录
- Interpreter：`.venv\Scripts\python.exe`
- Parameters：`--input-mode fusion --gaf-mode both --model both --horizon 1 --stride 1 --epochs 20 --seeds 42 43 44 45 46 --patience 5`

正式训练在 CPU 上运行，量子模拟可能耗时较长。原项目默认也使用 CPU。

PowerShell 中也可执行：

```powershell
.\.venv\Scripts\python.exe .\run_quick.py
```

如需先做快速验证，可缩小数据量和训练轮数：

```powershell
.\.venv\Scripts\python.exe .\run_quick.py --stride 500 --epochs 1 --seeds 42 --patience 1
```

Python 运行时保存在 `.runtime`，虚拟环境在 `.venv`。请保留这两个文件夹；如果移动整个项目，建议重新创建虚拟环境。
