# BOTDA 非周期编码的物理引导 CVAE

[English](README.en.md)

基于条件变分自编码器（CVAE）的 BOTDA 非周期编码生成方法。模型以能量增强因子 FE、编码长度和衰落系数为条件，将 EDFA 指数包络与 FFT 编码增益评估嵌入训练目标，通过潜变量采样生成二进制编码。

仓库包含训练程序、编码生成程序及完整训练集，共 **6,079 条编码**。

## 环境配置

Python 3.10 或更高版本，主要依赖 PyTorch、NumPy、Matplotlib、pandas 和 tqdm。建议使用独立虚拟环境：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS：

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

程序自动选择可用的 CUDA 设备，否则使用 CPU。使用 GPU 时，需安装与本机驱动兼容的 PyTorch 版本。

## 项目结构

```text
botda-cvae/
├── train.py              训练与检查点保存
├── generate.py           编码生成、评估与导出
├── requirements.txt      Python 依赖
├── data0/                训练集
│   ├── fading_0.00/
│   ├── fading_0.10/
│   ├── fading_0.20/
│   ├── fading_0.30/
│   ├── fading_0.40/
│   └── fading_0.50/
├── tests/                数据与计算检查
└── tools/                项目打包工具
```

## 训练数据

每个 `.txt` 文件保存一条二进制编码，所在目录指定衰落系数。编码长度覆盖 120、180、240、300、360、450、540、630 和 720。数据目录、文件名及编码内容保持原样。

训练输入为施加指数包络后的幅度序列，重构目标为原始二进制序列。条件向量为 `[FE/700, L/1024, fading/0.5]`，其中训练 FE 标签由编码中 `1` 的数量计算。各目录样本数见 [训练集说明](data0/README.md)。

## 模型训练

在项目目录运行：

```bash
python train.py
```

默认读取 `data0/`，检查点和训练曲线保存至 `checkpoints/`。也可以指定路径：

```bash
python train.py --data_root data0 --checkpoint_dir checkpoints
```

从最近一次保存的检查点继续训练：

```bash
python train.py --resume
```

主要训练参数位于 `train.py` 开头：

| 参数 | 默认值 |
| --- | --- |
| 最大编码长度 | 1024 |
| 隐藏层维度 | 1024 |
| 潜变量维度 | 128 |
| 条件维度 | 3 |
| Batch size | 32 |
| 训练轮数 | 900 |
| 优化器 | AdamW |
| 初始学习率 | `1e-4` |
| 验证集比例 | 10% |

检查点文件包括 `universal_cvae_v2_best.pt` 和 `universal_cvae_v2_last.pt`。模型按验证集平均 `Gc - Gr` 选择最佳检查点，历史最优检查点保存在 `checkpoints/history/`，曲线保存在 `checkpoints/plots/`。

## 编码生成

训练完成后，或将已训练的 `universal_cvae_v2_best.pt` 放入检查点目录后，运行：

```bash
python generate.py --checkpoint_dir checkpoints --output_base outputs --fe_list 40 60 80 100 120 150 180 210 --fading_list 0.0 0.15
```

默认编码长度为 `L = 3 × FE`。每个条件采样 10 轮，每轮 20,000 条候选，温度为 0.5。模型输出按概率和确定置 1 位数，经二值化、`200 × L` 点 FFT 评估、排序和去重后导出编码。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--fe_list` | 40–220，间隔 10 | 目标 FE 列表 |
| `--fading_list` | `0.0 0.15` | 衰落系数列表 |
| `--n_sample` | 20000 | 每轮采样数量 |
| `--n_rounds` | 10 | 采样轮数 |
| `--top_k` | 10 | 最多导出的编码数量 |
| `--temperature` | 0.5 | 解码温度 |
| `--target_ratio` | 1.0 | 幅度和筛选的目标比例 |
| `--amp_sum_tolerance` | 不启用 | 幅度和后筛选的绝对容差 |

`--amp_sum_tolerance` 对按增益差排序去重后的候选子集进行后筛选；若该子集中没有满足条件的编码，程序保留原排序结果。

结果按 `outputs/FE{FE}_L{L}/fading{值}/` 分目录保存，包含二进制 `.txt` 文件和 `summary.csv`；所有条件汇总为 `outputs/all_results.csv`。同一条件再次运行时会更新该目录中的编码文件。

| 结果字段 | 含义 |
| --- | --- |
| `FE_theory` | 输入目标 FE |
| `Binary_FE` | 编码中 `1` 的数量 |
| `AmpSum` | 衰落后的幅度和 |
| `Gc` | 实际编码增益，dB |
| `Gr` | 实际参考增益，dB |
| `Gr0` | 目标参考增益，dB |
| `Diff` | `Gc - Gr` |

## 检查与打包

```bash
python -m unittest discover -s tests -v
python train.py --help
python generate.py --help
python tools/build_release.py
```

打包文件包含源代码、说明和训练集。虚拟环境、模型检查点和运行输出由 `.gitignore` 排除。
