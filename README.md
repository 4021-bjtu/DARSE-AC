# DARSE-AC

**显著边缘与平均曲率协同的图像盲去卷积**

这是论文第一作者谭蒴（Shuo Tan）维护的 DARSE-AC 参考代码仓库。仓库把论文中的方法链条整理为可运行的 Python 版本，并提供流程说明、论文结果摘录、本地工程验证与研究记录。

> 论文：Tan, S., Dong, L., An, L. (2025). *The Regularization and Deconvolution Algorithm Combining Salient Edges and Average Curvature in High-quality Visual Communication*. **Information Technology and Control, 54(1), 234–253**. [DOI: 10.5755/j01.itc.54.1.38163](https://doi.org/10.5755/j01.itc.54.1.38163)

## 研究主线

古中山国出土文物拓印资料存在模糊、噪声和细节损失，难以直接用于设计素材提取。本研究将这一应用需求转化为图像盲去卷积问题：在模糊核未知的情况下，同时估计清晰图像和模糊核，尽可能保留文字、纹样等主要结构。

我的主要工作包括算法编程实现、实验运行、参数整理、图像处理、结果可视化和论文主体工作；研究方向、理论模型和关键数学问题在 Lingye Dong 老师指导下完成，Limei An 老师促成了研究联系。

## 方法流程

观测模型为：

```text
A ≈ I ⊗ k + n
```

其中 `A` 是模糊观测，`I` 是待恢复图像，`k` 是模糊核，`n` 是噪声。实现流程如下：

1. 灰度化并计算梯度，提取主要结构 `z`。
2. 用显著边缘保留主要轮廓，减少细碎纹理对核估计的干扰。
3. 将梯度 `L0` 和平均曲率（AC）约束加入联合目标。
4. 用 HQS 引入辅助变量 `alpha`、`beta`，交替更新图像和约束变量。
5. 固定辅助变量后用 FFT 求解图像二次子问题。
6. 在梯度域估计模糊核，进行非负化与归一化。
7. 采用粗到细图像金字塔，逐级更新图像与模糊核。
8. 用估计核完成最终非盲复原，并保存图像、核和迭代轨迹。

![DARSE-AC flow](assets/darse_ac_flow_diagram.png)

## 论文报告结果

下面的数字是论文在公开数据集上的报告值，不是古中山国拓印图像的实测值。

| 验证场景 | 论文报告结果 | 实验含义 |
|---|---:|---|
| iNaturalist 2021 | PSNR **31.03 dB**, SSIM **0.96**, ER **1.61**, 成功率 **99.54%** | 跨方法定量比较 |
| ICDAR2019 | PSNR **30.29 dB**, SSIM **0.96** | 文本图像与模块消融 |
| GLADNet | PSNR **25.16 dB**, SSIM **0.90** | 低照度跨数据集验证 |
| RESID | 总体 PSNR **20.56 dB** | 真实模糊视觉与定量验证 |

论文报告的核心结论是：显著边缘有助于保留主要轮廓，AC 正则化有助于抑制非边缘区域的异常起伏，二者结合后在论文设置下取得更好的去模糊表现；改进后的目标函数约在 10 次迭代后趋于平稳。强噪声和空间变化模糊仍是方法的主要挑战。

## 论文视觉结果

下图是论文 Figure 8 的带出处摘录，用于展示 ICDAR2019 文本消融结果。它不是本地代码生成的结果，也不是古中山国文物图像。

![Published Figure 8](assets/paper_fig8_full_attributed.png)

论文图像归 Shuo Tan、Lingye Dong、Limei An 及相应权利人所有。期刊页面未确认 CC 授权；论文图摘录不适用本仓库代码许可证，对外再分发请核查权利。

## 快速开始

需要 Python 3.11 或兼容版本；算法使用 CPU 即可运行，不需要 CUDA 或预训练权重。

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# 合成示例：输出 truth / blurred input / reconstruction / estimated kernel
.venv/bin/python experiment.py demo --output results/demo

# 固定小型基准、模块消融和测试
.venv/bin/python experiment.py benchmark --output results/benchmark
.venv/bin/python experiment.py ablation --output results/ablation
.venv/bin/python -m pytest -q --junitxml=results/pytest.xml
```

处理自己的灰度图像：

```bash
.venv/bin/python experiment.py deblur input.png \
  --output results/your_image --kernel-size 15 --levels 3 --outer 5
```

输出包括 `restored.png`、浮点结果与模糊核的 `restored.npz`、`trace.json` 和 `config.json`。输入图像会转换为灰度；当前实现不覆盖 HDR、彩色联合复原和空间变化模糊核。

## 本地工程验证

固定合成基准包含 3 类图像、2 类模糊核和 2 个噪声种子，共 12 个案例；另有 5 组模块消融。当前版本的数值与 CLI 测试为 **30 项全部通过**。本地运行的完整方法平均 PSNR 为 21.668 dB、平均 SSIM 为 0.6227；这些数字只用于验证这份参考实现的工程链条，不能与论文的公开数据集结果直接比较。

![Local synthetic example](results/demo/text_geometry_motion_seed2026/comparison.png)

上图是人工生成的文本与几何合成样例，不是文物拓印，也不是论文原始数据。完整逐例指标、配置、迭代轨迹和消融结果保存在 `results/`。

## 代码结构

| 文件 | 内容 |
|---|---|
| `darse_ac.py` | 梯度与伴随算子、周期卷积、AC 离散近似、L0 硬阈值、FFT 图像更新、核估计、金字塔和最终复原 |
| `experiment.py` | `demo`、`benchmark`、`ablation`、`deblur` 命令及结果落盘 |
| `test_darse_ac.py` | 算子、FFT 正规方程、核归一化、确定性和 CLI 测试 |
| `RECONSTRUCTION.md` | 论文公式与代码实现的逐项对照、离散约定和替代选择 |
| `assets/` | 流程图、实验链图和带出处的论文图摘录 |

## 复现范围说明

本仓库是依据发表论文整理的作者维护参考实现，不是原始实验环境的归档镜像。论文链接没有附带历史源代码、完整样本清单、数据划分、退化生成协议或全部后处理细节，因此无法声称逐位重现论文历史运行结果。

代码对以下实现约定作了明确记录：

- 灰度浮点图像 `[0, 1]`、周期边界和中心化奇数尺寸核；
- 从论文联合目标推导 HQS 与 FFT 图像更新；
- 以论文给出的 3×3 AC 模板构造可复现的离散近似；
- 用冻结权重的结构引导迭代近似互引导滤波；
- 梯度域岭核估计、支持域裁剪、非负归一化和有限尺度日程；
- 论文没有完整定义的 ER 和成功率不被重新命名为论文指标；本地验证使用单独标明的 `ER_proxy`。

本地合成基准用于验证代码工程链条，不替代 ICDAR2019、iNaturalist 2021、GLADNet、RESID，也不替代真实文物数据。增强图像可能产生伪影；真实拓印应用必须结合来源记录、专家盲评、笔画保真度和失败案例审查。

## 说明

本仓库仅包含论文研究代码、实验说明与公开结果摘录。答辩 PPT、讲稿等个人汇报材料不随仓库公开。论文全文可通过上述 DOI 获取。

## 引用

```bibtex
@article{tan2025darseac,
  author  = {Tan, Shuo and Dong, Lingye and An, Limei},
  title   = {The Regularization and Deconvolution Algorithm Combining Salient Edges and Average Curvature in High-quality Visual Communication},
  journal = {Information Technology and Control},
  volume  = {54},
  number  = {1},
  pages   = {234--253},
  year    = {2025},
  doi     = {10.5755/j01.itc.54.1.38163}
}
```

代码与本地合成示例按仓库许可证发布；论文图摘录、期刊排版和第三方样例图像不自动继承该许可证。详见 `NOTICE.md`。
