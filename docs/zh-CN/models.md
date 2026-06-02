# 模型参考

ModernTSF 共内置 172 个模型。每个模型位于 `src/models/<name>/` 目录下，并带有以 YAML front matter 开头的本地 `README.md`，同时包含 runner 使用的实现文件：

- `model.py` — `torch.nn.Module` 实现或适配器
- `schema.py` — 用于校验 `model.params` 的 Pydantic `ModelParameterConfig`
- `registry.py` — `register()` 函数，注册模型工厂

模型参数由各模型单独定义，在配置加载时进行校验。具体字段请参考对应的 `schema.py`。

模型目录按预测数据设定分类，而不是按架构家族分类。

---

## 时间序列

普通单变量或多变量时间序列预测，输入通常是 `(B, T, C)` 历史值。
这一类包含线性基线、传统机器学习、统计预测器、Transformer、MLP / patch、CNN、RNN、状态空间、滤波方法、近年 2025/2026 会议模型和其他架构变体。

| 名称 | 配置 | 说明 |
|---|---|---|
| `Linear` | `configs/models/Linear.toml` | 按通道对 `seq_len → pred_len` 做线性投影 |
| `DLinear` | `configs/models/DLinear.toml` | 将序列分解为趋势 + 季节性，分别做线性投影 |
| `NLinear` | `configs/models/NLinear.toml` | 先减去最后一个值归一化，再做线性投影 |
| `RLinear` | `configs/models/RLinear.toml` | 带 RevIN（可逆实例归一化）的线性模型 |
| `CrossLinear` | `configs/models/CrossLinear.toml` | 带跨通道交互的线性模型 |
| `MixLinear` | `configs/models/MixLinear.toml` | 时间维与通道维混合线性投影 |
| `PatchTST` | `configs/models/PatchTST.toml` | 将序列分为 patch，按通道应用 Transformer |
| `iTransformer` | `configs/models/iTransformer.toml` | 倒置 Transformer：对通道做注意力，对时间做 FFN |
| `TimeXer` | `configs/models/TimeXer.toml` | 内生变量分块嵌入 + 外生变量倒置嵌入，通过全局 token 做交叉注意力 |
| `Crossformer` | `configs/models/Crossformer.toml` | 对分块片段做跨维度注意力，采用两阶段注意力路由 |
| `Informer` | `configs/models/Informer.toml` | ProbSparse 自注意力 + 蒸馏，面向高效长序列预测 |
| `Autoformer` | `configs/models/Autoformer.toml` | 用自相关机制替代自注意力 |
| `FEDformer` | `configs/models/FEDformer.toml` | 频域增强的分解 Transformer |
| `Reformer` | `configs/models/Reformer.toml` | 高效 Transformer，使用 LSH 注意力降低显存与计算开销 |
| `Pyraformer` | `configs/models/Pyraformer.toml` | 在多分辨率金字塔树上做注意力，捕捉长程依赖 |
| `ETSformer` | `configs/models/ETSformer.toml` | 指数平滑注意力，分解为水平/增长/季节性分量 |
| `NSTransformer` | `configs/models/NSTransformer.toml` | 非平稳 Transformer，结合去平稳注意力与序列平稳化 |
| `MultiPatchFormer` | `configs/models/MultiPatchFormer.toml` | 多尺度 patch 嵌入，配合跨 patch Transformer 注意力 |
| `PAttn` | `configs/models/PAttn.toml` | 将 patch 嵌入直接送入单个自注意力块——极简的 patch Transformer 基线 |
| `CARD` | `configs/models/CARD.toml` | 通道对齐的鲁棒双注意力 Transformer，融合 token 与通道注意力 |
| `Fredformer` | `configs/models/Fredformer.toml` | 频率去偏 Transformer，对各频率 patch 做注意力以抑制低频偏置 |
| `DUET` | `configs/models/DUET.toml` | 在时间维与通道维上做双重聚类，并配以融合模块 |
| `Pathformer` | `configs/models/Pathformer.toml` | 多尺度 Transformer，自适应路径在不同时间分辨率间路由 patch |
| `DSFormer` | `configs/models/DSFormer.toml` | 双采样 Transformer，使用 TVA（时间-变量注意力）编解码块 |
| `DTAF` | `configs/models/DTAF.toml` | patch 嵌入 Transformer，结合分解稳定化与频率差分波建模 |
| `TimePerceiver` | `configs/models/TimePerceiver.toml` | Perceiver 风格架构：对 patch 做迭代式交叉/自注意力，并以 query 解码未来 patch |
| `Transformer` | `configs/models/Transformer.toml` | 标准编解码器 Transformer，使用完整点积自注意力 |
| `PatchMLP` | `configs/models/PatchMLP.toml` | 基于 patch 的 MLP |
| `xPatch` | `configs/models/xPatch.toml` | 扩展版 patch MLP |
| `TSMixer` | `configs/models/TSMixer.toml` | 时间序列 MLP-Mixer，交替做时间与通道混合 |
| `LightTS` | `configs/models/LightTS.toml` | 轻量级 MLP，基于分块处理 |
| `WPMixer` | `configs/models/WPMixer.toml` | 小波 patch MLP-Mixer，在多层分解的子序列上混合 |
| `MTSMixer` | `configs/models/MTSMixer.toml` | 分解式 MLP-Mixer，解耦时间维与通道维交互以做多变量预测 |
| `UMixer` | `configs/models/UMixer.toml` | U-Net 风格的多尺度混合，配以平稳性校正模块 |
| `NHiTS` | `configs/models/NHiTS.toml` | 神经分层插值：多速率采样 + 分层插值 MLP 堆栈 |
| `NBeats` | `configs/models/NBeats.toml` | 全连接基扩展块的深层堆叠，带 backcast/forecast 残差 |
| `HDMixer` | `configs/models/HDMixer.toml` | 分层 patch mixer，采用可扩展长度的 patch 做多变量预测 |
| `SRSNet` | `configs/models/SRSNet.toml` | 选择性表示空间：双 patch 视图（选择性 + 动态）配 MLP 预测头 |
| `TimesNet` | `configs/models/TimesNet.toml` | 将一维时序重塑为二维，应用视觉风格卷积 |
| `SCINet` | `configs/models/SCINet.toml` | 样本卷积与交互网络 |
| `MICN` | `configs/models/MICN.toml` | 多尺度等距卷积，兼顾局部与全局时序模式 |
| `ModernTCN` | `configs/models/ModernTCN.toml` | 现代化时序卷积网络，采用大核深度可分卷积 |
| `WaveNet` | `configs/models/WaveNet.toml` | 堆叠膨胀因果卷积，带门控激活与残差/跳跃连接 |
| `SegRNN` | `configs/models/SegRNN.toml` | 分段 RNN — 以固定长度分段替代逐步处理 |
| `DeepAR` | `configs/models/DeepAR.toml` | 自回归循环网络，产生概率预测 |
| `MambaSimple` | `configs/models/MambaSimple.toml` | 选择性状态空间（Mamba）序列模型——纯 PyTorch 实现选择性扫描，无需依赖 CUDA 算子 |
| `S_Mamba` | `configs/models/S_Mamba.toml` | iTransformer 风格的倒置嵌入，在通道维上叠加 Mamba 块；无需 CUDA 算子的选择性扫描 |
| `BiMamba` | `configs/models/BiMamba.toml` | 双向 Mamba，对序列正向与反向各扫描一次；无需 CUDA 算子的选择性扫描 |
| `S4` | `configs/models/S4.toml` | 结构化状态空间（S4D 对角化）序列模型，使用频域卷积核 |
| `TimeMixer` | `configs/models/TimeMixer.toml` | 多尺度时序混合 |
| `FITS` | `configs/models/FITS.toml` | 频域插值 — 在频域压缩后重建 |
| `SparseTSF` | `configs/models/SparseTSF.toml` | 基于周期对齐采样的稀疏跨周期预测 |
| `CycleNet` | `configs/models/CycleNet.toml` | 从残差中分离周期模式 |
| `TiDE` | `configs/models/TiDE.toml` | 时序稠密编解码器，支持协变量 |
| `FiLM` | `configs/models/FiLM.toml` | 频率增强的 Legendre 记忆单元，结合低秩近似 |
| `FreTS` | `configs/models/FreTS.toml` | 在频域实部/虚部分量上应用 MLP |
| `Koopa` | `configs/models/Koopa.toml` | 基于 Koopman 理论的算子，分离时不变与时变动态 |
| `SOFTS` | `configs/models/SOFTS.toml` | 序列-核融合，通过 STar 聚合-再分配模块实现通道交互 |
| `TimeKAN` | `configs/models/TimeKAN.toml` | Kolmogorov-Arnold 网络，结合多尺度频率分解进行预测 |
| `Amplifier` | `configs/models/Amplifier.toml` | 基于放大器的预测器 |
| `TimeBase` | `configs/models/TimeBase.toml` | 时间基础架构 |
| `TimeBridge` | `configs/models/TimeBridge.toml` | 桥接架构 |
| `TimeEmb` | `configs/models/TimeEmb.toml` | 增强时间戳嵌入的模型 |
| `PaiFilter` | `configs/models/PaiFilter.toml` | 可学习滤波模型 |
| `TexFilter` | `configs/models/TexFilter.toml` | 纹理启发的滤波模型 |
| `SVTime` | `configs/models/SVTime.toml` | 基于奇异值分解 |
| `CMoS` | `configs/models/CMoS.toml` | 通道混合结构 |
| `PWS` | `configs/models/PWS.toml` | 分块时序模型 |
| `Sumba` | `configs/models/Sumba.toml` | 动态图卷积预测器，配合膨胀 inception 时序块 |
| `CrossGNN` | `configs/models/CrossGNN.toml` | 跨尺度、跨变量图网络，无需外部邻接矩阵即可建模多尺度交互 |
| `MSGNet` | `configs/models/MSGNet.toml` | 多尺度序列间图网络——通过 FFT 选择周期，并在内部自适应构建变量图（无需外部邻接矩阵） |
| `TimeFilter` | `configs/models/TimeFilter.toml` | patch 级时空图过滤，内部学习 patch 图（无需外部邻接矩阵） |
| `MoFo` | `configs/models/MoFo.toml` | 周期模式 Transformer，周期对齐 patch |
| `PHAT` | `configs/models/PHAT.toml` | 周期异质性 Transformer；`PHAT_Attention` ⚠️ **未验证**的论文重建（arXiv:2602.00654），非论文复现 |
| `CATS` | `configs/models/CATS.toml` | 查询自适应掩码 Transformer，对未来 token 做交叉注意力 |
| `RidgeRegressionTS` | `configs/models/RidgeRegressionTS.toml` | PyTorch 原生岭回归风格滞后窗口预测器，带 L2 正则 |
| `LassoRegressionTS` | `configs/models/LassoRegressionTS.toml` | PyTorch 原生 Lasso 风格滞后窗口预测器，带 L1 正则 |
| `ElasticNetTS` | `configs/models/ElasticNetTS.toml` | Elastic-Net 风格滞后窗口预测器，组合 L1 与 L2 正则 |
| `BayesianRidgeTS` | `configs/models/BayesianRidgeTS.toml` | Bayesian Ridge 启发的线性预测器，使用收缩正则 |
| `PolynomialRegressionTS` | `configs/models/PolynomialRegressionTS.toml` | 多项式滞后预测器，使用原始、平方和平方根历史特征 |
| `KNNForecasterTS` | `configs/models/KNNForecasterTS.toml` | 可微 KNN 风格原型预测器，使用 RBF 权重 |
| `SVRForecasterTS` | `configs/models/SVRForecasterTS.toml` | SVR 启发的 RBF 原型预测器，带线性跳连 |
| `GaussianProcessTS` | `configs/models/GaussianProcessTS.toml` | Gaussian Process 启发的原型核预测器 |
| `DecisionTreeTS` | `configs/models/DecisionTreeTS.toml` | 基于滞后特征的单棵可微软决策树 |
| `RandomForestTS` | `configs/models/RandomForestTS.toml` | 随机森林风格软树集成 |
| `ExtraTreesTS` | `configs/models/ExtraTreesTS.toml` | Extra-Trees 风格随机浅层软树集成 |
| `GradientBoostingTS` | `configs/models/GradientBoostingTS.toml` | 梯度提升风格残差软树集成 |
| `XGBoostTS` | `configs/models/XGBoostTS.toml` | XGBoost 风格残差软树集成，以 Torch 预测器注册 |
| `LightGBMTS` | `configs/models/LightGBMTS.toml` | LightGBM 风格轻量残差软树集成 |
| `CatBoostTS` | `configs/models/CatBoostTS.toml` | CatBoost 风格有序残差软树集成 |
| `ARIMATS` | `configs/models/ARIMATS.toml` | ARIMA 启发的可微分预测器，基于历史差分外推 |
| `AutoRegressiveTS` | `configs/models/AutoRegressiveTS.toml` | 自回归滞后窗口预测器 |
| `ExpSmoothingTS` | `configs/models/ExpSmoothingTS.toml` | 指数平滑启发预测器，带可学习衰减和趋势外推 |
| `KalmanFilterTS` | `configs/models/KalmanFilterTS.toml` | Kalman Filter 启发的 alpha-beta 平滑器，带可学习更新增益 |
| `MLPForecasterTS` | `configs/models/MLPForecasterTS.toml` | 基础 MLP 滞后窗口预测器，带通道混合 |
| `RNNForecasterTS` | `configs/models/RNNForecasterTS.toml` | 基础 vanilla-RNN 序列预测器 |
| `GRUForecasterTS` | `configs/models/GRUForecasterTS.toml` | 基础 GRU 序列预测器 |
| `LSTMForecasterTS` | `configs/models/LSTMForecasterTS.toml` | 时间序列设定下的基础 LSTM 序列预测器 |
| `TCNForecasterTS` | `configs/models/TCNForecasterTS.toml` | 小型时序卷积预测器 |
| `Aurora` | `configs/models/Aurora.toml` | 通用多模态时间序列基础模型适配器，融合相位、频域与通道上下文。 |
| `TimeAlign` | `configs/models/TimeAlign.toml` | 分布感知对齐预测器，将预测窗口统计量匹配到近期上下文。 |
| `GTR` | `configs/models/GTR.toml` | 全局时间检索适配器，将局部窗口与长周期时间上下文混合。 |
| `PhaseFormer` | `configs/models/PhaseFormer.toml` | 相位域预测器，聚合周期对齐的历史模式。 |
| `PMDformer` | `configs/models/PMDformer.toml` | Patch 均值解耦预测器，分离局部形状与趋势水平。 |
| `MMPD` | `configs/models/MMPD.toml` | 多模态 patch 扩散启发适配器，用于多样化时间序列预测。 |
| `COSA` | `configs/models/COSA.toml` | 上下文感知输出空间适配预测器，用于测试时预测校正。 |
| `DistDF` | `configs/models/DistDF.toml` | 联合分布对齐适配器，受 Wasserstein 预测-标签匹配启发。 |
| `Sonnet` | `configs/models/Sonnet.toml` | 谱算子神经预测器，强调平滑谐波成分。 |
| `APN` | `configs/models/APN.toml` | 自适应周期网络风格预测器，使用相位投影。 |
| `TimeCAP` | `configs/models/TimeCAP.toml` | 通道感知预训练启发适配器，使用上下文感知时间提示。 |
| `GOTSF` | `configs/models/GOTSF.toml` | 目标导向预测器，可让预测偏向应用指定的目标区间。 |
| `FTP` | `configs/models/FTP.toml` | FusionTimePatch 风格适配器，联合通道独立与通道混合时间视角。 |
| `OccamVTS` | `configs/models/OccamVTS.toml` | 视觉模型蒸馏启发预测器，这里以多模态时间门控表示。 |
| `HN_MVTS` | `configs/models/HN_MVTS.toml` | HyperNetwork 风格层次适配器，用于多变量时间序列预测。 |
| `SEMPO` | `configs/models/SEMPO.toml` | 轻量时间序列基础模型适配器，结合频谱分解与提示专家路由。 |
| `InterPDN` | `configs/models/InterPDN.toml` | 逐步概率分布建模适配器，使用稳定的序数式预测窗口。 |
| `TimeO1` | `configs/models/TimeO1.toml` | 转换标签对齐启发适配器，用于解码后的预测校正。 |
| `FeTS` | `configs/models/FeTS.toml` | 特征感知预测适配器，学习稀疏时间重要性掩码。 |
| `SymTime` | `configs/models/SymTime.toml` | 符号化时间序列基础模型适配器，将预测约束在近期水平与尺度附近。 |
| `ImplicitForecaster` | `configs/models/ImplicitForecaster.toml` | 隐式神经解码器，从潜在时间坐标形成预测窗口。 |
| `AMRC` | `configs/models/AMRC.toml` | 自适应掩码损失适配器，结合表示一致性的时间核心保留机制。 |
| `HMformer` | `configs/models/HMformer.toml` | 层次多尺度 Transformer 风格适配器，用于长期预测。 |
| `TiRex` | `configs/models/TiRex.toml` | 零样本 xLSTM 启发预测适配器，以时间专家组合实现。 |
| `LatentTSF` | `configs/models/LatentTSF.toml` | 潜在状态预测适配器，从紧凑隐藏状态解码未来数值。 |
| `CoRA` | `configs/models/CoRA.toml` | 面向多变量预测基础模型的相关性感知适配器。 |
| `DynamicTMoE` | `configs/models/DynamicTMoE.toml` | 漂移感知动态专家混合适配器，用于非平稳预测。 |
| `PULSE` | `configs/models/PULSE.toml` | 生成式相位演化适配器，用于非平稳时间序列预测。 |
| `OLinear` | `configs/models/OLinear.toml` | 正交变换线性预测适配器，带归一化通道混合。 |
| `MAFS` | `configs/models/MAFS.toml` | 多智能体预测适配器，组合专门化时间专家。 |
| `TSRAG` | `configs/models/TSRAG.toml` | 检索增强时间序列基础模型适配器，用于零样本预测。 |
| `TimeMosaic` | `configs/models/TimeMosaic.toml` | 自适应粒度 patch 与分段解码适配器，用于异质时间序列。 |
| `Kronos` | `configs/models/Kronos.toml` | 大规模时间序列基础模型适配器，使用提示式时间条件。 |

---

## 时空学习

节点结构化或图预测模型，同时建模时间动态与空间 / 节点关系。
这类模型通过 `spatiotemporal` 数据设定接收历史值以及节点 / 日历协变量。

| 名称 | 配置 | 说明 |
|---|---|---|
| `BiST` | `configs/models/BiST.toml` | 轻量双向 MLP，自适应图 |
| `MAGE` | `configs/models/MAGE.toml` | 自适应图专家混合 |
| `STOP` | `configs/models/STOP.toml` | 解耦基座 MLP + Core_Adaptive 残差校正 |
| `STID` | `configs/models/STID.toml` | 时空身份 MLP，含节点 / 时刻 / 星期嵌入 |
| `GWNet` | `configs/models/GWNet.toml` | Graph WaveNet：自适应邻接 + 膨胀因果卷积 |
| `STGCN` | `configs/models/STGCN.toml` | 时空图卷积网络（图卷积 + 时间卷积块） |
| `DCRNN` | `configs/models/DCRNN.toml` | 扩散卷积循环网络（GRU 内做双向随机游走图卷积） |
| `MTGNN` | `configs/models/MTGNN.toml` | 联合学习图结构 + mix-hop 图卷积 + 膨胀时间卷积 |
| `AGCRN` | `configs/models/AGCRN.toml` | 自适应图卷积 GRU，节点自适应参数（从节点嵌入学邻接） |
| `STNorm` | `configs/models/STNorm.toml` | WaveNet 主干上做空间 + 时间归一化（无需外部图） |
| `StemGNN` | `configs/models/StemGNN.toml` | 谱-时序 GNN（图 + 离散傅里叶变换），学习潜在关联图 |
| `STGODE` | `configs/models/STGODE.toml` | 图神经 ODE，建模连续时空动态 |
| `STAEformer` | `configs/models/STAEformer.toml` | 时空自适应嵌入 Transformer（在时间与节点维上做注意力） |
| `GTS` | `configs/models/GTS.toml` | 联合学习离散图结构 + DCRNN 风格的循环预测器 |
| `DGCRN` | `configs/models/DGCRN.toml` | 动态图卷积循环网络（GRU 内使用随时间变化的邻接） |
| `STDN` | `configs/models/STDN.toml` | 时空解耦网络 |
| `DFDGCN` | `configs/models/DFDGCN.toml` | 数据驱动频域动态图卷积网络（移植自 GestaltCogTeam/DFDGCN，MIT 许可） |
| `STPGNN` | `configs/models/STPGNN.toml` | 时空关键节点图神经网络 |
| `D2STGNN` | `configs/models/D2STGNN.toml` | 解耦动态时空图网络（用动态图分离扩散信号与固有信号） |
| `MegaCRN` | `configs/models/MegaCRN.toml` | 元图卷积循环网络，配合记忆增强的图学习器 |
| `HimNet` | `configs/models/HimNet.toml` | 面向时空预测的分层交互记忆网络 |
| `BigST` | `configs/models/BigST.toml` | 线性复杂度时空 GNN，通过随机特征线性注意力扩展到大规模图 |
| `STWave` | `configs/models/STWave.toml` | 解耦趋势/事件的时空 Transformer，使用离散小波分解 |
| `STTN` | `configs/models/STTN.toml` | 时空 Transformer 网络（解耦的空间 + 时间注意力） |
| `DSTAGNN` | `configs/models/DSTAGNN.toml` | 动态时空感知 GNN（数据驱动动态图 + 多头注意力） |
| `HL` | `configs/models/HL.toml` | Historical Last——重复最后一个观测步（朴素基线） |
| `LSTM` | `configs/models/LSTM.toml` | 逐节点的普通 LSTM 序列预测器 |
| `RPMixer` | `configs/models/RPMixer.toml` | 随机投影 MLP-Mixer |

---

## 协变量预测

对应原空气质量预测模型族。它们预测节点目标值，并使用历史协变量；带解码端协变量块的模型还会通过 `covariate` 数据设定使用已知未来协变量。

| 名称 | 配置 | 说明 |
|---|---|---|
| `CauAir` | `configs/models/CauAir.toml` | 因果协变量注意力，使用未来协变量 |
| `AirCade` | `configs/models/AirCade.toml` | 因果解耦，使用未来协变量，默认 `freq_mae` 损失 |
| `ASTGCN` | `configs/models/ASTGCN.toml` | 基于注意力的时空 GCN（在 Chebyshev 图卷积上叠加空间 + 时间注意力） |
| `GCLSTM` | `configs/models/GCLSTM.toml` | 图卷积 LSTM（在 LSTM 门内嵌入 Chebyshev 图卷积） |
| `DeepAir` | `configs/models/DeepAir.toml` | 基于融合的深度空气质量预测器 |
| `GAGNN` | `configs/models/GAGNN.toml` | 组感知图神经网络（组/城市级注意力加 GNN） |
| `PM25_GNN` | `configs/models/PM25_GNN.toml` | GNN + GRU 的 PM2.5 预测器，使用领域知识构边 |
| `AirFormer` | `configs/models/AirFormer.toml` | 因果时间注意力加随机隐变量的空气质量模型 |
| `PCDCNet` | `configs/models/PCDCNet.toml` | 物理/因果引导的动态卷积网络 |
| `AirPhyNet` | `configs/models/AirPhyNet.toml` | 物理信息网络，基于扩散/平流 ODE（需 `torchdiffeq`） |
| `AirDualODE` | `configs/models/AirDualODE.toml` | 双 ODE 系统（物理 + 数据驱动）加知识融合（需 `torchdiffeq`） |
| `MGSFformer` | `configs/models/MGSFformer.toml` | 多粒度时空融合 Transformer |

---

## 共享模块

可复用的构建模块位于 `src/models/module/`：

| 模块 | 内容 |
|---|---|
| `embed.py` | 位置编码、时间特征嵌入、patch 嵌入 |
| `self_attention_family.py` | 点积、加性、Autoformer、FEDformer 注意力变体 |
| `fourier_correlation.py` | 频域互相关 |
| `auto_correlation.py` | 自相关计算 |
| `positional_encoding.py` | 正弦位置编码 |
| `revin.py` | RevIN — 可逆实例归一化 |
| `masking.py` | 三角因果掩码 |
| `conv_blocks.py` | 卷积构建块 |
| `transformer_encdec.py` | 标准 Transformer 编解码器层 |
| `autoformer_encdec.py` | Autoformer 专用编解码器 |
| `tst_transformer.py` | PatchTST Transformer 层 |
| `standard_norm.py` | InstanceNorm 封装 |

---

## 模型接口

所有模型遵循统一接口：

```python
# 构造器接收解包后的 model.params
model = Model(c_in=7, seq_len=512, pred_len=96, **other_params)

# forward 签名 — 不使用的参数用 *args 接收
def forward(self, x, x_mark, dec_inp, dec_mark):
    ...
```

`registry.py` 中注册的工厂接收 `(cfg: RootConfig, params: dict)`：

```python
def register() -> None:
    MODEL_REGISTRY.register(
        "MyModel",
        lambda cfg, params: Model(
            c_in=cfg.dataset.params.get("enc_in", 7),
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            **params,
        ),
        ModelParameterConfig,
    )
```
