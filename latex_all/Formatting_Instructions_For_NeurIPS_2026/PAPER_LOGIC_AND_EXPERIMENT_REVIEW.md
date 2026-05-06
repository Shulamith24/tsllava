# ChronoMorph 论文写作逻辑与实验设置复盘

本文当前的主线可以概括为：少样本时间序列分类不是简单地把序列转成文字交给 LLM，而是需要解决三个错配问题。第一，类别证据同时存在于时间顺序依赖和波形形态结构中，单一时序编码器可能在少样本条件下遗漏一部分判别线索。第二，连续时间序列特征和 LLM token embedding 分布不同，需要先学习稳定的 time-series-to-language interface。第三，TSC 是闭集分类，而自回归生成是开放式输出，因此最终决策应限制在有效类别集合内。

围绕这条主线，论文的方法叙事现在是：ChronoMorph 先用 temporal branch 捕获顺序依赖，用 pseudo-image visual branch 捕获形态结构；两路 token 经过轻量投影后拼接，形成连续 time-series tokens；Stage I 用无监督/自监督目标训练双视图编码器，Stage II 用 TSQA 学习接入冻结 causal LLM 的生成接口；下游少样本分类不使用外部分类头，而是为每个数据集创建 class tokens，并在有效类别 token 上做 one-pass scoring。对于有语义标签的数据集，label verbalizer prototype 进一步初始化并正则化这些 class-token rows。因此第三个创新点应写成“supervised closed-set decoding with semantic-prior class-token rows”，而不是把语义先验降级为边缘补充。

## 当前实验设置

UCR 主实验采用 UCR Archive 2018 的 128 个数据集，协议是 full-label-space few-shot：每个目标数据集包含所有类别；对每个类别从 official TRAIN split 中最多采样 1、2、5、10 个样本作为 support；在 official TEST split 的全部测试样本上评估。所有方法共享同一组 support indices，主表报告 5 次独立运行后的平均准确率，并进一步对数据集做 macro average。主指标包括每个 shot 的平均准确率、四个 shot 的 Avg，以及按数据集和 shot 计算的 AvgRank。

主表 baseline 组包括 ResNet、TapNet、COSCO-ResNet，PatchTST、DLinear、TimesNet、Autoformer、Crossformer、FEDformer、Informer，以及 GPT4TS。当前结果显示 ChronoMorph 在四个 shot 下都有最高平均准确率和最佳 AvgRank。相对最强 CNN/prototype baseline 的提升是 modest，但相对 Transformer-style 和 LLM/foundation baselines 的提升明显更大。这个口径很适合本文，因为 ChronoMorph 本身也包含 Transformer-style temporal branch 和 LLM pathway，实验要强调的是：不是“LLM 全面击败所有 CNN”，而是“在同类 Transformer/LLM 路线里，正确的时序-语言接口、双视图证据和闭集监督解码非常关键，同时整体结果还能与强 CNN/prototype 方法竞争并取得平均最优”。

预训练设置已经明确：Stage I 使用 TSQA-train raw series、M4-train raw series，以及来自固定 98 个 UCR 数据集 official TRAIN split 的无标签序列；不使用 UCR TEST split，也不使用 UCR 标签。正文应持续保持这个清楚边界：这不是测试集泄漏，也没有使用 UCR 类别标签；但它确实是同一 archive 的 unlabeled-source transfer，而不是严格 leave-dataset-out pretraining。

外部实验包括 MIT-BIH、SleepEDF、CinC2017，使用 full-label-space 10/20/30-shot 设置。外部主表显示 ChronoMorph 在 MIT-BIH 和 CinC2017 上平均最好，整体外部平均第二，SleepEDF 上落后于 ResNet/TapNet。因此外部主实验仍应定位为 transfer checks beyond UCR，不写成跨所有外部域的 uniform dominance。

语义先验实验需要单独强调。根据已更新的 `tables/ablations/m2_semantic_prior_ablation_accuracy.tex`，semantic-prior class tokens 在 10/20/30-shot 平均后对三个语义标签数据集均有提升：MIT-BIH 从 61.04 到 64.18，SleepEDF 从 47.16 到 50.83，CinC2017 从 34.63 到 36.27，整体平均从 47.61 到 50.43，提升 +2.82。细粒度 shot-level 结果仍有波动，例如 CinC2017 的 10-shot 和 30-shot 提升、20-shot 下降，但平均结论支持“语义先验 class-token rows 是监督闭集解码的有效扩展”。

效率实验的主结论是参数更新量小但部署和推理成本高。ChronoMorph 更新约 12.3M 参数，但总模型约 1.28B 参数，推理延迟远高于紧凑 CNN/prototype baselines。正文和 limitation 已经说明 runtime/memory 来自 representative UCR runtime subset，而不是全 128 数据集逐个测得的运行时均值。

## 已做的针对性修改

1. 摘要和贡献段保留“best average accuracy and average rank”，同时说明相对最强 CNN/prototype baseline 的提升是 modest，而对 Transformer-style 和 LLM/foundation baselines 的提升更大。

2. 第三个创新点已改写为“supervised closed-set decoding with semantic-prior class-token rows”：UCR 上验证 class-token scoring；外部语义标签数据集上验证 semantic-prior row initialization and regularization。

3. Method 中补充了 temporal branch、pseudo-image transform、fusion、Stage I loss 的关键细节，包括 PatchTST-style backbone、DINOv2 visual branch、逐样本 median/IQR normalization、stride ratio 0.1、MLP projectors、token concatenation、masked patch reconstruction 和 VICReg-style SSL。

4. 实验设置中明确 full-label-space 协议：所有类别都参与，每类采样 up to S 个 support，并在全部 official TEST examples 上评估。

5. 预训练数据段显式说明 Stage I 使用固定 98-dataset UCR TRAIN pool 的无标签序列；同时强调不使用 UCR TEST split 和 UCR labels。

6. 删除了“缺少 ROCKET/InceptionTime/distance baselines”作为风险或 limitation 的口径。当前论文只围绕已完成且协议一致的对比模型展开。

7. Ablation 表和正文口径恢复为全 128 UCR、5-run、full-label-space few-shot 协议；实验叙述以论文文件夹中已迁移的服务器完整表格为准。

8. 语义先验 summary 表已根据新的服务器真实数据同步：MIT-BIH 64.18、SleepEDF 50.83、CinC2017 36.27、Avg 50.43，整体提升 +2.82。

9. LLM backbone ablation 的解释保持克制：它证明当前实现和训练日程下 language pathway 有帮助，但不单独归因于 pretrained knowledge、参数量或 token-scoring interface 的某一个因素。

10. 附录中补充了论文 Stage I/II/III 与代码 stage 名称的对应关系：`stage0_encoder_ssl`、`stage1_tsqa_transfer`、`stage2_semantic_bridge` 以及后续 caption specialization。

## 建议的第三创新点写法

建议把第三创新点写成：

> We introduce a supervised closed-set decoding interface for generative few-shot TSC. Instead of attaching an external classifier head or relying on free-form label generation, ChronoMorph learns dataset-specific class-token rows and performs one-pass scoring only over valid class tokens. For semantic-label datasets, label verbalizer prototypes initialize and regularize the same class-token rows, injecting label semantics without abandoning the stable single-token decision rule.

中文理解是：这不是单纯的“constrained decoding 小技巧”，而是一个把 LLM 生成接口改造成闭集分类接口的设计。它同时解决两个问题：UCR 这类匿名标签数据集需要稳定的闭集决策；MIT-BIH/SleepEDF/CinC2017 这类标签有语义的数据集可以利用标签名称，但不能直接做多 token phrase likelihood，因为那会引入长度偏置、shared-prefix 问题和语言先验不稳定。class-token rows 把两者统一起来：匿名标签时它是 learned identifiers，语义标签时它是 semantic-prior identifiers。

## 当前最稳妥的论文定位

这篇论文现在最适合定位为：一个面向 LLM-based few-shot TSC 的完整接口设计，而不是单纯堆大模型。核心贡献是双视图证据建模、阶段式 time-series-to-language bridging，以及监督闭集解码/语义先验 class-token label space。实验上，UCR 主结果显示它在 128 个数据集、4 个 shot、5 runs 下取得最佳平均准确率和 AvgRank；相对 CNN/prototype 强基线是 modest gain，但相对 Transformer-style 和 LLM/foundation baselines 是更明显的 gain。这个叙事稳健，也更能解释为什么一个同样包含 Transformer 和 LLM 的方法优于已有 Transformer/LLM baselines。
