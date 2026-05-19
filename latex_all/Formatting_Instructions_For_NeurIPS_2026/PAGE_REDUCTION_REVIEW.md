# TimeMorph 主文页数压缩审稿报告

本文档只检查 `neurips_2026.tex` 中 `\appendix` 之前的正文内容，并参考了主文中 `\input` 的表格文件。未修改论文源码、表格源码或 PDF。当前 PDF 总页数为 46 页，其中正文在参考文献前已经进入第 10 页；若按常见 NeurIPS 主文 9 页限制理解，当前主要需要从 Method 和 Experiments 中稳定省出约 1 页以上。

> 重要提醒：NeurIPS 2026 的正式页数、参考文献、checklist、附录计入规则请以官方最新要求为准。这里的判断只针对当前稿件的主文压缩。

## 一句话结论

最值得压缩的不是逐句润色，而是把正文中“训练细节型公式”和“表格数字复述”移到附录或改成紧凑叙述。推荐优先压缩：

1. Stage I 预训练公式细节，第 353-441 行。
2. Semantic initialization 公式，第 301-344 行。
3. Few-shot adaptation 的参数集合公式，第 467-503 行。
4. Ablation 逐项 shot-level 数字复述，第 557-574 行。
5. Figure/table caption 和 Main Results 中重复表格数字的句子。

预计推荐组合可节省约 1.3-1.8 页；激进组合可节省约 1.7-2.2 页。

## 当前主文冗余来源

### 1. 双视图动机重复出现

重复位置包括 Introduction 第 36、38、42、44、46 行，Related Work 第 53-54 行，Method 第 62、101-106 行，Ablation 第 561、574 行。  

这条主线必须保留，但建议只在 Introduction 中完整论证一次。Related Work 和 Method 只需短承接，否则审稿人会感觉叙事在反复绕同一个点。

### 2. LLM interface / closed-set scoring 问题陈述重复

重复位置包括 Introduction 第 40、42、44、46 行，Related Work 第 56-57 行，Method 第 62、203-206、247-250、296-299 行。  

“continuous tokens aligned to LLM + avoid unconstrained decoding + class-token scoring”已经在 Introduction 讲清楚，Method 中应直接定义接口，不必多次解释为什么。

### 3. Method 公式密度过高

Stage I 的 VICReg-style 公式、semantic row regularizer、warm-up/LoRA 参数集合公式，对复现有帮助，但对主文说服力的边际贡献较低。正文保留设计理由和总目标即可，完整定义放 appendix 更符合会议主文节奏。

### 4. Experiments 复述表格数字

Main Results 和 Ablation 中反复复述四个 shot 的准确率。表格已经承载数字，正文应主要解释趋势、关键差距和审慎限定。

## 必须保留的核心证据链

以下内容不建议删除，否则会削弱投稿说服力：

1. Few-shot TSC 标签稀缺，单一证据空间不足，需要 temporal dynamics 和 waveform morphology。
2. LLM 接入时间序列存在连续特征对齐和 closed-set 分类两个关键难点。
3. TimeMorph 的核心结构：dual-view encoder + time-series-to-language projection/alignment + closed-set class-token scoring。
4. UCR 主实验协议：128 个 UCR 数据集、1/2/5/10-shot、相同 support sampling、官方 TEST split、五次运行、宏平均和 AvgRank。
5. UCR 主结果表。它是整篇论文最核心的性能证据。
6. 对强 CNN/prototype baseline 提升有限的审慎表述。TimeMorph 对 COSCO-ResNet 的 Avg 是 59.29 vs. 58.39，不能只写“显著优于所有方法”。
7. W/T/L 或等价的 dataset-level robustness 证据。因为平均优势较小，WTL 能说明结果不是少数数据集拉高。
8. 关键消融：curriculum pretraining、dual-view、class-token scoring、no-LLM backbone。
9. 效率表和 limitation 中对 LLM 成本的承认。LLM 方法如果不报告成本，很容易被审稿人质疑回避部署代价。
10. UCR pretraining 数据说明，尤其是“不用 TEST split / 不用 UCR class label / 可能使用部分 UCR TRAIN unlabeled sequences”。这是敏感但必须透明的信息。

## 高优先级压缩建议

### P1. 压缩 Stage I 预训练公式细节

位置：`neurips_2026.tex` 第 353-441 行。  
建议：将 pooled representation、完整 IVC 展开式、masked reconstruction 展开式压成一个段落和一个总损失。VICReg 三项用一句话说明，详细公式放 appendix。  
预计节省：45-60 行，约 0.5-0.8 页。  
风险：低。正文主线更清楚，复现细节仍可在附录给出。

可替换文本：

```latex
\paragraph{Stage I: dual-view representation pretraining.}
Stage I pretrains the dual-view encoder without class labels using two self-supervised signals. First, masked temporal patch reconstruction encourages the temporal branch to preserve local waveform content. Second, an invariance--variance--covariance regularizer \citep{bardes2022vicreg} is applied to pooled temporal, morphology, and fused representations from two augmented views, encouraging augmentation invariance while avoiding collapse and redundant dimensions. The overall objective is
\begin{equation}
\label{eq:stage1-loss}
\mathcal{L}_{\mathrm{I}}
=
\lambda_{\mathrm{rec}}\mathcal{L}_{\mathrm{rec}}
+
\lambda_t\mathcal{L}_{\mathrm{ivc}}^{t}
+
\lambda_m\mathcal{L}_{\mathrm{ivc}}^{m}
+
\lambda_f\mathcal{L}_{\mathrm{ivc}}^{f}.
\end{equation}
We also use branch dropout to prevent the fused representation from collapsing onto a single view. Full definitions of the reconstruction and regularization terms are provided in Appendix~\ref{app:details}.
```

### P2. 压缩 semantic initialization 公式

位置：第 301-344 行。  
建议：UCR 主实验不用 semantic stage，因此正文不需要完整 prototype averaging 和 cosine regularization 公式。保留“label verbalizers initialize and regularize class-token rows”即可。  
预计节省：25-35 行，约 0.3-0.45 页。  
风险：低到中。若 contribution 中保留 semantic label prior，则正文仍需留一句定义和一处实验结果。

可替换文本：

```latex
\paragraph{Semantic initialization for named labels.}
For anonymous-label benchmarks, class tokens are learned as dataset-specific identifiers. When meaningful class names are available, we initialize each class-token row from the average embedding of template-consistent label verbalizers and regularize the input and output rows toward this semantic prototype during adaptation. The resulting loss adds a cosine regularization term to $\mathcal{L}_{\mathrm{ct}}$:
\begin{equation}
\label{eq:adapt-loss}
\mathcal{L}_{\mathrm{adapt}}
=
\mathcal{L}_{\mathrm{ct}}
+
\lambda_{\mathrm{sem}}\mathcal{L}_{\mathrm{sem}},
\end{equation}
with $\lambda_{\mathrm{sem}}=0$ for datasets without meaningful label names. The prototype construction and row regularizer are detailed in Appendix~\ref{app:details}.
```

### P3. 压缩 Dual-View Encoder 小节

位置：第 101-198 行。  
建议：temporal view 的 patch indexing、morphology view 的 median/IQR、`sqrt(T)`、resize/RGB/contrast adjustment 等实现细节可移附录。正文保留双分支、frozen vision encoder、fusion 定义即可。  
预计节省：25-40 行，约 0.35-0.55 页。  
风险：中。若压得太短，方法会显得像工程拼接；建议保留 morphology map 的直观作用。

可替换文本：

```latex
\subsection{Dual-View Time-Series Encoder}
\label{sec:dual-view-encoder}

TimeMorph represents each sequence through two complementary views. The temporal
branch extracts overlapping patches from the raw waveform and applies a
Transformer encoder to preserve chronological dynamics. In parallel, the
morphology branch robustly normalizes the sequence, unfolds it into a windowed
two-dimensional morphology map, and feeds the resized map to a frozen vision
backbone; we use its patch tokens and discard the class token. The two streams
are projected to a shared dimension and concatenated,
$Z_x=[Q_t(Z^t(x));Q_m(Z^m(x))]\in\mathbb{R}^{N_x\times d_e}$, yielding the
continuous time-series tokens passed to the language interface. Detailed patching
and morphology-map construction are given in Appendix~\ref{app:details}.
```

### P4. 压缩 Time-Series-to-Language Interface

位置：第 203-242 行。  
建议：标准 answer-token causal LM loss 可移附录。正文保留 conditioned embedding 公式和 Stage II calibration 的一句说明。  
预计节省：9-12 行。  
风险：低。

可替换文本：

```latex
\subsection{Time-Series-to-Language Interface}
\label{sec:ts-token-conditioning}

Rather than serializing numeric values as text, TimeMorph inserts continuous
time-series tokens directly into the LLM embedding stream. Let $P_\psi$ map
encoder tokens to the LLM hidden dimension and let $E_{\mathrm{lm}}$ denote the
token embedding matrix. For a prompt $p=(p^{\mathrm{pre}},p^{\mathrm{post}})$,
the conditioned input is
\[
H(x,p)=
[E_{\mathrm{lm}}(\tau(p^{\mathrm{pre}}));P_\psi(Z_x);
E_{\mathrm{lm}}(\tau(p^{\mathrm{post}}))].
\]
Stage II calibrates this interface with an answer-token language-modeling loss
on time-series question-answering examples; downstream classification uses the
same conditioned representation but replaces free-form decoding with the
closed-set rule below.
```

### P5. 压缩 Closed-Set Class-Token Prediction

位置：第 247-344 行。  
建议：删除 class-token set 公式，把 `v_c` 在文字中定义；保留 score 和 cross entropy，argmax 可内联。semantic prior 细节移附录。  
预计节省：25-35 行。  
风险：低。

可替换文本：

```latex
\subsection{Closed-Set Class-Token Prediction}
\label{sec:class-token-classification}

Few-shot TSC requires choosing among a fixed label set, whereas free-form
generation can leave the label space and is sensitive to tokenization and label
length. TimeMorph therefore assigns each class $c$ in dataset $D$ a single
dataset-specific class token with vocabulary index $v_c$. Given the final hidden
state $h_\Theta(x,p_D)$ at the classification prompt, the class score is the
corresponding next-token logit,
\[
s_c(x)=[W_{\mathrm{out}}h_\Theta(x,p_D)]_{v_c},
\qquad
\hat y=\arg\max_{c\in\{0,\ldots,C-1\}}s_c(x).
\]
Training minimizes cross entropy over only these valid class tokens. For datasets
with meaningful class names, we optionally initialize and regularize the class
token rows using label-verbalizer prototypes; anonymous-label benchmarks set this
regularizer to zero. Details are provided in Appendix~\ref{app:details}.
```

### P6. 压缩 Stage-wise Pretraining and Few-Shot Adaptation

位置：第 349-504 行。  
建议：Stage I/II/optional semantic/few-shot adaptation 统一写成紧凑训练流程。删去 warm-up 参数集合公式和 LoRA 参数集合公式。  
预计节省：50-75 行，约 0.7-1.0 页。  
风险：中。需确保附录有完整 objectives 和 hyperparameters。

可替换文本：

```latex
\subsection{Stage-wise Pretraining and Few-Shot Adaptation}
\label{sec:stage-wise-transfer}

TimeMorph is trained in stages. Stage I learns time-series representations
without class labels by combining masked temporal patch reconstruction with a
VICReg-style invariance, variance, and covariance objective over the temporal,
morphology, and fused streams. Stage II attaches the projector and frozen causal
LLM, then optimizes an answer-token language-modeling loss on time-series
question-answering examples, first updating the projector and then jointly
updating the encoder and projector. An optional semantic stage is used only when
meaningful label names or textual descriptions are available; it is not used for
the anonymous-label UCR benchmark.

For each downstream dataset, adaptation first updates the dual-view encoder, the
projector, and the dataset-specific class-token rows. A second phase additionally
updates LoRA adapters in the language model while keeping the pretrained LLM
weights frozen. Inference requires one forward pass per query and ranks only the
valid class tokens. Full objectives, hyperparameters, and stage-wise data sources
are reported in Appendix~\ref{app:details}.
```

### P7. 合并 Introduction 中的 dual-view 和三挑战叙事

位置：第 38 和 42 行。  
建议：第 38 行已经说明 dual-view，第 42 行又以 challenge 1 重复一次。可合并为更紧的 problem-to-design paragraph。  
预计节省：8-12 行。

可替换文本：

```latex
Discriminative evidence in time series is heterogeneous: some labels depend on ordered temporal dynamics such as trends, amplitudes, phase shifts, and long-range evolution, whereas others rely on waveform morphology such as peaks, cycles, motifs, or discriminative subsequences \citep{tang2020interpretable,ye2009time,dempster2020rocket,fawaz2020inceptiontime,wang2015encoding}. Existing CNN, Transformer, and prototype-based methods capture parts of this evidence, but few-shot supervision makes it difficult to preserve both views while avoiding unstable cues \citep{snell2017prototypical,tapnet,dyformer}. For LLM-based few-shot TSC, this creates three interface requirements: dual-view time-series representations, continuous-to-language alignment, and closed-set prediction over valid dataset labels.
```

然后可删除第 42 行整段，避免重复。

### P8. 压缩 Related Work 第二段

位置：第 53-54 行。  
建议：保留文献定位，不再完整重复 dual-space hypothesis。  
预计节省：5-8 行。

可替换文本：

```latex
\paragraph{Temporal and morphological evidence in TSC.}
TSC methods capture discriminative structure at different levels: shapelet methods focus on local subsequences \cite{ye2009shapelets,tang2020interpretable}, convolutional models such as InceptionTime and ROCKET exploit multi-scale local responses \cite{fawaz2020inceptiontime,dempster2020rocket}, and patch-based Transformers model longer temporal contexts \cite{nietime}. Complementary to these chronological representations, time-series imaging methods convert signals into two-dimensional structures for visual recognition \cite{wang2015imaging}, with recent work showing benefits from visualization and frozen vision transformers \cite{liu2025picture,roschmann2025time}. TimeMorph combines these two evidence spaces before the LLM interface.
```

### P9. 压缩 Experimental Setup

位置：第 511-525 行。  
建议：保留 protocol、公平性和 pretraining 敏感点；baseline 列表让 Table 1 承担；实现细节放附录。  
预计节省：10-18 行。

可替换文本：

```latex
\subsection{Experimental Setup}

We evaluate on all 128 UCR Archive 2018 datasets under a full-label-space
few-shot protocol. For each support size $S\in\{1,2,5,10\}$, we sample up to
$S$ labeled examples per class from the official training split and evaluate on
the official test split; all methods use the same support indices across five
runs. We compare with the CNN/prototype, Transformer-style, decomposition-based,
and LLM-style baselines listed in Table~\ref{tab:ucr_fewshot_paper_current-fewshot-main}.
Accuracy is averaged over runs and macro-averaged over datasets.

The main benchmark uses the Stage II checkpoint. Pretraining never uses official
UCR TEST examples or UCR class labels, although Stage I may include unlabeled
TRAIN sequences from a fixed UCR subset. Implementation details, checkpoint
selection rules, and baseline hyperparameters are provided in
Appendix~\ref{app:baseline-details}.
```

### P10. 压缩 Main Results

位置：第 529-540 行。  
建议：不逐 shot 复述准确率。保留 highest all shots、best AvgRank、strongest CNN/prototype margin modest、Transformer/LLM-style margin larger。  
预计节省：5-10 行。

可替换文本：

```latex
\paragraph{Benchmark-level comparison.}
Table~\ref{tab:ucr_fewshot_paper_current-fewshot-main} shows that TimeMorph
achieves the best mean accuracy at every support size and the best average rank.
The gains over the strongest CNN/prototype baseline are modest but consistent,
whereas the margins over Transformer-style and LLM-style baselines are larger.
Win/tie/loss counts in Table~\ref{tab:ucr_fewshot_paper_current-paper-wtl}
confirm this pattern: TimeMorph wins substantially more often than it loses
against PatchTST, TimesNet, and GPT4TS, while remaining closer to COSCO-ResNet
and ResNet.

\paragraph{External datasets.}
Table~\ref{tab:external-fewshot-avg} reports external few-shot results averaged
over 10-, 20-, and 30-shot settings. TimeMorph performs best on the reported
external average, supporting transfer beyond UCR, but these results should not be
read as uniform dominance across all domains.
```

注意：上述 external paragraph 适用于主表只报告两个外部数据集的情况。如果要保留 SleepEDF 叙述，必须先修正文和表的不一致。

### P11. 压缩 Ablation Studies

位置：第 553-574 行。  
建议：summary table 已经给出四个 shot 数字，正文只报告 Avg drop 和结论。  
预计节省：15-30 行。

可替换文本：

```latex
\subsection{Ablation Studies}

Table~\ref{tab:ablation-summary} summarizes the main UCR ablations under the
same five-run protocol. Removing curriculum pretraining lowers \texttt{Avg} by
3.35 points, indicating that the Stage I+II initialization is important when
downstream labels are scarce. Single-view variants also underperform the full
model, with average drops of 5.08 points without the morphology branch and 4.20
points without the temporal branch, supporting the complementarity of the two
views. Replacing class-token scoring with unconstrained generative adaptation
reduces \texttt{Avg} by 4.15 points, suggesting that the closed-set interface is
more stable for few-shot classification.

For external datasets with meaningful labels,
Table~\ref{tab:semantic-prior-summary} shows that semantic-prior class tokens
improve the averaged result, although the effect is not uniform at every shot.
The no-LLM variant in Table~\ref{tab:ablation-llm-backbone} reduces \texttt{Avg}
by 7.83 points, showing that the language pathway contributes beyond a linear
classifier on the same dual-view features. This ablation does not isolate whether
the gain comes from pretrained knowledge, additional capacity, or the token
scoring interface.
```

### P12. 压缩 Limitations

位置：第 578-585 行。  
建议：当前 limitation 很诚实，建议保留核心限定，但删掉与 Main Results/Efficiency 重复的长解释。  
预计节省：4-8 行。

可替换文本：

```latex
TimeMorph does not establish universal dominance of LLM-based TSC: gains over the strongest CNN/prototype baselines are modest in very low-shot UCR settings, and SleepEDF remains challenging in external evaluation. Its frozen LLM backbone also increases memory and inference cost, making it less suitable for compact deployment despite updating only a small fraction of parameters. In addition, the UCR pretraining setup uses unlabeled official TRAIN sequences from a fixed subset of UCR datasets, so it is not a strict leave-dataset-out transfer evaluation for every target. Semantic label priors should be viewed as regularization for class-token rows rather than a replacement for supervised adaptation. Deployment in clinical, industrial, or safety-critical settings would require domain-specific validation, privacy review, and failure analysis beyond this study.
```

如果最终决定正文不再提 SleepEDF，则这里第一句也应相应删除 SleepEDF。

## 图表和 caption 压缩建议

### Figure 1 caption

位置：第 86-94 行。  
问题：caption 同时承担 overview、branch 解释、Stage I/II/semantic stage 解释，太长。图中已有很多文字，caption 可压缩为 2 句。  
预计节省：4-6 行。

可替换文本：

```latex
\caption{Overview of TimeMorph. A temporal encoder and a frozen vision-based morphology encoder produce complementary time-series tokens, which are fused and projected into a pretrained causal LLM. Classification is performed by closed-set scoring over dataset-specific class tokens.}
```

### UCR main table caption

文件：`tables/main/ucr_fewshot_main.tex`。  
建议：删掉常规的 bold/underline 解释或放表下注。  
预计节省：1-2 行。

可替换文本：

```latex
\caption{Few-shot accuracy (\%) on 128 UCR datasets. Means are over five runs; \texttt{AvgRank} is averaged over dataset-level ranks across shots.}
```

### W/T/L table

文件：`tables/main/ucr_fewshot_wtl.tex`。  
建议：不建议直接删除，因为主结果与 COSCO-ResNet 的平均差距较小，WTL 能支撑 dataset-level robustness。若篇幅非常紧，可移附录，正文保留一句 “Win/tie/loss counts are reported in Appendix.”  
预计节省：约 0.15-0.25 页。  
风险：中。删掉后主文对“平均值是否由少数数据集驱动”的回应会弱一些。

### External table caption 和内容一致性

文件：`tables/main/external_fewshot_avg.tex`。  
当前 caption 写 “two external datasets”，表中只有 MIT-BIH 和 CinC2017；但正文第 539-540 行写 MIT-BIH、SleepEDF、CinC2017，并说 SleepEDF 上落后 ResNet/TapNet。这个不是单纯篇幅问题，而是审稿风险。建议二选一：

1. 主文只报告 MIT-BIH 和 CinC2017，SleepEDF 全量移附录，并删除正文中 SleepEDF 的结论。
2. 主文表补上 SleepEDF 列，并同步 caption 与正文数字。

若保留紧凑 caption，可用：

```latex
\caption{External few-shot accuracy (\%), averaged over 10-, 20-, and 30-shot settings.}
```

### Efficiency table caption

文件：`tables/main/efficiency_compact.tex`。  
建议：正文已解释 measurement subset 和 trade-off，caption 可短。  
预计节省：1-2 行。

可替换文本：

```latex
\caption{Compact efficiency comparison under the 10-shot UCR protocol.}
```

### Ablation table caption

文件：`tables/main/ucr_fewshot_ablation_summary.tex`。  
建议：caption 只说明 protocol 和 metric，anonymous-label 解释放正文或附录。  
预计节省：2-3 行。

可替换文本：

```latex
\caption{Top-level UCR ablations under the five-run full-label-space few-shot protocol. Values are mean accuracy (\%).}
```

### Semantic prior table

文件：`tables/main/semantic_prior_summary.tex`。  
建议：如果 contribution 中保留 semantic label prior，正文至少要留一处结果；如果篇幅紧，可移附录并在正文一句提及。若保留，caption 压缩为：

```latex
\caption{Semantic-prior ablation on external datasets, averaged over 10-, 20-, and 30-shot settings.}
```

## 需要优先修正的呈现风险

### 1. 外部数据集正文与主表不一致

正文写 “MIT-BIH, SleepEDF, and CinC2017”，但当前主表 `external_fewshot_avg.tex` 只包含 MIT-BIH 和 CinC2017。正文还写 “clearly behind ResNet and TapNet on SleepEDF”，但表中没有 SleepEDF。  

压缩建议：若目标是省页数，主文只保留两个外部数据集表，把 SleepEDF 相关完整结果和分析放附录；或补全主表。不能维持当前不一致。

### 2. Semantic prior 数值不一致

正文第 567 行写 anonymous average 从 47.61 到 50.43，但当前 `semantic_prior_summary.tex` 表中是 47.84 到 50.23。  

压缩建议：修改时以最终表格为准。正文可避免写具体起止数字，改成 “improves the averaged result by 2.39 points”，减少同步错误。

### 3. no-LLM 表在正文中被引用但似乎没有主文 input

正文第 571-572 行引用 `Table~\ref{tab:ablation-llm-backbone}`，但当前主文在 ablation 处只 input 了 `tables/main/ucr_fewshot_ablation_summary.tex` 和 `tables/main/semantic_prior_summary.tex`，未看到 `\input{tables/main/m2_llm_backbone_ablation_summary.tex}`。PDF 主文中也未显示 no-LLM 表，而是引用到附录 Table 23。  

压缩建议：不要只在文字里保留 no-LLM 证据。可把 no-LLM 作为 summary table 的一行并入 Table 5，或正文明确写 “reported in Appendix”，但 contribution 证据会弱一些。理想做法是合并入主消融表，这可能不增加太多版面。

### 4. Overclaim 风险

TimeMorph 对最强 CNN/prototype baseline 的提升很小，尤其低 shot。压缩时不要删掉 “modest but consistent” 和 limitation 中相关表述。否则审稿人会认为摘要和结论过度宣传。

## 三档压缩方案

### 保守方案：约省 0.35-0.55 页

执行 P7、P8、P10、P11 的短版，压缩 Figure/table caption，Limitations 简化。  
优点：几乎无风险。  
缺点：大概率仍不够回到 9 页以内。

### 推荐方案：约省 1.3-1.8 页

执行 P1、P2、P6、P7、P10、P11、P12，并压缩主要 captions。  
优点：不伤核心贡献，主文更像会议论文，能够稳定省出一页以上。  
缺点：需要确保附录补齐被移走的公式定义。

### 激进方案：约省 1.7-2.2 页

在推荐方案基础上，把 Dual-View Encoder 的 temporal/morphology 具体公式也大幅移附录，并考虑将 W/T/L 表或 semantic-prior 表移附录。  
优点：给 9 页限制留更大余量。  
缺点：方法细节和 dataset-level robustness 在主文中变弱，需要在附录和正文一句话中补偿。

## 建议的最终主文保留结构

1. Introduction：保留完整问题动机、LLM 接口挑战、贡献；删重复例子。
2. Related Work：每段只做定位，不展开方法细节。
3. Method：保留 dual-view、embedding insertion、class-token scoring、stage-wise training 的核心定义；公式从“完整实现”改为“必要接口”。
4. Experiments：保留 UCR protocol、主表、WTL 或一句 dataset-level robustness、效率表、主消融表。
5. Limitations：保留审慎声明，但压缩重复解释。

## 不建议删除的内容

1. UCR 主结果表。
2. UCR protocol 和 fairness/pretraining 数据说明。
3. “modest but consistent” 这类审慎表述。
4. dual-view、class-token scoring、no-LLM backbone 消融。
5. efficiency table 和 limitation 中关于成本的说明。
6. UCR pretraining 使用部分 TRAIN unlabeled sequences 的透明说明。

## 下一步建议

如果开始实际改稿，建议顺序是：

1. 先移动/压缩 Method 中 Stage I、semantic prior、few-shot adaptation 公式。
2. 再压缩 ablation 段落和 Main Results 中的数字复述。
3. 修正外部数据集和 semantic prior 数值不一致。
4. 重新编译，确认主文是否回到 9 页内。
5. 如果仍超页，再考虑移动 W/T/L 或 semantic-prior 表。

