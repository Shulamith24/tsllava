# Method 部分写作检查与修改建议

检查对象：`neurips_2026.tex` 的 Section 3 Method，主要对应第 68--488 行。  
检查方式：结合 Introduction、Related Work、Experiments、Appendix 中已有叙事一起看，重点关注 Method 是否能独立支撑读者理解 TimeMorph 的技术链条、训练阶段、数据来源和实验结论。

## 总体判断

当前 Method 的整体结构是成立的：先定义 few-shot TSC 问题，再给出框架，随后展开 dual-view encoding、time-series token projection、closed-set class-token classification，以及 curriculum pretraining/few-shot adaptation。它已经覆盖了论文贡献中的三个核心点：双视图表示、LLM embedding-space alignment、closed-set class-token scoring。

但从 NeurIPS 风格的主文 Method 来看，目前主要问题不是技术缺失，而是叙事重心不够均衡。Dual-view encoder 和 class-token scoring 写得相对充分，公式也比较完整；相比之下，`Embedding-space alignment pretraining` 和 `Optional label-semantic alignment` 明显偏短，尤其缺少数据来源、阶段目的、训练样本形式、与主实验/附录的对应关系。这会让审稿人产生两个疑问：

1. Stage I/II/III 到底用了哪些数据？是否接触了目标数据集？是否使用了 UCR label 或 TEST split？
2. label-semantic alignment、label semantic priors、semantic-prior ablation 三者之间是什么关系？哪些是主方法默认使用的，哪些只用于外部语义标签数据集？

因此，我建议 Method 不要大改技术公式，但要强化“训练阶段和数据边界”的说明，并把 semantic 部分的术语关系理顺。

## 优先级最高的修改

### 1. 在 Method 开头或 Framework 末尾提前说明三阶段训练和默认 checkpoint

位置建议：第 103--108 行 Framework 段落之后，或第 343 行 `Curriculum Pretraining and Few-Shot Adaptation` 开头。

当前 Method 直到第 433 行才进入 embedding alignment，而且数据源只写了 `\mathcal{D}_{\mathrm{TSQA}}`。真正的数据源说明在 Experiments 第 499 行和 Appendix 第 647--649 行。这对实验部分是充分的，但 Method 里过于隐含。

建议在 Method 中加入一个很短的阶段说明，让读者先知道：

- Stage I：raw TSQA-train series、raw M4-train series、98-dataset UCR pool 的 official TRAIN raw series，用于无监督/自监督 dual-view representation pretraining。
- Stage II：TSQA train split，用于 time-series QA 语言建模式 alignment。
- Stage III：可选，TSQA/M4 captioning/synthetic attribute supervision，用于 semantic enrichment，不用于默认 UCR 主结果。
- 默认 UCR 主结果使用 Stage II checkpoint；不使用 UCR TEST split 或 UCR class labels。

推荐写法：

```tex
In the default benchmark, the curriculum contains two active pretraining stages before target-dataset adaptation. Stage I trains the dual-view encoder on unlabeled raw series from TSQA-train, M4-train, and official TRAIN-split sequences from a fixed UCR pretraining pool. Stage II uses the TSQA training split to align projected time-series tokens with the frozen LLM embedding space through answer-token language modeling. An optional Stage III can further use caption or attribute-style supervision when semantic descriptions are available, but it is not used for the default UCR results. No stage uses any official UCR TEST split or UCR class labels.
```

这段可以放在 `Curriculum Pretraining and Few-Shot Adaptation` 开头，也可以拆成 Stage I/II/III 三个 paragraph 内部。

### 2. 扩展 `Embedding-space alignment pretraining`

位置：第 433--448 行。

你的直觉是对的，这个小节现在体量偏简略。它只有一个目标函数和一句两阶段优化，缺少读者最关心的上下文：TSQA 是什么形式的数据、为什么 QA loss 能完成 embedding-space alignment、哪些模块训练/冻结、与 Stage II checkpoint 的关系。

建议最少扩展到 1.5--2 个自然段。可以不加入新公式，但要解释：

- 输入样本形式：`(x,p,a)`，其中 `x` 是 raw time series，`p` 是 question/prompt，`a` 是 textual answer。
- 数据集：Stage II 使用 TSQA train split；Stage I 中也用了 raw TSQA/M4/UCR TRAIN raw series。
- 训练目标：只在 answer tokens 上计算 LM loss，不要求模型生成自由标签；目的是让 LLM 的 frozen layers 能把 projected time-series tokens 当作可用条件。
- 参数更新：先固定 encoder 更新 projector，再联合更新 encoder+projector；LLM backbone frozen。
- 数据边界：不使用 UCR TEST split 或 UCR class labels。

推荐替换文本：

```tex
\paragraph{Embedding-space alignment pretraining.}
The second pretraining phase attaches the projector $P_\psi$ and the pretrained causal LLM to the Stage-I dual-view encoder. We use the TSQA training split, where each example is represented as $(x,p,a)$: a raw time series $x$, a natural-language or template-based prompt $p$ about the series, and an answer string $a$. The projected tokens $P_\psi(Z_x)$ are inserted between the prompt tokens as in Eq.~\eqref{eq:conditioned-input}, and the loss is applied only to the answer tokens:
...
This stage does not train the model to perform target-dataset classification. Instead, it teaches the projection interface to produce continuous tokens that the frozen LLM can condition on under its standard next-token objective. We first update $P_\psi$ with the dual-view encoder fixed, then jointly optimize the encoder and projector while keeping the LLM backbone frozen. The resulting Stage-II checkpoint is the default initialization for the UCR few-shot benchmark; no UCR TEST sequence or UCR class label is used in this phase.
```

其中公式 `\mathcal{L}_{\mathrm{align}}` 可以保留现有写法。

### 3. 扩展或重命名 `Optional label-semantic alignment`

位置：第 450--453 行。

这个小节目前确实过短，而且与第 302--341 行的 `Label semantic priors` 容易混淆。前者看起来是一个“预训练阶段”，后者是“下游分类时 class-token rows 的初始化和正则化”。但现在两者都叫 semantic alignment/semantic priors，读者可能不清楚：

- `Optional label-semantic alignment` 是否就是 semantic-prior ablation？
- 它是否用于外部 MIT-BIH/SleepEDF/CinC2017？
- 它是否用于 Table 1 的 UCR 主结果？
- 它的数据来自哪里？

建议二选一：

**方案 A：如果 Stage III 不是主方法核心，只保留简短但清晰的数据边界。**

将标题改为 `Optional semantic enrichment pretraining.`，避免和 downstream label semantic priors 混淆。

推荐替换文本：

```tex
\paragraph{Optional semantic enrichment pretraining.}
When auxiliary training data provide textual descriptions, captions, or attribute-style supervision, TimeMorph can include an additional semantic-enrichment stage after embedding-space alignment. In our implementation, this optional stage uses mixtures derived from TSQA and M4 raw series, including captioning or synthetic attribute supervision, to encourage time-series-conditioned hidden states to support label-related language. This stage is not used for the default UCR benchmark because UCR labels are anonymous and the main results use the Stage-II checkpoint. We therefore treat semantic enrichment as an optional extension rather than a required component of the core UCR method.
```

**方案 B：如果外部语义标签实验只用了 class-row semantic priors，没有使用 Stage III。**

那就更应该把 Stage III 写弱一点，并明确“semantic-prior results in experiments refer to class-token row initialization/regularization, not necessarily this optional pretraining stage”。否则审稿人可能误以为外部语义结果来自额外预训练。

推荐补一句：

```tex
The semantic-prior ablation in Section~\ref{sec:experiments} evaluates the downstream class-token prior in Eq.~\eqref{eq:semantic-row-reg}; it should be distinguished from this optional pretraining stage.
```

如果这句话事实不准确，就不要加；但一定要在 Method 里把两者关系讲清楚。

### 4. 明确 Stage I 的数据源，不要只在 Experiments/Appendix 里说

位置：第 348--431 行 `Representation pretraining`。

当前该小节讲了 masked reconstruction、IVC、branch dropout，但没有说明训练数据是什么。Appendix 已经写了 Stage I 数据源，但 Method 中完全缺位，会让第 348 行 “without class labels” 变得过于抽象。

建议在第 349 行后加一句：

```tex
In the default setting, this phase uses unlabeled raw series from TSQA-train, M4-train, and official TRAIN-split sequences from a fixed UCR pretraining pool, without using any UCR TEST sequence or class label.
```

这句话很重要，因为它主动处理了一个潜在审稿风险：UCR TRAIN raw series 被用于无监督 pretraining。你已经在 Experiments 和 Limitations 里诚实说明了这一点，Method 中也应该前后一致。

## 中等优先级的写作与逻辑问题

### 5. `Problem Setup` 中 K-shot 与 Experiments 中 S-shot 命名不统一

位置：第 85--96 行；Experiments 第 496 行。

Method 用 `$K$-shot`，Experiments 用 `$S\in\{1,2,5,10\}$`。这不是严重问题，但建议统一为 `$K$`，或者在 Experiments 中写 `K\in\{1,2,5,10\}`。论文中 few-shot support size 的符号最好保持一致。

### 6. `Framework` 的四组件叙述略显平铺，可以更贴近贡献主线

位置：第 103--108 行。

当前写法是组件枚举，清楚但不够有“为什么这样设计”的力度。建议改成“evidence extraction -> modality interface -> closed-set decision -> curriculum initialization”的因果链。

推荐写法：

```tex
TimeMorph separates the interface problem into four steps. The dual-view encoder first preserves complementary chronological and morphology-oriented evidence. The projector then converts the fused continuous tokens into the hidden dimension expected by a pretrained causal LLM. The class-token classifier restricts the autoregressive vocabulary head to the valid label set of the target dataset. Finally, curriculum pretraining initializes the encoder and projector before few-shot adaptation, reducing the burden of learning both time-series structure and the LLM interface from only a few labeled examples.
```

### 7. `Morphology encoder` 需要补充一点“为什么这个 map 合理”

位置：第 160--197 行。

现在 morphology map 的公式比较完整，但解释仍偏工程描述。建议增加一句把 map 与论文主张连接起来：它不是把时序变成随意图像，而是把 local subsequence windows 排列成二维结构，让视觉 backbone 看到 repeated shape、cross-window transition、motif-like patterns。

推荐补充：

```tex
This construction is intended to expose repeated local shapes and cross-window transitions rather than to treat the time series as a natural image.
```

此外，`$\rho$` 的默认值没有在 Method 中说明，Appendix 里实际是 `0.1`。如果篇幅允许，可以写 `with $\rho=0.1$ in our experiments`，或统一放到 implementation details。

### 8. `P_\psi` 在 Method 中写成 linear projector，但 Appendix 说是 MLP projector

位置：第 221 行；Appendix table 第 612 行。

Method 第 221 行：

```tex
Let $P_\psi$ be a linear projector ...
```

Appendix 写的是：

```tex
Language projector & MLP projector into the LLM hidden dimension followed by output LayerNorm
```

这是一个需要修正的一致性问题。建议 Method 改为更稳妥的：

```tex
Let $P_\psi$ be a trainable projection module from encoder dimension $d_e$ to the LLM hidden dimension ...
```

这样既兼容 linear，也兼容 MLP。

### 9. `Optional textual descriptions` 这句话位置有点悬空

位置：第 233 行。

当前写：

```tex
Optional textual descriptions, when available, are appended to the prompt and encoded by the same tokenizer.
```

但 Method 前面没有说明哪些阶段有 textual descriptions，后面 optional semantic alignment 又很短。因此这句话读起来像突然引入一个未定义输入。建议移到 semantic enrichment 或 label-semantic prior 小节，或者改成：

```tex
When a pretraining or downstream setting provides auxiliary textual descriptions, they are included as part of $p$ and encoded by the same tokenizer.
```

### 10. Class-token notation `\langle c_c\rangle` 容易误读

位置：第 258--271 行。

`\langle c_c\rangle` 这种写法容易让人误解为 “class c subscript c”。建议改成更清楚的 token notation，例如：

```tex
\mathcal{C}_D=\{\langle y_0\rangle,\ldots,\langle y_{C-1}\rangle\}
```

或者：

```tex
\mathcal{C}_D=\{\langle \mathrm{cls}_0\rangle,\ldots,\langle \mathrm{cls}_{C-1}\rangle\}
```

后者更符合“dataset-specific class-token identifiers”的语义。

### 11. 需要说明 class tokens 如何保证 single-token

位置：第 255--300 行。

你写了 “single-token class identifiers”，但没有解释这是通过新增 vocabulary rows / tokenizer special tokens 实现的。Appendix 里有 `Tokenizer training mode & class_rows`，但 Method 主文最好补一句：

```tex
We implement these identifiers as newly added special tokens, so each class corresponds to exactly one vocabulary index.
```

这能直接回应 generative classification 中常见的 label-tokenization bias 问题。

### 12. input/output class-token rows 是否共享或分别训练，需要表达更精确

位置：第 300、317--331 行。

当前写 “input embeddings and output rows”，这对 untied LM head 是清楚的；但 Llama 系模型常见实现中 input/output embeddings 是否 tied 取决于模型配置。建议避免引发细节追问，写成：

```tex
For models with separate input and output embeddings, both rows are trainable; when the embeddings are tied, this reduces to updating the shared class-token row.
```

如果你的实现明确 untied，则可不用这句，但最好在 Method 或 Appendix 中说清楚。

## 低到中优先级的表达润色

### 13. Method 开头可以更强地承接 Introduction 的三挑战

位置：第 71--73 行。

当前开头很清楚，但可以更紧贴 Introduction 第 51 行的三挑战：“dual evidence, continuous-to-language alignment, closed-set prediction”。推荐改成：

```tex
TimeMorph addresses the three interface challenges identified above: preserving complementary temporal and morphological evidence, aligning continuous time-series tokens with the LLM embedding space, and restricting prediction to a valid closed label set.
```

这会让论文主线更统一。

### 14. `Representation pretraining` 的公式篇幅偏长，可考虑压缩说明

位置：第 351--431 行。

这部分目前公式密度较高，而 Stage II/III 反而很短，导致 Method 的视觉重心偏向 Stage I。建议不是删除公式，而是考虑把 Eq. `\mathcal{L}_{\mathrm{ivc}}` 或 reconstruction 的细节略压缩，把篇幅让给 Stage II 数据和 semantic 部分。如果页数紧张，可以把 VICReg 三项公式保留，但减少文字解释；如果页数不紧张，则不必动。

### 15. `Few-shot adaptation` 建议补充 warm-up/joint 的动机

位置：第 455--487 行。

目前写了两阶段更新参数，但动机只写了一句。建议强调 warm-up 是先校准 encoder/projector/class rows，joint phase 再用 LoRA 轻量调整 LLM-side computation，避免在极少支持样本上直接扰动语言侧适配器。

推荐补一句：

```tex
This staged adaptation avoids immediately fitting the LLM adapters before the newly introduced class-token rows and projection interface have been calibrated on the target label space.
```

## 需要特别回答的问题

### `Embedding-space alignment pretraining` 是否太简略？

是。它是论文贡献链条里的关键一环，因为 Introduction 和 Contributions 都强调 “transferring continuous tokens into a pretrained causal LLM through stage-wise pretraining”。但当前第 433--448 行只说明了 TSQA loss，没有说明：

- TSQA 是什么样的数据；
- Stage II 使用哪个 split；
- Stage I checkpoint 如何进入 Stage II；
- 训练哪些模块、冻结哪些模块；
- 为什么 answer-token LM loss 能实现 alignment；
- 是否使用 UCR TEST 或 labels。

建议扩展为两个段落，并显式写出 “Stage II / TSQA train split / frozen LLM / no UCR TEST or labels / default downstream checkpoint”。

### `Optional label-semantic alignment` 是否太简略？

也是。更准确地说，它不仅太简略，而且与 `Label semantic priors` 概念重叠。建议先确定你想表达的是下面哪一个：

1. 一个可选的 Stage III 预训练阶段，使用 TSQA/M4 captioning 或 synthetic attribute supervision；
2. 下游 external semantic-label datasets 上的 class-token row initialization/regularization；
3. 两者都有，但分别服务于不同目的。

如果是第 3 种，Method 必须明确区分：

- `Label semantic priors`：下游分类机制的一部分，定义在 Eq. `\eqref{eq:semantic-row-reg}`，用于 meaningful label names。
- `Optional semantic enrichment pretraining`：可选预训练阶段，使用 TSQA/M4 caption/attribute supervision，不用于默认 UCR 主结果。

目前的写法会让读者把两者混在一起。

### 是否需要说明用的数据集？

需要。Method 中至少应给出主文级别的数据来源，不必像 Appendix 那样列 license/source，但要写清：

- Stage I：TSQA-train raw series、M4-train raw series、fixed UCR TRAIN pool raw series；
- Stage II：TSQA train split；
- Stage III：optional TSQA/M4 captioning and synthetic attribute supervision；
- downstream UCR：只使用 sampled support labels；UCR TEST 只用于 evaluation；
- external semantic-prior evaluation：MIT-BIH、SleepEDF、CinC2017 可在 Experiments 里详细说，Method 里只需说 “datasets with meaningful label names”。

这类说明不只是实验设置，而是方法可复现性和审稿信任的一部分。

## 建议的 Method 结构微调

保持现有大结构，但可以把最后一节改得更明确：

```tex
\subsubsection{Curriculum Pretraining and Few-Shot Adaptation}

\paragraph{Stage I: representation pretraining.}
说明目标、数据源、loss。

\paragraph{Stage II: embedding-space alignment.}
说明 TSQA train split、样本形式、answer-token LM loss、冻结/更新策略、default checkpoint。

\paragraph{Optional Stage III: semantic enrichment.}
说明 TSQA/M4 caption/attribute 数据、不是 UCR 默认结果、与 downstream semantic priors 的区别。

\paragraph{Target few-shot adaptation.}
说明 warm-up、joint LoRA、class-token scoring inference。
```

这个结构比当前标题更能呼应 Appendix 的 Stage I/II/III 表，也能减少读者在 Method、Experiments、Appendix 之间来回寻找信息的负担。

## 一份可执行的改稿清单

1. 在 Method 中新增一句或一小段总结 Stage I/II/III 的数据来源和默认 checkpoint。
2. 在 `Representation pretraining` 中补充 Stage I 数据源，以及“不使用 UCR TEST/class labels”。
3. 将 `Embedding-space alignment pretraining` 扩展为两个自然段，写清 TSQA train split、样本形式、训练/冻结模块、default Stage-II checkpoint。
4. 将 `Optional label-semantic alignment` 改名为 `Optional semantic enrichment pretraining`，并说明数据源和不用于默认 UCR。
5. 明确区分 `Label semantic priors` 与 optional Stage III semantic enrichment。
6. 把 `P_\psi` 从 “linear projector” 改为 “trainable projection module”，与 Appendix 的 MLP projector 保持一致。
7. 统一 `$K$` / `$S$` shot 符号。
8. 将 class token 记号从 `\langle c_c\rangle` 改成 `\langle \mathrm{cls}_c\rangle` 或类似更清晰的形式。
9. 补充 class identifiers 是 newly added special tokens，因此每个 class 对应单个 vocabulary index。
10. 如篇幅允许，在 morphology map 处补一句解释它捕捉 repeated local shapes and cross-window transitions 的动机。

## 推荐优先改写片段

如果只优先改一处，我建议先改第 343--453 行，也就是 `Curriculum Pretraining and Few-Shot Adaptation` 的前三个 paragraph。这里是目前 Method 最容易被审稿人追问的地方，也是与你的实验设置、limitations、appendix 最需要保持一致的地方。

最小改动版本可以是：

```tex
TimeMorph uses curriculum pretraining so that the model first learns reusable time-series structure, then aligns continuous time-series tokens with the LLM embedding space, and finally adapts to each target few-shot label space. In the default benchmark, Stage I uses unlabeled raw series from TSQA-train, M4-train, and official TRAIN-split sequences from a fixed UCR pretraining pool; Stage II uses the TSQA training split for time-series question answering; and an optional Stage III uses caption or attribute-style supervision derived from TSQA/M4 when semantic enrichment is desired. The default UCR results use the Stage-II checkpoint and no pretraining stage uses UCR TEST sequences or UCR class labels.
```

然后分别在 Stage I、Stage II、Stage III 的 paragraph 中展开即可。

