# NeurIPS 2026 Checklist Compliance Review

检查对象：`neurips_2026.tex` 与 `checklist.tex`  
检查日期：2026-05-05  
官方依据：NeurIPS Paper Checklist Guidelines，https://nips.cc/public/guides/PaperChecklist  

## 总体结论

当前草稿已经保留 NeurIPS checklist，并且位置基本正确：正文与参考文献之后进入 appendix，appendix 结束后再输入 `checklist.tex`。这符合官方说明中“paper -> optional technical appendices -> checklist”的顺序要求。`checklist.tex` 中 16 个问题均有 yes/no/n/a 回答和简短 justification，没有 TODO。

我重新编译了论文：

- 编译命令：`latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=out neurips_2026.tex`
- 输出 PDF：`out/neurips_2026.pdf`
- PDF 状态：41 页，US Letter，PDF 1.5
- 编译状态：无 undefined citation/reference；仅有若干 underfull box 和 PDF inclusion version warning，不影响 checklist 合规性
- 页数观察：正文主体到第 8 页，References 从第 9 页开始，Checklist 在 appendix 后出现。按当前模板规则，正文页数看起来没有超出主文页限制；正式提交前仍应核对 NeurIPS 2026 最新 CFP/author kit 中的页数定义。

总体判断：**形式上基本满足 NeurIPS checklist 要求，但有 4 项建议补强后再提交**：Compute resources、Broader impacts、Licenses、New assets。另外，Open access to data and code 当前回答为 No 是可接受的，但若能匿名释放代码或补充复现实验命令，会明显提升可审查性。

## 逐项检查

### 1. Claims

当前回答：Yes  
判断：满足。

正文中的 claims 与实验范围总体一致。Abstract 和 Introduction 明确将主张限定在 few-shot UCR classification、dual-view modeling、curriculum transfer、class-token scoring；Experiments 中也谨慎说明与强 CNN/prototype baseline 的差距是 modest but consistent，并把 external datasets 作为 transfer check，而不是统一优势声明。Limitations 也呼应了这些边界。

建议：保持当前回答。若后续加强摘要，继续避免使用 “dominates all baselines/domains” 这类超出证据的表述。

### 2. Limitations

当前回答：Yes  
判断：满足。

论文已有独立 `Limitations` section，覆盖了与强 baseline 的差距有限、LLM 推理成本较高、UCR pretraining 不是严格 leave-dataset-out、semantic prior 效果有数据集依赖、临床/安全关键部署需要额外验证等问题。这些内容对 NeurIPS checklist 是有力支撑。

建议：保持当前回答。

### 3. Theory Assumptions and Proofs

当前回答：N/A  
判断：满足。

论文是经验研究，没有定理、理论保证或 formal proof。当前 N/A 合理。

建议：保持当前回答。

### 4. Experimental Result Reproducibility

当前回答：Yes  
判断：基本满足，但建议补一句更强的复现说明。

论文已经说明了数据划分、support sampling、pretraining data usage、baseline families、metrics、模型配置、主要超参数、完整 per-dataset 结果和 baseline recipes。对于“论文中是否充分披露可复现实验所需信息”这一项，Yes 可以成立。

风险点：当前文本没有集中列出 exact commands、random seed 列表、环境版本、checkpoint 获取方式。虽然第 5 项已经回答 No，但第 4 项的 “fully disclose” 容易被审稿人按更高标准理解。

建议：在 appendix 增加一个短小 subsection，例如 `Reproducibility Checklist for Experiments`，列出：

- exact scripts / commands for main method and baselines
- Python/PyTorch/CUDA 版本或环境文件位置
- random seeds / support-set index generation rule
- checkpoint 使用方式：Stage II checkpoint 是否随 supplement 匿名提供，或如何重训得到
- 哪些实验可以完全复现，哪些只提供结果表

### 5. Open Access to Data and Code

当前回答：No  
判断：回答可接受，但从投稿说服力看建议尽量补强。

官方 checklist 明确允许没有开放代码时回答 No，只要给出合理说明；NeurIPS 通常不会仅因没有代码而拒稿，除非论文贡献本身是开放 benchmark/dataset 等。当前 justification 说明“匿名 code/data release instructions 尚未作为完整复现包提供”，是诚实且合规的。

建议：如果时间允许，最好改成 Yes 或至少更强的 No：

- 准备 anonymized supplement zip 或匿名仓库
- 包含 `requirements.txt` / `pyproject.toml`、数据下载说明、主实验命令、baseline 命令
- 若部分实验暂不能开放，明确说明哪些实验不可复现以及原因

### 6. Experimental Setting / Details

当前回答：Yes  
判断：满足。

论文主体和 appendix 已包含 protocol、pretraining data、baselines/metrics、implementation details、ChronoMorph hyperparameters、baseline training recipes、optimizer、epoch、batch size、LR、checkpoint selection rule 等关键信息。

建议：保持当前回答。若做第 4 项复现补充，可与本项形成互相支撑。

### 7. Experiment Statistical Significance

当前回答：Yes  
判断：基本满足，但建议强化标准差说明。

论文说明 main protocol 使用 five independent runs，appendix full tables 报告 mean ± standard deviation。这个足以支持 Yes。

风险点：主表多为平均值，不一定直接显示 error bars；appendix 有标准差，但 checklist guideline 还希望说明 variability factor 和计算方式。当前 justification 写得较简略。

建议：在实验设置或 appendix 增加一句：

> For all UCR few-shot results, we report the mean accuracy over five independently sampled support sets and use the sample standard deviation across these five runs as the reported uncertainty.

如果没有做 significance test，不要声称 statistical significance，只说 standard deviation / uncertainty。

### 8. Experiments Compute Resources

当前回答：Yes  
判断：需补强；当前 Yes 略偏乐观。

论文已经报告了 runtime subset、single RTX 3090、batch size 8、Peak GB、train ms/step、infer ms/example、throughput 和 parameter counts。这对 efficiency probe 是充分的。

主要缺口：官方问题问的是 “for each experiment” 是否提供复现所需 compute resources，并要求说明 individual experimental runs、估计 total compute、是否有未报告的 preliminary/failed experiments 产生额外 compute。当前正文主要覆盖 efficiency measurement，不足以完全覆盖 Stage I/II pretraining、UCR full benchmark、external datasets、ablation studies 和 baselines 的总计算资源。

建议二选一：

1. 保持 Answer: Yes，但必须在 appendix 增加 compute budget 表，至少包括：
   - Stage I/II/III pretraining：GPU 型号、GPU 数量、显存、训练时长或 GPU-hours
   - UCR main benchmark：每个 dataset/shot/run 的大致耗时或总 GPU-hours
   - external experiments 与 ablations 的估计总 compute
   - 是否包含 exploratory/failed runs

2. 暂时改成 Answer: No，并解释：
   - The paper reports runtime/memory for the efficiency probe, but a complete project-level compute budget for all experiments is not yet included.

更推荐方案 1，因为这项通常很容易补表，补完后 Yes 更稳。

### 9. Code of Ethics

当前回答：Yes  
判断：形式上满足，但需要作者本人确认。

论文使用公开时间序列数据和公开/可访问 pretrained backbones，未涉及欺骗性部署、新的人体实验或高风险内容生成。当前 justification 合理。

注意：这一题本质上包含“作者是否已经阅读并确认遵守 NeurIPS Code of Ethics”。我无法替作者确认阅读行为，最终提交前作者需要实际核对官方 Code of Ethics。

建议：保持当前回答，但提交前作者应自行确认。

### 10. Broader Impacts

当前回答：Yes  
判断：部分满足，建议补一个更明确的 broader impacts 段落。

当前 Limitations 中已经提到潜在正向应用，如 physiological monitoring / industrial inspection，也提醒 clinical or safety-critical deployment 需要 validation、privacy review、failure analysis。这是好的开始。

主要缺口：官方问题更关注 potential negative societal impacts，尤其是 privacy、fairness、错误预测导致的风险、misuse。当前文本对这些风险的展开还偏短，回答 Yes 略显单薄。

建议：在 Limitations 末尾或新增 `Broader Impacts` 小段，加入 3-5 句更直接的负面影响讨论，例如：

> Because ChronoMorph can be applied to physiological and sensor time series, incorrect predictions could affect downstream decision support if the model were deployed without domain validation. The use of patient- or worker-generated time series may also raise privacy and fairness concerns, especially when datasets underrepresent specific populations, devices, or acquisition conditions. We therefore view the present work as a benchmark study rather than a deployment-ready system, and any real-world use should require dataset-specific auditing, privacy review, calibrated uncertainty reporting, and human oversight.

补完后 Yes 会更稳。

### 11. Safeguards

当前回答：N/A  
判断：基本满足。

如果论文不释放新的 high-risk pretrained generative model、scraped dataset 或可被明显滥用的模型资产，N/A 是合理的。ChronoMorph 使用 LLM 作为分类接口，但研究任务不是通用内容生成或开放式高风险生成。

注意：如果最终要公开 ChronoMorph checkpoint，尤其是包含 Llama 相关权重或可生成文本的模型，需要重新考虑是否应描述 release safeguards、usage restrictions、base model license restrictions 等。

建议：若不发布模型权重，保持 N/A。若发布 checkpoint，改为 Yes 并补充 release policy。

### 12. Licenses for Existing Assets

当前回答：No  
判断：当前诚实但未满足，建议优先补。

论文使用了多个 existing assets：UCR Archive、TSQA、M4、MIT-BIH、SleepEDF、CinC2017、`meta-llama/Llama-3.2-1B`、`facebook/dinov2-base`、以及多个 baseline implementations / libraries。当前草稿只在方法和实验中引用了部分资产，没有完整列出版本、URL、license 和 terms of use。

建议：在 appendix 增加 `Existing Assets, Licenses, and Terms of Use` 表。至少包含：

- Asset name
- Role in this paper
- Version / source URL
- Citation
- License / terms of use
- Whether redistributed

补齐后可以把 checklist 改成 Yes。若某些数据集 license 无法确认，应保留 No 并在 justification 中说明哪些资产 license unavailable，以及已经如何处理。

### 13. New Assets

当前回答：No  
判断：建议改写；当前回答可能不够精确。

官方这一项问的是“如果释放新资产，是否提供文档”。如果当前 submission 不释放新 dataset/code/model/checkpoint，那么更合适的回答通常是 N/A，而不是 No。当前 justification 写“method and experimental scripts constitute new research artifacts, but complete public documentation and release packaging are not yet included”，这容易让审稿人理解为你计划释放但文档不完整。

建议二选一：

1. 如果不释放新资产：改成 Answer: N/A，并写：
   > The submission does not release new datasets, models, or code assets; it reports a method and experimental results only.

2. 如果释放匿名代码/supplement：改成 Answer: Yes，并提供 README、license、training instructions、limitations、asset card/model card，且保持匿名。

### 14. Crowdsourcing and Research with Human Subjects

当前回答：N/A  
判断：基本满足。

论文没有新的 crowdsourcing 或直接 human-subject study。使用 public physiological datasets 通常不等价于作者开展人体实验，但要注意与第 12 项 license/terms 和第 15 项 IRB 区分清楚。

建议：保持 N/A，但可以把 justification 改得更精确：

> The work uses existing public datasets and does not involve new crowdsourcing or direct interaction with human participants.

### 15. IRB Approvals

当前回答：N/A  
判断：基本满足。

如果作者没有进行新的人体受试者数据收集、标注或实验，N/A 合理。由于 MIT-BIH、SleepEDF、CinC2017 等是公开生理数据，建议说明是 secondary use of public/de-identified datasets，而不是笼统说“不涉及 human-subject research”。

建议：可将 justification 改为：

> The work uses existing public/de-identified datasets and does not collect new human-subject data or interact with participants; therefore no new IRB approval is required for the reported study.

注意不要写具体机构信息，以免破坏匿名。

### 16. Declaration of LLM Usage

当前回答：Yes  
判断：满足。

LLM 是方法核心组成部分，正文多处说明 causal LLM、Llama-3.2-1B、token-scoring interface、LoRA adaptation 等内容。当前 Yes 和 justification 合理。

建议：保持当前回答。

## 建议优先修改清单

优先级 1：补 `Existing Assets, Licenses, and Terms of Use` appendix 表，并将第 12 项从 No 改为 Yes，或明确列出无法确认 license 的资产。

优先级 2：补 compute budget 表，覆盖 pretraining、main UCR benchmark、external experiments、ablation studies、baseline runs 和 exploratory compute；之后第 8 项 Yes 会更稳。

优先级 3：扩展 broader impacts/limitations，明确 privacy、fairness、错误预测和安全关键场景风险；之后第 10 项 Yes 更有支撑。

优先级 4：根据是否释放匿名代码/模型，重写第 13 项：

- 不释放新资产：Answer 改为 N/A
- 释放代码/模型：Answer 改为 Yes，并提供匿名 release documentation

优先级 5：为第 4/5/7 项补充复现实验命令、环境、seed 和 standard deviation 计算方式。即使第 5 项仍回答 No，这些信息也会显著降低 reviewer 对可复现性的疑虑。

## 当前可提交风险评估

- Desk rejection 风险：低。Checklist 存在、顺序正确、无 TODO，编译成功。
- Reviewer 质疑风险：中等。主要集中在 licenses、compute budget、broader impacts 和 code release。
- 最值得立刻修的点：第 12 项 licenses。它最容易被 checklist reviewer 或 ethics reviewer 抓住，而且补表成本相对低。

