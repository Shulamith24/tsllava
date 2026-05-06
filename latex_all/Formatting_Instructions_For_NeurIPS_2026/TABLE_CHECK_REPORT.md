# Table Consistency Check Report

本次检查只使用了 `latex_all/Formatting_Instructions_For_NeurIPS_2026 (1)` 目录内已有的 TeX/CSV 文件，没有重新读取本地 `results` 目录。

## 主要结论

- UCR 主实验仍保持 5-run 协议：正文、主表和四个 UCR 附录表中的 `five runs` / `5 runs` 描述未改动。
- MIT-BIH、SleepEDF、CinC2017 三个外部数据集统一为 10-run 协议：外部主表、外部均值表、semantic-prior 表和对应 CSV 的 `num_runs` 已同步。
- Macro-F1 没有从 accuracy 或 accuracy std 反推；当前 TeX 表仍只展示 accuracy。
- `未处理` 中的 TeX 草稿已合并进 `tables` 目录，并从 `未处理` 中删除，避免后续误用重复表。

## 修改内容

- `未处理/external_fewshot_comparison_values.csv`
  - 将 M2 在 MIT-BIH、SleepEDF、CinC2017 的 `num_runs` 从旧的 2/3 统一为 10。
  - 保留已有 accuracy mean/std 和 display percent 数值，未重新读取外部结果文件。

- `未处理/m2_semantic_prior_ablation_values.csv`
  - 将 Anonymous labels 和 Semantic prior 的全部外部数据集行 `num_runs` 统一为 10。
  - 根据 `accuracy_mean * 100` 重算 `accuracy_percent`。
  - 根据 `accuracy_std * 100` 重算 `accuracy_std_percent`。
  - 未修改 `macro_f1_*` 列，因为 Macro-F1 不能由 accuracy/std 严谨反推。

- `tables/main/external_fewshot_main.tex`、`tables/main/external_fewshot_avg.tex`
  - caption 改为明确三外部数据集为 10-run setting。
  - 合并 `未处理/external_fewshot_comparison_full.tex` 和 `未处理/external_fewshot_comparison_avg.tex` 的 accuracy 数值；正式表保留 `ChronoMorph` 命名和正式 label。

- `tables/main/semantic_prior_summary.tex`、`tables/ablations/m2_semantic_prior_ablation_accuracy.tex`
  - caption 改为明确 semantic-prior 外部实验为 10-run setting。
  - 修正 Anonymous labels 在 CinC2017 30-shot 的 std：`3.73` -> `9.73`，与外部主表和 CSV 保持一致。
  - 合并 `未处理/m2_semantic_prior_ablation_accuracy.tex` 的 accuracy 数值；正式表不包含 Macro-F1 字段。

- 已删除的未处理 TeX 草稿：`未处理/external_fewshot_comparison_full.tex`、`未处理/external_fewshot_comparison_avg.tex`、`未处理/m2_semantic_prior_ablation_accuracy.tex`
  - 同步 10-run caption。
  - 修正未处理 semantic-prior 表中 CinC2017 30-shot std：`3.73` -> `9.73`。
  - 重新核对正式 semantic-prior 表中 CinC2017 三个 shot-level delta：10-shot 为 `+1.92`，20-shot 为 `-1.74`，30-shot 为 `+4.73`。
  - 这些 TeX 草稿的内容已合并到 `tables` 后删除。

- `neurips_2026.tex`、`checklist.tex`
  - 删除“外部实验 M2 reruns 少于 baseline”“semantic-prior unfinished”的旧叙述。
  - 改为说明三外部数据集采用 10 runs，并作为 UCR 之外的 transfer check。

- `未处理/README.md`、`未处理/m2_semantic_prior_ablation_README.md`
  - 更新说明，标明当前一致性检查只基于论文目录内 CSV/TeX。
  - 删除/标注过时的 M2 fewer-runs 说明。

## 校验

- 已检查两个外部 CSV 中所有 `accuracy_percent == accuracy_mean * 100`、`std_percent/accuracy_std_percent == accuracy_std * 100`。
- 已检查外部 M2/semantic-prior CSV 行的 `num_runs` 均为 10。
- 已检查 UCR 相关表格仍保留 5-run 描述。
- 已检查 `tables` 目录中的 TeX 表没有 Macro-F1 字段；`ACSF1` 数据集名中的 `F1` 不是指标字段。
- 已按 `neurips_2026.tex` 中的 `\input{tables/...}` 清点：`tables/` 下 19 个 TeX 文件全部被论文实际引用，且没有缺失或未引用的 TeX 表。
- 已确认 `未处理/` 下不再保留 TeX 草稿，只保留 CSV 和说明文件。
- 已运行 `latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=out neurips_2026.tex`，当前 PDF 已是最新状态。
