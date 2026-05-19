# Appendix and Experiments Compliance Review

Review target: `latex_all/Formatting_Instructions_For_NeurIPS_2026/neurips_2026.tex` and its included appendix/checklist/table files.

Review scope requested by the author: appendix compliance, consistency of the Experiments section, NeurIPS-style writing, and logical risks. This report is diagnostic only. No LaTeX manuscript, table, bibliography, or checklist source was modified.

Official references checked during review:

- NeurIPS 2026 Main Track Handbook: https://neurips.cc/Conferences/2026/MainTrackHandbook
- NeurIPS Paper Checklist Guidelines: https://neurips.cc/public/guides/PaperChecklist

## Executive Summary

The paper already has a reasonably mature experimental narrative: it reports all 128 UCR datasets, includes full per-dataset appendix tables, discloses that UCR Stage I uses unlabeled TRAIN sequences from a 98-dataset pool, separates total parameters from adapted parameters, includes asset/license and compute sections, and writes the main UCR result paragraph in a relatively careful way.

The main risks are not small wording issues. They are version-consistency and claim-calibration problems that a NeurIPS reviewer would likely catch quickly:

1. The paper text repeatedly says the external evaluation uses three datasets, `MIT-BIH`, `SleepEDF`, and `CinC2017`, but the current external-result tables only contain two datasets, `MIT-BIH` and `CinC2017`.
2. The semantic-prior paragraph reports numbers and dataset counts that do not match the actual semantic-prior tables.
3. The compute table says GPT4TS coverage is a completed 60-dataset run, while the main UCR table and W/T/L table present GPT4TS as if it covers all 128 datasets.
4. The compute table says the TimeMorph main UCR benchmark has 2,507 completed downstream runs, while the stated protocol implies 128 datasets x 4 shots x 5 runs = 2,560 runs.
5. The abstract and some captions say TimeMorph "consistently outperforms strong task-specific baselines", but the actual evidence against COSCO-ResNet and ResNet is modest and sometimes close at the dataset level.
6. The UCR pretraining setup is disclosed, but it should be framed more explicitly as unlabeled in-archive / in-domain pretraining, not strict leave-dataset-out transfer.

The appendix order itself appears broadly consistent with the official NeurIPS guidance: paper, references, optional appendices, then checklist. The current PDF is 46 pages and 648 KB, so file size is not an issue. The checklist is present after the appendix.

## Must Fix Before Submission

### 1. External Dataset Count Mismatch

Locations:

- `neurips_2026.tex`, external-results paragraph around line 540.
- `neurips_2026.tex`, compute table around lines 740-741.
- `neurips_2026.tex`, external appendix intro around line 757.
- `tables/main/external_fewshot_avg.tex`.
- `tables/main/external_fewshot_main.tex`.
- `tables/ablations/m2_semantic_prior_ablation_accuracy.tex`.

Current issue:

The main text says:

> We also evaluate MIT-BIH, SleepEDF, and CinC2017 ...

It also says TimeMorph is behind ResNet and TapNet on SleepEDF. However, the current main and appendix external tables contain only `MIT-BIH` and `CinC2017`, and their captions explicitly say "two external datasets".

Why this is serious:

This looks like selective reporting or a stale manuscript/table mismatch. If SleepEDF was evaluated and TimeMorph underperforms there, omitting it from the table weakens trust. If SleepEDF was not part of the final table, then the text is wrong.

Recommended resolution:

Choose one of these paths:

- If SleepEDF results are valid and part of the claimed external evaluation, add SleepEDF to `external_fewshot_avg.tex`, `external_fewshot_main.tex`, `semantic_prior_summary.tex`, and `m2_semantic_prior_ablation_accuracy.tex`; then recompute all averages and revise the text.
- If the current table version is final and only covers MIT-BIH and CinC2017, remove SleepEDF from the text and compute table, and change all "three/all three" claims to "two/both".

Suggested safer replacement if only two datasets remain:

```latex
We also evaluate MIT-BIH and CinC2017 under full-label-space 10-, 20-, and 30-shot protocols, using 10 runs for each setting. Table~\ref{tab:external-fewshot-avg} reports the average over these shot settings, with the full table in Appendix~\ref{app:external-results}. TimeMorph obtains the best average accuracy on both datasets. We therefore treat these experiments as evidence of transfer beyond UCR, not as a claim of uniform dominance across all time-series domains.
```

### 2. Semantic-Prior Numbers Do Not Match Tables

Locations:

- `neurips_2026.tex`, semantic-prior paragraph around line 567.
- `neurips_2026.tex`, limitations around line 583.
- `neurips_2026.tex`, semantic-prior appendix around line 764.
- `tables/main/semantic_prior_summary.tex`.
- `tables/ablations/m2_semantic_prior_ablation_accuracy.tex`.

Current issue:

The text says semantic priors improve the average on `MIT-BIH, SleepEDF, and CinC2017` from `47.61` to `50.43`. The actual summary table reports only two datasets and gives `47.84` to `50.23`.

Why this is serious:

This is a direct numerical inconsistency between text and table. Reviewers often check exactly these deltas.

Recommended resolution:

If two-dataset tables are final, revise the text to use `47.84 -> 50.23` and say "two external semantic-label datasets". If SleepEDF is added, recompute the values.

Suggested safer replacement if only two datasets remain:

```latex
For external datasets with meaningful label names, Table~\ref{tab:semantic-prior-summary} compares anonymous class-token rows with rows initialized and regularized from label-verbalizer prototypes. Semantic priors improve the 10/20/30-shot average on MIT-BIH and CinC2017, increasing the overall average from 47.84 to 50.23. The full appendix table shows that the benefit is not uniform at every shot, including a decrease on CinC2017 20-shot, but the averaged result supports the semantic-prior class-token space as a useful extension of supervised closed-set decoding.
```

### 3. GPT4TS Coverage Conflict

Locations:

- `neurips_2026.tex`, compute table around line 739.
- `tables/main/ucr_fewshot_main.tex`.
- `tables/main/ucr_fewshot_wtl.tex`.
- UCR appendix tables.

Current issue:

The compute table says:

> GPT4TS coverage is reported for the completed 60-dataset run.

But the main UCR table presents GPT4TS inside the 128-dataset benchmark, and the W/T/L table reports 128 dataset comparisons per shot against GPT4TS.

Why this is serious:

This can invalidate the main comparison if GPT4TS actually covers only a subset. It also creates doubt about whether the macro-averages are over a common dataset set.

Recommended resolution:

- If GPT4TS results are in fact complete over all 128 UCR datasets, remove the "60-dataset run" note from the compute table.
- If GPT4TS covers only 60 datasets, the main table and W/T/L table must explicitly say they use a shared 60-dataset subset for GPT4TS, or GPT4TS should be removed from the 128-dataset aggregate table.

Suggested wording if GPT4TS is complete:

```latex
UCR baseline runs & ResNet, TapNet, COSCO-ResNet, PatchTST, GPT4TS, and TSLib-family baselines under the same few-shot protocol & 6 single-GPU workers & 0.40--0.49 GPU-h/run & 10,937 & Includes the retained main UCR baseline logs. \\
```

### 4. TimeMorph Run Count Conflict

Locations:

- `neurips_2026.tex`, protocol around line 512.
- `neurips_2026.tex`, compute table around line 738.

Current issue:

The stated protocol implies:

```text
128 datasets x 4 shot settings x 5 runs = 2,560 downstream runs
```

The compute table says `2,507 completed downstream runs`.

Why this is serious:

This implies 53 missing runs. If the table averages are based on five runs for every dataset-shot, the compute table is stale or wrong. If there were missing runs, the paper must not claim every cell uses five independent runs without explaining how missing runs were handled.

Recommended resolution:

- If results are complete, correct the compute table to 2,560 runs.
- If 2,507 is correct, state which runs are missing and whether the affected entries were recomputed, excluded, or averaged over fewer seeds.

### 5. Abstract Claim Is Stronger Than Evidence

Locations:

- `neurips_2026.tex`, abstract around line 31.
- `neurips_2026.tex`, main-result paragraph around line 530.
- `neurips_2026.tex`, limitations around lines 578-579.
- `tables/main/ucr_fewshot_wtl.tex`.

Current issue:

The abstract says TimeMorph "consistently outperforms strong task-specific, Transformer-style, and LLM-style baselines." The main result section is more careful and says the gains over COSCO-ResNet are modest. The W/T/L table also shows close comparisons against strong CNN/prototype baselines, e.g. COSCO-ResNet is better on more datasets at 5-shot.

Why this is serious:

Reviewers may interpret the abstract as overstating the paper. The evidence supports "best average accuracy and rank" more strongly than "consistent superiority over all strong task-specific baselines".

Recommended replacement:

```latex
Extensive few-shot experiments on the UCR benchmark and additional external datasets show that TimeMorph achieves the best average accuracy and average rank on UCR, with clear gains over Transformer-style and LLM-style baselines and competitive or modestly better performance against strong CNN/prototype baselines. Further analyses support the contributions of dual-view modeling, stage-wise pretraining, and class-token scoring.
```

## Appendix Compliance and NeurIPS-Style Issues

### 6. Appendix Order Is Mostly Correct

The official checklist guidance says the single PDF should include the paper, optional technical appendices, and then the NeurIPS checklist. The official 2026 handbook similarly lists submitted paper content, references, appendices, and the checklist. Your current order is:

```text
main paper -> references -> appendix -> checklist
```

This is appropriate. The checklist is not missing. The current PDF size is also far below the official 50 MB limit.

Remaining caution:

The main content page count still needs a final manual check against the official style output. The compiled PDF has 46 pages total, but references, appendices, and checklist do not count as content pages. The current main text appears to end before references and appendices, but final submission should verify the main content does not exceed the current official page limit.

### 7. Checklist "Open Access to Data and Code = Yes" Needs Artifact Verification

Locations:

- `checklist.tex`, open access to data and code.
- `neurips_2026.tex`, `Code Release and Assets`.

Current issue:

The appendix says "We will release an anonymized code package with the submission." The checklist says the anonymized supplementary code package is submitted with the paper. These are not the same commitment.

Why this matters:

The official handbook says supplementary code/data should be anonymized, and linked or submitted material must preserve double-blind review. If no anonymous zip/link is actually submitted, the checklist answer can look inaccurate.

Recommended resolution:

- If an anonymized artifact is submitted with the paper, change "will release" to "we include/submit".
- If the code is only planned for later release, answer the checklist more conservatively and explain the plan.

Suggested artifact-present wording:

```latex
We include an anonymized code package with the submission. The package contains the TimeMorph implementation, few-shot adaptation and evaluation scripts, baseline launchers, reporting scripts, environment files, and README instructions for preparing the public datasets used in our experiments.
```

### 8. Asset/License Table Is Useful But Needs a Few Safer Phrases

Locations:

- `neurips_2026.tex`, asset table around lines 700-709.

Good aspects:

- Major datasets and pretrained models are listed.
- Redistribution status is clear.
- Gated Llama weights are not redistributed.
- PhysioNet assets are correctly treated as external downloads.

Risks:

- `OpenTSLM base code` with "Project repository and release package" is vague and might either fail to credit an external source or reveal an identifiable project source.
- If OpenTSLM is a third-party project, cite the paper or official anonymization-safe source clearly.
- If OpenTSLM is the authors' prior code, avoid identifiable repository names during double-blind review unless it is already a non-anonymous public preprint that the submission can cite safely under NeurIPS policy.

Suggested safer wording depends on the true provenance:

```latex
OpenTSLM-style base implementation & Starting codebase and time-series language-model components & Anonymized implementation derived from the cited OpenTSLM framework & MIT-compatible terms where applicable & Modified code is released in anonymized form, preserving applicable notices. \\
```

Use only if factually correct.

### 9. Compute Reporting Is Present But Some Granularity Is Weak

Locations:

- `neurips_2026.tex`, compute budget table.
- `tables/main/efficiency_table.tex`.
- `tables/appendix/efficiency_runtime_details.tex`.

Good aspects:

- Hardware type, GPU memory, CPU, RAM, total GPU-hours, and runtime/memory probes are reported.
- Total parameters and updated parameters are separated.

Risks:

- Some rows use `--` for mean per run.
- Pretraining rows do not give wall-clock time.
- Baseline compute is grouped broadly and has the GPT4TS coverage conflict noted above.
- The efficiency probe row has no actual total or run count.

Recommended additions:

- For Stage I-II and Stage III, give wall-clock time or per-stage GPU-hours.
- For baseline groups, either list per-family run counts or point to an appendix/reporting script.
- For efficiency probe, state it is a measurement pass rather than a training experiment, and list the representative subset size.

### 10. Appendix Structure Is Clear Enough But Could Be More Reader-Friendly

Current structure:

- Full UCR Benchmark Results
- Additional Experimental Details
  - Pretraining Stages and Data Sources
  - Code Release and Assets
  - Asset and License Table
  - Computational Resource Reporting
  - External Dataset Results
  - Semantic-Prior Decision Ablation
  - Efficiency Summary
  - Baseline Training Recipes and Hyperparameters
  - Computational Efficiency Probe Details
- Additional Plot
- Ablation Tables
- Checklist

Issue:

The `Additional Experimental Details` section becomes a large mixed container for implementation, reproducibility, licensing, compute, external results, and baselines.

Recommended appendix organization:

```latex
\section{Full UCR Benchmark Results}
\section{Implementation and Reproducibility Details}
\section{Pretraining Data and Stage-wise Training}
\section{Baseline Recipes and Hyperparameters}
\section{Additional Results and Ablations}
\section{Compute, Assets, and Licenses}
\section{Additional Figures}
\input{checklist.tex}
```

This is not mandatory, but it would make the appendix feel more NeurIPS-polished.

## Experiments Consistency Review

### 11. Main UCR Results Are Mostly Internally Consistent

The following pieces are aligned:

- Main table values match the prose values for TimeMorph and COSCO-ResNet.
- The ablation summary values match the prose values for no pretraining, single-view variants, class-token scoring, and no-LLM backbone.
- W/T/L rows sum to 128 for each selected baseline and shot.
- UCR appendix tables report 128 datasets plus summary rows.

The main UCR experimental paragraph around line 530 is one of the strongest parts of the paper because it is honest about the small margin over COSCO-ResNet and emphasizes the clearer gains over Transformer-style and LLM-style baselines.

Recommended action:

Keep this cautious tone and make the abstract, introduction contribution sentence, and figure caption match it.

### 12. Statistical Support Is Adequate for Reporting But Weak for Strong Claims

Locations:

- Main result table.
- Appendix UCR per-dataset tables.
- Checklist statistical significance item.

Current state:

The appendix reports mean plus standard deviation over five runs. This is acceptable as error-bar-style reporting. However, the main claims against COSCO-ResNet and ResNet depend on small average margins, and no paired significance test or bootstrap confidence interval is reported.

Reviewer risk:

For comparisons like TimeMorph vs COSCO-ResNet, average differences under 1 point at 1-, 2-, and 5-shot may not be statistically convincing. Reviewers may accept "best average" but reject "significantly better" or "consistently better".

Recommended addition:

Add paired bootstrap confidence intervals or a Wilcoxon signed-rank test over dataset-level paired accuracies, at least for:

- TimeMorph vs COSCO-ResNet.
- TimeMorph vs ResNet.
- TimeMorph vs PatchTST.
- TimeMorph vs GPT4TS.

Safe checklist wording if no hypothesis test is added:

```latex
The main protocol averages over five independent support-set draws. Appendix tables report the sample standard deviation over these runs; we do not assume normality or perform a formal hypothesis test.
```

### 13. UCR Pretraining Should Be Framed as In-Archive Unlabeled Pretraining

Locations:

- `neurips_2026.tex`, pretraining data paragraph around line 515.
- `neurips_2026.tex`, limitations around line 582.
- `neurips_2026.tex`, pretraining-stage appendix around lines 663-671.

Current state:

The manuscript already states that no UCR TEST split or UCR class labels are used during pretraining, and it acknowledges that unlabeled TRAIN sequences from some UCR targets may have been seen.

Remaining risk:

This is a sensitive few-shot evaluation point. Reviewers may call it semi-supervised or transductive if the target dataset's unlabeled TRAIN split appears in Stage I.

Recommended action:

Add a concise framing sentence in the main experimental setup:

```latex
Thus, the main UCR benchmark evaluates few-shot supervised adaptation after unlabeled in-archive pretraining, rather than strict leave-dataset-out pretraining for every target dataset.
```

Recommended appendix addition:

- List the 98 UCR Stage I pool members.
- Optionally report separate average results for UCR targets inside vs outside the Stage I pool.

### 14. Stage I Monitoring Uses "Validation/Test" Language

Location:

- `neurips_2026.tex`, pretraining stage table around line 663.

Current issue:

The table says Stage I monitoring uses raw TSQA validation/test and M4 validation/test series. Even if this does not touch UCR TEST, "test series" for monitoring can raise questions.

Recommended resolution:

If test splits were only used for diagnostic reporting and not checkpoint selection, state that explicitly. If they were used for model selection, consider changing the pipeline or the wording to validation-only.

Suggested wording:

```latex
TSQA and M4 validation splits for pretraining diagnostics; held-out test splits are used only for post-hoc diagnostics and never for downstream model selection.
```

Use only if true.

### 15. No-LLM Ablation Is Useful But Does Not Isolate LLM Knowledge

Locations:

- `neurips_2026.tex`, LLM-backbone ablation paragraph around line 572.
- `tables/main/m2_llm_backbone_ablation_summary.tex`.

Current state:

The paper correctly says the ablation does not isolate pretrained knowledge, capacity, or token-scoring interface. This is good.

Recommended action:

Keep this caveat. Avoid adding stronger claims elsewhere that the gain is due to "LLM knowledge" alone.

Suggested safe phrasing:

```latex
The result shows that the proposed language pathway improves over a lightweight classifier on the same dual-view features under our adaptation schedule, but it does not by itself isolate whether the gain comes from pretrained computation, model capacity, or the token-scoring interface.
```

## Writing and Claim Calibration

### 16. Figure Caption Should Avoid "Strongest" If W/T/L Is Mixed

Location:

- `neurips_2026.tex`, few-shot trend caption around line 818.

Current issue:

The caption says TimeMorph "remains the strongest method". Because W/T/L against COSCO-ResNet is mixed, "strongest" is slightly broad.

Suggested replacement:

```latex
\caption{Shot-scaling trend on the 128 UCR datasets. Each point reports the mean accuracy over five runs. TimeMorph has the highest macro-averaged accuracy across all four shot settings.}
```

### 17. Contribution Statement Is Mostly Fine But Should Mirror Evidence

Location:

- `neurips_2026.tex`, contributions around line 46.

Current issue:

The contribution paragraph says TimeMorph achieves the best average accuracy and rank. This is supported by the main table. It also says external experiments validate the roles of the components. That is a little broad because external semantic-prior experiments are only on two current table datasets.

Suggested safer ending:

```latex
Across 128 UCR datasets, TimeMorph achieves the best average accuracy and average rank under 1-, 2-, 5-, and 10-shot settings. Additional external experiments and ablations further support the roles of dual-view modeling, stage-wise pretraining, and closed-set class-token scoring, while showing that gains over specialized CNN/prototype baselines are modest in the lowest-shot regimes.
```

### 18. Limitations Are Honest But Could Be More Specific

Location:

- `neurips_2026.tex`, limitations section.

Good aspects:

- It acknowledges modest gains over strong CNN/prototype baselines.
- It acknowledges LLM inference cost.
- It acknowledges UCR unlabeled TRAIN exposure.
- It avoids clinical deployment claims.

Recommended additions:

- If SleepEDF remains, explicitly state that external transfer is not uniformly positive and that SleepEDF is a failure/underperformance domain.
- State that aggregate ablations do not fully isolate causal mechanisms.
- State that results are on univariate UCR plus selected external datasets; claims about multivariate or domain-shifted settings need more evidence if applicable.

## Lower-Priority Cleanup

### 19. Table Labels Contain Draft-Like Names

Examples:

- `tab:ucr_fewshot_paper_current-fewshot-main`
- `tab:ucr_fewshot_paper_current-paper-wtl`
- `tab:ucr_fewshot_paper_current-paper-shot-1`

This does not affect compilation, but final papers usually use cleaner labels:

- `tab:ucr-main`
- `tab:ucr-wtl`
- `tab:ucr-shot-1`

This can wait until after content-level fixes.

### 20. Some Captions Sound Like Generated/Script Output

Location:

- `tables/appendix/efficiency_runtime_details.tex`.

Current caption starts with "Printed per-dataset runtime details...". This is a little engineering-like.

Suggested replacement:

```latex
\caption{Per-dataset runtime details for ACSF1 from the representative UCR runtime subset. Aggregate runtime and memory statistics over the representative subset are reported in Tables~\ref{tab:efficiency-compact} and \ref{tab:computational-efficiency}.}
```

### 21. Appendix Could Add a Failure/Diagnostic Summary

The full UCR per-dataset tables are complete, but they are hard to read. A short diagnostic subsection would strengthen the appendix:

- Average delta by class count.
- Average delta by series length.
- Average delta by training-set size.
- Top datasets where TimeMorph loses to COSCO-ResNet/ResNet.
- Top datasets where TimeMorph gains most.

This would help reviewers understand when the method works, not just whether the average is best.

## Suggested Fix Priority

1. Decide whether external experiments are two datasets or three; make text, tables, semantic-prior results, limitations, and compute table consistent.
2. Resolve GPT4TS coverage and TimeMorph run count contradictions.
3. Weaken abstract/caption claims to match the cautious main-result paragraph.
4. Clarify UCR Stage I as unlabeled in-archive pretraining and list the 98-dataset pool in the appendix.
5. Verify the anonymized code artifact before answering checklist "Open access to data and code" as `Yes`.
6. Add statistical confidence/significance for comparisons against COSCO-ResNet and ResNet, or avoid any wording that implies statistical superiority.
7. Improve appendix structure and polish captions/labels.

## Short Reviewer-Style Verdict

If I were reviewing this version, I would view the method and UCR result package as potentially solid, but I would raise a serious concern about experimental reporting consistency because the external dataset claims do not match the tables. I would also ask whether the UCR setup is semi-supervised due to unlabeled TRAIN exposure and whether the small gains over COSCO-ResNet are statistically meaningful.

The paper can become much more defensible with relatively contained changes: unify the external-result story, resolve run-count/coverage notes, and align the abstract with the measured evidence. The current main UCR result paragraph already has the right tone; the rest of the paper should be pulled toward that level of precision.
