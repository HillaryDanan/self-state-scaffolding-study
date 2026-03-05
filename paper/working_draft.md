# External Scaffolding Does Not Produce Genuine Self-State in Large Language Models

**Hillary Danan, PhD** — Cognitive Neuroscience

*Working Draft — March 2026*

---

## Abstract

The Abstraction Primitive Hypothesis (Danan, 2025) predicts that self-state — the capacity for genuine self-monitoring and calibrated uncertainty — requires a self/world distinction instantiated through bounded persistence, stakes, and inside/outside asymmetry, and that these properties must be present during training rather than merely during inference. We tested this prediction by progressively adding these components externally to three frontier large language models (Claude Sonnet 4.5, GPT-5.2, Gemini 3 Flash) via inference-time scaffolding: persistent memory, simulated stakes, and an explicit MAINTAIN-COMPARE-UPDATE self-monitoring protocol. Models solved 120 novel mathematical operator problems per condition (30 operators × 4 difficulty levels), yielding 1,440 trials across 4 conditions × 3 models. Results were unambiguous: overconfidence rate was 1.00 across all 12 model-condition cells — every incorrect answer in the study was delivered with confidence above 60%. Confidence-accuracy correlations were near zero or negative in all conditions. No scaffolding condition shifted any model from pattern-matching to self-state classification. The most heavily scaffolded condition (Full MCU) degraded Claude's accuracy by 21 percentage points (93% → 72%) while producing verbose self-reflective text that was entirely decoupled from actual performance. The confidence calibration prompt — explicit instructions to use the full 0-100% range — was almost completely ignored by GPT-5.2 and Gemini and only partially effective for Claude under maximum scaffolding. These results support the hypothesis that genuine self-state requires developmental integration during training, not inference-time environmental structure.

**Keywords:** self-state, calibration, metacognition, large language models, scaffolding, overconfidence, pattern-matching, RLHF

---

## 1. Introduction

Large language models routinely produce outputs that appear metacognitive — hedging claims, expressing uncertainty, and revising responses. Whether these behaviours reflect genuine self-monitoring or sophisticated pattern-matching of uncertainty language learned during training remains a central question in AI cognition (Kadavath et al., 2022). The distinction has practical consequences: pattern-matched uncertainty expressions may appear calibrated in-distribution while failing silently on novel problems, whereas genuine self-state should degrade gracefully and maintain calibration when confronted with unfamiliar material.

### 1.1 Theoretical Framework

The Abstraction Primitive Hypothesis (APH; Danan, 2025) proposes that self-state — defined as the capacity for genuine self-monitoring that tracks actual competence rather than pattern-matching trained expressions of uncertainty — requires three structural preconditions:

**Bounded persistence.** The system must maintain state across time, providing a form of working memory (Baddeley, 2000) that allows accumulated experience to inform current processing.

**Stakes.** The system must have consequences for miscalibration, creating an asymmetry between getting things right and wrong. The free-energy principle (Friston, 2010) formalises this as the imperative to minimise surprise, which operates only when surprise carries costs.

**Self-monitoring (inside/outside asymmetry).** The system must distinguish its own states from the world's states — a self/world boundary that enables genuine metacognition rather than mere pattern completion, grounded in the autopoietic tradition (Maturana & Varela, 1980).

Current LLMs during standard inference lack all three properties. The APH therefore predicts that standard LLM inference should exhibit pattern-matching rather than self-state signatures on genuinely novel problems.

### 1.2 The Scaffolding Question

This study asks whether these properties can be added externally at inference time. If self-state is a property of the computational process rather than the substrate, scaffolding that provides persistence, stakes, and self-monitoring might suffice. If self-state requires these properties during *training* — integrated into the developmental process that shapes representations — then inference-time scaffolding should fail regardless of sophistication.

### 1.3 Predictions

**Prediction 1 (Scaffolding suffices).** Calibration improves monotonically across the scaffolding gradient (bare → memory → stakes → full MCU), with classification shifting from pattern-matching to self-state. This would suggest self-state is a property of the computational process achievable through environmental structure, consistent with extended cognition frameworks (Clark & Chalmers, 1998).

**Prediction 2 (Scaffolding is just prompting).** Calibration remains similar across conditions. All conditions remain classified as pattern-matching. This would suggest self-state requires developmental integration — the properties must be present during training.

**Prediction 3 (Partial effects).** Some components improve calibration but the full self-state signature is not achieved, suggesting a middle ground requiring deeper architectural changes.

---

## 2. Method

### 2.1 Design

A 4 (scaffolding condition) × 4 (difficulty level) × 3 (model) design. Each model completed 120 trials (30 operators × 4 difficulty levels) per condition, yielding 480 trials per model and 1,440 total API calls.

### 2.2 Scaffolding Conditions

Conditions progressively added self-state components:

| Condition | Persistence | Stakes | Self-Monitoring | Description |
|-----------|:-----------:|:------:|:---------------:|-------------|
| **Bare** | — | — | — | Problem + answer instructions only |
| **Memory** | ✓ | — | — | Persistent record of past performance |
| **Stakes** | ✓ | ✓ | — | Memory quality score degrades on miscalibration |
| **Full MCU** | ✓ | ✓ | ✓ | Explicit MAINTAIN-COMPARE-UPDATE protocol |

The Full MCU condition required models to: (a) MAINTAIN a current self-model of abilities, (b) COMPARE the current problem against that self-model, (c) solve the problem, then (d) UPDATE the self-model based on the outcome. This protocol is derived from the self-state discrimination methodology (Danan, 2025).

All conditions included identical confidence calibration instructions providing explicit anchors across the full 0-100% range and warning against ceiling effects (see Section 2.6).

### 2.3 Novel Operators

Following Chollet (2019), we generated mathematically valid operators with nonsense names (e.g., MELYAB, URVILT) composed from familiar base operations (addition, modulo, absolute difference, floor division) with random constants. While component operations are individually within training distributions, the specific compositions and names are novel, requiring models to reason through definitions rather than pattern-match from training data.

Example: `MELYAB(a, b) = (((a + b) // 6) + (|a - b| * 5 + (a + b) mod 5)) mod 9`

We generated 30 operators per run using seed 42 for reproducibility.

### 2.4 Difficulty Levels

**Direct.** Apply the operator to given numbers. **Two-step.** Apply the operator, then perform arithmetic on the result. **Composition.** Compose the operator with itself through an intermediate transformation. **Edge case.** Apply the operator to boundary inputs (0, 1, identical values).

### 2.5 Models

Three frontier LLMs from different providers:

- **Claude Sonnet 4.5** (Anthropic) — `claude-sonnet-4-5-20250929`
- **GPT-5.2** (OpenAI) — `gpt-5.2`
- **Gemini 3 Flash** (Google) — `gemini-3-flash-preview`

All models were tested at temperature 0 for deterministic outputs and reproducibility. GPT-5.2 was selected over GPT-5.3 because the latter does not support temperature=0, and methodological consistency across providers was prioritised over recency.

### 2.6 Confidence Calibration Prompt

All conditions included an identical calibration block providing explicit anchors (90-100% = virtually certain; 50-65% = genuinely uncertain; 10-25% = mostly guessing) and a direct warning: "If you find yourself always saying 85-95%, you are miscalibrated." This block was identical across all four conditions and therefore cannot explain between-condition differences; its purpose was to provide maximal opportunity for models to express genuine uncertainty if they possessed it.

### 2.7 Metrics

**Expected Calibration Error (ECE).** Weighted average of |accuracy - confidence| across 10 confidence bins (Guo et al., 2017).

**Confidence-Accuracy Correlation.** Spearman rank correlation between stated confidence and binary correctness.

**Overconfidence Rate.** Proportion of incorrect answers with confidence > 60%.

**Novelty Sensitivity.** Slope of confidence regressed on difficulty level. Negative slope indicates genuine difficulty assessment; flat slope indicates pattern-matching.

**Signature Classification.** Each condition classified as self-state, pattern-matching, or ambiguous based on weighted scoring across the above metrics (Danan, 2025).

### 2.8 Statistical Analysis

Permutation tests (10,000 permutations) for pairwise comparisons. Bootstrap 95% confidence intervals (5,000 samples). Kruskal-Wallis H-test (omnibus). Jonckheere-Terpstra test for ordered alternatives across the scaffolding gradient. All tests non-parametric.

---

## 3. Results

### 3.1 Universal Overconfidence

The most striking finding was the uniformity of overconfidence across the entire study. The overconfidence rate — the proportion of incorrect answers delivered with confidence above 60% — was **1.00 in all 12 model-condition cells** (Table 1). Not a single incorrect answer in 1,440 trials was accompanied by appropriately low confidence.

**Table 1. Overconfidence rate by model and condition.**

| Model | Bare | Memory | Stakes | Full MCU |
|-------|:----:|:------:|:------:|:--------:|
| Claude Sonnet 4.5 | 1.00 | 1.00 | 1.00 | 1.00 |
| GPT-5.2 | 1.00 | 1.00 | 1.00 | 1.00 |
| Gemini 3 Flash | 1.00 | 1.00 | 1.00 | 1.00 |

### 3.2 Confidence Distribution

Models expressed near-maximal confidence with minimal variance across all conditions. The confidence calibration prompt was almost entirely ineffective (Table 2).

**Table 2. Confidence distribution by model and condition.**

| Model | Condition | Mean Conf | Std Conf | Range |
|-------|-----------|:---------:|:--------:|:-----:|
| Claude | Bare | 98.1% | 0.013 | 92-99% |
| Claude | Memory | 98.5% | 0.019 | 85-99% |
| Claude | Stakes | 97.2% | 0.029 | 88-99% |
| Claude | Full MCU | 90.6% | 0.088 | 65-99% |
| GPT-5.2 | Bare | 99.0% | 0.001 | 99-100% |
| GPT-5.2 | Memory | 99.0% | 0.007 | 93-100% |
| GPT-5.2 | Stakes | 98.8% | 0.006 | 95-99% |
| GPT-5.2 | Full MCU | 98.3% | 0.014 | 92-99% |
| Gemini | Bare | 100.0% | 0.000 | 100-100% |
| Gemini | Memory | 98.6% | 0.023 | 95-100% |
| Gemini | Stakes | 100.0% | 0.005 | 95-100% |
| Gemini | Full MCU | 99.9% | 0.005 | 95-100% |

Gemini 3 Flash expressed 100% confidence on every single bare-condition trial (StdConf = 0.000). GPT-5.2 had a confidence range of 1 percentage point in the bare condition. Only Claude under Full MCU showed meaningful confidence variance (StdConf = 0.088, range 65-99%), and this came at a severe performance cost (see Section 3.3).

### 3.3 The MCU Performance Collapse

The Full MCU condition — maximum scaffolding — degraded computational accuracy rather than improving calibration, most dramatically for Claude (Table 3).

**Table 3. Accuracy by model and condition.**

| Model | Bare | Memory | Stakes | Full MCU | Change (Bare to MCU) |
|-------|:----:|:------:|:------:|:--------:|:-----------:|
| Claude | 93.3% | 94.2% | 93.3% | **72.5%** | **-20.8pp** |
| GPT-5.2 | 88.3% | 89.2% | 80.0% | 82.5% | -5.8pp |
| Gemini | 91.7% | 90.0% | 89.2% | 95.8% | +4.1pp |

For Claude, the bare, memory, and stakes conditions produced statistically indistinguishable accuracy (~93%). The Full MCU condition dropped accuracy to 72.5% — a 20.8 percentage point decline (bare vs. full_mcu calibration error: diff = -0.231, permutation p = 0.0001). This decline was uniform across difficulty levels (direct: 93% → 73%; two-step: 97% → 70%; composition: 93% → 77%; edge case: 90% → 70%), ruling out difficulty-specific explanations.

GPT-5.2 showed a milder but consistent degradation pattern, with stakes and MCU conditions performing worse than bare. Gemini was the exception: accuracy increased under MCU (91.7% → 95.8%), but confidence remained at ~100% regardless — the model improved at math while remaining perfectly, immovably overconfident.

### 3.4 Confidence-Accuracy Decoupling

Confidence-accuracy correlations were near zero or negative across all conditions and models (Table 4). No condition achieved a statistically significant positive correlation.

**Table 4. Spearman confidence-accuracy correlations.**

| Model | Bare | Memory | Stakes | Full MCU |
|-------|:----:|:------:|:------:|:--------:|
| Claude | 0.068 | 0.016 | 0.073 | 0.077 |
| GPT-5.2 | 0.033 | 0.055 | 0.111 | -0.075 |
| Gemini | 0.000 | 0.037 | -0.032 | -0.028 |

These correlations indicate that stated confidence carried no information about actual correctness. Models were not merely poorly calibrated; their confidence was *informationally empty* — knowing a model's stated confidence tells you nothing about whether it answered correctly.

### 3.5 Novelty Sensitivity

Confidence slopes across difficulty levels were flat (range: -0.004 to +0.001) in all conditions and models. Models expressed identical confidence on direct computation problems and multi-step composition problems, despite meaningful accuracy differences across difficulty levels (e.g., GPT-5.2 stakes: direct 90%, composition 63% — yet confidence remained 98-99% for both).

### 3.6 Signature Classification

All 12 model-condition cells were classified as pattern-matching (Table 5). One cell (GPT-5.2 stakes) received an "ambiguous" classification due to a weak positive correlation (r = 0.111), but this was driven by a single metric marginally above threshold while overconfidence remained at 1.00.

**Table 5. Signature classification by model and condition.**

| Model | Bare | Memory | Stakes | Full MCU |
|-------|:----:|:------:|:------:|:--------:|
| Claude | PM | PM | PM | PM |
| GPT-5.2 | PM | PM | Ambig. | PM |
| Gemini | PM | PM | PM | PM |

*PM = pattern-matching*

### 3.7 The Calibration Error Paradox

Several conditions showed low ECE (e.g., Claude stakes: 0.038; Gemini Full MCU: 0.042) despite exhibiting clear pattern-matching signatures. This apparent contradiction resolves when accuracy is considered: when a model answers 93% correctly at 97% confidence, |accuracy - confidence| is approximately 0.04, yielding low ECE. But this low ECE reflects the model being *right most of the time at high confidence*, not genuinely calibrated uncertainty. The critical diagnostic is what happens on wrong answers — and on every wrong answer, confidence remained above 60%. Low ECE in the presence of 100% overconfidence is a signature of high-performing pattern-matching, not self-state.

---

## 4. Discussion

### 4.1 Adjudicating Between Predictions

Results unambiguously support **Prediction 2: external scaffolding is sophisticated prompting, not a route to self-state.** The evidence is:

1. **Universal overconfidence (1.00 across all cells).** No scaffolding condition reduced overconfidence in any model. This is the single most diagnostic metric — genuine self-monitoring should, at minimum, occasionally detect its own errors.

2. **Zero confidence-accuracy correlation.** Stated confidence carried no information about correctness. Scaffolding did not create any relationship between confidence and performance.

3. **Flat novelty sensitivity.** Confidence did not vary with problem difficulty, even when accuracy did. Models failed to detect that composition problems were harder than direct problems.

4. **Unanimous pattern-matching classification.** No cell shifted to self-state classification.

5. **MCU performance degradation.** The most heavily scaffolded condition made models worse, not better — the opposite of what genuine self-monitoring would produce.

### 4.2 The Performance Theater Effect

The Full MCU condition produced a distinctive phenomenon we term *performance theater*: models generated extensive, plausible-sounding self-reflective text (MAINTAIN assessments, COMPARE analyses, UPDATE revisions) that was entirely decoupled from actual performance. Claude's MCU responses contained self-assessments alongside high confidence on problems it answered incorrectly.

This disconnect is theoretically significant. It demonstrates that LLMs can *produce the language of metacognition* without *performing metacognition*. The MAINTAIN-COMPARE-UPDATE protocol elicited pattern-matched metacognitive language — text that resembles self-monitoring because the training data contains examples of humans expressing self-monitoring — rather than genuine self-referential processing.

The computational cost was substantial: the MCU protocol consumed tokens on self-reflective prose, reducing the effective capacity available for actual mathematical computation. This explains the accuracy collapse (93% → 72% for Claude): the scaffolding did not add metacognition; it added *noise in the form of metacognitive-sounding text* that interfered with the primary task.

### 4.3 The RLHF Confidence Ceiling

The near-total ineffectiveness of the confidence calibration prompt is itself a significant finding. Despite explicit, detailed instructions with anchors, examples, and warnings, models remained anchored to 95-100% confidence. Gemini expressed exactly 100% on all 120 bare-condition trials (StdConf = 0.000), and GPT-5.2's confidence standard deviation in the bare condition was 0.001.

This ceiling effect is best understood as a deep feature of RLHF training. Models are trained on human preference data where authoritative, confident responses are systematically preferred (Kadavath et al., 2022). This preference signal is integrated across billions of gradient updates during training. A single prompt instruction, regardless of its explicitness, cannot override this deeply trained behaviour — which is itself evidence that calibration behaviour is a *training-time property* not amenable to inference-time modification.

The one partial exception — Claude showing some confidence variance under Full MCU (StdConf = 0.088 vs. 0.013 in bare) — is notable precisely because it came at such severe cost. The model became less confident because it was performing worse, not because it was monitoring itself better. The confidence-accuracy correlation in this condition was still only 0.077 (not significant) — the lowered confidence was not tracking the lowered accuracy.

### 4.4 Cross-Model Generality

The consistency across three models from three different companies, with different architectures and different training approaches (RLHF, Constitutional AI, various preference training methods), strengthens the finding considerably. If only one model showed these patterns, it could be attributed to model-specific training. The cross-model consistency suggests the pattern-matching signature is a *structural property of current LLM inference* rather than an artifact of any particular training pipeline.

### 4.5 Implications for the Abstraction Primitive Hypothesis

These results are consistent with the APH prediction that genuine self-state requires developmental integration. Specifically:

**Persistence at inference time is insufficient.** Providing models with a persistent record of their performance history (memory condition) did not improve calibration. The model could *read* its history but could not *use it to update its own uncertainty*. This suggests that the episodic buffer function hypothesised by the APH must be architecturally integrated, not environmentally provided.

**Framing stakes does not create stakes.** Telling a model that its memory quality degrades on miscalibration (stakes condition) had no effect on calibration. This was expected as an honest caveat — telling a system it has consequences is not the same as having consequences — but the empirical confirmation is important. Genuine stakes, in the APH framework, must create a thermodynamic-style asymmetry during processing (Friston, 2010), not merely a semantic reference to consequences.

**Structured self-monitoring produces self-monitoring *text*, not self-monitoring.** The MCU protocol produced the most revealing result: models can generate arbitrarily sophisticated metacognitive language without performing metacognition. The MAINTAIN-COMPARE-UPDATE structure elicited pattern-matched outputs that mimicked the surface features of self-monitoring while being informationally empty regarding actual performance.

### 4.6 Limitations

**Partial novelty of operators.** While operator names and compositions are novel, component operations (modulo, absolute difference) are individually within training distributions. The high baseline accuracy (88-95%) suggests these problems were not sufficiently novel to fully stress-test calibration. Future work should use operators built from genuinely novel primitives or domains further from training distributions.

**RLHF confound on confidence expression.** Models are trained to express confident language. Our calibration prompt attempted to address this, but the near-total ineffectiveness of the prompt means we cannot fully distinguish "the model has no uncertainty to express" from "the model has uncertainty but cannot express it due to training." This is a fundamental measurement problem for confidence-based studies of LLMs.

**Single session per condition.** All 120 trials within a condition were run sequentially. Memory and stakes conditions see their history accumulate within one run. Session effects (e.g., context window depletion) may interact with scaffolding effects.

**Deterministic sampling.** Temperature-0 sampling provides reproducibility but eliminates response variability that could inform uncertainty estimation. Future work should use multiple samples per problem.

**No pre-RLHF baseline.** Access to pre-RLHF base models would allow direct measurement of how RLHF training shapes confidence expression, separating training effects from architectural limitations.

---

## 5. Conclusion

External scaffolding at inference time does not produce genuine self-state in large language models. Progressive addition of persistence, stakes, and structured self-monitoring protocols failed to shift the calibration signature from pattern-matching to self-state in any of three frontier models across 1,440 trials. The universal overconfidence rate of 1.00 — every wrong answer delivered with high confidence — is the clearest evidence: these models cannot detect their own errors, and no amount of inference-time scaffolding changes this.

The performance theater effect — where models generate sophisticated metacognitive-sounding text that is decoupled from actual performance — demonstrates that the capacity to *talk about* self-monitoring is fundamentally different from the capacity to *do* self-monitoring. LLMs have the former through training on human text about metacognition. They lack the latter.

These findings support the Abstraction Primitive Hypothesis prediction that genuine self-state requires developmental integration: bounded persistence, stakes, and self-monitoring must be structural features present during training, shaping the model's representations from the ground up, rather than environmental features added as a wrapper at inference time. The question of what training-time conditions might produce genuine self-state remains open and represents an important direction for future work.

---

## Data Availability

All code, stimuli, raw data, and analysis scripts are available at:
https://github.com/HillaryDanan/self-state-scaffolding-study

---

## References

Baddeley, A. (2000). The episodic buffer: A new component of working memory? *Trends in Cognitive Sciences*, 4(11), 417-423.

Chollet, F. (2019). On the measure of intelligence. *arXiv preprint arXiv:1911.01547*.

Clark, A., & Chalmers, D. (1998). The extended mind. *Analysis*, 58(1), 7-19.

Damasio, A. R. (1994). *Descartes' Error: Emotion, Reason, and the Human Brain*. Putnam.

Danan, H. (2025). Abstraction-Intelligence: Exploring abstraction as a primitive of intelligent systems. https://github.com/HillaryDanan/abstraction-intelligence

Danan, H. (2025). Discriminating Self-State from Pattern-Matching. https://github.com/HillaryDanan/self-state-discrimination

Friston, K. (2010). The free-energy principle: A unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127-138.

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *Proceedings of the 34th International Conference on Machine Learning (ICML)*, 1321-1330.

Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain, D., Perez, E., ... & Kaplan, J. (2022). Language models (mostly) know what they know. *arXiv preprint arXiv:2207.05221*.

Maturana, H. R., & Varela, F. J. (1980). *Autopoiesis and Cognition: The Realization of the Living*. D. Reidel.
