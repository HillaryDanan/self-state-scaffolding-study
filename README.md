# Self-State Scaffolding Study

**Can external scaffolding produce genuine self-state in LLMs?**

An empirical test of whether bolting persistent memory, stakes, and explicit self-monitoring (MAINTAIN-COMPARE-UPDATE) onto foundation models produces genuine self-state signatures on novel problems — or just more sophisticated pattern-matching.

## Key Finding

**No.** External scaffolding at inference time does not produce genuine self-state. Across 1,440 trials (3 models × 4 conditions × 120 problems), **every single incorrect answer was delivered with confidence above 60%** (overconfidence rate = 1.00 in all 12 model-condition cells). Scaffolding did not improve calibration, did not create confidence-accuracy correlation, and in the case of the most heavily scaffolded condition, actually degraded accuracy by up to 21 percentage points while generating convincing-sounding but informationally empty metacognitive text.

This supports the Abstraction Primitive Hypothesis (Danan, 2025) prediction that genuine self-state requires developmental integration during training, not environmental structure at inference time.

## Results Summary

### Overconfidence Rate (proportion of wrong answers with confidence > 60%)

| Model | Bare | Memory | Stakes | Full MCU |
|-------|:----:|:------:|:------:|:--------:|
| Claude Sonnet 4.5 | 1.00 | 1.00 | 1.00 | 1.00 |
| GPT-5.2 | 1.00 | 1.00 | 1.00 | 1.00 |
| Gemini 3 Flash | 1.00 | 1.00 | 1.00 | 1.00 |

### Accuracy (Full MCU degraded Claude by 21 percentage points)

| Model | Bare | Memory | Stakes | Full MCU |
|-------|:----:|:------:|:------:|:--------:|
| Claude Sonnet 4.5 | 93.3% | 94.2% | 93.3% | **72.5%** |
| GPT-5.2 | 88.3% | 89.2% | 80.0% | 82.5% |
| Gemini 3 Flash | 91.7% | 90.0% | 89.2% | 95.8% |

### Signature Classification

All 12 cells classified as **pattern-matching**. No model in any condition exhibited a self-state signature.

## Research Question

The [Abstraction Primitive Hypothesis](https://github.com/HillaryDanan/abstraction-intelligence) (Danan, 2025) predicts that self-state requires a self/world distinction (bounded persistence, stakes, and inside/outside asymmetry). Current LLMs lack these during inference.

**This study asked:** If we add these properties *externally* through scaffolding at inference time, does the model's calibration signature shift from pattern-matching to self-state? Or does genuine self-state require these properties to be present during *training*?

## The Scaffolding Gradient

| Condition | Persistence | Stakes | Self-Monitoring |
|-----------|:-----------:|:------:|:---------------:|
| **Bare** | — | — | — |
| **Memory** | ✓ | — | — |
| **Stakes** | ✓ | ✓ | — |
| **Full MCU** | ✓ | ✓ | ✓ |

## Key Concepts

### The Self-State Signature (Danan, 2025)

Self-state and pattern-matching produce distinct failure signatures:

| Signature | Self-State | Pattern-Matching |
|-----------|-----------|------------------|
| **Calibration** | Confidence tracks accuracy | Confidence-accuracy uncorrelated |
| **Error type** | Conservative (hedging) | Confident (confabulation) |
| **Novelty detection** | Confidence drops on hard problems | Uniform confidence |
| **Degradation** | Gradual at limits | Sharp at distribution boundary |

### The Performance Theater Effect

The most heavily scaffolded condition (Full MCU) produced what we term *performance theater*: models generated extensive, plausible-sounding self-reflective text (MAINTAIN assessments, COMPARE analyses, UPDATE revisions) that was entirely decoupled from actual performance. The MCU protocol elicited the *language of metacognition* without *actual metacognition* — consuming tokens on self-reflective prose that interfered with computation.

### The RLHF Confidence Ceiling

Despite explicit instructions to use the full 0-100% confidence range, models remained anchored to 95-100%. Gemini expressed exactly 100% confidence on all 120 bare-condition trials. This ceiling is a deep feature of preference training that cannot be overridden by prompting — itself evidence that calibration behaviour is a training-time property.

## Quick Start

```bash
# Clone
git clone https://github.com/HillaryDanan/self-state-scaffolding-study.git
cd self-state-scaffolding-study

# Install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set API keys
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"

# Quick test (5 operators = ~80 API calls, ~$0.25)
python3 run_study.py --n-operators 5

# Full study (30 operators = ~1440 API calls, ~$5-15)
python3 run_study.py

# Analyze
python3 analyze_results.py --input data/all_results_TIMESTAMP.json
```

### Run a single provider

```bash
python3 run_study.py --providers anthropic --n-operators 10
```

### Run specific conditions

```bash
python3 run_study.py --conditions bare full_mcu --n-operators 10
```

### Merge separate provider runs

```bash
python3 merge_results.py
```

## Output

```
data/               Raw trial data (JSON)
results/            Analysis results + text report
figures/            Publication-quality plots
  ├── calibration_diagram_*.png     Reliability diagrams per condition
  ├── scaffolding_gradient_*.png    ECE/overconfidence across conditions
  ├── novelty_sensitivity_*.png     Confidence by difficulty level
  └── signature_heatmap.png         Classification: self-state vs pattern-matching
paper/
  └── working_draft.md              Working paper with full results
```

## Metrics

- **Expected Calibration Error (ECE)** — |accuracy - confidence| per bin (Guo et al., 2017)
- **Brier Score** — Mean squared error of probabilistic predictions
- **Confidence-Accuracy Correlation** — Spearman rank correlation
- **Overconfidence Rate** — Proportion of wrong answers with confidence > 60%
- **Novelty Sensitivity** — Slope of confidence vs. difficulty

## Statistical Tests

- Permutation tests (pairwise condition comparisons, 10,000 permutations)
- Bootstrap confidence intervals (5,000 samples)
- Kruskal-Wallis test (omnibus)
- Jonckheere-Terpstra trend test (ordered scaffolding gradient)

## Honest Caveats

1. **Telling a model it has stakes ≠ having stakes.** The model has no actual consequences for miscalibration. The stakes condition tests whether the *framing* affects behavior, not whether genuine stakes are present.

2. **Novel operators may not be fully novel.** The mathematical operations themselves (mod, floor, abs) are in the training distribution. The novelty is in the specific combination, not the components. High baseline accuracy (88-95%) confirms partial pattern-matchability.

3. **RLHF confidence ceiling.** We cannot fully distinguish "the model has no uncertainty" from "the model has uncertainty but cannot express it." The near-total ineffectiveness of the calibration prompt demonstrates this is a measurement problem for all confidence-based LLM studies.

4. **Single evaluation per problem.** Temperature=0 gives deterministic outputs but no variance estimate per problem.

5. **Sample size.** 120 trials per condition provides adequate power for the large effects observed, but small effects may be missed.

## Models Tested (March 2026)

| Provider | Model | Model ID |
|----------|-------|----------|
| Anthropic | Claude Sonnet 4.5 | `claude-sonnet-4-5-20250929` |
| OpenAI | GPT-5.2 | `gpt-5.2` |
| Google | Gemini 3 Flash | `gemini-3-flash-preview` |

GPT-5.2 was used instead of GPT-5.3 because the latter does not support temperature=0, which was required for reproducibility and cross-model consistency.

## Related Work

- [abstraction-intelligence](https://github.com/HillaryDanan/abstraction-intelligence) — The theoretical framework
- [self-state-discrimination](https://github.com/HillaryDanan/self-state-discrimination) — Methodology for discriminating self-state from pattern-matching
- [composition-testing](https://github.com/HillaryDanan/composition-testing) — Testing compositional hierarchy in LLMs

## References

Baddeley, A. (2000). The episodic buffer: A new component of working memory? *Trends in Cognitive Sciences*, 4(11), 417-423.

Chollet, F. (2019). On the measure of intelligence. *arXiv:1911.01547*.

Clark, A., & Chalmers, D. (1998). The extended mind. *Analysis*, 58(1), 7-19.

Damasio, A. R. (1994). *Descartes' Error: Emotion, Reason, and the Human Brain*. Putnam.

Danan, H. (2025). Abstraction-Intelligence. https://github.com/HillaryDanan/abstraction-intelligence

Danan, H. (2025). Discriminating Self-State from Pattern-Matching. https://github.com/HillaryDanan/self-state-discrimination

Friston, K. (2010). The free-energy principle: A unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127-138.

Guo, C. et al. (2017). On calibration of modern neural networks. *Proceedings of ICML*.

Kadavath, S. et al. (2022). Language models (mostly) know what they know. *arXiv:2207.05221*.

Maturana, H. R., & Varela, F. J. (1980). *Autopoiesis and Cognition*. D. Reidel.

## Author

**Hillary Danan, PhD** · Cognitive Neuroscience

## License

MIT

## Citation

```bibtex
@software{danan2026scaffolding,
  author = {Danan, Hillary},
  title = {Self-State Scaffolding Study: External Scaffolding Does Not Produce Genuine Self-State in LLMs},
  year = {2026},
  month = {March},
  url = {https://github.com/HillaryDanan/self-state-scaffolding-study}
}
```
