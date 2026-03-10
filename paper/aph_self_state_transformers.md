# Abstraction, Self-State and the Limits of Predictive Artificial Intelligence

## Why Transformers Work — and Why Their Limits Appear Where They Do

Hillary Danan, PhD

-----

# Abstract

Large language models based on transformer architectures (Vaswani et al., 2017) have recently achieved remarkable performance in domains such as programming, mathematics and formal reasoning. These successes have prompted speculation that predictive language models may be approaching general intelligence. However, performance across cognitive domains remains uneven: systems that excel in deterministic reasoning often exhibit brittle failures in novelty detection, social inference and calibrated uncertainty (Kadavath et al., 2022; Xiong et al., 2024).

This article proposes a testable explanation grounded in the **Abstraction Primitive Hypothesis (APH)**. APH posits that intelligence develops through progressively deeper forms of abstraction culminating in **self-state**—a computational mechanism enabling systems to monitor and regulate their own reasoning.

Deterministic domains provide unusually strong external signals of correctness. These signals allow systems lacking intrinsic self-state to approximate metacognitive regulation through environmental verification. We argue that this property explains the disproportionate success of contemporary AI systems in deterministic domains and predicts systematic failure patterns as verification signals weaken.

Mapping APH onto transformer architectures reveals a structural correspondence: transformers efficiently implement the first three stages of abstraction—pattern extraction, symbol formation and recursive composition—while lacking intrinsic mechanisms corresponding to Stage 4 self-state. This relationship explains both the strengths and limitations of current AI systems and suggests that the next advances in artificial intelligence may depend on architectures capable of genuine self-regulation.

-----

# Introduction

Recent progress in artificial intelligence has been driven by large language models trained using transformer architectures (Vaswani et al., 2017). These systems demonstrate strong performance in domains such as programming, mathematics and formal reasoning, yet exhibit persistent weaknesses in novelty detection, calibrated uncertainty and social inference.

This uneven performance distribution raises a scientific question:

**What architectural properties determine where artificial reasoning systems succeed and where they fail?**

Several lines of empirical work have begun to characterize these patterns. Kadavath et al. (2022) demonstrated that language models exhibit partial self-knowledge—they can estimate their own accuracy above chance—but that this calibration degrades on questions requiring genuine novelty assessment. Xiong et al. (2024) and Geng et al. (2024) have documented systematic calibration failures across model families, finding that expressed confidence often tracks surface features of prompts rather than actual accuracy. Most strikingly, Huang et al. (2024) showed that large language models cannot reliably self-correct their own reasoning without external feedback, and Stechly, Valmeekam & Kambhampati (2025) demonstrated that LLMs fail systematically at verifying solutions they themselves generated.

These findings converge on a pattern: current systems can generate plausible reasoning but struggle to *evaluate* their own outputs. This generation–verification asymmetry demands explanation.

The **Abstraction Primitive Hypothesis (APH)** provides a framework for addressing this question. APH proposes that intelligence emerges through a hierarchy of abstraction capacities culminating in **self-state**, a computational mechanism enabling systems to monitor and regulate their own reasoning.

This paper advances three core hypotheses.

**H1. Abstraction hierarchy hypothesis**

Intelligent reasoning systems progress through a hierarchy of abstraction capacities:

pattern extraction → symbol formation → recursive composition → self-state

**H2. Deterministic scaffolding hypothesis**

Domains with explicit correctness criteria allow reasoning systems lacking intrinsic self-state to approximate metacognitive regulation through external verification.

**H3. Transformer alignment hypothesis**

Transformer architectures efficiently implement the first three stages of abstraction but lack mechanisms corresponding to Stage 4 self-state.

If these hypotheses are correct, several empirical patterns should follow:

- transformer systems should excel in symbolically structured domains
- scaling should improve abstraction capacity but not self-state
- reasoning failures should cluster in domains requiring intrinsic regulation
- the generation–verification gap should widen as external verification weakens

The remainder of this paper develops these hypotheses and derives empirically testable predictions.

-----

# The Abstraction Hierarchy

APH proposes four stages of abstraction capacity, drawing on established cognitive science: symbolic representation in reasoning (Fodor & Pylyshyn, 1988), compositional semantics (Fodor, 1975), relational complexity (Halford, Wilson & Phillips, 1998), working memory structure (Baddeley, 2000; Cowan, 2001) and the development of metacognition (Flavell, 1979).

|Stage  |Capacity             |Function                                |
|-------|---------------------|----------------------------------------|
|Stage 1|Pattern extraction   |Detect statistical regularities         |
|Stage 2|Symbol formation     |Represent discrete combinable structures|
|Stage 3|Recursive composition|Construct hierarchical abstractions     |
|Stage 4|Self-state           |Monitor and regulate reasoning          |

Stages 1–3 construct abstractions.
Stage 4 regulates them.

The developmental ordering of these stages is well supported in cognitive science. Pattern extraction is present in invertebrates. Symbol formation emerges in human development around 18–24 months (Piaget, 1952). Recursive composition appears around age 4 and is associated with the emergence of recursive syntax and planning (Hauser, Chomsky & Fitch, 2002). Metacognitive self-monitoring develops last, maturing gradually through adolescence in parallel with prefrontal cortex development (Flavell, 1979; Kuhn, 2000).

-----

## Figure 1 — The Abstraction Hierarchy

```
Pattern Extraction
       ↓
Symbol Formation
       ↓
Recursive Composition
       ↓
Self-State (metacognitive regulation)
```

Lower stages construct representations.
Self-state regulates their reliability.

-----

# Self-State as a Computational Operation

Self-state can be formalized as a feedback loop:

```
MAINTAIN(x)
     ↓
COMPARE(x, y)
     ↓
UPDATE(x | result)
```

Where:

- **MAINTAIN** — retain representations across reasoning steps
- **COMPARE** — evaluate processing relative to expectations
- **UPDATE** — modify representations in response to discrepancies

This structure resembles feedback control systems in engineering and the **central executive** component of Baddeley’s (2000) working memory model. It is distinct from passive storage (no active comparison) and from the *content* of metacognitive knowledge (self-state is the *mechanism*).

The MAINTAIN-COMPARE-UPDATE loop presupposes a *subject*—an entity whose states are being maintained, compared and updated. This constitutive requirement connects self-state to the self/world distinction explored in detail in the broader APH framework (Danan, 2021) and relates to autopoietic theory (Maturana & Varela, 1980) and the free energy principle’s distinction between internal and external states (Friston, 2010).

-----

# Deterministic Domains as Cognitive Scaffolds

Certain domains possess explicit correctness criteria.

|Domain     |Verification mechanism |
|-----------|-----------------------|
|Programming|compilation and tests  |
|Mathematics|formal derivation      |
|Logic      |proof consistency      |
|Physics    |dimensional constraints|

These domains support the loop:

```
generate → test → revise
```

The **test step is provided by the domain itself**.

This property has been exploited in recent AI systems. Lightman et al. (2024) demonstrated that process-based verification—checking individual reasoning steps against formal criteria—substantially improves mathematical problem-solving. The success of code-executing language models similarly depends on compilation and test suites providing external comparison signals that the model itself cannot generate internally.

-----

## Figure 2 — External Verification Loop

```
Model reasoning
       ↓
Candidate solution
       ↓
External verifier
       ↓
Error signal
       ↓
Revision
```

Deterministic domains provide the comparison signal externally.

-----

# External Verification as a Substitute for Self-State

Within the APH framework, deterministic domains effectively provide the **COMPARE** step of the self-state loop.

Instead of performing:

```
MAINTAIN → COMPARE → UPDATE
```

systems operating in deterministic domains perform:

```
MAINTAIN  (model)
COMPARE   (environment)
UPDATE    (model)
```

Deterministic domains therefore partially substitute for intrinsic self-state. This explains a finding that has puzzled researchers: why LLMs can improve their reasoning through tool use, code execution and iterative refinement in formal domains (Kambhampati et al., 2024) but fail to self-correct in domains without external verifiers (Huang et al., 2024; Kamoi et al., 2024).

The critical asymmetry is not in the model’s reasoning capacity but in the *availability of external comparison signals*. When the environment supplies the COMPARE step, systems with only Stages 1–3 abstraction can approximate reliable reasoning. When it does not, the absence of self-state becomes apparent.

-----

# Transformers as Partial Abstraction Systems

Transformer architectures (Vaswani et al., 2017) map naturally onto the first three abstraction stages.

-----

## Figure 3 — Mapping APH to Transformer Architecture

```
APH Stage 1 — Pattern Extraction
  → Embedding layers (statistical compression)

APH Stage 2 — Symbol Formation
  → Token representations (discrete symbolic units)

APH Stage 3 — Recursive Composition
  → Attention layers (relational composition)

APH Stage 4 — Self-State
  → largely absent
```

Lake & Baroni (2018) demonstrated that sequence-to-sequence networks achieve compositional generalization in structured tasks, providing evidence for substantial Stage 2 capacity. Halford, Wilson & Phillips (1998) characterized the relational complexity that Stage 3 composition must support, and transformer attention mechanisms appear well-suited to this function.

**Important caveat on attention.** Modern attention mechanisms with key-value caching do maintain information across processing steps. Whether this constitutes MAINTAIN in the self-state sense is an open empirical question. The hypothesis is that attention-based maintenance differs from self-state in lacking the COMPARE operation that evaluates processing quality against held standards. This distinction requires empirical validation.

-----

# Why Transformers Work

Transformers perform well in domains characterized by:

- discrete symbolic representations
- compositional syntax
- recursive structure

|Domain          |Structural properties          |
|----------------|-------------------------------|
|Natural language|compositional grammar          |
|Programming     |symbolic syntax                |
|Mathematics     |recursive symbolic manipulation|
|Logic           |rule-governed inference        |

These domains align naturally with Stages 1–3 abstraction.

-----

# Why Language Became the First Domain of General AI

Language possesses several properties that make it uniquely suitable for large-scale abstraction learning:

1. Massive availability of training data
1. Strong compositional structure
1. High-level conceptual content
1. Symbolic representation of knowledge across domains

Language therefore provides an unusually efficient substrate for **Stages 1–3 abstraction learning**.

This perspective aligns with recent findings from Xu et al. (2025), who demonstrated that language models without grounding recover non-sensorimotor conceptual features (the abstract, internally-referenced dimension of meaning) far better than sensorimotor features. Within APH, this is expected: language statistics encode the compositional and relational structure of abstract concepts but cannot encode the grounded perceptual features that depend on embodied interaction.

Domains such as robotics require grounded perception, sensorimotor interaction and sparse feedback signals, making large-scale abstraction learning significantly more difficult. Within APH, language therefore represents the domain in which large-scale abstraction learning first becomes computationally tractable.

-----

# Scaling Laws and Abstraction Growth

Large language models exhibit scaling laws: performance improves predictably with model size, data and compute (Kaplan et al., 2020).

Within APH, scaling primarily enhances the first three stages of abstraction.

Scaling improves:

- pattern extraction resolution (Stage 1)
- stability of symbolic representations (Stage 2)
- depth of recursive composition (Stage 3)

-----

## Figure 4 — Scaling Under the Abstraction Hierarchy

```
Model scale ↑

  Stage 1 improves  ████████████████████████
  Stage 2 stabilizes ███████████████████████
  Stage 3 deepens   ██████████████████████

  ─────────────────────────────────────────
  Stage 4 (self-state)
  Not improved by scale alone
```

-----

# Why Scaling Plateaus

Scaling increases abstraction capacity but does not automatically produce **self-state regulation**.

This prediction aligns with emerging empirical evidence. Villalobos et al. (2024) identified fundamental data limits on continued scaling, and Li, Kudugunta & Zettlemoyer (2025) surveyed scaling law research and documented diminishing returns across multiple domains. Sevilla et al. (2024) analyzed whether current scaling trajectories can continue through 2030, finding significant constraints in data, compute and energy availability.

Within APH, these plateaus are expected: scaling improves the resolution and depth of Stages 1–3 abstraction but cannot produce the qualitatively different computational operation of self-state. Self-state requires mechanisms enabling systems to:

- maintain internal reasoning state across steps
- compare reasoning outcomes against expectations
- update strategies based on discrepancies

These processes require architectural innovations beyond parameter scaling.

-----

## Figure 5 — Verification Gradient Across Domains

```
HIGH VERIFICATION
  │  Formal mathematics
  │  Programming
  │  Structured reasoning
  │
  │  Commonsense reasoning
  │
  │  Social inference
  │  Metacognition
LOW VERIFICATION

  Reliance on intrinsic self-state increases
  as external verification decreases.
```

-----

# The Generation–Verification Gap

APH provides a natural explanation for one of the most robust findings in contemporary AI research: language models are systematically better at generating solutions than verifying them.

Stechly, Valmeekam & Kambhampati (2025) demonstrated that LLMs fail to verify their own solutions on reasoning and planning tasks, even when they can generate correct solutions at above-chance rates. Kambhampati et al. (2024) argued that LLMs fundamentally cannot plan but can assist planning when embedded in verification frameworks. Huang et al. (2024) showed that LLMs cannot self-correct reasoning without external feedback, and Kamoi et al. (2024) surveyed the self-correction literature and concluded that reliable self-correction requires external signals.

Within APH, this asymmetry follows directly from the absence of Stage 4 self-state. **Generation** is a Stage 1–3 operation: extract patterns, form symbols, compose them recursively. **Verification** is a Stage 4 operation: maintain the generated output, compare it against correctness criteria, update the assessment. When those criteria are supplied externally (deterministic domains), verification succeeds. When they must be generated internally (open-ended reasoning), verification fails.

-----

# Failure Signatures

When self-state is absent, APH predicts distinct failure patterns:

|Signature        |Self-state systems        |Pattern-matching systems|
|-----------------|--------------------------|------------------------|
|Novelty detection|confidence decreases      |confidence unchanged    |
|Error profile    |conservative uncertainty  |confident confabulation |
|Calibration      |confidence tracks accuracy|weak correlation        |
|Capacity limits  |gradual degradation       |abrupt failure          |

These predictions align with documented behaviors of contemporary LLM systems. Kadavath et al. (2022) found that model confidence partially tracks accuracy on in-distribution questions but degrades on novel inputs. Xiong et al. (2024) demonstrated systematic miscalibration across prompting strategies. The pattern of *confident confabulation*—generating plausible but incorrect outputs with high expressed confidence—is one of the most widely reported failure modes of current systems.

-----

# Chain-of-Thought and Architectural Self-State

Modern AI systems increasingly approximate self-state through architectural mechanisms:

- **Chain-of-thought prompting** (Wei et al., 2022) externalizes intermediate reasoning steps, creating a partial trace that can be inspected. However, within APH, chain-of-thought is best understood as extending Stage 3 recursive composition rather than implementing Stage 4 self-state: it generates more reasoning but does not evaluate whether that reasoning is reliable.
- **Test-time compute scaling** (Snell et al., 2025) allocates additional computation during inference, allowing models to explore multiple solution paths. The OpenAI o1 system (OpenAI, 2024) represents the most prominent implementation, combining extended chain-of-thought with reinforcement learning to improve performance on reasoning tasks.
- **Process-based verification** (Lightman et al., 2024) trains separate reward models to evaluate individual reasoning steps, creating an external COMPARE signal.
- **Agentic architectures** such as Reflexion (Shinn et al., 2023) maintain verbal reasoning traces across episodes and use environmental feedback to update strategies. Multi-agent debate (Du et al., 2024) distributes the verification function across multiple model instances.

Within APH, these mechanisms create **architectural self-state**: the COMPARE step is supplied by external tools, separate models or environmental feedback rather than emerging from the reasoning system’s intrinsic architecture.

This distinction matters because architectural self-state depends on the *availability* of external comparison signals. It works when those signals can be constructed (deterministic domains, tool-augmented environments) but fails when they cannot (genuinely novel problems, social reasoning, open-ended inference).

-----

## Figure 6 — Unified APH–Transformer–Scaling Landscape

```
                    SELF-STATE (Stage 4)
             metacognitive regulation layer
                         │
          ┌──────────────┴──────────────┐
          │                             │
  Reasoning Generator          External Verification
  (transformer core:           (deterministic domains,
   Stages 1-3)                  tools, reward models)
          │                             │
          └──────────────┬──────────────┘
                         │
                 Recursive Composition
                   (attention layers)
                         │
                  Symbol Formation
                 (token representations)
                         │
                  Pattern Extraction
                   (embedding layers)
```

Transformers implement the lower abstraction stack.
Deterministic domains and architectural scaffolds provide external verification.
General intelligence likely requires intrinsic self-state.

-----

# Empirical Research Program

The Abstraction Primitive Hypothesis generates a set of experimentally testable predictions.

## Experiment 1 — Novelty Calibration Test

Construct reasoning tasks that fall outside known training distributions (e.g., randomized operators or synthetic formal systems).

**Measure:**

- accuracy
- confidence estimates

**Prediction:**

Systems lacking self-state will show **weak correlation between confidence and accuracy** on genuinely novel tasks, even when calibration is adequate on in-distribution tasks. This distinguishes genuine self-monitoring from learned hedging patterns trained through RLHF.

## Experiment 2 — Verification Gradient Study

Evaluate reasoning performance across domains with varying verification strength.

**Domains:**

1. Formal mathematics (strong verification)
1. Programming (strong verification)
1. Structured reasoning (moderate verification)
1. Commonsense reasoning (weak verification)
1. Social reasoning (minimal verification)

**Prediction:**

Performance should decline systematically as external verification decreases. Critically, the *generation–verification gap* should widen along this gradient: the difference between a model’s ability to produce solutions and its ability to evaluate them should increase as domains become less deterministic.

## Experiment 3 — Persistent Agent Comparison

Compare stateless language models with **persistent agent architectures** possessing:

- memory across episodes
- explicit objectives
- environmental feedback

**Prediction:**

Persistent agents should show improved **calibration and novelty detection**, reflecting partial emergence of self-state through architectural scaffolding. This prediction is partially supported by the improved performance of Reflexion-style agents (Shinn et al., 2023) and multi-agent debate systems (Du et al., 2024) on tasks where stateless models fail.

## Experiment 4 — Self-State Stress Test

Introduce tasks requiring reasoning monitoring:

- proof verification
- solution checking
- error detection in presented arguments

**Prediction:**

Models without self-state will perform significantly worse on **verification tasks than generation tasks** of equivalent difficulty. This asymmetry, documented by Stechly et al. (2025) and Huang et al. (2024), should be most pronounced in non-deterministic domains where external comparison signals are unavailable.

-----

# Implications for AI Architecture

If APH is correct, the next generation of AI systems would benefit from architectures that implement the full abstraction hierarchy, including intrinsic self-state. In practical terms, this suggests:

- **Persistent working memory** that maintains reasoning state across extended inference
- **Internal verification modules** that evaluate reasoning quality without external tools
- **Difficulty estimation** mechanisms that detect novelty and adjust confidence
- **Self-models of competence** that represent the system’s own capabilities and limitations

Recent developments in test-time compute scaling (Snell et al., 2025) and reasoning-focused training (OpenAI, 2024) represent steps toward architectural self-state but remain dependent on externally constructed verification signals. APH predicts that intrinsic self-state—systems that can evaluate their own reasoning without environmental feedback—would produce qualitatively different capabilities, particularly in open-ended domains where external verification is unavailable.

This is essentially a **cognitive control layer** operating over the transformer’s abstraction engine. The key prediction is that the next major capability advance in AI will come not from scaling the abstraction engine (larger transformers) but from adding a genuine self-state architecture on top of it.

-----

# Conclusion

Deterministic domains allow reasoning systems to approximate metacognitive regulation through external verification. This property explains the disproportionate success of contemporary artificial intelligence systems in mathematics, programming and formal reasoning.

However, intelligence in open-ended environments requires the ability to monitor and regulate abstraction under conditions of novelty and uncertainty.

Within the Abstraction Primitive Hypothesis, this capability corresponds to **self-state**.

The generation–verification gap documented across contemporary AI systems (Stechly et al., 2025; Huang et al., 2024; Kamoi et al., 2024), the calibration failures observed on novel inputs (Kadavath et al., 2022; Xiong et al., 2024), and the dependence of self-correction on external feedback all converge on a single architectural explanation: current systems implement powerful abstraction machinery (Stages 1–3) without the self-monitoring mechanism (Stage 4) needed to regulate that machinery under novel conditions.

Determining whether artificial systems can develop such mechanisms—and what architectural properties are required—represents a central challenge in the scientific study of intelligence.

-----

# References

Baddeley, A. (2000). The episodic buffer: A new component of working memory? *Trends in Cognitive Sciences*, *4*(11), 417–423.

Cowan, N. (2001). The magical number 4 in short-term memory: A reconsideration of mental storage capacity. *Behavioral and Brain Sciences*, *24*(1), 87–114.

Danan [Levinson], H. J. (2021). *The neural representation of abstract concepts in typical and atypical cognition* [Doctoral dissertation, Rutgers University–Newark].

Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2024). Improving factuality and reasoning in language models through multiagent debate. In *Proceedings of the 41st International Conference on Machine Learning (ICML 2024)*, *Proceedings of Machine Learning Research*, *235*, 11733–11763. PMLR.

Flavell, J. H. (1979). Metacognition and cognitive monitoring: A new area of cognitive–developmental inquiry. *American Psychologist*, *34*(10), 906–911.

Fodor, J. A. (1975). *The Language of Thought*. Harvard University Press.

Fodor, J. A., & Pylyshyn, Z. W. (1988). Connectionism and cognitive architecture: A critical analysis. *Cognition*, *28*(1–2), 3–71.

Friston, K. (2010). The free-energy principle: A unified brain theory? *Nature Reviews Neuroscience*, *11*(2), 127–138.

Geng, J., Cai, F., Wang, Y., Koeppl, H., Nakov, P., & Gurevych, I. (2024). A survey of confidence estimation and calibration in large language models. In *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL 2024)*, 6577–6595. Association for Computational Linguistics.

Gentner, D. (1983). Structure-mapping: A theoretical framework for analogy. *Cognitive Science*, *7*(2), 155–170.

Halford, G. S., Wilson, W. H., & Phillips, S. (1998). Processing capacity defined by relational complexity: Implications for comparative, developmental, and cognitive psychology. *Behavioral and Brain Sciences*, *21*(6), 803–831.

Hauser, M. D., Chomsky, N., & Fitch, W. T. (2002). The faculty of language: What is it, who has it, and how did it evolve? *Science*, *298*(5598), 1569–1579.

Huang, J., Chen, X., Mishra, S., Zheng, H. S., Yu, A. W., Song, X., & Zhou, D. (2024). Large language models cannot self-correct reasoning yet. In *The Twelfth International Conference on Learning Representations (ICLR 2024)*.

Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain, D., Perez, E., Schiefer, N., Hatfield-Dodds, Z., DasSarma, N., Tran-Johnson, E., Johnston, S., El-Showk, S., Jones, A., Elhage, N., Hume, T., Chen, A., Bai, Y., Bowman, S., Fort, S., Ganguli, D., Hernandez, D., Jacobson, J., Kernion, J., Kravec, S., Lovitt, L., Ndousse, K., Olsson, C., Ringer, S., Amodei, D., Brown, T., Clark, J., Joseph, N., Mann, B., McCandlish, S., Olah, C., & Kaplan, J. (2022). Language models (mostly) know what they know. *arXiv preprint arXiv:2207.05221*.

Kambhampati, S., Valmeekam, K., Guan, L., Verma, M., Stechly, K., Bhambri, S., Saldyt, L. P., & Murthy, A. B. (2024). Position: LLMs can’t plan, but can help planning in LLM-Modulo frameworks. In *Proceedings of the 41st International Conference on Machine Learning (ICML 2024)*, *Proceedings of Machine Learning Research*, *235*, 22895–22907. PMLR.

Kamoi, R., Zhang, Y., Zhang, N., Han, J., & Zhang, R. (2024). When can LLMs actually correct their own mistakes? A critical survey of self-correction of LLMs. *Transactions of the Association for Computational Linguistics*, *12*, 1417–1440.

Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., & Amodei, D. (2020). Scaling laws for neural language models. *arXiv preprint arXiv:2001.08361*.

Kuhn, D. (2000). Metacognitive development. *Current Directions in Psychological Science*, *9*(5), 178–181.

Lake, B. M., & Baroni, M. (2018). Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. In *Proceedings of the 35th International Conference on Machine Learning (ICML 2018)*, *Proceedings of Machine Learning Research*, *80*, 2873–2882. PMLR.

Li, M., Kudugunta, S., & Zettlemoyer, L. (2025). (Mis)Fitting: A survey of scaling laws. In *The Thirteenth International Conference on Learning Representations (ICLR 2025)*.

Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., & Cobbe, K. (2024). Let’s verify step by step. In *The Twelfth International Conference on Learning Representations (ICLR 2024)*.

Maturana, H. R., & Varela, F. J. (1980). *Autopoiesis and Cognition: The Realization of the Living*. D. Reidel.

OpenAI. (2024). OpenAI o1 system card. *arXiv preprint arXiv:2412.16720*.

Piaget, J. (1952). *The Origins of Intelligence in Children*. International Universities Press.

Sevilla, J., Besiroglu, T., Cottier, B., You, J., Roldán, E., Villalobos, P., & Erdil, E. (2024). Can AI scaling continue through 2030? *Epoch AI Technical Report*.

Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., & Yao, S. (2023). Reflexion: Language agents with verbal reinforcement learning. In *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*.

Snell, C., Lee, J., Xu, K., & Kumar, A. (2025). Scaling LLM test-time compute optimally can be more effective than scaling model parameters. In *The Thirteenth International Conference on Learning Representations (ICLR 2025)* [Oral].

Stechly, K., Valmeekam, K., & Kambhampati, S. (2025). On the self-verification limitations of large language models on reasoning and planning tasks. In *The Thirteenth International Conference on Learning Representations (ICLR 2025)*.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*, 6000–6010.

Villalobos, P., Ho, A., Sevilla, J., Besiroglu, T., Heim, L., & Hobbhahn, M. (2024). Position: Will we run out of data? Limits of LLM scaling based on human-generated data. In *Proceedings of the 41st International Conference on Machine Learning (ICML 2024)*, *Proceedings of Machine Learning Research*, *235*, 49523–49544. PMLR.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q. V., & Zhou, D. (2022). Chain-of-thought prompting elicits reasoning in large language models. In *Advances in Neural Information Processing Systems 35 (NeurIPS 2022)*, 24824–24837.

Xiong, M., Hu, Z., Lu, X., Li, Y., Fu, J., He, J., & Hooi, B. (2024). Can LLMs express their uncertainty? An empirical evaluation of confidence elicitation in LLMs. In *The Twelfth International Conference on Learning Representations (ICLR 2024)*.

Xu, Q., Peng, Y., Nastase, S. A., Chodorow, M., Wu, M., & Li, P. (2025). Large language models without grounding recover non-sensorimotor but not sensorimotor features of human concepts. *Nature Human Behaviour*, *9*(9), 1871–1886.
