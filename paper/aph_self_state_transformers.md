# Abstraction, Self-State and the Limits of Predictive Artificial Intelligence

## Why Transformers Work — and Why Their Limits Appear Where They Do

Hillary Danan, PhD

---

# Abstract

Large language models based on transformer architectures have recently achieved remarkable performance in domains such as programming, mathematics and formal reasoning. These successes have prompted speculation that predictive language models may be approaching general intelligence. However, performance across cognitive domains remains uneven: systems that excel in deterministic reasoning often exhibit brittle failures in novelty detection, social inference and calibrated uncertainty.

This article proposes a testable explanation grounded in the **Abstraction Primitive Hypothesis (APH)**. APH posits that intelligence develops through progressively deeper forms of abstraction culminating in **self-state**—a computational mechanism enabling systems to monitor and regulate their own reasoning.

Deterministic domains provide unusually strong external signals of correctness. These signals allow systems lacking intrinsic self-state to approximate metacognitive regulation through environmental verification. We argue that this property explains the disproportionate success of contemporary AI systems in deterministic domains and predicts systematic failure patterns as verification signals weaken.

Mapping APH onto transformer architectures reveals a structural correspondence: transformers efficiently implement the first three stages of abstraction—pattern extraction, symbol formation and recursive composition—while lacking intrinsic mechanisms corresponding to Stage 4 self-state. This relationship explains both the strengths and limitations of current AI systems and suggests that the next advances in artificial intelligence may depend on architectures capable of genuine self-regulation.

---

# Introduction

Recent progress in artificial intelligence has been driven by large language models trained using transformer architectures. These systems demonstrate strong performance in domains such as programming, mathematics and formal reasoning, yet exhibit persistent weaknesses in novelty detection, calibrated uncertainty and social inference.

This uneven performance distribution raises a scientific question:

**What architectural properties determine where artificial reasoning systems succeed and where they fail?**

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

* transformer systems should excel in symbolically structured domains
* scaling should improve abstraction capacity but not self-state
* reasoning failures should cluster in domains requiring intrinsic regulation

The remainder of this paper develops these hypotheses and derives empirically testable predictions.

---

# The Abstraction Hierarchy

APH proposes four stages of abstraction capacity.

| Stage   | Capacity              | Function                                 |
| ------- | --------------------- | ---------------------------------------- |
| Stage 1 | Pattern extraction    | Detect statistical regularities          |
| Stage 2 | Symbol formation      | Represent discrete combinable structures |
| Stage 3 | Recursive composition | Construct hierarchical abstractions      |
| Stage 4 | Self-state            | Monitor and regulate reasoning           |

Stages 1–3 construct abstractions.
Stage 4 regulates them.

---

# Figure 1 | The Abstraction Hierarchy

Pattern Extraction
↓
Symbol Formation
↓
Recursive Composition
↓
Self-State (metacognitive regulation)

Lower stages construct representations.
Self-state regulates their reliability.

---

# Self-State as a Computational Operation

Self-state can be formalized as a feedback loop.

MAINTAIN(x)
↓
COMPARE(x, y)
↓
UPDATE(x | result)

Where:

MAINTAIN — retain representations across reasoning steps
COMPARE — evaluate processing relative to expectations
UPDATE — modify representations in response to discrepancies

This structure resembles feedback control systems and the **central executive** component of working memory models.

---

# Deterministic Domains as Cognitive Scaffolds

Certain domains possess explicit correctness criteria.

| Domain      | Verification mechanism  |
| ----------- | ----------------------- |
| Programming | compilation and tests   |
| Mathematics | formal derivation       |
| Logic       | proof consistency       |
| Physics     | dimensional constraints |

These domains support the loop:

generate → test → revise

The **test step is provided by the domain itself**.

---

# Figure 2 | External Verification Loop

Model reasoning
↓
Candidate solution
↓
External verifier
↓
Error signal
↓
Revision

Deterministic domains provide the comparison signal externally.

---

# External Verification as a Substitute for Self-State

Instead of performing

MAINTAIN → COMPARE → UPDATE

systems operating in deterministic domains perform

MAINTAIN (model)
COMPARE (environment)
UPDATE (model)

Deterministic domains therefore partially substitute for intrinsic self-state.

---

# Transformers as Partial Abstraction Systems

Transformer architectures map naturally onto the first three abstraction stages.

---

# Figure 3 | Mapping APH to Transformer Architecture

APH Stage 1 — Pattern Extraction
→ Embedding layers (statistical compression)

APH Stage 2 — Symbol Formation
→ Token representations (discrete symbolic units)

APH Stage 3 — Recursive Composition
→ Attention layers (relational composition)

APH Stage 4 — Self-State
→ largely absent

---

# Why Transformers Work

Transformers perform well in domains characterized by:

* discrete symbolic representations
* compositional syntax
* recursive structure

| Domain           | Structural properties           |
| ---------------- | ------------------------------- |
| Natural language | compositional grammar           |
| Programming      | symbolic syntax                 |
| Mathematics      | recursive symbolic manipulation |
| Logic            | rule-governed inference         |

These domains align naturally with Stages 1–3 abstraction.

---

# Why Language Became the First Domain of General AI

Language possesses several properties that make it uniquely suitable for large-scale abstraction learning.

1. Massive availability of training data
2. Strong compositional structure
3. High-level conceptual content
4. Symbolic representation of knowledge

Language therefore provides an unusually efficient substrate for **Stages 1–3 abstraction learning**.

Domains such as robotics require grounded perception, sensorimotor interaction and sparse feedback signals, making large-scale abstraction learning significantly more difficult.

Within APH, language therefore represents the domain in which large-scale abstraction learning first becomes computationally tractable.

---

# Scaling Laws and Abstraction Growth

Large language models exhibit scaling laws: performance improves predictably with model size, data and compute.

Within APH, scaling primarily enhances the first three stages of abstraction.

Scaling improves:

* pattern extraction resolution
* stability of symbolic representations
* depth of recursive composition

---

# Figure 4 | Scaling Under the Abstraction Hierarchy

Model scale ↑

Stage 1 improves
Stage 2 stabilizes
Stage 3 deepens

---

Stage 4 self-state
(not improved by scale alone)

---

# Why Scaling Plateaus

Scaling increases abstraction capacity but does not automatically produce **self-state regulation**.

Self-state requires mechanisms enabling systems to:

* maintain internal reasoning state
* compare reasoning outcomes
* update strategies

These processes require architectural innovations beyond parameter scaling.

---

# Figure 5 | Verification Gradient Across Domains

High verification

Formal mathematics
Programming
Structured reasoning

Commonsense reasoning

Social inference
Metacognition

Low verification

Reliance on intrinsic self-state increases as verification decreases.

---

# Failure Signatures

When self-state is absent:

| Signature         | Self-state systems         | Pattern-matching systems |
| ----------------- | -------------------------- | ------------------------ |
| Novelty detection | confidence decreases       | confidence unchanged     |
| Error profile     | conservative uncertainty   | confident confabulation  |
| Calibration       | confidence tracks accuracy | weak correlation         |
| Capacity limits   | gradual degradation        | abrupt failure           |

These patterns resemble widely observed behaviors of contemporary LLM systems.

---

# Architectural Self-State

Modern AI systems approximate self-state through architectural mechanisms such as:

* code execution environments
* verification tools
* retry loops
* reinforcement learning feedback

These mechanisms create **architectural self-state**, even when intrinsic metacognitive regulation is absent.

---

# Figure 6 | Unified APH–Transformer–Scaling Landscape

```
          SELF-STATE (Stage 4)
    metacognitive regulation layer
                   │
                   │
    ┌──────────────┴──────────────┐
    │                             │
```

Recursive Composition        External Verification
(Stage 3 abstraction)        (deterministic domains)
│
Symbol Formation
│
Pattern Extraction

Transformers implement the lower abstraction stack.
Deterministic domains provide external verification.
General intelligence likely requires intrinsic self-state.

---

# Empirical Research Program

The Abstraction Primitive Hypothesis generates a set of experimentally testable predictions.

## Experiment 1 — Novelty Calibration Test

Construct reasoning tasks that fall outside known training distributions (e.g., randomized operators or synthetic formal systems).

Measure:

* accuracy
* confidence estimates

Prediction:

Systems lacking self-state will show **weak correlation between confidence and accuracy** on genuinely novel tasks.

---

## Experiment 2 — Verification Gradient Study

Evaluate reasoning performance across domains with varying verification strength.

Domains:

1. mathematics
2. programming
3. structured reasoning
4. commonsense reasoning
5. social reasoning

Prediction:

Performance should decline systematically as external verification decreases.

---

## Experiment 3 — Persistent Agent Comparison

Compare stateless language models with **persistent agent architectures** possessing:

* memory across episodes
* explicit objectives
* environmental feedback

Prediction:

Persistent agents should show improved **calibration and novelty detection**, reflecting partial emergence of self-state.

---

## Experiment 4 — Self-State Stress Test

Introduce tasks requiring reasoning monitoring:

* proof verification
* solution checking
* error detection

Prediction:

Models without self-state will perform significantly worse on **verification tasks than generation tasks**.

---

# Conclusion

Deterministic domains allow reasoning systems to approximate metacognitive regulation through external verification. This property explains the disproportionate success of contemporary artificial intelligence systems in mathematics, programming and formal reasoning.

However, intelligence in open-ended environments requires the ability to monitor and regulate abstraction under conditions of novelty and uncertainty.

Within the Abstraction Primitive Hypothesis, this capability corresponds to **self-state**.

Determining whether artificial systems can develop such mechanisms represents a central challenge in the scientific study of intelligence.

---

# References

Baddeley, A. (2000). The episodic buffer: A new component of working memory. *Trends in Cognitive Sciences.*

Flavell, J. (1979). Metacognition and cognitive monitoring. *American Psychologist.*

Jerry Fodor (1975). *The Language of Thought.*

Fodor, J., & Pylyshyn, Z. (1988). Connectionism and cognitive architecture. *Cognition.*

Karl Friston (2010). The free-energy principle. *Nature Reviews Neuroscience.*

Xu, Q. et al. (2025). Large language models without grounding recover non-sensorimotor concept features. *Nature Human Behaviour.*

Danan, H. (2021). *The neural representation of abstract concepts in typical and atypical cognition.* Rutgers University–Newark.
