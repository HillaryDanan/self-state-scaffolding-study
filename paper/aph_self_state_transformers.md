# Abstraction, Self-State and the Limits of Predictive Artificial Intelligence

## Why Transformers Work — and Why Their Limits Appear Where They Do

Hillary Danan, PhD

---

# Abstract

Large language models (LLMs) based on transformer architectures have recently achieved remarkable performance in domains such as programming, mathematics and formal reasoning. These successes have prompted speculation that predictive language models may be approaching general intelligence. However, performance across cognitive domains remains uneven: systems that excel in deterministic reasoning often exhibit brittle failures in novelty detection, social inference and calibrated uncertainty.

This article proposes a parsimonious explanation grounded in the **Abstraction Primitive Hypothesis (APH)**. APH posits that intelligence develops through progressively deeper forms of abstraction culminating in **self-state**—a computational mechanism enabling systems to monitor and regulate their own reasoning.

Deterministic domains provide unusually strong external signals of correctness. These signals allow systems lacking intrinsic self-state to approximate metacognitive regulation through environmental verification. We argue that this property explains the disproportionate success of contemporary AI systems in deterministic domains and predicts systematic failure patterns as verification signals weaken.

Mapping APH onto transformer architectures reveals a striking structural correspondence: transformers efficiently implement the first three stages of abstraction—pattern extraction, symbol formation and recursive composition—while lacking intrinsic mechanisms corresponding to Stage 4 self-state. This relationship explains both the strengths and limitations of current AI systems and suggests that the next advances in artificial intelligence may depend on architectures capable of genuine self-regulation.

---

# Introduction

Artificial intelligence systems based on transformer architectures have recently demonstrated impressive performance across a wide range of formal reasoning tasks. Models trained through large-scale next-token prediction can now generate working software, perform multi-step symbolic reasoning and solve advanced mathematical problems.

Despite these achievements, the distribution of performance across cognitive domains remains highly uneven. Systems that perform strongly on deterministic reasoning benchmarks frequently struggle with:

* calibrated uncertainty
* novelty detection
* figurative language
* social reasoning
* open-ended causal inference

This pattern raises a fundamental question: **why do artificial systems exhibit strong competence in some domains while remaining fragile in others?**

The **Abstraction Primitive Hypothesis (APH)** offers a potential explanation. APH proposes that intelligence emerges through a hierarchy of abstraction capacities culminating in **self-state**, a mechanism that enables systems to monitor and regulate their own reasoning under conditions of novelty.

In this framework, deterministic domains possess a critical property: they provide explicit signals of correctness that allow reasoning systems to detect and correct errors through environmental feedback. Such domains therefore allow systems lacking intrinsic self-monitoring to approximate reliable reasoning.

This perspective yields a parsimonious explanation for the observed distribution of AI capabilities and suggests a principled framework for understanding the limits of current architectures.

---

# The Abstraction Hierarchy

APH proposes four stages of abstraction capacity.

| Stage   | Capacity              | Function                                 |
| ------- | --------------------- | ---------------------------------------- |
| Stage 1 | Pattern extraction    | Detect statistical regularities          |
| Stage 2 | Symbol formation      | Represent discrete combinable structures |
| Stage 3 | Recursive composition | Construct hierarchical abstractions      |
| Stage 4 | Self-state            | Monitor and regulate reasoning           |

Stages 1–3 support the **construction of abstractions**.
Stage 4 introduces **regulation of abstraction itself**.

---

# Figure 1 | The Abstraction Hierarchy

Pattern Extraction
↓
Symbol Formation
↓
Recursive Composition
↓
Self-State
(metacognitive regulation)

Lower stages construct representations.
Self-state regulates their reliability.

---

# Self-State as a Computational Operation

Self-state can be operationalized through a feedback loop:

MAINTAIN(x)
↓
COMPARE(x, y)
↓
UPDATE(x | result)

Where:

MAINTAIN
retain a representation across reasoning steps

COMPARE
evaluate ongoing processing relative to stored representations

UPDATE
modify representations in response to discrepancies

This loop resembles feedback control systems in engineering and the central executive component of working memory models.

---

# Deterministic Domains as Cognitive Scaffolds

Certain domains possess an unusual property: **explicit correctness criteria**.

Examples include:

| Domain      | Verification mechanism  |
| ----------- | ----------------------- |
| Programming | compilation and tests   |
| Mathematics | formal derivation       |
| Logic       | proof consistency       |
| Physics     | dimensional constraints |

These domains enable iterative reasoning correction:

generate → test → revise

The critical property is that the **test stage is supplied by the domain itself**.

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

Deterministic domains supply the **comparison signal externally**.

---

# External Verification as a Substitute for Self-State

Within the APH framework, deterministic domains effectively provide the **COMPARE** step of the self-state loop.

Instead of performing:

MAINTAIN → COMPARE → UPDATE

the system performs:

MAINTAIN (model)
COMPARE (environment)
UPDATE (model)

This structure allows systems lacking intrinsic self-state to approximate metacognitive regulation.

Consequently, systems possessing only Stages 1–3 abstraction capacity can appear highly capable in deterministic environments.

---

# Transformers as Partial Abstraction Systems

Transformer architectures display a striking correspondence with the APH hierarchy.

---

# Figure 3 | Mapping APH to Transformer Architecture

APH Stage 1 — Pattern Extraction
→ Transformer embeddings (statistical feature compression)

APH Stage 2 — Symbol Formation
→ Token representations (discrete symbolic units)

APH Stage 3 — Recursive Composition
→ Attention layers (relational composition)

APH Stage 4 — Self-State
→ largely absent

Transformers efficiently implement the first three stages of abstraction.

---

# Why Transformers Work

Transformers excel in domains characterized by:

* discrete symbolic elements
* compositional structure
* recursive relations

Language, programming languages and formal mathematics possess precisely these properties.

| Domain           | Structural features             |
| ---------------- | ------------------------------- |
| Natural language | compositional syntax            |
| Programming      | symbolic grammar                |
| mathematics      | recursive symbolic manipulation |
| logic            | rule-governed inference         |

These domains are therefore naturally aligned with transformer architectures.

---

# Why Strengths Cluster Where They Do

APH predicts that systems implementing Stages 1–3 abstraction will perform best in **symbolically structured domains**.

These include:

* language
* programming
* formal mathematics

These domains are characterized by:

* discrete representations
* compositional syntax
* hierarchical structure

These properties allow transformers to exploit their strengths in statistical pattern extraction and relational composition.

---

# Why Limitations Appear Where They Do

Domains requiring **intrinsic reasoning regulation** expose the absence of Stage 4 self-state.

Examples include:

* novelty detection
* calibrated uncertainty
* social inference
* open-ended causal reasoning

In such contexts, the system must determine whether its reasoning is reliable **without external verification signals**.

---

# Figure 4 | Verification Gradient Across Domains

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

APH predicts distinct failure patterns when self-state is absent.

| Signature         | Self-state systems         | Pattern-matching systems |
| ----------------- | -------------------------- | ------------------------ |
| Novelty detection | confidence decreases       | confidence unchanged     |
| Error profile     | conservative uncertainty   | confident confabulation  |
| Calibration       | confidence tracks accuracy | weak correlation         |
| Capacity limits   | gradual degradation        | abrupt failure           |

These patterns resemble several widely observed behaviors of contemporary LLM systems.

---

# Architectural Self-State

Modern AI systems often compensate for limited intrinsic self-state through architectural mechanisms such as:

* code execution environments
* verification tools
* iterative retry loops
* reinforcement learning feedback

These mechanisms approximate **architectural self-state**, even if the underlying model lacks intrinsic metacognitive control.

---

# Predictions

The framework yields several empirical predictions.

Prediction 1
Performance should decline as external verification signals weaken.

Prediction 2
Confidence–accuracy correlation should decrease as task novelty increases.

Prediction 3
Persistent agent architectures possessing memory, objectives and environmental interaction may exhibit stronger calibration signatures than stateless language models.

---

# Conclusion

Deterministic domains provide an environment in which reasoning systems can approximate metacognitive regulation through external verification signals. This property explains the disproportionate success of contemporary artificial intelligence systems in mathematics, programming and formal reasoning.

However, intelligence in open-ended environments requires more than the construction of abstractions. It requires the ability to monitor and regulate abstraction under conditions of novelty and uncertainty.

Within the Abstraction Primitive Hypothesis, this capability corresponds to **self-state**.

Determining whether artificial systems can develop such mechanisms represents a central challenge in the scientific study of intelligence.

---

# References

Baddeley, A. (2000). The episodic buffer: A new component of working memory. *Trends in Cognitive Sciences.*

Flavell, J. (1979). Metacognition and cognitive monitoring. *American Psychologist.*

Fodor, J. (1975). *The Language of Thought.*

Fodor, J., & Pylyshyn, Z. (1988). Connectionism and cognitive architecture. *Cognition.*

Friston, K. (2010). The free-energy principle. *Nature Reviews Neuroscience.*

Xu, Q. et al. (2025). Large language models without grounding recover non-sensorimotor concept features. *Nature Human Behaviour.*

Danan, H. (2021). *The neural representation of abstract concepts in typical and atypical cognition.* Rutgers University–Newark.
