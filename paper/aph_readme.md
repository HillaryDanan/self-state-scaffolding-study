# Abstraction Primitive Hypothesis (APH)

**A minimal research program for intelligence in brains and AI**
**Hillary Danan, PhD**

---

## Thesis

The **Abstraction Primitive Hypothesis (APH)** proposes that intelligence emerges from the recursive interaction of **symbol formation** and **compositional structure**.

On this view, systems become more intelligent not simply by storing more patterns, but by:

- forming reusable symbolic units,
- composing those units into higher-order structures,
- and recursively refining those structures over time.

APH further proposes that a distinct higher-order mechanism, **self-state**, may be required for reliable abstraction under novelty.

**Self-state** is defined here as a computational operation that:

- maintains internal representations,
- compares them against new information or generated outputs,
- and updates them accordingly.

The central claim is:

> **Intelligence is the capacity to form, compose, recursively revise, and stabilize abstractions in ways that support coherent behavior across novel conditions.**

---

## Core Definitions

### 1. Abstraction

Abstraction is the recursive process by which a system:

1. extracts structure from input,
2. forms discrete reusable representations,
3. composes them into higher-order representations,
4. refines those representations through feedback.

It is not compression alone, and not composition alone, but their iterative interaction.

### 2. Developmental Stages of Abstraction

| Stage | Capacity | Function |
| --- | --- | --- |
| **1. Pattern Extraction** | Detect regularities | Compression, prediction |
| **2. Symbol Formation** | Create discrete recombinable units | Compositional generalization |
| **3. Recursive Composition** | Build abstractions from abstractions | Hierarchy, analogy, planning |
| **4. Self-State** | Monitor and regulate internal processing | Verification, calibration, metacognition |

**Stages 1–3 construct abstractions. Stage 4 regulates them.**

### 3. Self-State

Self-state is operationalized as:

```text
MAINTAIN(x) → COMPARE(x, y) → UPDATE(x | result)
```

Where:

- **MAINTAIN** = hold a representation active across steps
- **COMPARE** = evaluate it against incoming information, expectations, or outputs
- **UPDATE** = revise it based on the comparison

Self-state is a proposed **mechanism**, not merely a report of confidence or the use of self-referential language.

### 4. Self/World Distinction

APH hypothesizes that robust self-state may require a principled distinction between:

- what belongs to the system's own maintained state,
- and what belongs to the external world.

A substrate-neutral version of this claim is that self-state is more likely in systems with:

- **bounded persistence**,
- **stakes**,
- and an **inside/outside asymmetry**.

This remains a **hypothesis**, not an established fact.

---

## Diagram

```text
[Raw Input]
     ↓
[Pattern Extraction]
     ↓
[Symbol Formation]
     ↓
[Symbols]
     ↓
[Recursive Composition]
     ↓
[Composed Structures]
     ↓
[Self-State: maintain / compare / update]
     ↑
     └──────────── feedback / revision ────────────┘
```

A simpler formulation:

```text
pattern extraction → symbol formation → recursive composition → self-state
```

---

## Mapping APH to Current AI

APH suggests that many current large models strongly implement the lower stages:

| APH Stage | AI Analogue |
| --- | --- |
| Pattern Extraction | Statistical learning, embedding formation |
| Symbol Formation | Token or feature representations, latent concepts |
| Recursive Composition | Attention, multi-step reasoning, planning traces |
| Self-State | Weak or externally scaffolded |

On this view, transformer systems are especially strong in domains where external structure supports Stages 1–3:

- language
- programming
- mathematics
- formal reasoning

They are weaker where success depends on intrinsic regulation under novelty:

- calibrated uncertainty
- open-ended verification
- social inference
- robust self-correction without external feedback

---

## Predictions

APH makes five broad predictions.

### 1. Generation–Verification Asymmetry

Systems lacking robust self-state should be better at **generating** candidate solutions than at **verifying** them.

### 2. Verification Gradient Across Domains

Performance should be strongest in domains with strong external correctness signals, such as math or code, and weaker in domains with ambiguous or weak verification, such as commonsense or social reasoning.

### 3. Calibration Breakdown Under Novelty

A system may appear calibrated on familiar tasks yet fail to align confidence with accuracy on genuinely novel tasks.

### 4. Scaling Improves Stages 1–3 More Than Stage 4

Increasing model size, data, and compute should improve pattern extraction, symbol formation, and recursive composition more reliably than intrinsic self-state.

### 5. Persistent Agents May Show Stronger Self-State Signatures

Systems with persistent memory, explicit objectives, and feedback across time may show stronger novelty detection and calibration than stateless systems.

---

## Falsifiable Tests

APH is useful only if it can be tested and potentially falsified.

### Test 1: Novelty Calibration

Construct tasks outside likely training distributions, such as synthetic operators or randomized formal systems.

**Measure:**

- task accuracy
- confidence estimates

**APH prediction:**
Systems without robust self-state will show weak confidence–accuracy correlation on novelty, even if they perform well on familiar tasks.

**Would count against APH:**
A stateless predictive model showing strong calibration on genuinely novel tasks without external verification.

### Test 2: Generation vs Verification

Compare a model's ability to solve problems with its ability to evaluate candidate solutions of comparable difficulty.

**APH prediction:**
Verification should lag generation, especially in non-deterministic domains.

**Would count against APH:**
Verification matching or exceeding generation in open-ended domains without external scaffolds.

### Test 3: Verification Gradient

Evaluate the same system across domains ordered by strength of external correctness criteria:

1. formal mathematics
2. programming
3. structured reasoning
4. commonsense reasoning
5. social inference

**APH prediction:**
The generation–verification gap should widen as external verification weakens.

**Would count against APH:**
No systematic relationship between external verifiability and verification performance.

### Test 4: Persistent vs Stateless Systems

Compare standard stateless LLMs against persistent agentic systems with memory, feedback, and objective continuity.

**APH prediction:**
Persistent systems should show better novelty sensitivity, calibration, and adaptive correction.

**Would count against APH:**
No measurable advantage from persistence, stakes, or maintained internal state.

### Test 5: Biological Parallel

In humans and other biological systems, Stage 4-like capacities should mature later than Stages 1–3 and should depend more strongly on working memory and prefrontal control.

**APH prediction:**
Tasks requiring self-monitoring should show later developmental emergence and higher sensitivity to working-memory load than tasks requiring only pattern extraction or composition.

**Would count against APH:**
Robust self-monitoring emerging independently of working memory or developmental control systems.

---

## Why This Matters

APH offers a unifying framework for linking:

- abstraction
- reasoning
- memory
- verification
- metacognition
- AI architecture

Instead of treating calibration, self-correction, planning, and introspection as separate add-ons, APH treats them as different aspects of one broader problem:

> **How systems build and regulate abstractions over time.**

This framework suggests that the next major advances in AI may depend not only on scaling existing abstraction engines, but on building systems with stronger mechanisms for:

- persistent internal state,
- comparison against expectations,
- adaptive revision,
- calibrated self-monitoring under novelty.

---

## Minimal Conclusion

APH proposes that intelligence is not just better prediction. It is the organized capacity to create, compose, revise, and stabilize abstractions.

Current AI systems may already exhibit strong forms of abstraction in Stages 1–3. The key open question is whether systems can develop **Stage 4 self-state**: an intrinsic capacity to monitor and regulate their own reasoning under novel conditions.

That question is empirical. The value of APH is that it turns it into a concrete research program.
