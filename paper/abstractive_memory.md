# Abstractive Memory Architectures for Large Language Models: A Cognitive Neuroscience Perspective

**Hillary Danan, PhD**
Independent Researcher, AI Safety and Cognitive Architecture

---

## Abstract

Large language models (LLMs) face a fundamental bottleneck in persistent memory. Current approaches — context window expansion, retrieval-augmented generation (RAG), and parametric fine-tuning — store and retrieve information at the level of raw text or token-level embeddings. This stands in contrast to biological memory systems, which progressively compress experience into hierarchical, schematic, and increasingly abstract representations through well-characterized consolidation processes. Recent engineering efforts, including tiered memory systems (Packer et al., 2023), reflection-based architectures (Park et al., 2023), neurobiologically inspired retrieval (Gutiérrez et al., 2024), and tree-structured schemas (Rezazadeh et al., 2025), have begun to approximate aspects of biological consolidation with growing sophistication. However, no existing system integrates a full consolidation pipeline — from raw episodic encoding through relational binding, schematization, and metacognitive monitoring — within a unified theoretical framework specifying both representational targets and transition dynamics at each stage.

This paper argues that the LLM memory problem is fundamentally one of representational adequacy, not storage capacity or retrieval efficiency. Drawing on complementary learning systems theory, schema theory, reconsolidation dynamics, and recent computational models of memory consolidation (Spens & Burgess, 2024), I propose a framework for abstractive memory architectures — systems that store not raw information but structured representations organized hierarchically by level of compression and generalizability. I introduce a four-stage taxonomy of memory abstraction as a working hypothesis, map each stage onto known neural consolidation mechanisms with explicit attention to where the mapping is strong versus speculative, and derive testable predictions. I situate this framework within the rapidly expanding landscape of bio-inspired LLM memory systems, identify the specific gap it addresses, and confront directly the question of whether LLM-based consolidation can achieve genuine abstraction or merely lossy text compression — a distinction that recent empirical work has begun to operationalize (Ho et al., 2025; Zhao et al., 2024; Sarch et al., 2024).

**Epistemic status.** This paper is a theoretical perspective and research framework, not a report of empirical findings. Claims about biological memory are grounded in peer-reviewed neuroscience. Claims about current LLM limitations reflect published technical work. The proposed architecture represents a working hypothesis — theoretically motivated, empirically testable, but not yet validated. This distinction is maintained throughout.

### Key Points

- Current LLM memory mechanisms operate at the token or text-chunk level, lacking the progressive abstraction characteristic of biological memory consolidation.
- Recent systems (HippoRAG, MemTree, Larimar, EM-LLM, Generative Agents) have introduced neuroscience-inspired memory components, but each implements only a subset of the consolidation pipeline. No existing system integrates relational binding, hierarchical schematization, reconsolidation, and metacognitive monitoring within a single architecture.
- Biological memory systems compress experience through well-characterized stages: encoding, relational binding, schematization, and metacognitive monitoring — processes that are transformative, not merely storage operations.
- A four-stage taxonomy of memory abstraction (proposed here as a working hypothesis) offers a candidate framework for the representational targets at each consolidation phase. The correspondence to neuroscience is strongest for relational binding and schematization, moderate for episodic encoding, and most speculative for the metacognitive stage.
- The central challenge any abstractive memory architecture must confront is whether its consolidation engine achieves genuine structural abstraction or merely sophisticated summarization — a distinction that recent work on concept-level versus instance-level memory has begun to empirically adjudicate (Ho et al., 2025; Zhao et al., 2024).
- The framework generates testable predictions designed to discriminate it from both existing approaches and simpler alternatives.

---

## 1. Introduction

The capacity for persistent, flexible memory is a cornerstone of intelligent behavior. Biological agents encode experience, consolidate it over time, integrate it with existing knowledge, and retrieve it in context-sensitive ways that support adaptive action (Squire & Zola, 1996; Eichenbaum, 2004). This process is not mere storage and retrieval. It is fundamentally transformative: raw experience is progressively compressed into schematic, abstract representations that capture regularities while discarding incidental detail (Winocur & Moscovitch, 2011; Gilboa & Marlatte, 2017). Recent computational modeling has formalized this as a generative process in which hippocampal traces train neocortical generative models, producing the progressive episodic-to-semantic transformation observed empirically (Spens & Burgess, 2024).

Large language models, despite remarkable advances in reasoning, generation, and apparent comprehension, lack anything resembling this capacity. A model's "knowledge" is either frozen in its parameters at training time or temporarily available within a finite context window during inference. The field has recognized this limitation and produced a rapidly growing body of engineering solutions — retrieval-augmented generation (Lewis et al., 2020), extended context windows (Anthropic, 2024; Google, 2024), parameter-efficient fine-tuning (Hu et al., 2021), tiered memory systems (Packer et al., 2023), reflection-based agents (Park et al., 2023), neurobiologically grounded retrieval (Gutiérrez et al., 2024), and schema-inspired hierarchical storage (Rezazadeh et al., 2025). A comprehensive survey cataloguing over 100 such systems has recently been published (Hu et al., 2025).

This proliferation of approaches makes it essential to be precise about what gap remains. Current systems have individually addressed specific aspects of biological memory: HippoRAG (Gutiérrez et al., 2024) implements hippocampal pattern completion for associative retrieval; MemTree (Rezazadeh et al., 2025) implements hierarchical schematic organization; Larimar (Das et al., 2024) implements complementary learning systems dynamics for fast knowledge editing; EM-LLM (Fountas et al., 2025) implements surprise-based event segmentation for episodic memory. What no existing system does is integrate these components into a unified consolidation pipeline with (a) explicitly specified representational targets at each abstraction level, (b) principled transition dynamics governing when and how representations move between levels, (c) reconsolidation mechanics allowing schema updating through use, and (d) metacognitive indexing supporting confidence-calibrated retrieval.

This paper proposes such a framework. I introduce a four-stage taxonomy of memory abstraction — pattern encoding, relational binding, hierarchical schematization, and metacognitive monitoring — as a working hypothesis for the representational targets of an abstractive memory architecture. This taxonomy is derived from, and constrained by, established neuroscience of memory consolidation, but it is presented here as a novel theoretical proposal, not as a claim about established biological fact. The distinction between well-supported mappings and speculative extensions is maintained throughout.

**A note on scope and novelty.** The field of bio-inspired LLM memory has expanded rapidly since 2023, and this paper enters a crowded landscape. I acknowledge this explicitly and aim for a specific contribution: not the first proposal to draw on neuroscience for LLM memory design, but rather a framework that uniquely integrates the full consolidation pipeline — from encoding through metacognition — with explicit representational specifications at each stage and testable predictions designed to discriminate the framework from simpler alternatives. Section 2.5 and Section 8 provide detailed comparison with the most closely related systems.

---

## 2. Current Approaches to LLM Memory: Capabilities and Limitations

### 2.1 Context Windows

The most basic form of LLM "memory" is the context window: the sequence of tokens available to the model during a single inference pass. Early transformer architectures were limited to 2,048 tokens (Vaswani et al., 2017); current frontier models support windows of 128K to 1M+ tokens (Anthropic, 2024; Google, 2024).

Expanding context windows has proven useful but faces fundamental constraints. Computational cost scales quadratically with sequence length under standard self-attention (Vaswani et al., 2017), though efficient attention variants partially mitigate this (Dao et al., 2022). More critically, larger context windows do not solve the representation problem: all information is stored as raw tokens, and empirical work has demonstrated a U-shaped performance curve — degraded performance on information located in the middle of long contexts — suggesting that raw token storage does not produce reliable memory even within the supported window (Liu et al., 2024).

*Biological analogy and its limits.* The context window functions roughly as an externalized working memory buffer — a temporary store of currently active information (cf. Baddeley, 2000). However, biological working memory is actively maintained through sustained neural firing, selectively filtered by attention, and continuously integrated with long-term memory representations (Cowan, 2001; D'Esposito & Postle, 2015). LLM context windows lack this selective, integrative character. The analogy holds for the temporariness of the store but breaks down for the active maintenance and bidirectional interaction with long-term memory.

### 2.2 Retrieval-Augmented Generation (RAG)

RAG systems (Lewis et al., 2020) address context window limitations by maintaining an external vector database of text passages. At inference time, the model's query is embedded, similar passages are retrieved via approximate nearest-neighbor search, and retrieved text is injected into the context window.

RAG represents a meaningful advance: it decouples memory capacity from context window size and allows dynamic updating without retraining. However, RAG systems store and retrieve text chunks — typically 256–512 token passages embedded as dense vectors. The embedding captures semantic similarity at the passage level but does not represent the structure or abstraction level of the contained knowledge. A passage describing a specific event and a passage articulating a general principle are stored in the same format and retrieved by the same mechanism.

*Biological analogy and its limits.* RAG functions roughly as an external episodic memory with content-addressable retrieval — analogous to hippocampal pattern completion triggering recall of specific stored episodes (Eichenbaum, 2004). But biological episodic memory does not remain episodic. Through consolidation, specific episodes are transformed into generalized knowledge structures stored in neocortical networks (Frankland & Bontempi, 2005; Preston & Eichenbaum, 2013). RAG has no such consolidation process. Moreover, the hippocampus does not merely store episodes — it constructs relational maps representing the structure of experience (Eichenbaum, 2004; Behrens et al., 2018), a function absent from vector similarity search.

GraphRAG (Edge et al., 2024) partially addresses this limitation by building knowledge graphs from source documents and applying community detection algorithms to produce hierarchical summaries. This introduces graph structure into retrieval but operates as a static index-building step rather than a dynamic consolidation process — the graph is constructed once from source documents and does not evolve through interaction or experience.

### 2.3 Parametric Memory (Fine-Tuning)

Knowledge can be integrated into model parameters through fine-tuning — either full fine-tuning or parameter-efficient methods such as LoRA (Hu et al., 2021). This produces stable, persistent "memory" that does not require context window space. However, parametric memory is inflexible: it cannot be selectively updated, is prone to catastrophic forgetting of previously learned information (McCloskey & Cohen, 1989; French, 1999), and offers no mechanism for the model to distinguish the source, recency, or confidence level of stored knowledge.

*Biological analogy and its limits.* Parametric memory resembles neocortical long-term memory in its stability and generalization, but lacks the dynamic updating and source monitoring of biological systems (Moscovitch et al., 2016; Schacter et al., 1998). It also lacks reconsolidation dynamics — the finding that retrieved memories become labile and modifiable, allowing updating in light of new experience (Nader et al., 2000; Nader & Hardt, 2009). Parametric memory, once written, cannot be selectively revised without risking interference with other stored knowledge. CLS-inspired continual learning methods (Arani et al., 2022) partially address catastrophic forgetting by maintaining dual semantic memories with different consolidation rates, but these operate at the level of model parameters rather than explicit memory representations.

### 2.4 Tiered and Reflective Memory Systems

Recent work has introduced memory hierarchy and consolidation-like operations into LLM architectures, representing meaningful and increasingly sophisticated progress.

**MemGPT** (Packer et al., 2023) implements a tiered memory system with a main context (analogous to working memory), recall storage, and archival storage, with an LLM-controlled memory manager that moves information between tiers. This introduces multi-level storage with active management, but the stored representations at each tier remain text-based — the format does not change across tiers, only the accessibility.

**Generative Agents** (Park et al., 2023) introduces a memory stream architecture with a reflection mechanism that represents genuine abstraction progress. Periodically, the agent generates higher-level observations from accumulated episodic memories (e.g., "Klaus Mueller is writing a research paper" derived from multiple observations). Ablation studies demonstrated that reflection was critical to agent believability, and the reflection process generates multi-level "reflection trees" with increasing abstraction at each depth. This is a meaningful step toward biological consolidation. However, reflections are stored alongside raw observations in an undifferentiated memory stream — there is no episodic-to-semantic separation, no structured relational representation, no confidence metadata, and no forgetting mechanism. The system retains all raw observations indefinitely.

### 2.5 Neuroscience-Inspired Memory Architectures

Since 2023, a growing body of work has explicitly drawn on neuroscience to design LLM memory systems. Accurately situating the present contribution requires detailed engagement with these systems.

**HippoRAG** (Gutiérrez et al., 2024; NeurIPS 2024) operationalizes hippocampal indexing theory. An LLM extracts knowledge graph triples (neocortex analog), a parahippocampal region encoder detects synonymy across passages, and a knowledge graph with Personalized PageRank retrieval serves as the hippocampal index, mimicking pattern completion. HippoRAG demonstrates that neuroscience-grounded design can substantially improve retrieval — up to 20% gains on multi-hop QA over standard RAG. However, HippoRAG is a retrieval architecture, not a consolidation architecture: the knowledge graph is built during indexing and does not evolve through use or undergo progressive abstraction.

**Larimar** (Das et al., 2024; ICML 2024) pairs a slow-learning LLM with a fast-learning external episodic memory matrix, directly inspired by complementary learning systems theory (McClelland et al., 1995). The memory uses a distributed associative memory framework enabling one-shot write operations via pseudo-inverse computation. Larimar achieves gradient-free knowledge editing at inference time — a significant advance. However, its memory representations are fixed-dimensional latent vectors; there is no hierarchical abstraction or schema formation.

**EM-LLM** (Fountas et al., 2025; ICLR 2025) introduces surprise-based event segmentation using Bayesian surprise to detect event boundaries, graph-theoretic boundary refinement, and a two-stage retrieval mechanism inspired by the contiguity and asymmetry effects observed in human free recall. EM-LLM achieves retrieval across 10 million tokens and shows strong correlations between its event boundaries and human-annotated event structures. This is the most biologically grounded episodic memory system for LLMs, but it does not implement consolidation — episodes remain in their original form.

**MemTree** (Rezazadeh et al., 2025; ICLR 2025) implements a dynamic tree-structured memory inspired by cognitive schemas, where higher nodes store increasingly abstract summaries and leaf nodes store specific interaction details. New information is integrated by traversing the tree and dynamically restructuring. MemTree outperforms MemGPT on multiple benchmarks and provides O(log n) insertion complexity. Of all existing systems, MemTree most closely approximates the hierarchical schematization proposed in this framework. However, MemTree's abstraction is achieved through text summarization at each level, and the system does not include relational binding, reconsolidation, or metacognitive indexing.

**Titans** (Behrouz et al., 2025; NeurIPS 2025) introduces a neural long-term memory module distinct from standard attention. Memory updates are governed by a surprise-based mechanism: a per-token associative memory loss quantifies novelty, and its gradient drives memory storage via gradient descent with momentum (capturing both momentary and past surprise) and weight decay (adaptive forgetting). Titans scales beyond 2 million tokens and outperforms Transformers and state-space models. However, Titans operates at the architectural level — it modifies the model's internal computation rather than maintaining explicit, inspectable memory representations that can be audited or structured.

**Memora** (Xia et al., 2026; preprint) organizes memory through primary abstractions indexing concrete memory values, with cue anchors enabling contextualized access. The authors prove that standard RAG and knowledge graph memory systems are special cases of their framework and achieve substantial token reduction. Memora explicitly addresses the abstraction-specificity trade-off, but its dual-layer structure does not implement the full multi-stage consolidation pipeline proposed here.

**Continuum Memory Architectures (CMA)** (Logan, 2026; preprint) articulates behavioral requirements for memory systems including persistent storage, selective retention, retrieval-driven mutation, associative routing, temporal chaining, and consolidation. CMA draws on Ebbinghaus forgetting curves and spreading activation theory. Of existing proposals, CMA is closest to the present framework in its scope of ambition, but it specifies requirements rather than representational targets — it describes *what* a memory system should do without specifying *what form* representations should take at each abstraction level.

**FadeMem** (Wei et al., 2026; preprint) implements a dual-layer memory hierarchy with differential decay rates, adaptive forgetting modulated by semantic relevance and access frequency, and LLM-guided conflict resolution. This provides a principled approach to forgetting but does not include hierarchical abstraction or schema formation.

**CoALA** (Sumers et al., 2024; TMLR 2024) provides a comprehensive cognitive architecture framework mapping cognitive science memory types — working, episodic, semantic, procedural — to LLM agent components. CoALA retrospectively organizes existing agents and identifies research gaps but does not specify how episodic memory should be consolidated into semantic memory — the transformation process itself.

**Zep/Graphiti** (Rasmussen et al., 2025) uses a temporal knowledge graph engine with a bi-temporal data model enabling point-in-time queries and temporal conflict resolution. Zep introduces graph-structured, temporally aware memory — a significant advance — but does not implement progressive abstraction across consolidation stages.

### 2.6 Summary of the Landscape and Remaining Gap

| Approach | Storage Unit | Key Advance | What It Lacks |
|---|---|---|---|
| Context Window | Raw tokens | Immediate availability | No compression; no persistence |
| RAG | Text-chunk embeddings | Scalable; updatable | No abstraction; flat structure |
| Fine-tuning | Model parameters | Persistent; generalized | Inflexible; catastrophic forgetting |
| MemGPT | Text across tiers | Multi-level; managed | Same format across tiers |
| Generative Agents | Text + reflections | Genuine abstraction via reflection | No structured representation; no forgetting |
| HippoRAG | KG triples + PageRank | Neuroscience-grounded retrieval | No consolidation; static graph |
| Larimar | Latent vectors | CLS-inspired fast editing | No hierarchical abstraction |
| EM-LLM | Event-segmented episodes | Surprise-based segmentation | No consolidation of episodes |
| MemTree | Hierarchical tree nodes | Schema-inspired hierarchy | No relational binding; no reconsolidation |
| Titans | Neural memory parameters | Learned surprise-based storage | Not inspectable; no explicit structure |
| Memora | Abstraction-value pairs | Formal abstraction-specificity balance | Two-layer; no full consolidation pipeline |
| CMA | Behavioral specification | Comprehensive requirements | No representational targets specified |
| CoALA | Cognitive architecture | Maps cognitive science to agents | No consolidation transformation specified |

**Working hypothesis:** The remaining gap is not any single capability but the *integration* of these capabilities into a consolidation pipeline with explicitly specified representational targets at each stage. Individual components — hippocampal retrieval (HippoRAG), hierarchical storage (MemTree), fast episodic writing (Larimar), surprise-based segmentation (EM-LLM), graph structure (Zep), forgetting (FadeMem) — exist. What is missing is a framework specifying how they compose: what the representational format should be at each level, what governs transitions between levels, and how to evaluate whether consolidation achieves genuine abstraction versus mere text compression.

---

## 3. Biological Memory Consolidation: From Episode to Abstraction

A substantial body of evidence from cognitive neuroscience characterizes memory consolidation as a transformative process that progressively extracts structure from experience. This section reviews key findings relevant to the design of artificial memory systems.

**Important methodological caveat.** Biological memory consolidation involves concurrent, bidirectional, and interactive processes. Encoding, relational binding, schematization, and metacognitive monitoring operate in parallel and influence each other — schemas modulate encoding from the outset (Gilboa & Marlatte, 2017), and consolidation can be rapid for schema-consistent information (Tse et al., 2007). The staged decomposition presented in Section 4 of this paper is a *computational simplification for engineering purposes*, not a claim that biological consolidation proceeds in discrete, sequential phases. This caveat is critical: the value of the decomposition lies in its utility for specifying representational targets, not in its fidelity to biological temporal dynamics.

### 3.1 Complementary Learning Systems

The complementary learning systems (CLS) framework (McClelland et al., 1995; Kumaran et al., 2016) proposes that mammalian memory depends on two interacting systems with distinct computational properties:

1. **Hippocampal system:** Rapid encoding of specific episodes using sparse, pattern-separated representations. High learning rate; low interference between memories. The hippocampus also constructs relational representations that bind together elements of experience and situate them within a broader cognitive map (Eichenbaum, 2004; Cohen & Eichenbaum, 1993).

2. **Neocortical system:** Slow extraction of statistical regularities across many episodes. Low learning rate; distributed, overlapping representations encoding the structure common to many experiences.

The interaction between these systems produces consolidated memory. The hippocampus replays encoded episodes during sleep and rest, allowing the neocortex to gradually extract shared structure and integrate it with existing knowledge (Rasch & Born, 2013; Frankland & Bontempi, 2005). Recent evidence suggests that replay involves creative recombination of experienced elements, potentially supporting construction of novel relational structures (Gupta et al., 2010; Lewis & Durrant, 2011).

*Relevance to LLMs:* Current approaches map onto one system or the other but not the interaction. Context windows and RAG resemble hippocampal episodic storage (specific, temporary). Fine-tuning resembles neocortical parametric storage (general, stable, inflexible). Larimar (Das et al., 2024) explicitly implements CLS dynamics for fast knowledge editing, while CLS-ER (Arani et al., 2022) applies dual-memory principles to continual learning. However, neither implements the full consolidation transformation — the progressive, structure-preserving compression from specific to general. Spens and Burgess (2024) have recently formalized this transformation computationally using modern Hopfield networks (hippocampal encoding) training variational autoencoders (neocortical generative models), demonstrating how replay produces the progressive episodic-to-semantic transformation observed empirically. This computational model provides a formal target for artificial consolidation systems.

### 3.2 Schema Theory and Memory Transformation

Schema theory, originating with Bartlett (1932) and developed extensively in cognitive neuroscience (Ghosh & Gilboa, 2014; Gilboa & Marlatte, 2017), proposes that memory is organized around structured knowledge representations — schemas — that capture the general form of repeated experience while abstracting away incidental details.

Key properties of schemas relevant to memory architecture design:

- **Compression:** Schemas represent the invariant structure of many episodes in a compact format. This can be formalized through rate-distortion theory (Shannon, 1959): schemas represent a near-optimal trade-off between representational fidelity and compactness for a given distortion tolerance.
- **Hierarchical organization:** Schemas nest within superordinate schemas, forming a hierarchical knowledge structure (Piaget, 1952; Rumelhart, 1980).
- **Assimilation and accommodation:** New experience is either integrated into existing schemas (assimilation) or triggers schema modification (accommodation) when sufficiently discrepant (Piaget, 1952). The medial prefrontal cortex appears to mediate this gating function (van Kesteren et al., 2012).
- **Active reconstruction and distortion:** Retrieval is not replay of stored content but active reconstruction guided by schematic knowledge, which both fills gaps and introduces *systematic distortions* (Bartlett, 1932; Schacter & Addis, 2007). This is critical: schemas are not merely efficient — they are biased. Schema-consistent details are preferentially recalled; schema-inconsistent details are often lost or distorted. Both facilitative and distortive effects are well-documented (Gilboa & Marlatte, 2017).

Schema-consistent information is more rapidly consolidated into neocortical representations (Tse et al., 2007; van Kesteren et al., 2012). Importantly, this means consolidation can proceed rapidly when existing schemas support integration — the slow neocortical learning rate of CLS theory applies to schema-*inconsistent* information, not universally.

### 3.3 Reconsolidation: Memory as Dynamic and Updateable

A critical feature of biological memory absent from nearly all artificial memory systems is reconsolidation. Nader et al. (2000) demonstrated that retrieved memories return to a labile state and must be re-stabilized — a process during which they can be modified, strengthened, or updated. Subsequent work has established reconsolidation as a general property of memory across species and memory types (Nader & Hardt, 2009; Lee et al., 2017).

Reconsolidation has implications for memory architecture design:

- **Retrieval is an opportunity for updating.** Each time a memory is accessed, it can be modified in light of current context, providing a natural mechanism for schema refinement that does not require a separate consolidation process.
- **Memories are not fixed once consolidated.** Even schematic, well-established memories can be updated when retrieved, allowing the system to correct errors and integrate new information into existing knowledge structures.
- **Boundary conditions constrain updating.** Reconsolidation is triggered primarily when retrieved information encounters prediction error — a mismatch between the retrieved memory and current experience (Exton-McGuinness et al., 2015). This selectivity ensures that stable memories are not gratuitously destabilized. Not all retrieval events trigger reconsolidation; a prediction-error signal appears necessary for destabilization.

*Relevance to LLMs:* CMA (Logan, 2026) includes "retrieval-driven mutation" as a behavioral requirement, and Zep/Graphiti (Rasmussen et al., 2025) implements temporal conflict resolution through edge invalidation. These approximate aspects of reconsolidation but neither implements prediction-error-gated selective updating of consolidated schemas.

### 3.4 Emotional Modulation of Consolidation

Biological memory consolidation is not uniform across all experience. Emotionally significant events are preferentially consolidated through modulation of hippocampal-neocortical interactions by the amygdala and stress hormone systems (McGaugh, 2004; LaBar & Cabeza, 2006). This produces a natural prioritization: experiences with greater relevance to the organism's goals are more robustly encoded and more rapidly consolidated.

*Relevance to LLMs:* An abstractive memory system without a salience signal treats all episodes as equally worthy of consolidation. Introducing a salience function — based on task relevance, user feedback, prediction error magnitude, or downstream utility — would allow prioritization of informative experiences. EM-LLM's surprise-based segmentation (Fountas et al., 2025) and Titans' surprise-based storage (Behrouz et al., 2025) implement versions of this principle at the encoding level; extending it to consolidation dynamics is a natural next step. FadeMem (Wei et al., 2026) also implements relevance-weighted retention through adaptive decay.

### 3.5 Cognitive Maps and Relational Structure

The hippocampus encodes not only specific episodes but also the relational structure of experience — spatial layouts, temporal sequences, and abstract relational graphs (O'Keefe & Nadel, 1978; Eichenbaum, 2004; Behrens et al., 2018). This "cognitive map" function means that biological memory is organized not just hierarchically (schemas within schemas) but also graphically — as a network of relational connections between entities, events, and abstractions.

Recent computational work has formalized this as a relational graph in which nodes represent entities or events and edges represent relationships, with the hippocampus computing relational structure through successor representations and related mechanisms (Stachenfeld et al., 2017; Behrens et al., 2018).

*Relevance to LLMs:* This suggests that the optimal storage format for abstracted memory is graph-structured — a relational knowledge graph in which entities, schemas, and episodes are connected by typed relationships. HippoRAG (Gutiérrez et al., 2024) implements this insight for retrieval, and Zep/Graphiti (Rasmussen et al., 2025) implements it for temporally-aware storage. The present framework proposes extending graph structure across all abstraction levels, not only the episodic/retrieval level.

### 3.6 Abstraction in Neural Representation

Evidence from neuroimaging and neuropsychology points to several principles of neural abstraction:

- **Graded abstraction:** Neural populations in anterior temporal and prefrontal regions encode increasingly abstract, modality-independent representations compared to posterior sensory regions (Patterson et al., 2007; Lambon Ralph et al., 2017; Binder et al., 2009).
- **Relational coding:** Hippocampal and prefrontal populations encode relational structure — the relationships between entities — not merely entity features (Eichenbaum, 2004; Frankland & Greene, 2020).
- **Dimensionality reduction:** Cortical representations of complex stimuli occupy lower-dimensional manifolds than the input space, suggesting active compression toward task-relevant dimensions (DiCarlo & Cox, 2007).

These findings converge on a picture in which biological memory consolidation is fundamentally a process of dimensionality reduction through abstraction: high-dimensional episodic representations are progressively compressed into lower-dimensional schematic representations that capture invariant structure.

---

## 4. A Four-Stage Taxonomy of Memory Abstraction

### 4.1 Overview

I propose that the representational targets for an abstractive memory architecture can be organized into four stages of increasing abstraction. This taxonomy is a **working hypothesis** — theoretically motivated by the neuroscience reviewed in Section 3 and intended to generate testable predictions, but not yet empirically validated. The stages are:

- **Stage 1 — Episodic Encoding:** Storage of specific experience with minimal transformation. The system records what happened, preserving detail and contextual binding.
- **Stage 2 — Relational Binding:** Representation of relationships between encoded elements. The system encodes not just "A" and "B" but "A relates to B in manner X," enabling structural comparison and analogical transfer.
- **Stage 3 — Hierarchical Schematization:** Construction of nested, hierarchical knowledge structures that organize relational abstractions into coherent frameworks. The system represents categories, taxonomies, causal models, and behavioral patterns.
- **Stage 4 — Metacognitive Monitoring:** Representation of the system's own knowledge states. The system can monitor, evaluate, and calibrate its own abstractions — tracking where its schemas are well-supported versus tentative.

### 4.2 Mapping to Biological Consolidation: A Qualified Assessment

The strength of the correspondence between these stages and biological memory consolidation varies, and precision about where the mapping is strong versus speculative is essential.

| Stage | Memory Analog | Biological Basis | Mapping Strength |
|---|---|---|---|
| Stage 1: Episodic Encoding | Hippocampal rapid encoding | Yassa & Stark, 2011; Eichenbaum, 2004 | **Moderate** |
| Stage 2: Relational Binding | Hippocampal relational memory | Cohen & Eichenbaum, 1993; Eichenbaum, 2004 | **Strong** |
| Stage 3: Hierarchical Schematization | Neocortical schema formation | van Kesteren et al., 2012; Gilboa & Marlatte, 2017 | **Strong** |
| Stage 4: Metacognitive Monitoring | Metamemory | Fleming & Dolan, 2012; Vaccaro & Fleming, 2018 | **Speculative** |

**Where the mapping is strong.** Stage 2 maps cleanly onto hippocampal relational binding — the well-established finding that the hippocampus encodes relationships between elements of experience, not just the elements themselves (Eichenbaum, 2004; Cohen & Eichenbaum, 1993). Stage 3 maps well onto neocortical schema formation — the extensively documented process by which repeated relational patterns are extracted into generalized knowledge structures mediated by medial prefrontal cortex (Gilboa & Marlatte, 2017; van Kesteren et al., 2012; Tse et al., 2007).

**Where the mapping requires qualification.** Stage 1 ("episodic encoding") does not map straightforwardly onto hippocampal encoding. The hippocampus performs sophisticated computational operations — sparse coding, pattern separation, conjunctive binding — that go well beyond simple storage (Yassa & Stark, 2011). The correspondence is better understood as: Stage 1 represents the storage of specific experience with *relatively minimal transformation compared to subsequent consolidation stages*, even though the encoding process itself is computationally rich.

**Where the mapping is speculative.** Stage 4 (metacognitive monitoring) maps onto metamemory and metacognition, which are related but not identical constructs. Metamemory — the capacity to monitor and evaluate one's own memory states — is a specific form of metacognition (Fleming & Dolan, 2012). Stage 4 as defined here is narrower than "self-referential abstraction" in some formulations: it specifically concerns the system's ability to represent *confidence in its own stored knowledge*, enabling calibrated retrieval and strategic information-seeking. Even this narrower formulation remains speculative as a memory architecture component — biological metamemory involves anterior prefrontal cortex and default mode network activity (Vaccaro & Fleming, 2018), but the computational mechanisms are less well-characterized than those underlying Stages 2–3.

**Critical caveat on sequentiality.** As noted in Section 3, these stages should *not* be interpreted as a strict temporal sequence in which one phase completes before the next begins. In biological memory, schemas influence encoding from the outset (Gilboa & Marlatte, 2017), relational binding occurs during initial encoding (Eichenbaum, 2004), and consolidation can be rapid for schema-consistent information (Tse et al., 2007). The four stages describe a hierarchy of *representational complexity*, not a temporal pipeline. For engineering purposes, implementing them as a pipeline is a practical simplification, and the predictions in Section 6 are designed to evaluate whether this simplification is productive.

### 4.3 Relationship to Existing Frameworks

Several existing frameworks overlap with aspects of this taxonomy, and the specific contribution must be clearly delineated.

**CoALA** (Sumers et al., 2024) maps cognitive science memory types (working, episodic, semantic, procedural) onto LLM agent components. The present taxonomy is complementary: CoALA specifies *what kinds* of memory exist; this framework specifies *how representations should transform* as they move from episodic to semantic memory — the consolidation process itself.

**MemTree** (Rezazadeh et al., 2025) implements hierarchical abstraction through tree-structured storage, approximating Stages 1 and 3 (episodic leaves, abstract higher nodes). The present framework extends this with explicit relational binding (Stage 2), reconsolidation mechanics, metacognitive monitoring (Stage 4), and graph structure at each level.

**Generative Agents** (Park et al., 2023) implements Stages 1 and partial Stage 3 (raw observations plus reflections that generate higher-level observations). The present framework adds structured relational representation, hierarchical organization, confidence metadata, and forgetting.

**CMA** (Logan, 2026) specifies behavioral requirements (persistence, selective retention, consolidation) that the present architecture aims to satisfy, while adding representational specificity at each abstraction level.

**Memora** (Xia et al., 2026) implements a dual-layer abstraction-specificity balance, but the present framework proposes four distinct representational levels with explicit transition dynamics.

The specific contribution, therefore, is the *integration* of these elements: a single framework specifying representational format, transition dynamics, and evaluation criteria across the full consolidation pipeline. Whether this integration produces value beyond its components is an empirical question addressed by the predictions in Section 6.

### 4.4 Implications for Memory Architecture Design

If this taxonomy is productive, it suggests that an effective LLM memory system should implement a consolidation pipeline that progressively transforms stored representations:

**Stage 1 → Stage 2:** Raw interaction logs are processed to extract relational structure. "User said X, model responded Y, user corrected to Z" becomes a structured relation: [topic: X, user_preference: Z > Y, confidence: high, context: technical_discussion]. This also involves building relational graphs — connecting entities and events through typed relationships, as the hippocampus constructs cognitive maps (Behrens et al., 2018). HippoRAG's knowledge graph extraction (Gutiérrez et al., 2024) provides an existence proof for this transition using LLMs.

**Stage 2 → Stage 3:** Relational abstractions accumulated across many interactions are integrated into hierarchical schemas. Multiple instances of user preference for direct communication consolidate into a schema: [communication_style: {preference: direct, evidence_count: n, contexts: [technical, emotional, planning], exceptions: [formal_email]}]. This transition should be governed by a consistency threshold: sufficient convergent evidence across diverse episodes before schematization, paralleling the brain's requirement for repeated replay before consolidation (Rasch & Born, 2013). MemTree's hierarchical merging (Rezazadeh et al., 2025) provides an existence proof for this transition.

**Stage 3 → Stage 4:** The system develops representations of its own knowledge states — what it knows well versus poorly, where its schemas are well-supported versus tentative, which domains show stable patterns versus ongoing change. This enables metacognitive memory operations: targeted retrieval, confidence-calibrated responses, and strategic knowledge-seeking.

**Reconsolidation across stages:** When retrieved Stage 2–3 representations encounter prediction error — new evidence contradicting the stored representation — the representation re-enters a modifiable state, paralleling biological reconsolidation (Nader et al., 2000). This provides a mechanism for schema updating triggered by use, not only by scheduled offline consolidation.

---

## 5. Proposed Architecture: Abstractive Memory System (AMS)

**Epistemic status:** This section describes a proposed architecture. It is speculative, theoretically motivated, and intended to generate testable predictions rather than to report an implemented or validated system.

### 5.1 Design Principles

An Abstractive Memory System (AMS) would be organized around seven principles derived from biological memory consolidation:

1. **Multi-level representation:** Information is stored at multiple levels of abstraction simultaneously, not at a single level.
2. **Progressive consolidation:** Representations are actively transformed from lower to higher abstraction levels over time or usage.
3. **Compression with structure preservation:** Higher-level representations are more compact but preserve relational and hierarchical structure.
4. **Graph-structured storage:** Representations are organized as relational graphs, not flat vectors or linear hierarchies, reflecting the cognitive map function of biological memory (Behrens et al., 2018).
5. **Reconsolidation on retrieval:** Retrieved representations that encounter prediction error re-enter a modifiable state, enabling schema refinement through use (Nader et al., 2000).
6. **Salience-weighted consolidation:** Not all episodes are consolidated equally. A salience function prioritizes informative experiences (cf. McGaugh, 2004), using signals such as prediction error magnitude, task relevance, or downstream utility.
7. **Metacognitive indexing:** The system maintains representations of its own knowledge states, supporting confidence calibration and strategic information-seeking.

### 5.2 Architectural Components

**Memory Store:** A structured knowledge graph organized by abstraction level. Each memory entry has: a defined abstraction level (Stage 1–4), typed relational links to other entries, a confidence/evidence metric (number of supporting episodes, recency, consistency), temporal metadata, and a salience score. This extends GraphRAG's static knowledge graphs (Edge et al., 2024) and Zep/Graphiti's temporal knowledge graphs (Rasmussen et al., 2025) with abstraction-level metadata and consolidation dynamics.

**Consolidation Engine:** A process that periodically transforms lower-level representations into higher-level abstractions — the functional analog of hippocampal-neocortical replay during sleep (Rasch & Born, 2013).

*The circularity problem.* A central challenge: if the consolidation engine is itself an LLM, we are using a system that may lack genuine abstraction capacity to perform the abstraction operations the framework requires. This is not unique to AMS — it applies to any LLM-based reflection or summarization system, including Generative Agents' reflection mechanism and MemTree's hierarchical summarization. Three approaches to mitigation:

(a) *Structured extraction rather than free summarization.* The consolidation engine fills structured templates with typed fields and relational links rather than producing free-text summaries. This constrains the output format and allows structural properties to be verified independently of the process that produced them. HippoRAG's knowledge graph extraction (Gutiérrez et al., 2024) demonstrates feasibility.

(b) *Convergence-based validation.* Consolidation is accepted only when multiple independent passes produce consistent relational structure, reducing the probability that statistical artifacts are reified as schemas. This parallels the biological requirement for repeated replay before consolidation (Rasch & Born, 2013).

(c) *Evaluating the output empirically.* The testable predictions in Section 6 provide criteria for distinguishing genuine abstractive memory from sophisticated summarization. If AMS shows the predicted signatures — distinct error profiles, superior transfer on structurally similar but surface-dissimilar tasks — this constitutes evidence that functional abstraction is occurring regardless of mechanism. Recent work on concept-level versus instance-level memory (Ho et al., 2025) demonstrates that this distinction is empirically tractable.

**Retrieval Controller:** A mechanism that determines which abstraction level to query based on current task context. Specific factual questions trigger Stage 1–2 retrieval; planning and reasoning tasks trigger Stage 3; self-evaluation and calibration tasks trigger Stage 4. The controller also implements reconsolidation gating: when retrieved information encounters prediction error exceeding a threshold, the retrieved representation is flagged for updating.

**Forgetting Mechanism:** Consistent with biological memory, lower-level (Stage 1) representations decay over time once their relational structure has been consolidated into higher-level representations. This parallels the finding that hippocampal dependence decreases as memories are consolidated into neocortical schemas (Frankland & Bontempi, 2005), supported by evidence that forgetting of detail after schematization is functional, not pathological (Richards & Frankland, 2017). FadeMem (Wei et al., 2026) provides a recent engineering precedent for principled forgetting in LLM memory.

### 5.3 Worked Example

Consider an AI assistant that interacts with a user over many sessions. Under the proposed AMS:

**Stage 1 (episodic):** Stores specific interaction records with relational links. "On Feb 10, user expressed frustration with corporate work culture and described preference for direct-impact family business." Linked to: [user_emotion: frustrated], [topic: career_decisions], [timestamp: 2026-02-10].

**Stage 2 (relational):** Extracts structured relations into the knowledge graph. [user_values: {direct_impact > institutional_prestige}, {family > career}, {autonomy > security}]. Relations linked to supporting episodes and annotated with confidence: {confidence: 0.85, evidence_episodes: [feb10, feb14, mar02], last_updated: 2026-03-02}.

**Stage 3 (schematic):** Consolidated into user schema. [user_profile: values-driven decision maker; core_motivation: family_welfare; work_style: independent; communication_preference: direct, informal]. Schema annotated with provenance: {evidence_count: 47, schema_stability: high, last_modified: 2026-03-15, domains_covered: [career, parenting, health, technology]}.

**Stage 4 (metacognitive):** System represents its own confidence in different schema aspects. [confidence_map: {values: high, n=47}, {technical_skills: moderate, n=12}, {health_history: low, n=3}]. Enables: targeted questions where knowledge is thin, flags when schema may be outdated, and confidence-calibrated responses hedged in low-confidence domains.

**Reconsolidation example:** The system retrieves the schema communication_preference: direct during an interaction where the user explicitly requests a gentle, diplomatic approach. Prediction error triggers reconsolidation: the schema is updated to communication_preference: {default: direct, exception: interpersonal_conflict, evidence_for_exception: [mar20]}.

---

## 6. Testable Predictions

A critical requirement for any theoretical framework is that it generate predictions distinguishable from alternatives. The following predictions are designed to discriminate AMS from both existing approaches *and* from simpler alternatives (e.g., a two-stage system, or Generative Agents' reflection mechanism applied to structured storage). For each prediction, I specify what outcome would *disconfirm* the framework.

### 6.1 Efficiency Predictions

**Prediction 1 (Compression scaling).** An AMS storing Stage 2–3 abstracted representations will require fewer tokens in the context window to achieve equivalent task performance compared to a RAG system storing the same interaction history as raw text, with the efficiency ratio increasing as interaction history grows. Specifically, the relationship between interaction count and required token budget should follow a sublinear function for AMS (reflecting increasing compression as more episodes consolidate into shared schemas) but a roughly linear function for RAG (reflecting proportional growth of stored passages).

*Rationale:* Biological consolidation achieves compression that scales with experience as more episodes contribute to schematic representations (Winocur & Moscovitch, 2011). The sublinear scaling prediction is specific to the multi-stage consolidation pipeline — a system with only summarization (e.g., Generative Agents' reflection without hierarchical structure) would also compress, but without the structured schema formation that produces accelerating returns in high-regularity domains.

*Disconfirmation criterion:* If AMS's token-performance curve is linear or shows no compression advantage over a RAG baseline at any interaction count, or if the compression advantage is matched by a simple summarization baseline (e.g., periodic LLM summarization without structured relational extraction), this would constitute evidence against the framework's claim that structured multi-stage consolidation provides benefits beyond summarization.

**Prediction 2 (Domain-dependent compression).** The compression ratio achieved by AMS will correlate positively with the regularity of the underlying domain (measured by the entropy of the ground-truth relational structure). Highly regular domains (e.g., consistent user preferences) will achieve greater compression than irregular domains (e.g., preferences that vary unpredictably by context).

*Rationale:* Schema formation is experience-dependent and structure-dependent (Gilboa & Marlatte, 2017). Rate-distortion theory predicts that compression efficiency depends on the statistical structure of the source (Shannon, 1959).

*Disconfirmation criterion:* If compression ratio is uncorrelated with domain regularity, this would suggest the consolidation engine is not extracting genuine statistical structure.

### 6.2 Generalization Predictions

**Prediction 3 (Schema transfer).** Systems using Stage 3 (schematic) memory will show superior performance on novel tasks that share *structural* similarity with prior experience but differ in *surface* features, compared to systems using Stage 1 (episodic) or Stage 2 (relational) memory alone. The advantage will be largest when surface similarity is low but structural similarity is high.

*Rationale:* Schemas support generalization by representing invariant structure (Bartlett, 1932; Tse et al., 2007). This prediction is directly motivated by recent empirical findings: Ho et al. (2025) demonstrated that concept-level memories (abstractions) outperformed instance-level memories on novel ARC-AGI tasks, and Zhao et al. (2024) showed that extracted abstract insights outperformed raw trajectory retrieval on tasks requiring generalization (HotpotQA).

*Discriminating power:* This is the most critical prediction for distinguishing genuine abstraction from lossy compression. A summarization system would show performance that correlates with *surface* similarity to prior experience (because summaries preserve surface content in compressed form), whereas a genuinely abstractive system would show performance that correlates with *structural* similarity (because schemas preserve relational structure while discarding surface features).

*Disconfirmation criterion:* If Stage 3 memory shows no transfer advantage over Stage 1 memory on structurally similar but surface-dissimilar tasks, or if the transfer advantage correlates with surface rather than structural similarity, this would constitute evidence that the consolidation engine produces summarization rather than abstraction.

**Prediction 4 (Metacognitive calibration).** Systems with Stage 4 (metacognitive) memory will show improved calibration — better alignment between expressed confidence and actual accuracy — compared to systems without metacognitive memory, as measured by expected calibration error (ECE). The improvement will be specific to domains where the metacognitive index has high evidence counts and minimal for low-evidence domains.

*Rationale:* Metamemory enables monitoring of one's own knowledge states (Fleming & Dolan, 2012).

*Disconfirmation criterion:* If metacognitive memory produces no calibration improvement, or if the improvement is not modulated by evidence count (i.e., is equally present for high- and low-evidence domains), this would suggest the metacognitive index is not tracking genuine knowledge-state information.

### 6.3 Consolidation Process Predictions

**Prediction 5 (Consolidation advantage over accumulation).** Introducing a consolidation step (periodic abstraction of Stage 1 memories into Stage 2–3 representations) will improve long-horizon task performance on a composite battery (recall, inference, personalization) compared to simply accumulating raw memories in a RAG system of equivalent total storage capacity.

*Discriminating power:* This prediction must be evaluated against multiple baselines to be informative: (a) RAG with equivalent storage, (b) Generative Agents-style reflection without structured relational extraction, and (c) MemTree-style hierarchical summarization without explicit relational binding. The framework predicts AMS > (b) > (a) and AMS > (c) on transfer tasks specifically (Prediction 3). If AMS shows no advantage over baseline (b) or (c), this would suggest that the specific multi-stage consolidation pipeline does not add value beyond simpler abstraction mechanisms.

**Prediction 6 (Functional forgetting).** Systems with a forgetting mechanism for consolidated Stage 1 memories will show equivalent performance on schema-dependent tasks (inference, prediction, personalization) while using substantially less storage than systems retaining all raw memories. They will show degraded performance on tasks requiring specific episodic detail (verbatim recall, source attribution).

*Rationale:* Biological forgetting after schematization is functional but produces characteristic detail loss (Winocur & Moscovitch, 2011; Richards & Frankland, 2017). FadeMem (Wei et al., 2026) provides engineering precedent, demonstrating 45% storage reduction with maintained task performance.

*Disconfirmation criterion:* If forgetting degrades schema-dependent task performance proportionally to storage savings (i.e., no efficiency gain), this would suggest that Stage 1 memories are not being successfully consolidated into higher-level representations.

### 6.4 Discriminating Predictions

**Prediction 7 (Interaction-length scaling).** The performance advantage of AMS over RAG will be negligible for short interaction histories, measurable at moderate interaction histories, and large for long interaction histories, following a monotonically increasing function. The inflection point — where AMS's advantage becomes reliably measurable — should depend on domain regularity (earlier in high-regularity domains, later in low-regularity domains).

*Rationale:* The value of consolidation scales with the volume of experience to be compressed, and the rate of useful schema formation depends on the statistical structure of the domain.

*Disconfirmation criterion:* If AMS's advantage does not scale with interaction length, or scales identically regardless of domain regularity, this would undermine the framework's core claim.

**Prediction 8 (Error profile dissociation).** AMS and RAG will show qualitatively different error profiles. RAG errors will be dominated by retrieval failures: relevant information exists in storage but is not retrieved (false negatives on cued recall). AMS errors will be dominated by schema-driven distortions: the system produces responses consistent with its consolidated schema but inconsistent with specific episodic details (systematic biases toward schema-typical responses).

*Rationale:* This mirrors the dissociation between hippocampal retrieval failures and neocortical schema-driven distortions in biological systems (Schacter & Addis, 2007; Bartlett, 1932). Schema-driven distortion is both a strength (supporting generalization) and a risk (introducing systematic bias). Gilboa and Marlatte (2017) emphasize that schemas produce both facilitative and distortive effects.

*Disconfirmation criterion:* If AMS and RAG show the same error profiles, this would suggest that AMS's consolidation is not producing qualitatively different representations from raw storage.

**Prediction 9 (Reconsolidation advantage).** AMS with reconsolidation (schema updating on retrieval-triggered prediction error) will show superior accuracy on tasks involving changed user preferences compared to AMS without reconsolidation (consolidation only during offline periods), with the advantage increasing with the recency and magnitude of the preference change.

*Rationale:* Biological reconsolidation allows memories to be updated through use, providing a faster correction mechanism than offline consolidation alone (Nader & Hardt, 2009).

*Disconfirmation criterion:* If reconsolidation-enabled AMS shows no advantage on preference-change detection tasks, or if offline-only consolidation matches reconsolidation's update speed, this would suggest that reconsolidation dynamics are not necessary for schema maintenance.

---

## 7. The Central Challenge: Abstraction Versus Summarization

Any implementation of AMS must confront the distinction between genuine abstractive memory and sophisticated text summarization. This is the central theoretical question the framework raises, and recent empirical work has begun to provide tools for adjudicating it.

### 7.1 The Problem

When an LLM-based consolidation engine processes a set of Stage 1 episodic memories and produces a Stage 3 schema, two possibilities exist:

1. **Genuine structural abstraction:** The system has extracted the invariant relational structure underlying the episodes, producing a representation that captures generalizable patterns and supports transfer to novel situations.

2. **Lossy text compression:** The system has produced a shorter natural language description that captures the gist of the episodes but does not represent structural information in a way that supports novel inference.

These two possibilities may produce similar outputs on familiar situations but will diverge on novel situations requiring genuine structural generalization.

### 7.2 Recent Empirical Progress

The abstraction-versus-summarization distinction has been partially operationalized in recent work:

**ArcMemo** (Ho et al., 2025) explicitly distinguishes instance-level memories (specific solution traces, analogous to summarization) from concept-level memories (reusable, modular abstractions distilled from traces). On ARC-AGI tasks, concept-level memories consistently outperformed instance-level memories and scaled better with inference compute — direct evidence that abstract representations provide qualitatively different utility than compressed episodes.

**ExpeL** (Zhao et al., 2024; AAAI 2024) demonstrates a two-mode system: retrieval of raw trajectories versus extraction of cross-task abstract insights. For tasks requiring generalization (HotpotQA), abstracted insights dominated; for action-sequence tasks (ALFWorld), raw trajectory retrieval was better. This dissociation is consistent with the prediction that schema-level memory supports structural transfer while episodic memory supports surface-similar retrieval.

**ICAL** (Sarch et al., 2024; NeurIPS 2024 Spotlight) shows that VLM agents achieve superior performance by distilling experience into generalized programs ("embodied programs of thought") rather than storing raw demonstrations, with approximately double the scaling efficiency.

**Position paper on episodic memory** (Modarressi et al., 2025; ICML 2025) directly addresses consolidation from episodic to semantic memory in LLM agents, framing it as producing generalization rather than mere compression — convergent with the present framework's core argument.

These findings collectively suggest that the abstraction-summarization distinction is not merely philosophical — it produces measurable behavioral differences, and systems that achieve genuine abstraction show characteristic advantages on transfer and scaling tasks.

### 7.3 Implications for AMS

For AMS, the critical test is Prediction 3 (schema transfer): performance on novel, structurally similar but surface-dissimilar tasks. If AMS's consolidation engine produces representations that support this transfer pattern, this constitutes functional evidence for abstraction — regardless of the mechanism. The architectural choice of structured extraction rather than free summarization (Section 5.2) is designed to bias toward structural representation, and the convergence-based validation step is designed to filter statistical artifacts.

However, the field currently lacks a **formal evaluation framework** specifically for measuring whether memory consolidation produces genuine abstraction versus lossy compression. Existing benchmarks (LoCoMo, LongMemEval, MemoryBench) evaluate retrieval accuracy, temporal reasoning, and multi-hop QA, but do not specifically test for generalization quality of consolidated memories. Developing such a benchmark — inspired by ArcMemo's concept-level versus instance-level comparison — is a critical need that this framework highlights.

---

## 8. Connections to Active Research Programs

### 8.1 Neuroscience-Informed Architectural Design

The AMS framework extends a growing tradition of neuroscience-informed AI design (Lake et al., 2017). Within the LLM memory space specifically, HippoRAG (Gutiérrez et al., 2024) has demonstrated that grounding retrieval architecture in hippocampal indexing theory produces measurable performance gains. Larimar (Das et al., 2024) has shown that CLS-inspired dual-system design enables capabilities (gradient-free knowledge editing) that are difficult to achieve through purely engineering-driven approaches. EM-LLM (Fountas et al., 2025) has shown that surprise-based event segmentation, grounded in Bayesian accounts of perception, produces event boundaries that correlate with human-annotated event structures. The AMS framework aims to extend this approach from individual memory components to the full consolidation pipeline.

The Spens and Burgess (2024) computational model of memory consolidation — using modern Hopfield networks for hippocampal encoding and variational autoencoders for neocortical generative models — provides a particularly relevant formal target. AMS's consolidation engine is functionally analogous to the VAE training process in their model: extracting statistical regularities from replayed episodes to build generative schemas. Whether LLM-based consolidation can achieve the same computational objectives as VAE training is an empirical question, but the formal framework provides benchmarks for evaluating the quality of consolidation.

### 8.2 Memory-Augmented Neural Networks

The proposed framework extends prior work on memory-augmented neural networks, including Neural Turing Machines (Graves et al., 2014) and Differentiable Neural Computers (Graves et al., 2016). These architectures introduced external memory with learned read/write operations but store fixed-dimensional vectors without hierarchical abstraction or consolidation. Titans (Behrouz et al., 2025) represents a more recent architectural innovation — neural long-term memory with surprise-based storage — that operates within the model's computation rather than as explicit external memory. AMS proposes that explicit, inspectable, graph-structured memory with abstraction-level metadata complements (rather than replaces) these architectural innovations.

### 8.3 Continual Learning and Catastrophic Forgetting

The catastrophic forgetting problem (McCloskey & Cohen, 1989; French, 1999) arises because neural networks lack the complementary learning systems that biological brains use to segregate rapid learning from slow consolidation. CLS-ER (Arani et al., 2022; ICLR 2022) demonstrated that maintaining dual semantic memories with different consolidation rates — directly inspired by CLS theory — produces state-of-the-art continual learning by combining episodic replay with slowly consolidated semantic representations. The AMS framework's forgetting mechanism (Section 5.2) provides a principled approach to managing plasticity-stability trade-offs in explicit memory systems, complementing the parametric approaches of CLS-ER.

### 8.4 Knowledge Graphs and Structured Memory

AMS's graph-structured storage connects to a substantial literature on knowledge graph construction and reasoning (Ji et al., 2022). The distinction is that AMS proposes knowledge graphs that are dynamically constructed through consolidation rather than statically defined, and that include abstraction-level metadata, confidence annotations, and reconsolidation dynamics. GraphRAG (Edge et al., 2024) represents a step toward dynamic graph construction from documents but does not incorporate multi-stage consolidation. Zep/Graphiti (Rasmussen et al., 2025) introduces temporal dynamics to knowledge graphs — a significant advance — but does not implement progressive abstraction across consolidation stages.

### 8.5 Memory Operating Systems

MemOS (Li et al., 2025) proposes an operating system abstraction for managing heterogeneous memory types (parametric, activation, plaintext) through standardized memory units (MemCubes). This systems-level approach is complementary to AMS's cognitive-level approach: MemOS specifies the infrastructure for memory management, while AMS specifies the representational targets and transformation dynamics that such infrastructure should support. An AMS could potentially be implemented on top of a MemOS-like infrastructure layer.

### 8.6 Calibration and Trust

AMS Stage 4 (metacognitive memory) addresses a specific gap in current systems: the disconnect between LLMs' internal uncertainty representations and their expressed confidence. Maintaining explicit, persistent representations of evidence counts and schema stability at each abstraction level could bridge this gap by grounding confidence in accumulated evidence rather than in-context heuristics.

---

## 9. Limitations and Open Questions

### 9.1 Computational Feasibility

The consolidation engine requires computation. Whether efficiency gains from compressed memory offset consolidation costs is an empirical question. Biological consolidation exploits offline periods (sleep) when the organism is not actively processing input (Rasch & Born, 2013); an analogous "offline consolidation" schedule for AI systems is feasible — Letta's "sleep-time" agents represent a commercial implementation of this principle. The computational cost could be managed through amortization: consolidation cost is paid once while retrieval benefits are realized across all subsequent queries.

### 9.2 The Abstraction-Summarization Boundary

As discussed in Section 7, the central challenge is whether LLM-based consolidation achieves genuine structural abstraction or merely lossy text compression. The testable predictions in Section 6 are designed to empirically adjudicate this question. Recent work (Ho et al., 2025; Zhao et al., 2024; Sarch et al., 2024) provides initial evidence that the distinction is empirically tractable and that some forms of LLM-based abstraction do produce qualitatively different behavior than summarization. However, the theoretical resolution remains open. If the predictions are not borne out — if structured consolidation shows no advantage over simple summarization on transfer tasks — this would constitute evidence against the framework's core claim.

### 9.3 Fidelity-Compression Trade-Off

Biological memory consolidation is lossy — schematic representations discard detail that may later prove relevant (Bartlett, 1932). An AMS would face the same trade-off. Gilboa and Marlatte (2017) emphasize that schemas produce systematic distortions, not merely information loss. An AMS could exhibit analogous biases: consolidated user schemas that resist updating when preferences change, or world-model schemas that filter out disconfirming evidence. The reconsolidation mechanism (Section 5.2) is designed to mitigate this, but schema rigidity remains a risk — and a safety concern (Section 10.2). The optimal compression schedule, and whether the Minimum Description Length principle (Rissanen, 1978) can be operationalized for structured representations, are open design questions.

### 9.4 Evaluation Methodology

Evaluating whether AMS achieves abstractive memory requires metrics that distinguish abstraction from summarization. The error profile analysis (Prediction 8), transfer performance (Prediction 3), and calibration measures (Prediction 4) provide initial approaches. ArcMemo's concept-level versus instance-level comparison (Ho et al., 2025) offers a template. However, more refined methodologies — potentially including analysis of the internal structure of consolidated representations — may be necessary. The development of a dedicated benchmark for evaluating consolidation quality is a critical need this framework highlights.

### 9.5 Scope of Biological Analogy

Biological memory consolidation evolved under constraints (energy efficiency, embodiment, mortality) that differ fundamentally from those facing artificial systems. Importing biological principles should be done selectively, guided by computational analysis of *why* a given principle is adaptive, not by blanket biomimicry (Lake et al., 2017). The principle of progressive abstraction may be adaptive for any information-processing system facing storage constraints and generalization demands — a computational argument. The specific neural mechanisms (hippocampal replay, sleep spindles, sharp-wave ripples) are biological implementation details that need not be copied. This paper aims for the former (computational principles) while drawing on the latter (implementation details) only for inspiration.

### 9.6 Abstraction Grounding

In biological systems, abstract representations are ultimately grounded in sensorimotor experience (Barsalou, 2008). LLMs lack this grounding, and it remains an open question whether genuinely abstract (as opposed to statistically derived) representations can emerge in systems without embodied experience. The AMS framework does not resolve this concern, but the prediction framework (Section 6) provides a context for empirically probing it: if AMS produces representations that support the kind of generalization that grounded schemas support (Prediction 3), this would be evidence — though not proof — that functional abstraction can occur without embodied grounding.

### 9.7 Novelty and Crowded Landscape

This paper enters a rapidly evolving field with over 100 published systems addressing LLM memory (Hu et al., 2025). The specific contribution is the integration of a full consolidation pipeline with explicit representational targets, transition dynamics, and discriminating predictions — not any single component. Whether this integration produces empirical value beyond its components remains to be demonstrated. The framework's utility should be judged by its predictions: if the discriminating predictions (particularly Predictions 3, 7, and 8) are confirmed and cannot be explained by simpler alternatives, this validates the integrative approach. If they are not, simpler approaches should be preferred on grounds of parsimony.

---

## 10. Implications and Future Directions

### 10.1 For Long-Horizon Agents

The emergence of long-horizon AI agents makes the memory problem acute. An agent operating over days or weeks cannot rely on context windows alone and will accumulate experience far faster than flat storage can efficiently manage. AMS offers a principled approach: consolidating experience into increasingly abstract representations that support planning and decision-making without requiring retrieval of every prior episode. The reconsolidation mechanism additionally allows the agent to refine its world model through experience.

### 10.2 For AI Safety

Abstractive memory has direct implications for AI safety. Metacognitive memory (Stage 4) supports calibration — critical for safe deployment in high-stakes domains. A forgetting mechanism provides a natural approach to privacy: specific personal details decay after their structural information has been consolidated.

However, abstractive memory also introduces novel risks. Schema-driven distortions — the tendency to assimilate new information into existing schemas even when it contradicts them — are a well-documented feature of biological memory (Bartlett, 1932; Gilboa & Marlatte, 2017). An AMS could exhibit analogous biases: consolidated user schemas that resist updating, or world-model schemas that filter out disconfirming evidence. Monitoring for schema rigidity and ensuring that the reconsolidation mechanism is sufficiently sensitive to prediction error would be essential safety components.

### 10.3 For the Science of Intelligence

If the AMS framework proves empirically viable, it would support the broader claim that certain computational principles of intelligence — progressive abstraction, hierarchical knowledge organization, reconsolidation dynamics — may be substrate-independent: useful for biological and artificial systems alike, not because artificial systems are biological, but because the computational problems they solve have similar structure (cf. Marr, 1982). This would constitute evidence for a convergent computational account of memory and abstraction across substrates — a claim that can be tested through the prediction framework in Section 6.

---

## 11. Conclusion

The LLM memory problem is widely recognized but has been addressed primarily through individual engineering innovations — larger windows, better retrieval, hierarchical storage, surprise-based segmentation, knowledge graph construction — each implementing a subset of the capabilities that biological memory consolidation integrates. This paper argues that the remaining gap is not any single missing capability but the integration of a full consolidation pipeline with explicit representational targets at each abstraction level.

The proposed four-stage taxonomy of memory abstraction — episodic encoding, relational binding, hierarchical schematization, and metacognitive monitoring — provides a working hypothesis for these representational targets, derived from and constrained by established neuroscience of memory consolidation. The correspondence to biology is strongest for relational binding and schema formation, moderate for episodic encoding, and speculative for metacognitive monitoring. The staged decomposition is a computational simplification for engineering purposes, not a claim about biological temporal dynamics.

The proposed Abstractive Memory System operationalizes this taxonomy into an architecture enriched by reconsolidation dynamics, salience-weighted consolidation, graph-structured storage, and metacognitive indexing. It generates predictions designed to discriminate the framework from both existing approaches and simpler alternatives — particularly on transfer tasks (Prediction 3), error profiles (Prediction 8), and interaction-length scaling (Prediction 7).

The central open question — whether LLM-based consolidation can achieve genuine structural abstraction or merely lossy text compression — is acknowledged directly. Recent empirical work (Ho et al., 2025; Zhao et al., 2024; Sarch et al., 2024) provides initial evidence that the distinction is tractable and that abstract representations produce qualitatively different behavioral signatures than compressed episodes. The framework provides the theoretical structure and discriminating predictions necessary to make further empirical work on this question productive.

Whether this specific proposal proves correct is less important than the reframing it represents: treating LLM memory as a cognitive science problem requiring integration across the full consolidation pipeline, not merely individual engineering improvements to storage and retrieval.

---

## References

Arani, E., Sarfraz, F., & Zonooz, B. (2022). Learning fast, learning slow: A general continual learning method based on complementary learning system. *Proceedings of the International Conference on Learning Representations (ICLR 2022)*.

Baddeley, A. (2000). The episodic buffer: A new component of working memory? *Trends in Cognitive Sciences, 4*(11), 417–423.

Barsalou, L. W. (2008). Grounded cognition. *Annual Review of Psychology, 59*, 617–645.

Bartlett, F. C. (1932). *Remembering: A Study in Experimental and Social Psychology*. Cambridge University Press.

Behrens, T. E. J., Muller, T. H., Whittington, J. C. R., Mark, S., Baram, A. B., Stachenfeld, K. L., & Kurth-Nelson, Z. (2018). What is a cognitive map? Organizing knowledge for flexible behavior. *Neuron, 100*(2), 490–509.

Behrouz, A., Zhong, P., & Mirrokni, V. (2025). Titans: Learning to memorize at test time. *Advances in Neural Information Processing Systems 38 (NeurIPS 2025)*. arXiv:2501.00663.

Binder, J. R., Desai, R. H., Graves, W. W., & Conant, L. L. (2009). Where is the semantic system? A critical review and meta-analysis of 120 functional neuroimaging studies. *Cerebral Cortex, 19*(12), 2767–2796.

Cohen, N. J., & Eichenbaum, H. (1993). *Memory, Amnesia, and the Hippocampal System*. MIT Press.

Cowan, N. (2001). The magical number 4 in short-term memory: A reconsideration of mental storage capacity. *Behavioral and Brain Sciences, 24*(1), 87–114.

D'Esposito, M., & Postle, B. R. (2015). The cognitive neuroscience of working memory. *Annual Review of Psychology, 66*, 115–142.

Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and memory-efficient exact attention with IO-awareness. *Advances in Neural Information Processing Systems, 35*.

Das, P., Chaudhury, S., Nelson, E., Melnyk, I., Swaminathan, S., Dai, S., Lozano, A., Kollias, G., Chenthamarakshan, V., Navrátil, J., Dan, S., & Chen, P.-Y. (2024). Larimar: Large language models with episodic memory control. *Proceedings of the 41st International Conference on Machine Learning (ICML 2024)*.

DiCarlo, J. J., & Cox, D. D. (2007). Untangling invariant object recognition. *Trends in Cognitive Sciences, 11*(8), 333–341.

Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., Metropolitansky, D., Ness, R. O., & Larson, J. (2024). From local to global: A graph RAG approach to query-focused summarization. arXiv:2404.16130.

Eichenbaum, H. (2004). Hippocampus: Cognitive processes and neural representations that underlie declarative memory. *Neuron, 44*(1), 109–120.

Exton-McGuinness, M. T. J., Lee, J. L. C., & Reichelt, A. C. (2015). Updating memories — the role of prediction errors in memory reconsolidation. *Behavioural Brain Research, 278*, 375–384.

Fleming, S. M., & Dolan, R. J. (2012). The neural basis of metacognitive ability. *Philosophical Transactions of the Royal Society B, 367*(1594), 1338–1349.

Fountas, Z., Benfeghoul, M. A., Oomerjee, A., Christopoulou, F., Lampouras, G., Bou-Ammar, H., & Wang, J. (2025). Human-inspired episodic memory for infinite context LLMs. *Proceedings of the International Conference on Learning Representations (ICLR 2025)*. arXiv:2407.09450.

Frankland, P. W., & Bontempi, B. (2005). The organization of recent and remote memories. *Nature Reviews Neuroscience, 6*(2), 119–130.

Frankland, S. M., & Greene, J. D. (2020). Concepts and compositionality: In search of the brain's language of thought. *Annual Review of Psychology, 71*, 273–303.

French, R. M. (1999). Catastrophic forgetting in connectionist networks. *Trends in Cognitive Sciences, 3*(4), 128–135.

Ghosh, V. E., & Gilboa, A. (2014). What is a memory schema? A historical perspective on current neuroscience literature. *Neuropsychologia, 53*, 104–114.

Gilboa, A., & Marlatte, H. (2017). Neurobiology of schemas and schema-mediated memory. *Trends in Cognitive Sciences, 21*(8), 618–631.

Graves, A., Wayne, G., & Danihelka, I. (2014). Neural Turing machines. arXiv:1410.5401.

Graves, A., Wayne, G., Reynolds, M., Harley, T., Danihelka, I., Grabska-Barwińska, A., ... & Hassabis, D. (2016). Hybrid computing using a neural network with dynamic external memory. *Nature, 538*(7626), 471–476.

Gupta, A. S., van der Meer, M. A. A., Touretzky, D. S., & Redish, A. D. (2010). Hippocampal replay is not a simple function of experience. *Neuron, 65*(5), 695–705.

Gutiérrez, B. J., Shu, Y., Gu, Y., Yasunaga, M., & Su, Y. (2024). HippoRAG: Neurobiologically inspired long-term memory for large language models. *Advances in Neural Information Processing Systems 37 (NeurIPS 2024)*. arXiv:2405.14831.

Ho, M., et al. (2025). ArcMemo: Abstract reasoning composition with lifelong LLM memory. arXiv:2509.04439.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2021). LoRA: Low-rank adaptation of large language models. arXiv:2106.09685.

Hu, Y., Liu, S., Yue, Y., Zhang, G., et al. (2025). Memory in the age of AI agents. arXiv:2512.13564.

Ji, S., Pan, S., Cambria, E., Marttinen, P., & Yu, P. S. (2022). A survey on knowledge graphs: Representation, acquisition, and applications. *IEEE Transactions on Neural Networks and Learning Systems, 33*(2), 494–514.

Kumaran, D., Hassabis, D., & McClelland, J. L. (2016). What learning systems do intelligent agents need? Complementary learning systems theory updated. *Trends in Cognitive Sciences, 20*(7), 512–534.

LaBar, K. S., & Cabeza, R. (2006). Cognitive neuroscience of emotional memory. *Nature Reviews Neuroscience, 7*(1), 54–64.

Lake, B. M., Ullman, T. D., Tenenbaum, J. B., & Gershman, S. J. (2017). Building machines that learn and think like people. *Behavioral and Brain Sciences, 40*, e253.

Lambon Ralph, M. A., Jefferies, E., Patterson, K., & Rogers, T. T. (2017). The neural and computational bases of semantic cognition. *Nature Reviews Neuroscience, 18*(1), 42–55.

Lee, J. L. C., Nader, K., & Schiller, D. (2017). An update on memory reconsolidation updating. *Trends in Cognitive Sciences, 21*(7), 531–545.

Lewis, P. A., & Durrant, S. J. (2011). Overlapping memory replay during sleep builds cognitive schemata. *Trends in Cognitive Sciences, 15*(8), 343–351.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems, 33*.

Li, Z., Song, S., et al. (2025). MemOS: An operating system for memory-augmented generation in large language models. arXiv:2505.22101.

Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2024). Lost in the middle: How language models use long contexts. *Transactions of the Association for Computational Linguistics, 12*, 157–173.

Logan, J. (2026). Continuum memory architectures for long-horizon LLM agents. arXiv:2601.09913.

Marr, D. (1982). *Vision: A Computational Investigation into the Human Representation and Processing of Visual Information*. W.H. Freeman.

McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. In G. H. Bower (Ed.), *The Psychology of Learning and Motivation* (Vol. 24, pp. 109–165). Academic Press.

McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex. *Psychological Review, 102*(3), 419–457.

McGaugh, J. L. (2004). The amygdala modulates the consolidation of memories of emotionally arousing experiences. *Annual Review of Neuroscience, 27*, 1–28.

Modarressi, A., et al. (2025). Position: Episodic memory is the missing piece for long-term LLM agents. *Proceedings of the 42nd International Conference on Machine Learning (ICML 2025)*. arXiv:2502.06975.

Moscovitch, M., Cabeza, R., Winocur, G., & Nadel, L. (2016). Episodic memory and beyond: The hippocampus and neocortex in transformation. *Annual Review of Psychology, 67*, 105–134.

Nader, K., & Hardt, O. (2009). A single standard for memory: The case for reconsolidation. *Nature Reviews Neuroscience, 10*(3), 224–234.

Nader, K., Schafe, G. E., & Le Doux, J. E. (2000). Fear memories require protein synthesis in the amygdala for reconsolidation after retrieval. *Nature, 406*(6797), 722–726.

O'Keefe, J., & Nadel, L. (1978). *The Hippocampus as a Cognitive Map*. Oxford University Press.

Packer, C., Wooders, S., Lin, K., Fang, V., Patil, S. G., Stoica, I., & Gonzalez, J. E. (2023). MemGPT: Towards LLMs as operating systems. arXiv:2310.08560.

Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative agents: Interactive simulacra of human behavior. *Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology (UIST 2023)*.

Patterson, K., Nestor, P. J., & Rogers, T. T. (2007). Where do you know what you know? The representation of semantic knowledge in the human brain. *Nature Reviews Neuroscience, 8*(12), 976–987.

Piaget, J. (1952). *The Origins of Intelligence in Children*. International Universities Press.

Preston, A. R., & Eichenbaum, H. (2013). Interplay of hippocampus and prefrontal cortex in memory. *Current Biology, 23*(17), R764–R773.

Rasch, B., & Born, J. (2013). About sleep's role in memory. *Physiological Reviews, 93*(2), 681–766.

Rasmussen, P., Paliychuk, P., Beauvais, T., Ryan, J., & Chalef, D. (2025). Zep: A temporal knowledge graph architecture for agent memory. arXiv:2501.13956.

Rezazadeh, A., Li, Z., Wei, W., & Bao, Y. (2025). From isolated conversations to hierarchical schemas: Dynamic tree memory representation for LLMs. *Proceedings of the International Conference on Learning Representations (ICLR 2025)*. arXiv:2410.14052.

Richards, B. A., & Frankland, P. W. (2017). The persistence and transience of memory. *Neuron, 94*(6), 1071–1084.

Rissanen, J. (1978). Modeling by shortest data description. *Automatica, 14*(5), 465–471.

Rumelhart, D. E. (1980). Schemata: The building blocks of cognition. In R. J. Spiro, B. C. Bruce, & W. F. Brewer (Eds.), *Theoretical Issues in Reading Comprehension*. Lawrence Erlbaum.

Sarch, G., et al. (2024). VLM agents generate their own memories: Distilling experience into embodied programs of thought. *Advances in Neural Information Processing Systems 37 (NeurIPS 2024, Spotlight)*.

Schacter, D. L., & Addis, D. R. (2007). The cognitive neuroscience of constructive memory. *Philosophical Transactions of the Royal Society B, 362*(1481), 773–786.

Schacter, D. L., Norman, K. A., & Koutstaal, W. (1998). The cognitive neuroscience of constructive memory. *Annual Review of Psychology, 49*(1), 289–318.

Shannon, C. E. (1959). Coding theorems for a discrete source with a fidelity criterion. *IRE National Convention Record, 7*(4), 142–163.

Spens, E., & Burgess, N. (2024). A generative model of memory construction and consolidation. *Nature Human Behaviour, 8*(3), 526–543.

Squire, L. R., & Zola, S. M. (1996). Structure and function of declarative and nondeclarative memory systems. *Proceedings of the National Academy of Sciences, 93*(24), 13515–13522.

Stachenfeld, K. L., Botvinick, M. M., & Gershman, S. J. (2017). The hippocampus as a predictive map. *Nature Neuroscience, 20*(11), 1643–1653.

Sumers, T. R., Yao, S., Narasimhan, K., & Griffiths, T. L. (2024). Cognitive architectures for language agents. *Transactions on Machine Learning Research (TMLR), 2024*.

Tse, D., Langston, R. F., Kakeyama, M., Bethus, I., Spooner, P. A., Wood, E. R., ... & Morris, R. G. (2007). Schemas and memory consolidation. *Science, 316*(5821), 76–82.

Vaccaro, A. G., & Fleming, S. M. (2018). Thinking about thinking: A coordinate-based meta-analysis of neuroimaging studies of metacognitive judgements. *Brain and Neuroscience Advances, 2*, 1–14.

van Kesteren, M. T., Ruiter, D. J., Fernández, G., & Henson, R. N. (2012). How schema and novelty augment memory formation. *Trends in Neurosciences, 35*(4), 211–219.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems, 30*.

Wei, L., et al. (2026). FadeMem: Biologically-inspired forgetting for efficient agent memory. arXiv:2601.18642.

Winocur, G., & Moscovitch, M. (2011). Memory transformation and systems consolidation. *Journal of the International Neuropsychological Society, 17*(5), 766–780.

Xia, M., Zhang, X., Dixit, S., Harimurugan, P., Wang, R., Ruhle, V., Sim, R., Bansal, C., & Rajmohan, S. (2026). Memora: A harmonic memory representation balancing abstraction and specificity. arXiv:2602.03315.

Yassa, M. A., & Stark, C. E. (2011). Pattern separation in the hippocampus. *Trends in Neurosciences, 34*(10), 515–525.

Zhao, A., Huang, D., Xu, Q., Lin, M., Liu, Y.-J., & Huang, G. (2024). ExpeL: LLM agents are experiential learners. *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2024)*. arXiv:2308.10144.

---

**Correspondence:** Hillary Danan, PhD — github.com/HillaryDanan

**Competing interests:** The author declares no competing interests.

This paper represents independent research conducted without institutional funding or affiliation.