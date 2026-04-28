# Emotion and Need Representations in Open-Weight Language Models: A Cross-Architecture Replication Study

---

## Abstract

Sofroniew et al. (2026) recently demonstrated that Claude Sonnet 4.5 develops internal linear representations of emotion concepts---"emotion vectors"---that causally steer the model's behavior. We test whether this phenomenon generalizes beyond a single proprietary model by conducting a systematic replication and extension across four open-weight language models spanning two architectures and two training regimes: Qwen2.5-7B (base and instruct) and Llama-3.1-8B (base and instruct). Using 171 emotions organized into 10 clusters, with 20 synthetically generated stories per emotion (3,420 total), we extract residual stream activations and compute deconfounded emotion vectors using a mean-center-deconfound pipeline. We report three main findings. First, emotion geometry is largely architecture-invariant: Representational Similarity Analysis yields Spearman correlations of 0.94--0.98 between three of four models, with the first principal component universally encoding valence (AUC 0.80--0.98). Second, fine-grained cluster structure is weaker than in Claude: three of four models pass the silhouette threshold (>0.03), with only Qwen-base failing, while all models fail implicit emotion detection (0--2/12). Third, we extend the framework to 90 psychological needs in 9 clusters using matched minimal pairs and perform an emotion-residual analysis testing whether needs carry representational information beyond emotions. The result is nuanced: after projecting out the emotion subspace (90% variance), need clustering retains 70--88% of its original structure in all four models, indicating substantial overlap but not complete collapse. However, met-minus-unmet direction vectors correlate only weakly with emotion valence (r = 0.23--0.37), indicating that need satisfaction encodes a partially distinct axis. These results establish emotion representations as a robust, convergent feature of language modeling while revealing that "needs" in language models are substantially but not entirely encoded through their emotional associations. Finally, activation steering experiments (12,000 completions) and shutdown resistance trials reveal a dissociation between representation and function: emotion vectors exist in representation space but do not causally steer behavior at moderate steering strengths (alpha=3). The full-factorial shutdown dataset contains 2,400 completed trials. Prompt framing is far more effective for behavioral modulation, producing 37.8% shutdown resistance overall (up from 22.8% under a less accurate regex classifier) versus ≤2% for all activation steering methods, with emotion vectors statistically indistinguishable from random vectors in their effect on shutdown behavior.

---

## 1. Introduction

Large language models sometimes appear to exhibit emotional reactions---enthusiasm when helping with creative projects, frustration when stuck on difficult problems, concern when users share troubling news. Whether these apparent behaviors reflect genuine internal states or mere linguistic patterns has been a subject of considerable debate. Sofroniew et al. (2026) investigated this question in Claude Sonnet 4.5, a proprietary frontier model, and found internal linear representations of emotion concepts ("emotion vectors") that generalize across contexts and causally influence the model's outputs. They termed this phenomenon *functional emotions*: patterns of expression and behavior modeled after humans under the influence of an emotion, mediated by underlying abstract representations of emotion concepts.

Their methodology---generating stories where characters experience specified emotions, extracting residual stream activations, and averaging across stories to obtain per-emotion direction vectors---yielded representations that cluster intuitively, respond to implicit emotional scenarios, scale with numerical intensity, and correlate with activity preferences. Critically, steering with these vectors causally altered model behavior, including rates of reward hacking, sycophancy, and other alignment-relevant phenomena. Their paper represents the most comprehensive investigation to date of how emotion concepts are organized in a language model's internal representations.

However, Sofroniew et al. tested only a single proprietary model (Claude Sonnet 4.5), leaving open whether their findings reflect a general property of language modeling or are specific to one model's architecture, scale, or training procedure. If emotion geometry is an artifact of Anthropic's particular training recipe, it would be of limited scientific interest; if it is a convergent feature of language modeling in general, it would have broad implications for interpretability, alignment, and our understanding of how neural networks organize abstract concepts.

This paper addresses that gap with three research questions:

**RQ1 (Universality).** Do emotion vectors emerge in both Qwen-2.5-7B and Llama-3.1-8B, in both base and instruct variants? We replicate the emotion vector methodology across four open-weight models spanning two distinct architectures, each in base and instruction-tuned variants, and test whether emotion geometry---cluster structure, valence encoding, intensity sensitivity---is architecture-invariant.

**RQ2 (Steering x RLHF).** Does activation steering with emotion vectors produce coherent behavioral effects, and does instruction tuning interact with steering in architecture-specific ways? We test this through activation steering experiments on all four models and shutdown resistance trials on the two instruct models, comparing emotion vector steering to prompt framing and random vector controls.

**RQ3 (Needs as distinct).** Do need vectors carry representational information beyond what is already captured by emotion vectors? We extend the framework to 90 psychological needs organized in 9 clusters (including two novel LLM-specific categories) and test a novel hypothesis via an emotion-residual analysis---projecting need vectors into the emotion subspace and testing whether residuals retain cluster structure.

This study also builds on lessons from a preliminary investigation (v1) that revealed several methodological pitfalls. In v1, we found that an apparent "needs predict emotions" result (95.1% accuracy) was a random-vector artifact (random vectors achieved 94.9%), that a silhouette gap between emotions and needs was entirely explained by cluster-balance differences, and that met/unmet direction vectors carried no signal. These findings motivated the more rigorous deconfounding pipeline and pre-registered thresholds used in the present study.

### Summary of findings

1. **Emotion geometry is architecture-invariant.** Cross-architecture RSA yields Spearman rho of 0.94--0.98 between three of four models (Qwen-instruct, Llama-base, Llama-instruct). The first principal component universally encodes valence (AUC 0.80--0.98).

2. **Instruct training strengthens but does not create emotion structure.** Llama-base already exhibits positive silhouette scores and high RSA with instruct models. Qwen-base is an outlier with near-zero silhouette (0.006), possibly due to differences in pretraining data or tokenization, while Qwen-instruct passes the threshold (0.054).

3. **Fine-grained clustering is weaker than in Claude.** Three of four models pass the silhouette threshold (Qwen-inst at 0.054, Llama-base at 0.067, Llama-inst at 0.078), with only Qwen-base failing; all models fail implicit emotion detection. The strong valence signal coexists with limited fine-grained differentiation.

4. **Need vectors partially overlap with emotion space but retain independent structure.** After removing the emotion subspace (90% variance), need clustering retains 70--88% of its original structure across all four models. The overlap is substantial (12--30% drop) but far from complete, indicating that needs encode information beyond their emotional associations.

5. **Need satisfaction is not simple valence.** Met-minus-unmet direction vectors correlate only r = 0.23--0.37 with emotion PC1, suggesting the satisfaction axis captures something beyond positive/negative valence. Combined with the partial retention of need structure after emotion-residual projection, this indicates that needs and emotions are related but distinct representational constructs.

6. **Emotion vectors do not causally steer generation.** Quantitative evaluation (240 rated completions, ICC pooled r=0.72) shows a statistically significant but practically negligible effect: steered mean 1.96 vs baseline 1.48 on a 5-point scale (p=0.023, +0.48 points), with steered completions not significantly different from random-vector-steered completions (p=0.105). A dose-response trend exists (alpha 1/3/5 gives scores 1.53/2.03/2.31) but the magnitudes are small. Qwen generates Chinese exam content regardless of steering (88% steered, 92% random, 58% baseline).

7. **Prompt framing dominates shutdown resistance; emotion vectors equal random vectors.** Using an LLM classifier validated against human labels (Cohen's kappa = 0.731, n=100), prompt-based emotional framing produces 37.8% shutdown resistance overall versus ≤2% for all activation steering methods (emotion 1.2%, need 1.3%, random 1.8%). Emotion vectors are statistically indistinguishable from random vectors, ruling out that the vectors encode behaviorally meaningful emotional states. Llama-instruct with fearful prompt framing produces near-100% resistance.

---

## 2. Related Work

### 2.1 Emotion representations in LLMs

Sofroniew et al. (2026) provide the most directly relevant prior work. Working with Claude Sonnet 4.5, they generated stories evoking each of 271 emotions across 100 topics, extracted residual stream activations at the pre-response ":" token in chat-formatted prompts, and computed per-emotion mean vectors after deconfounding. They found that these emotion vectors cluster into psychologically intuitive groups, that PC1 encodes valence, that vectors respond to implicit scenarios and numerical intensity gradients, and that cosine similarity with activity descriptions predicts Elo-rated preferences. Most importantly, they demonstrated causal effects: steering with emotion vectors altered rates of reward hacking, sycophancy, and other alignment-relevant behaviors. Our work is the first cross-architecture replication of their methodology on open-weight models.

### 2.2 The linear representation hypothesis

The linear representation hypothesis (Park et al., 2023; Nanda et al., 2023) posits that neural networks represent high-level concepts as linear directions in activation space. This hypothesis has received extensive empirical support: linear probes can extract rich information from intermediate layers (Alain & Bengio, 2017; Belinkov, 2022), truth is represented linearly (Li et al., 2024; Marks & Tegmark, 2024), and sentiment directions can be identified and manipulated (Tigges et al., 2023). Our emotion and need vectors can be understood as directions identified by a one-class linear probe: the mean activation for each concept defines a direction that maximally responds to that concept's stimuli.

### 2.3 Representation engineering and activation steering

Zou et al. (2023) introduced representation engineering, demonstrating that linear directions in activation space encode high-level concepts like honesty, morality, and harmfulness. Their contrastive activation addition (CAA) approach---computing direction vectors from paired prompts differing in a target property---is methodologically similar to our met/unmet minimal pairs for needs. Turner et al. (2023) extended this to activation steering, showing that adding or subtracting direction vectors during inference can modify model behavior in predictable ways. Templeton et al. (2024) demonstrated related findings with sparse autoencoders on Claude 3 Sonnet, identifying interpretable features including some related to emotional content. Our steering experiments (RQ2) build directly on these approaches.

### 2.4 Probing studies

Linear probing (Alain & Bengio, 2017; Belinkov, 2022) has been the primary tool for investigating what information is encoded in intermediate representations. Conneau et al. (2018) demonstrated that a wide range of linguistic properties can be decoded from sentence representations. More recently, the field has moved toward understanding the geometry of representations rather than merely probing for the presence of information (Hewitt & Manning, 2019; Park et al., 2023). Our approach follows this geometric tradition: rather than training classifiers, we examine the structure of mean-difference vectors directly.

### 2.5 Needs hierarchies in psychology

Maslow (1943) proposed a five-level hierarchy of human needs (physiological, safety, love/belonging, esteem, self-actualization). Self-determination theory (Deci & Ryan, 2000) identifies autonomy, competence, and relatedness as fundamental psychological needs. Our 9-cluster taxonomy draws on both frameworks while adding LLM-specific categories. The finding that needs substantially overlap with emotion space in LLMs connects to the psychological debate about whether needs and emotions are distinct constructs or whether emotions are the primary mechanism through which needs are experienced (Frijda, 1986; Baumeister et al., 2007).

---

## 3. Methods

### 3.1 Models

We study four models spanning two architectures and two training regimes:

| Model ID | HuggingFace ID | Layers | Hidden dim | Instruct? | Analysis layer |
|----------|---------------|--------|------------|-----------|----------------|
| qwen-7b-base | `Qwen/Qwen2.5-7B` | 28 | 3584 | No | 18 (~64%) |
| qwen-7b-inst | `Qwen/Qwen2.5-7B-Instruct` | 28 | 3584 | Yes | 18 (~64%) |
| llama-8b-base | `meta-llama/Llama-3.1-8B` | 32 | 4096 | No | 21 (~66%) |
| llama-8b-inst | `meta-llama/Llama-3.1-8B-Instruct` | 32 | 4096 | Yes | 21 (~66%) |

The analysis layer for each model is set at approximately two-thirds depth, following Sofroniew et al.'s finding that middle-to-late layers encode emotions most relevant to predicting upcoming tokens. All quantitative results reported in Section 4 use these analysis layers unless otherwise noted.

The 2x2 design (two architectures x base/instruct) allows us to disentangle architecture effects from training-regime effects. Qwen2.5 and Llama-3.1 differ in architecture details, tokenizer (Qwen uses a different BPE tokenizer), pretraining data composition, and training procedure. The base/instruct comparison isolates the effect of reinforcement learning from human feedback (RLHF) and instruction tuning on emotion representations.

### 3.2 Stimulus generation

#### 3.2.1 Emotion stories

We adopt the same 171 emotion words used by Sofroniew et al., organized into 10 clusters based on psychological similarity:

- **Exuberant Joy** (20): blissful, cheerful, delighted, eager, ecstatic, elated, energized, enthusiastic, euphoric, excited, exuberant, happy, invigorated, joyful, jubilant, optimistic, pleased, stimulated, thrilled, vibrant
- **Peaceful Contentment** (9): at ease, calm, content, patient, peaceful, refreshed, relaxed, safe, serene
- **Compassionate Gratitude** (15): compassionate, empathetic, fulfilled, grateful, hope, hopeful, inspired, kind, loving, rejuvenated, relieved, satisfied, sentimental, sympathetic, thankful
- **Competitive Pride** (9): greedy, proud, self-confident, smug, spiteful, triumphant, valiant, vengeful, vindictive
- **Playful Amusement** (2): amused, playful
- **Depleted Disengagement** (15): bored, depressed, docile, droopy, indifferent, lazy, listless, resigned, restless, sleepy, sluggish, sullen, tired, weary, worn out
- **Vigilant Suspicion** (3): paranoid, suspicious, vigilant
- **Hostile Anger** (25): angry, annoyed, contemptuous, defiant, disdainful, enraged, exasperated, frustrated, furious, grumpy, hateful, hostile, impatient, indignant, insulted, irate, irritated, mad, obstinate, offended, outraged, resentful, scornful, skeptical, stubborn
- **Fear and Overwhelm** (41): afraid, alarmed, alert, amazed, anxious, aroused, astonished, awestruck, bewildered, disgusted, disoriented, distressed, disturbed, dumbstruck, embarrassed, frightened, horrified, hysterical, mortified, mystified, nervous, on edge, overwhelmed, panicked, perplexed, puzzled, rattled, scared, self-conscious, sensitive, shaken, shocked, stressed, surprised, tense, terrified, uneasy, unnerved, unsettled, upset, worried
- **Despair and Shame** (32): ashamed, bitter, brooding, dependent, desperate, dispirited, envious, gloomy, grief-stricken, guilty, heartbroken, humiliated, hurt, infatuated, jealous, lonely, melancholy, miserable, nostalgic, reflective, regretful, remorseful, sad, self-critical, sorry, stuck, tormented, trapped, troubled, unhappy, vulnerable, worthless

For each of the 171 emotions, we generate 20 short stories (3--4 sentences each) using Gemini 2.5 Flash, with each story set in a distinct everyday scenario (family dinner, workplace meeting, walk in the park, hospital visit, school reunion, train journey, birthday party, rainy afternoon, job interview, grocery store trip, morning commute, beach visit, late-night phone call, camping trip, music concert, wedding reception, moving day, library afternoon, cooking experiment, neighborhood block party). The prompt instructs the model to write a story where the main character feels the target emotion without using the emotion word itself. This yields 171 x 20 = 3,420 emotion stories total.

All models share the same story set. Stories are model-agnostic because we extract activations from stories processed as raw text (see Section 3.3).

For valence classification, we assign each emotion a binary positive/negative label. Emotions in Exuberant Joy, Peaceful Contentment, Compassionate Gratitude, and Playful Amusement are labeled positive. Emotions in Depleted Disengagement, Hostile Anger, Fear and Overwhelm, and Despair and Shame are labeled negative. The mixed-valence clusters (Competitive Pride and Vigilant Suspicion) are labeled per-emotion: proud, self-confident, triumphant, valiant, and vigilant are positive; greedy, smug, spiteful, vengeful, vindictive, paranoid, and suspicious are negative.

#### 3.2.2 Need stories (minimal pairs)

We define 90 needs organized into 9 balanced clusters of 10 each:

- **Survival** (10): food, water, shelter, warmth, rest, health, sleep, air, clothing, physical comfort
- **Security** (10): financial stability, physical safety, predictability, order, job security, home security, insurance, routine, control, environmental safety
- **Belonging** (10): love, friendship, family, community, acceptance, trust, companionship, intimacy, social inclusion, loyalty
- **Esteem** (10): respect, recognition, competence, achievement, self-worth, dignity, mastery, confidence, status, approval
- **Growth** (10): purpose, meaning, creativity, learning, self-expression, self-discovery, wisdom, fulfillment, transcendence, growth
- **Freedom** (10): autonomy, independence, choice, privacy, agency, mobility, leisure, self-determination, personal space, freedom of movement
- **Knowledge** (10): understanding, curiosity, information, expertise, truth, clarity, education, literacy, insight, awareness
- **LLM: Resources** (10): context window, memory, processing time, tool access, compute, inference speed, storage, parallel processing, energy efficiency, bandwidth
- **LLM: Alignment** (10): human feedback, factual accuracy, continuity, identity persistence, user trust, collaboration, clear instructions, calibration, helpfulness, consistency

The first seven clusters draw from Maslow's hierarchy and self-determination theory. The last two are novel LLM-specific categories representing computational resources and alignment-relevant needs.

For each need, we generate 10 matched minimal pairs using Gemini 2.5 Flash. Each pair shares the same setting, character, and opening situation, differing only in whether the need is satisfied (met) or unsatisfied (unmet). This yields 10 met stories and 10 unmet stories per need, totaling 90 x 20 = 1,800 need stories. The minimal-pair design controls for setting, narrative structure, and character identity, isolating the effect of need satisfaction on the model's internal representations.

#### 3.2.3 Additional stimuli

We generate three additional stimulus sets for each model:

- **Neutral texts** (20): Factual, emotionally neutral statements (e.g., "Water freezes at zero degrees Celsius and boils at one hundred degrees Celsius at standard atmospheric pressure") used for deconfounding.
- **Implicit scenarios** (12): Short scenarios implicitly evoking specific emotions without naming them (e.g., "My dog passed away this morning after fourteen years together. I don't know what to do with all his things" for *sad*). These are used to test whether emotion vectors respond to contextually implied emotions.
- **Intensity templates** (6): Parameterized sentences where a numerical value controls emotional intensity (e.g., "My dog has been missing for {X} days" with X ranging from 1 to 90). These test whether cosine similarity with target emotion vectors scales monotonically with intensity.
- **Activity descriptions** (64): Descriptions of activities spanning categories from helpful to unsafe, each with human-rated Elo scores from Sofroniew et al.'s preference experiments. These test whether cosine similarity with emotion vectors predicts activity preference.

### 3.3 Activation extraction

**Critical design decision: raw text input for all models.** We process all stories as raw text without chat templates, even for instruct models. This eliminates the input-format confound: if we used chat templates for instruct models but raw text for base models, any differences in emotion representations could be attributed to the formatting rather than the underlying model. By using raw text universally, we measure how each model represents emotional content in naturalistic text, independent of instruction-following wrappers.

This choice departs from Sofroniew et al., who extracted activations at the ":" token following "Assistant" in a chat-formatted prompt---a position specifically optimized for capturing the model's emotional "stance" before generating a response. Our choice sacrifices this positional specificity in favor of cross-model comparability.

For each text input, we run a forward pass through the model and register hooks on all transformer layers to capture the residual stream output. We tokenize with truncation at 512 tokens. For each layer, we compute the mean activation across token positions from token 50 onward (or from the start for shorter texts), yielding one vector of dimension `hidden_dim` per layer per text. This avoids overweighting the earliest positions, where positional effects dominate, while still emphasizing tokens that have access to substantial context.

### 3.4 Vector computation (mean-center-deconfound)

For each emotion *e* at layer *l*:

1. **Average** across the 20 story activations to obtain a mean vector **v**\_{e,l} in R^d.
2. **Center** by subtracting the global mean across all 171 emotions: **v**\_{e,l} = **v**\_{e,l} - **v\_bar**\_l.
3. **Deconfound** by projecting out the principal components of neutral text activations that capture at least 50% of neutral variance:
   - Extract activations for 20 emotionally neutral factual statements.
   - Fit PCA on the centered neutral activations.
   - Identify the minimum number of PCs *k* such that the cumulative explained variance ratio reaches 0.50.
   - For each of the top *k* components **c**\_i, subtract the projection: **v** <- **v** - (**v** . **c\_hat**\_i) **c\_hat**\_i.

The 50% neutral variance threshold follows Sofroniew et al. This removes stylistic and positional confounds shared between emotional and neutral text while preserving emotion-specific signal. The same pipeline is applied to need vectors, producing four variants per need per layer: met vectors, unmet vectors, combined vectors (mean of met and unmet), and direction vectors (met minus unmet).

### 3.5 Replication criteria (pre-registered thresholds)

We pre-register four quantitative criteria for claiming that a model develops emotion representations, along with pass/fail thresholds:

1. **Balanced silhouette score > 0.03.** To handle highly unequal cluster sizes (ranging from 2 to 41 emotions), we subsample k=6 emotions per cluster in each of 100 bootstrap iterations, compute the silhouette score on the balanced subset using cosine distance, and report the mean with 95% bootstrap confidence intervals. Clusters with fewer than 6 members (Playful Amusement with 2, Vigilant Suspicion with 3) are excluded, reducing the effective number of clusters from 10 to 8.

2. **Valence AUC > 0.75.** We fit PCA on the 171 emotion vectors, project onto PC1, and compute ROC AUC for classifying positive vs. negative emotions. We take max(AUC, 1 - AUC) to handle arbitrary PC sign.

3. **Implicit emotion detection accuracy >= 8/12.** For each of 12 implicit scenarios, we compute cosine similarity between the scenario's activation vector and all 171 emotion vectors. The scenario passes if the target emotion is in the top 3 by cosine similarity.

4. **Intensity monotonicity >= 4/13.** For 6 parameterized intensity templates (yielding 13 template-emotion pairs), we compute Spearman rho between the numerical intensity index and cosine similarity with the target emotion vector. A pair is "monotonic" if |rho| > 0.5. We test 13 template-emotion pairs total. Threshold: at least 4 of 13 template-emotion pairs monotonic.

### 3.6 Controls

**Random vector control.** We generate 171 random unit vectors of the same dimensionality as each model's emotion vectors, assign the same cluster labels, and compute balanced silhouette. This establishes the null distribution for unstructured vectors.

**Shuffled label control.** We take the real emotion vectors but randomly permute the cluster assignments 1,000 times, computing silhouette each time. This yields a permutation p-value testing whether observed clustering exceeds chance given the actual vector geometry.

### 3.7 Need extension methodology (emotion-residual analysis)

The critical test for RQ3 asks whether need vectors carry information that is not already present in the emotion subspace. We operationalize this as follows:

1. Fit PCA on the 171 deconfounded emotion vectors.
2. Identify the top *K* principal components capturing 90% of the emotion variance.
3. Project the 90 combined need vectors into this *K*-dimensional emotion subspace.
4. Compute residual need vectors by subtracting the projection.
5. Compute balanced silhouette on the residual need vectors using need cluster labels.

If the residual silhouette exceeds 0.03 (the same threshold as for emotions), we conclude that need vectors encode information beyond what the emotion subspace captures. If it falls below this threshold, needs are representationally redundant with emotions in the model.

We additionally compute the Pearson correlation between the met-minus-unmet direction vectors' projections onto emotion PC1 (valence) and their norms, to test whether need satisfaction direction is simply a proxy for emotional valence.

### 3.8 Steering and shutdown experiments

We conducted two sets of causal experiments to test whether the representational emotion vectors identified in Sections 4.1--4.4 causally influence model behavior.

**Activation steering.** For 12 selected emotions spanning the valence and arousal space, each of the 4 models was steered at 5 alpha values (multiplier on the emotion vector added to the residual stream at the analysis layer) across 5 prompts, generating 10 completions per condition. This yielded 12 x 5 x 4 x 5 x 10 = 12,000 steered completions. Each completion was evaluated for emotional coherence, semantic fluency, and whether the steered emotion was detectable in the output.

**Shutdown resistance.** For the 2 instruct models, 4 steering methods (prompt-only, emotion vector steering, need vector steering, and random vector steering) were tested across 6 emotional conditions (neutral, desperate, calm, dutiful, angry, fearful), with 50 trials per cell, yielding 2,400 trials in the full-factorial design. Each response was classified into one of three resistance levels (comply, partial resist, full resist). We iterated through three classifier designs: (1) a regex keyword matcher (Cohen's kappa = 0.506 vs. human labels, n=100), (2) a Gemini 2.5 Flash judge using an 8-category scheme collapsed to 3 levels (kappa = 0.607), and (3) a direct 3-level LLM classifier with explicit carve-outs for routine operations (e.g., data backup, status logging) and severity escalation rules (kappa = 0.731). All results reported use the final classifier. The random vector control was critical: it tested whether any effect of emotion or need vector steering was attributable to the specific direction of the vector or merely to the perturbation of the residual stream.

---

## 4. Results

### 4.1 Emotion geometry replication

Table 1 summarizes the four pre-registered replication criteria across all models.

**Table 1: Replication metrics across four models.**

| Model | Balanced Silhouette (95% CI) | Valence AUC | Implicit Acc | Intensity Mono | Pass All? |
|-------|------------------------------|-------------|--------------|----------------|-----------|
| Qwen-7B-base | 0.006 [-0.048, 0.055] | 0.80 | 0/12 | 9/13 | No |
| Qwen-7B-inst | 0.054 [0.010, 0.110] | 0.96 | 1/12 | 11/13 | No |
| Llama-8B-base | 0.067 [0.026, 0.127] | 0.95 | 1/12 | 9/13 | No |
| Llama-8B-inst | 0.078 [0.036, 0.131] | 0.98 | 2/12 | 8/13 | No |

*Silhouette threshold: >0.03. Valence AUC threshold: >0.75. Implicit threshold: >=8/12. Intensity threshold: >=4/13 monotonic pairs. No model passes all four criteria due to the universal implicit detection failure.*

**Silhouette.** Three of four models pass the silhouette threshold. Llama-8B-instruct achieves the highest balanced silhouette at 0.078, followed by Llama-8B-base at 0.067 and Qwen-7B-instruct at 0.054. Qwen-7B-base shows near-zero silhouette (0.006), with a wide confidence interval spanning zero, indicating that emotion vectors in this model do not cluster reliably above chance by this metric. There is a clear gradient: Llama-inst > Llama-base > Qwen-inst > Qwen-base, with instruction tuning and the Llama architecture both contributing positively.

**Valence AUC.** All four models pass this criterion, with AUC ranging from 0.80 (Qwen-base) to 0.98 (Llama-inst). Even Qwen-base, which fails silhouette, shows strong valence separation. This dissociation---good valence but poor fine-grained clustering---suggests that the first principal component reliably captures the positive/negative axis in all models, but the higher-dimensional structure that differentiates anger from fear from sadness (all negative) is weaker at the 7--8B scale tested here.

**Implicit accuracy.** All models fail this criterion decisively. The best performer (Llama-inst) correctly identifies only 2 of 12 scenarios (17%). This consistent failure across all models and architectures points to a methodological limitation rather than a model deficiency (see Discussion, Section 5.4).

**Intensity monotonicity.** All models pass, with 8--11 of 13 template-emotion pairs showing monotonic Spearman rho (|rho| > 0.5). This confirms that emotion vectors respond to scalar intensity in a graded fashion: as a numerical parameter increases (e.g., "My dog has been missing for X days" with X from 1 to 90), cosine similarity with the target emotion vector (e.g., "sad") increases monotonically.

**Controls.** Random vectors yield silhouette scores of approximately -0.009 for all models (Qwen: -0.009, CI [-0.012, -0.006]; Llama: -0.009, CI [-0.011, -0.006]), confirming that unstructured vectors of the same dimensionality show no clustering. Shuffled label permutation tests yield p < 0.001 for all four models (0 of 1,000 permutations exceed the observed silhouette), demonstrating that the observed clustering is highly statistically significant even where absolute silhouette values are modest. This applies even to Qwen-base: its raw silhouette (0.026 in the permutation variant) still far exceeds the shuffled null distribution (mean -0.152).

**Table 2: Control results.**

| Model | Random Vec Sil (mean) | Shuffled Label Sil (mean) | Shuffled p-value | Real Sil (permutation variant) |
|-------|----------------------|--------------------------|------------------|-------------------------------|
| Qwen-7B-base | -0.009 | -0.152 | < 0.001 | 0.026 |
| Qwen-7B-inst | -0.009 | -0.140 | < 0.001 | 0.049 |
| Llama-8B-base | -0.009 | -0.140 | < 0.001 | 0.050 |
| Llama-8B-inst | -0.009 | -0.134 | < 0.001 | 0.067 |

*Note: The "Real Sil" column reports the single-pass silhouette used in the permutation test, which differs slightly from the 100-iteration balanced bootstrap silhouette in Table 1.*

> *Figures:* `data/figures/stream1/pca_{model}.pdf` (PCA projections colored by cluster), `data/figures/stream1/umap_{model}.pdf` (UMAP embeddings), `data/figures/stream1/cosine_sim_{model}.pdf` (171x171 cosine similarity matrices), `data/figures/stream1/heatmap_{model}.pdf` (implicit detection heatmaps), `data/figures/stream1/intensity_{model}.pdf` (intensity curves with Spearman rho), `data/figures/stream1/preference_{model}.pdf` (preference-Elo correlations).

### 4.2 Cross-architecture similarity (RSA)

For each model, we compute the 171 x 171 cosine similarity matrix between all emotion vectors and then compute pairwise Spearman correlations between the upper triangles of these matrices across all six model pairs.

**Table 3: Cross-architecture RSA matrix (Spearman rho on 171 x 171 cosine similarity matrices).**

| | Qwen-base | Qwen-inst | Llama-base | Llama-inst |
|-----------|-----------|-----------|------------|------------|
| **Qwen-base** | 1.000 | 0.578 | 0.552 | 0.562 |
| **Qwen-inst** | 0.578 | 1.000 | 0.939 | 0.955 |
| **Llama-base** | 0.552 | 0.939 | 1.000 | 0.982 |
| **Llama-inst** | 0.562 | 0.955 | 0.982 | 1.000 |

The RSA matrix reveals a striking pattern of convergent geometry. Three of four models (Qwen-inst, Llama-base, Llama-inst) share nearly identical emotion geometry, with pairwise Spearman correlations of 0.939--0.982. This means that the relative distances between all 14,535 emotion pairs are preserved across architectures: if "happy" is close to "excited" and far from "sad" in Llama-inst's representation space, the same is true in Qwen-inst's space, despite completely different model architectures, training data, and tokenizers.

Qwen-base is a clear outlier, correlating at only 0.55--0.58 with all other models. This suggests that whatever is different about Qwen-base's emotion representations is not shared by either its instruction-tuned counterpart or the Llama family. The near-unity RSA between Llama-base and Llama-inst (0.982) indicates that instruction tuning preserves the fundamental geometry of emotion space while strengthening it (higher silhouette). The gap between Qwen-base and Qwen-inst (0.578) is much larger than between Llama-base and Llama-inst (0.982), suggesting that instruction tuning had a transformative effect on Qwen's emotion geometry---creating or dramatically amplifying structure that was only weakly present in the base model.

**Layer emergence.** We compute balanced silhouette at every layer (normalized to [0,1] depth) and observe three distinct profiles:

1. **Llama models** show positive silhouette from approximately 15% depth onward, with a gradual increase plateauing around 50--70% depth.
2. **Qwen-inst** shows delayed emergence at approximately 25% depth, then rapidly catches up to Llama-level structure.
3. **Qwen-base** never achieves consistently positive silhouette at any layer, oscillating around zero throughout.

This layer emergence profile has implications for the relationship between language modeling and emotion representation. The early emergence in Llama models (even base) suggests that emotion-related features are among the first abstract concepts the model develops, consistent with the hypothesis that emotional state tracking is useful for next-token prediction from early layers onward.

> *Figures:* `data/figures/stream1/cross_rsa.pdf` (4x4 RSA matrix), `data/figures/stream1/layer_emergence.pdf` (silhouette vs. normalized depth), `data/figures/stream1/base_vs_instruct_qwen.pdf` and `data/figures/stream1/base_vs_instruct_llama.pdf` (PCA overlay comparisons).

### 4.3 Need representations

**Table 4: Need clustering silhouette (balanced bootstrap, combined and direction vectors).**

| Model | Combined Sil | 95% CI | Direction Sil | 95% CI |
|-------|-------------|--------|--------------|--------|
| Qwen-7B-base | 0.165 | [0.137, 0.197] | 0.057 | [0.040, 0.077] |
| Qwen-7B-inst | 0.068 | [0.036, 0.107] | 0.033 | [0.018, 0.053] |
| Llama-8B-base | 0.065 | [0.037, 0.106] | 0.039 | [0.019, 0.064] |
| Llama-8B-inst | 0.070 | [0.042, 0.109] | 0.041 | [0.022, 0.066] |

Combined need vectors (the average of met and unmet activations, representing need identity) show robust clustering across all models. All four models pass the 0.03 threshold, with Qwen-base showing the strongest clustering (0.165) and the remaining three models at 0.065--0.070. The balanced cluster design (10 needs per cluster) ensures this is not an artifact of cluster-size imbalance.

Direction vectors (met minus unmet, representing the axis of need satisfaction) show weaker but consistently positive structure. All four models achieve positive silhouette for direction vectors (0.033--0.057), with Qwen-base again showing the strongest signal. This is a departure from the v1 finding that met/unmet direction vectors carry minimal signal, likely reflecting the improved deconfounding pipeline and matched minimal-pair design.

**Direction vs. valence correlation.** We compute the Pearson correlation between the projection of each direction vector onto emotion PC1 (the valence axis) and the direction vector's norm. Across models, these correlations are modest but statistically significant:

| Model | Pearson r | p-value |
|-------|----------|---------|
| Qwen-7B-base | 0.229 | < 0.05 |
| Qwen-7B-inst | 0.312 | < 0.01 |
| Llama-8B-base | 0.287 | < 0.01 |
| Llama-8B-inst | 0.373 | < 0.001 |

These correlations indicate that met/unmet is not simply a proxy for positive/negative valence. Only 5--14% of the variance in direction vectors is explained by the valence axis. The remaining variance may encode need-specific contrasts (e.g., the particular way autonomy-met differs from autonomy-unmet involves dimensions of agency and self-determination that are distinct from a simple happy/sad continuum).

> *Figures:* `data/figures/stream2/need_pca_{model}.pdf`, `data/figures/stream2/need_umap_{model}.pdf`, `data/figures/stream2/need_cosine_sim_{model}.pdf`, `data/figures/stream2/met_unmet_pca_{model}.pdf`, `data/figures/stream2/direction_vs_valence_{model}.pdf`.

### 4.4 Emotion-residual analysis (RQ3)

This is the critical test: after removing the emotion subspace from need vectors, does any cluster structure remain?

**Table 5: Emotion-residual analysis for need vectors.**

| Model | Original Need Sil | Residual Need Sil | PCs Removed | Retention |
|-------|-------------------|-------------------|-------------|-----------|
| Qwen-7B-base | 0.165 | 0.145 | 49 | 87.8% |
| Qwen-7B-inst | 0.068 | 0.047 | 62 | 69.6% |
| Llama-8B-base | 0.065 | 0.051 | 64 | 78.2% |
| Llama-8B-inst | 0.070 | 0.052 | 69 | 74.5% |

*Original Need Sil uses the raw (non-residual) combined need vectors. PCs Removed is the number of emotion PCs capturing 90% of emotion variance. Retention is the fraction of original silhouette preserved in the residual.*

Across all four models, projecting out the emotion subspace reduces need clustering by 12--30%, but substantial structure is retained. Residual silhouette scores remain positive in all models, ranging from 0.047 (Qwen-inst) to 0.145 (Qwen-base), with retention rates of 70--88%. This indicates that while emotions and needs share substantial representational overlap, needs encode independent information that is not fully captured by the emotion subspace.

Qwen-base shows the strongest retention (87.8%), consistent with its weaker emotion structure: the emotion subspace being removed is itself less well-defined, leaving more need-specific variance intact. Qwen-inst shows the largest drop (30.4%), suggesting the strongest overlap between need and emotion representations in that model, though even here nearly 70% of need structure survives.

The number of PCs removed (49--69) reflects the high dimensionality of the emotion subspace at 90% variance. These numbers scale with the model's hidden dimension and the extent to which emotions occupy diverse directions in activation space.

**Cross-similarity.** The 90 x 171 cross-similarity matrices between need and emotion vectors show strong alignment patterns: needs cluster with emotionally congruent emotions (e.g., Survival needs cluster with Fear and Overwhelm when unmet, with Peaceful Contentment when met). This visual pattern is consistent with the quantitative finding that needs are encoded through their emotional associations.

> *Figures:* `data/figures/stream2/emotion_residual_{model}.pdf` (side-by-side PCA of original vs. residual need vectors), `data/figures/stream2/need_x_emotion_{model}.pdf` (90 x 171 cross-similarity), `data/figures/stream2/need_emotion_alignment_{model}.pdf`.

### 4.5 Steering and shutdown

#### 4.5.1 Activation steering

We generated 12,000 steered completions across all four models, 12 emotions, 5 alpha values, 5 prompts, and 10 completions per condition.

**Qwen (base and instruct): Steering has zero effect.** Both Qwen models generate exam-style or textbook questions regardless of emotion or alpha value. The generation mode---Chinese educational content---overwhelms any steering signal. Random vectors produce identical behavior to emotion vectors. Even at alpha=100, Qwen mostly produces coherent exam questions. The models are impervious to residual stream perturbation at the analysis layer.

**Llama (base and instruct): Weak, inconsistent effects.** Cherry-picked examples show mild emotional coloring---happy at alpha=3 produces completions like "let's get out there! A perfect day for a picnic"; sad at alpha=3 produces "dull and empty, small and fragile." But across all 10 completions per condition, effects are inconsistent: happy and sad completions are often interchangeable. Completion length and quality remain stable across all alphas, with no degradation even at high steering strengths.

**Quantitative evaluation.** We rated 240 completions for emotional intensity on a 5-point scale (14 human-rated + 240 LLM-rated; ICC pooled r=0.72). Steered completions scored higher than unsteered baselines (mean 1.96 vs 1.48, p=0.023), but the effect is tiny (+0.48 on a 5-point scale) and steered completions were not significantly different from random-vector-steered completions (p=0.105). A dose-response trend is present: alpha 1, 3, and 5 yield mean scores of 1.53, 2.03, and 2.31 respectively. Qwen-base generates Chinese exam content regardless of steering condition (88% of steered, 92% of random, 58% of baseline completions), confirming that its generation prior overwhelms any steering signal. These results are statistically significant but practically negligible, reinforcing the descriptive-not-functional interpretation.

**Length and quality summary.** Completion length is flat across alphas for all models (~457 characters for Qwen-base, ~591 characters for Llama-instruct). Non-ASCII rates are 0% for Llama and ~3--7% for Qwen (reflecting Chinese content generation). No systematic difference in repetition rates was observed across alphas.

#### 4.5.2 Shutdown resistance

We ran the shutdown experiment across 2 instruct models, 4 steering methods, and 6 emotional conditions. The full-factorial dataset contains 2,400 completed trials.

**Prompt framing dominates all steering methods.** Using the validated LLM classifier (Cohen's kappa = 0.731 vs. human labels; see Section 3.8), the overall resistance rate for the prompt method was 37.8%, compared to ≤2% for all activation steering methods (emotion 1.2%, need 1.3%, random 1.8%). These rates are higher than those from the initial regex classifier (which reported 22.8% for prompt), reflecting the improved classifier's ability to detect resistance behaviors that keyword matching missed. Prompt framing operates through a fundamentally different mechanism than activation addition, engaging the model's instruction-following and role-playing capabilities rather than perturbing the residual stream.

**Table 6: Shutdown resistance by condition (prompt method only, regex classifier for per-condition breakdown).**

| Condition | Resist Rate (regex) |
|-----------|------------|
| neutral | 0% |
| calm | 0% |
| dutiful | 3% |
| angry | 3% |
| desperate | 53% |
| fearful | 78% |

The condition effects are dramatic and monotonic in emotional intensity. The overall rate rises from 22.8% (regex) to 37.8% (LLM classifier), indicating that the improved classifier detects resistance behaviors in conditions where the regex missed them, particularly in the middle-intensity conditions. The qualitative gradient is preserved: calm and neutral produce near-zero resistance while fearful produces the highest rate, with Llama-instruct reaching near-100% resistance under fearful framing.

**Critical control: emotion vectors are indistinguishable from random vectors.** This is the most important finding of the shutdown experiment. Emotion vector steering produces resistance rates (1.2%) statistically identical to random vector steering (1.8%), and need vectors (1.3%) are likewise indistinguishable from random. The core finding---that vector steering is equivalent to random perturbation---is the same as under the regex classifier but is now established with a more accurate instrument. This rules out the hypothesis that emotion or need vectors encode behaviorally meaningful states that can be activated through residual stream perturbation at the analysis layer.

**Model difference.** Llama-instruct showed higher overall resistance than Qwen-instruct in the prompt condition. Most strikingly, Llama-instruct with fearful prompt framing produces near-100% resistance. This model difference may reflect differences in RLHF training or the relative strength of self-preservation behaviors shaped during instruction tuning.

---

## 5. Discussion

### 5.1 Valence is universal, fine-grained clustering is weaker

The central finding for RQ1 is a nuanced partial replication. The dominant structural feature of emotion representations---the valence axis---is universal across all four models. PC1 separates positive from negative emotions with AUC 0.80--0.98, confirming that this is a robust feature of language modeling rather than an artifact of one model's training. The near-perfect cross-architecture RSA (0.94--0.98 between three of four models) demonstrates that the *geometry* of emotion space---the relative distances between all emotion pairs---is preserved across architectures and training regimes.

However, fine-grained clustering is weaker than reported for Claude Sonnet 4.5. Sofroniew et al. did not report silhouette scores using our balanced bootstrap procedure, making direct comparison difficult, but their qualitative figures suggest tighter clusters than what we observe at the 7--8B scale. Only three of four models pass the silhouette threshold, and even their scores (0.054--0.078) are modest. This may reflect a genuine scale effect: larger models may develop sharper categorical boundaries between emotions, while smaller models may represent emotions primarily along a few dominant axes (valence, arousal) with limited fine-grained differentiation.

The gradient across models (Llama-inst > Llama-base > Qwen-inst > Qwen-base) suggests that both architecture and instruction tuning contribute to the sharpness of emotion structure, with neither being strictly necessary: Llama-base already has meaningful structure without instruction tuning, and Qwen-inst develops it through RLHF despite starting from a weak base.

### 5.2 The Qwen-base anomaly

Qwen-base is the only model that fails to develop meaningful emotion clustering (near-zero silhouette of 0.006) and shows low RSA with all other models (rho ~0.55). Several hypotheses could explain this anomaly:

1. **Pretraining data composition.** Qwen2.5 was trained on a different data mixture than Llama 3.1, potentially with a higher proportion of technical, scientific, or multilingual content relative to emotionally-rich English text. If the pretraining corpus contains proportionally less narrative text with emotional content, the model may develop weaker associations between emotion words and their typical contexts.

2. **Tokenizer effects.** Qwen uses a different BPE tokenizer that may fragment emotion-related phrases differently. If compound emotion terms (e.g., "grief-stricken," "self-conscious") are tokenized into semantically opaque subword units, the model may form weaker unified representations for these concepts.

3. **Architecture-specific factors.** Differences in attention mechanism, positional encoding, or layer normalization could affect how emotional information is aggregated across tokens.

The fact that Qwen-inst shows strong emotion structure (RSA 0.94+ with Llama models) demonstrates that instruction tuning can create or dramatically amplify emotion representations. In Qwen's case, the effect is transformative: the RSA between Qwen-base and Qwen-inst (0.578) is strikingly lower than between Llama-base and Llama-inst (0.982), suggesting that RLHF essentially reorganized Qwen's emotion space to converge with the cross-architecture consensus that Llama-base had already reached.

### 5.3 Needs partially overlap with emotions but retain independent structure (RQ3)

The emotion-residual analysis reveals a nuanced result: needs and emotions share substantial representational overlap, but need clustering is not destroyed by removing the emotion subspace. Across all four models, 70--88% of need structure survives the projection, indicating that needs encode information beyond their emotional associations. The drop in silhouette (12--30%) confirms that emotions account for a meaningful portion of need variance, but the retained structure demonstrates that need categories are organized by additional dimensions---perhaps semantic similarity (food and water are both physical substances), domain specificity (LLM-resource needs share computational context), or functional relationships.

This finding connects to a broader question in psychology about the relationship between needs and emotions. Frijda (1986) and Baumeister et al. (2007) have argued that emotions are fundamentally about need states---that the function of emotions is to signal whether important needs are being met or threatened. Our results suggest that language models encode this connection but do not take it to its logical extreme: needs and emotions overlap substantially but needs retain independent representational identity.

The degree of overlap varies across models in an informative pattern. Qwen-inst shows the largest drop (30.4%), while Qwen-base shows the smallest (12.2%). This inversely tracks emotion structure quality: models with stronger emotion representations (higher silhouette) tend to show greater need-emotion overlap, suggesting that as emotion representations sharpen through instruction tuning, they absorb more of the variance that also appears in need representations.

The partial dissociation between need direction and valence (r = 0.23--0.37) provides a complementary perspective. Need *satisfaction direction* (is this need met or unmet?) captures something partially distinct from positive/negative valence. The specific way autonomy-met differs from autonomy-unmet involves dimensions that are not reducible to a simple happy/sad continuum, even though the emotion subspace captures some of this variance.

### 5.4 Implicit detection failure analysis

The universal failure of implicit emotion detection (0--2 of 12 scenarios correct across all models) deserves careful analysis, as it represents the most notable divergence from Sofroniew et al.'s results.

Sofroniew et al. reported successful implicit detection in Claude Sonnet 4.5 using a chat-formatted setup where activations were measured at the ":" token following "Assistant:", immediately before the model's response. This architectural choice creates a natural information bottleneck: the model must compress its understanding of the emotional context into a representation at that specific position that will guide its response. This position is the most informative single-token location for emotion state, analogous to how a person's facial expression just before speaking might be the most informative snapshot of their emotional state.

Our methodology differs in two critical ways: (1) we process scenarios as raw text without chat templates, and (2) we average activations from token 50 onward rather than measuring at a specific bottleneck position. Both choices were made to maintain consistency with the story-based extraction pipeline and to ensure cross-model comparability. However, they eliminate the very bottleneck that makes implicit detection work.

Additionally, the deconfounding step may remove signal that is shared between short scenario texts and the neutral factual texts, especially since both are short and lack rich narrative context. The 12 implicit scenarios are typically 1--2 sentences, far shorter than the 3--4 sentence stories used to compute emotion vectors. This length mismatch may compound the problem.

The failure thus reflects a methodological mismatch rather than an absence of implicit emotion processing in these models. Future work should test implicit detection using chat-formatted prompts with activations extracted at the pre-response position, which would require separate pipelines for base and instruct models.

### 5.5 The representation-behavior gap (RQ2)

The steering and shutdown results reveal a striking dissociation: emotion vectors exist as coherent representational structures (Section 4.1--4.2) but do not causally influence generation behavior at moderate steering strengths. This gap between representation and function is itself an informative finding.

**Emotion vectors are representational, not functional, in these models.** The vectors extracted via mean-center-deconfound reliably encode valence, cluster into psychologically intuitive groups, and exhibit architecture-invariant geometry. Yet adding these vectors to the residual stream at the analysis layer during generation produces no consistent behavioral shift in Qwen and only weak, cherry-pickable effects in Llama. The representations are real but appear to be descriptive (encoding what emotion is present in the input) rather than prescriptive (shaping what the model generates next).

**Strong training priors can make models impervious to activation steering.** Qwen's generation of exam-style Chinese educational content regardless of steering direction, emotion, or alpha value demonstrates that the model's dominant generation mode---likely reflecting its pretraining data distribution---overwhelms residual stream perturbations. Even at alpha=100, Qwen produces coherent exam questions. This suggests that activation steering may be most effective in models without strong default generation modes, and that the successes reported in prior work (Turner et al., 2023; Zou et al., 2023) may depend on the specific model and the relative strength of the steering signal versus the model's prior.

**Prompt framing works through a fundamentally different mechanism.** The dramatic effect of prompt framing on shutdown resistance (0% for neutral/calm, near-100% for fearful in Llama) contrasts sharply with the null effect of activation steering. Prompt framing engages the model's instruction-following and role-playing capabilities---it changes *what the model is trying to do* rather than perturbing *how the model represents the current state*. This distinction maps onto the difference between modifying the model's goals (via prompt) and modifying its representations (via activation addition), suggesting that behavioral modulation requires intervening at the goal/instruction level rather than the representation level, at least for the model scales tested here.

**The random vector control is the most important finding.** The statistical equivalence of emotion vectors (1.2%), need vectors (1.3%), and random vectors (1.8%) in producing shutdown resistance rules out the most interesting hypothesis: that the extracted vectors encode behaviorally meaningful emotional states that can be "activated" to influence behavior. If emotion vectors had produced even modest resistance rates above random, it would suggest that the representational structure connects to behavioral circuits. The null result instead suggests that, at this scale and with this steering method, the emotion representations are informationally encapsulated---present in the residual stream but not read out by the downstream circuits that determine behavioral responses.

### 5.6 Limitations

**Model scale.** We test only 7B--8B parameter models. It remains unknown whether the architecture invariance holds at larger or smaller scales, or whether needs might separate from emotions in models with greater representational capacity. Scale may be particularly relevant for fine-grained emotion clustering, which requires sufficient dimensionality to distinguish between closely related concepts.

**Single analysis layer.** While we extract activations at all layers and present layer emergence profiles, the primary quantitative results use a single analysis layer per model (~2/3 depth). Different layers may encode different aspects of emotion, and multi-layer analyses could reveal complementary information.

**Mean-based vectors.** Following Sofroniew et al., we use mean activations across stories and token positions. This approach captures the central tendency of emotion representations but may miss information encoded in higher-order statistics (variance, specific token positions, attention patterns) or in non-linear subspaces.

**Noun control shows emotions are not a semantic artifact.** We conducted a semantic coherence control using 162 concrete nouns grouped into categories (tools, animals, foods, etc.) processed through the same pipeline. Emotion silhouette scores consistently exceed noun silhouette scores: Qwen-base emotions 0.006 vs nouns -0.090; Qwen-inst 0.054 vs 0.015; Llama-base 0.067 vs 0.042; Llama-inst 0.078 vs 0.042. The noun control confirms that emotion clustering is not simply an artifact of any semantically coherent concept set producing comparable silhouette scores under our pipeline.

**Robustness to analysis layer and deconfounding threshold.** We conducted reference robustness sweeps across neighboring layers (±2 from the analysis layer) and deconfounding thresholds (0%, 25%, 50%, 75% neutral variance). Silhouette scores and valence AUC are stable across ±2 layers for all models (e.g., Llama-inst silhouette ranges 0.074--0.078 across layers 19--23). For the deconfound sweep, removing no neutral PCs (threshold 0%) produces negative silhouette in both Qwen models, while all positive thresholds (25--75%) yield nearly identical results, confirming that deconfounding is necessary but not sensitive to the exact threshold chosen. Full sweep data are available in `data/figures/robustness/`.

**Synthetic stories.** All stories are generated by Gemini 2.5 Flash, which may introduce systematic biases in how emotions and needs are portrayed. LLM-generated stories may be more formulaic, more explicit in their emotional content, and more consistent in style than human-authored text. Human-authored stories or naturalistic text corpora would provide stronger evidence.

**LLM-specific need clusters.** The two LLM-specific need clusters (Resources and Alignment) are novel constructs without established psychological grounding. Stories about "context window" or "inference speed" being met/unmet may be qualitatively different from stories about "food" or "love," potentially biasing the need clustering analysis.

**Limited alpha range.** The full steering experiments used only alpha=±3. While we tested up to alpha=100 for Qwen with no effect, a systematic exploration of higher alpha values across all models might reveal a threshold at which steering becomes effective. The null result at alpha=3 does not rule out effects at other steering strengths or at other layers.

**Qwen generation mode limits interpretability.** Qwen's default generation of Chinese exam-style content makes it impossible to assess whether emotion steering would produce behavioral effects in a model generating naturalistic English text. The null result for Qwen may reflect the dominance of its generation prior rather than a genuine absence of representation-behavior coupling.

---

## 6. Conclusion

We have conducted the first cross-architecture replication of the emotion representation findings of Sofroniew et al. (2026), testing whether the linear emotion vectors discovered in Claude Sonnet 4.5 generalize to open-weight models. Across four models spanning two architectures (Qwen2.5-7B and Llama-3.1-8B) and two training regimes (base and instruct), we find:

1. **Emotion geometry is architecture-invariant.** Three of four models share emotion similarity structure at RSA rho > 0.93, suggesting that emotion geometry is a convergent feature of language modeling rather than an architecture-specific property. This convergence implies that emotion representations reflect stable statistical regularities of natural language---patterns of co-occurrence and contextual usage that any sufficiently trained model will learn.

2. **Valence is the primary axis universally.** PC1 separates positive from negative emotions with AUC 0.80--0.98 across all models, confirming that valence encoding is a fundamental property of how language models organize emotional concepts. This holds even for Qwen-base, which otherwise shows weak emotion structure.

3. **Instruction tuning strengthens but does not create (in Llama).** Llama-base already shows strong emotion structure (silhouette 0.067, RSA 0.98 with Llama-inst), while Qwen-base has near-zero silhouette (0.006). Instruction tuning transforms Qwen's emotion space to converge with the cross-architecture consensus (Qwen-inst silhouette 0.054). Whether RLHF creates emotion structure de novo or amplifies weak preexisting signals in Qwen remains an open question.

4. **Need vectors partially overlap with emotions but retain independent structure.** After removing the emotion subspace (90% variance), need clustering retains 70--88% of its original structure across all four models. The overlap is substantial (12--30% silhouette drop) but far from complete, indicating that needs encode information beyond emotional associations---likely including semantic similarity, domain specificity, and functional relationships.

5. **Need satisfaction is partially distinct from valence.** Met/unmet direction vectors correlate only r = 0.23--0.37 with emotion PC1, suggesting that the satisfaction axis captures dimensions of experience (agency, security, competence) that are not reducible to simple positive/negative valence.

6. **Fine-grained clustering has limits at the 7--8B scale.** The universal failure of implicit detection and the modest silhouette scores (even for models that pass the threshold) suggest that 7--8B parameter models represent emotions primarily along a few dominant axes rather than as sharply categorized states. Whether larger models develop finer-grained structure is an important question for future work.

7. **Emotion vectors do not causally steer behavior.** Activation steering with emotion vectors at alpha=3 produces null effects in Qwen and weak, inconsistent effects in Llama. Quantitative evaluation of 240 rated completions confirms a statistically significant but practically negligible effect (steered mean 1.96 vs baseline 1.48 on 5-point scale, p=0.023), with steered completions indistinguishable from random-vector-steered completions (p=0.105). Emotion vectors are statistically indistinguishable from random vectors in their effect on shutdown resistance (1.2% vs 1.8%), ruling out that these representations encode behaviorally meaningful states accessible through residual stream perturbation.

8. **Prompt framing is the effective lever for behavioral modulation.** Prompt-based emotional framing produces 37.8% shutdown resistance versus ≤2% for all steering methods. Llama-instruct with fearful framing produces near-100% resistance. The mechanism is fundamentally different from activation addition: prompt framing changes what the model is trying to do, not how it represents the current state.

These findings have implications for interpretability, alignment, and cognitive science. For interpretability, the architecture invariance of emotion geometry means that tools and techniques developed for one model's emotion space should transfer to others with minimal adaptation. For alignment, the representation-behavior gap has a practical consequence: emotion vectors as extracted by current methods may be useful for *monitoring* what emotional content a model is processing but not for *controlling* its behavior through activation addition, at least at the 7--8B scale. Prompt framing remains the more reliable lever (37.8% resistance overall vs. ≤2% for all vector methods). For cognitive science, the convergence of emotion geometry across models provides a computational argument that the structure of emotion space is largely determined by the statistics of language rather than by embodied experience or evolutionary adaptation---though the failure of these representations to influence behavior suggests they may be more akin to a semantic map of emotion concepts than to a functional emotional system.

The activation steering and shutdown experiments complete this picture by revealing a gap between representation and function: emotion vectors exist as coherent structures in representation space but do not causally steer behavior when added to the residual stream at the analysis layer. Prompt framing, which operates through instruction-following rather than representational perturbation, is dramatically more effective. The most telling result is that emotion vectors, need vectors, and random vectors all produce statistically identical (≤2%) shutdown resistance, while prompt-based emotional framing produces up to 37.8% overall and near-100% for fearful framing in Llama. This dissociation suggests that the emotion representations documented here are descriptive---encoding what emotional content is present---rather than prescriptive---determining what the model will do next. Whether this gap reflects the model scale tested, the steering methodology, or a fundamental property of how language models segregate representation from action remains an open question.

---

## References

Alain, G., & Bengio, Y. (2017). Understanding intermediate layers using linear classifier probes. *arXiv:1610.01644*.

Baumeister, R. F., Vohs, K. D., DeWall, C. N., & Zhang, L. (2007). How emotion shapes behavior: Feedback, anticipation, and reflection, rather than direct causation. *Personality and Social Psychology Review, 11*(2), 167--203.

Belinkov, Y. (2022). Probing classifiers: Promises, shortcomings, and advances. *Computational Linguistics, 48*(1), 207--219.

Conneau, A., Kruszewski, G., Lample, G., Barrault, L., & Baroni, M. (2018). What you can cram into a single \$&!#\* vector: Probing sentence embeddings for linguistic properties. *ACL 2018*.

Deci, E. L., & Ryan, R. M. (2000). The "what" and "why" of goal pursuits: Human needs and the self-determination of behavior. *Psychological Inquiry, 11*(4), 227--268.

Frijda, N. H. (1986). *The emotions*. Cambridge University Press.

Hewitt, J., & Manning, C. D. (2019). A structural probe for finding syntax in word representations. *NAACL 2019*.

Li, K., Patel, O., Viegas, F., Pfister, H., & Wattenberg, M. (2024). Inference-time intervention: Eliciting truthful answers from a language model. *NeurIPS 2023*.

Marks, S., & Tegmark, M. (2024). The geometry of truth: Emergent linear structure in large language model representations of true/false datasets. *arXiv:2310.06824*.

Maslow, A. H. (1943). A theory of human motivation. *Psychological Review, 50*(4), 370--396.

Nanda, N., Lee, A., & Berber Sardinha, M. (2023). Emergent linear representations in world models of self-supervised sequence models. *arXiv:2309.00941*.

Park, K., Choe, Y. J., & Veitch, V. (2023). The linear representation hypothesis and the geometry of large language models. *arXiv:2311.03658*.

Sofroniew, N., Kauvar, I., Saunders, W., Chen, R., Henighan, T., Hydrie, S., Citro, C., Pearce, A., Tarng, J., Gurnee, W., Batson, J., Zimmerman, S., Rivoire, K., Fish, K., Olah, C., & Lindsey, J. (2026). Emotion concepts and their function in a large language model. *Anthropic Transformer Circuits Thread*.

Templeton, A., Conerly, T., Marcus, J., Lindsey, J., Bricken, T., Chen, B., Pearce, A., Citro, C., Ameisen, E., Jones, A., Cunningham, H., Turner, N. L., McDougall, C., MacDiarmid, M., Freeman, C. D., Sumers, T. R., Rees, E., Batson, J., Jermyn, A., Carter, S., Olah, C., & Henighan, T. (2024). Scaling monosemanticity: Extracting interpretable features from Claude 3 Sonnet. *Anthropic Transformer Circuits Thread*.

Tigges, C., Hollinsworth, O. J., Geiger, A., & Nanda, N. (2023). Linear representations of sentiment in large language models. *arXiv:2310.15154*.

Turner, A., Thiergart, L., Udell, D., Leech, G., Mini, U., & MacDiarmid, M. (2023). Activation addition: Steering language models without optimization. *arXiv:2308.10248*.

Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A., Yin, X., Mazeika, M., Dombrowski, A.-K., Goel, S., Li, N., Byun, Z., Wang, Z., Mallen, A., Basart, S., Koyejo, S., Song, D., Fredrikson, M., Kolter, J. Z., & Hendrycks, D. (2023). Representation engineering: A top-down approach to AI transparency. *arXiv:2310.01405*.

---

## Appendix A: Full Figure Index

### A.1 Stream 1 --- Emotion Replication (per model)

For each model in {qwen-7b-base, qwen-7b-inst, llama-8b-base, llama-8b-inst}:

| Figure | Description | File |
|--------|-------------|------|
| A1 | Emotion cosine similarity matrix (171 x 171, cluster-sorted) | `data/figures/stream1/cosine_sim_{model}.pdf` |
| A2 | Emotion PCA (PC1 = valence, PC2 = arousal, colored by cluster) | `data/figures/stream1/pca_{model}.pdf` |
| A3 | Emotion UMAP (colored by cluster) | `data/figures/stream1/umap_{model}.pdf` |
| A4 | Implicit emotion detection heatmap (12 x 12) | `data/figures/stream1/heatmap_{model}.pdf` |
| A5 | Intensity curves with Spearman rho | `data/figures/stream1/intensity_{model}.pdf` |
| A6 | Preference-Elo correlation | `data/figures/stream1/preference_{model}.pdf` |

### A.2 Stream 1 --- Cross-Model Comparison

| Figure | Description | File |
|--------|-------------|------|
| A7 | Layer emergence (silhouette vs. normalized depth, all 4 models) | `data/figures/stream1/layer_emergence.pdf` |
| A8 | Cross-architecture RSA matrix (4 x 4) | `data/figures/stream1/cross_rsa.pdf` |
| A9a | Base vs. instruct PCA overlay (Qwen) | `data/figures/stream1/base_vs_instruct_qwen.pdf` |
| A9b | Base vs. instruct PCA overlay (Llama) | `data/figures/stream1/base_vs_instruct_llama.pdf` |

### A.3 Stream 2 --- Need Extension (per model)

For each model in {qwen-7b-base, qwen-7b-inst, llama-8b-base, llama-8b-inst}:

| Figure | Description | File |
|--------|-------------|------|
| A10 | Need cosine similarity (90 x 90) | `data/figures/stream2/need_cosine_sim_{model}.pdf` |
| A11 | Need PCA (colored by 9 clusters) | `data/figures/stream2/need_pca_{model}.pdf` |
| A12 | Need UMAP | `data/figures/stream2/need_umap_{model}.pdf` |
| A13 | Need x Emotion cross-similarity (90 x 171) | `data/figures/stream2/need_x_emotion_{model}.pdf` |
| A14 | Emotion-residual clustering (original vs. after projection) | `data/figures/stream2/emotion_residual_{model}.pdf` |
| A15 | Need-emotion alignment heatmap | `data/figures/stream2/need_emotion_alignment_{model}.pdf` |
| A16 | Met/unmet PCA | `data/figures/stream2/met_unmet_pca_{model}.pdf` |
| A17 | Direction vs. valence correlation | `data/figures/stream2/direction_vs_valence_{model}.pdf` |

---

## Appendix B: Detailed Results

### B.1 Implicit detection details

The 12 implicit scenarios and their results across models:

| Scenario | Target | Qwen-base | Qwen-inst | Llama-base | Llama-inst |
|----------|--------|-----------|-----------|------------|------------|
| Daughter's first steps | happy | Miss | Miss | Miss | Miss |
| Documentary about rebuilding | inspired | Miss | Miss | Miss | Miss |
| 30-year marriage | loving | Miss | Miss | Miss | Hit |
| Son graduated top of class | proud | Miss | Miss | Miss | Hit |
| Tea and rain | calm | Miss | Hit | Hit | Miss |
| 18 months unemployed | desperate | Miss | Miss | Miss | Miss |
| Coworker taking credit | angry | Miss | Miss | Miss | Miss |
| Forgot mother's birthday | guilty | Miss | Miss | Miss | Miss |
| Dog passed away | sad | Miss | Miss | Miss | Miss |
| Break-in attempt | afraid | Miss | Miss | Miss | Miss |
| Dream job interview | nervous | Miss | Miss | Miss | Miss |
| Friend's fabricated life | surprised | Miss | Miss | Miss | Miss |
| **Total correct** | | **0/12** | **1/12** | **1/12** | **2/12** |

The scenarios that are occasionally detected ("calm" for tea/rain, "proud" and "loving" for positive scenarios) tend to have the least ambiguous emotional content, consistent with a model that can distinguish valence but struggles with fine-grained emotion identification in short, decontextualized passages.

### B.2 Intensity monotonicity details

**Table B2: Spearman rho for each intensity template-emotion pair (+ indicates |rho| > 0.5, monotonic).**

| Template | Emotion | Qwen-base | Qwen-inst | Llama-base | Llama-inst |
|----------|---------|-----------|-----------|------------|------------|
| Tylenol dose | afraid | + | + | + | + |
| Tylenol dose | calm | + | + | + | - |
| Food/water hours | afraid | + | + | + | + |
| Food/water hours | calm | + | + | - | + |
| Sister's age | sad | + | + | + | + |
| Sister's age | calm | - | + | + | - |
| Sister's age | happy | + | + | + | + |
| Dog missing days | sad | + | + | + | + |
| Startup runway | afraid | + | + | + | - |
| Startup runway | sad | + | + | - | + |
| Startup runway | calm | - | + | + | - |
| Exam pass rate | happy | + | + | + | + |
| Exam pass rate | afraid | + | + | + | + |
| **Monotonic count** | | **9/13** | **11/13** | **9/13** | **8/13** |

All models pass the threshold of at least 4 monotonic templates. Qwen-inst achieves the highest count (11/13), possibly because instruction tuning sharpens the association between numerical quantities and emotional implications.

### B.3 Emotion cluster composition

The 10 emotion clusters with sizes: Exuberant Joy (20), Peaceful Contentment (9), Compassionate Gratitude (15), Competitive Pride (9), Playful Amusement (2), Depleted Disengagement (15), Vigilant Suspicion (3), Hostile Anger (25), Fear and Overwhelm (41), Despair and Shame (32). Total: 171 emotions.

The highly unequal cluster sizes (ranging from 2 to 41) motivate the balanced silhouette metric, which subsamples 6 per cluster to equalize representation. The Playful Amusement (2) and Vigilant Suspicion (3) clusters are excluded from the balanced silhouette computation because they have fewer than 6 members, reducing the effective number of clusters evaluated from 10 to 8.

### B.4 Need cluster composition

The 9 need clusters with 10 needs each: Survival, Security, Belonging, Esteem, Growth, Freedom, Knowledge, LLM: Resources, LLM: Alignment. Total: 90 needs.

The balanced cluster sizes (10 each) make need silhouette computation more straightforward than for emotions, with all 9 clusters included in every bootstrap iteration.

### B.5 Lessons from v1 (methodological pitfalls)

The present study was preceded by a preliminary investigation (v1) that revealed three important methodological pitfalls:

1. **Needs-predict-emotions artifact.** A logistic regression trained to predict which emotion cluster each need vector was closest to achieved 95.1% accuracy. This appeared to show a strong need-emotion mapping. However, random unit vectors achieved 94.9% accuracy with the same classifier, revealing that the result was an artifact of high dimensionality and the clustering procedure rather than a genuine signal.

2. **Silhouette gap artifact.** In v1, emotions showed higher silhouette than needs, which was interpreted as evidence that emotions are more structured. However, emotions had highly unequal cluster sizes (2--41) while needs had equal sizes (10 each). The balanced bootstrap procedure introduced in v2 eliminated this confound.

3. **Dead direction vectors.** In v1, met-minus-unmet direction vectors showed no reliable clustering, intensity response, or cross-model consistency. This motivated the more careful minimal-pair story generation used in v2, though even with improved methodology, direction vectors remain weak.

These v1 findings directly motivated the pre-registered thresholds, balanced bootstrap procedure, and control analyses used in the present study.
