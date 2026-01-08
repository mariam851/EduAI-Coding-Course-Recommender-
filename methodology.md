# Methodology  
## Intelligent Recommendation System for Coding Courses Based on Learner Behavior

---

## 1. Problem Definition

Online coding education platforms generate large volumes of sequential learner interaction data. Each learner progresses at a different pace, encounters different difficulties, and exhibits unique learning patterns. Traditional recommendation systems often fail to model the **temporal evolution of student knowledge**, resulting in generic or suboptimal learning paths.

This project addresses the problem of:

> **Modeling student knowledge evolution over time and recommending personalized next learning activities based on historical learner behavior.**

The core objectives are:
- To estimate a learner’s latent knowledge state from interaction sequences.
- To detect learning instability and risk patterns early.
- To recommend the most relevant next coding problems tailored to each learner.

---

## 2. Dataset Description

The system is evaluated using a **large-scale educational interaction dataset** consisting of student-problem attempts collected from an online coding learning environment.

Each interaction sequence includes:
- A student identifier  
- A problem (item) identifier  
- A timestamp  
- A binary correctness label (correct / incorrect)

The dataset is inherently:
- **Sequential**: interactions are time-ordered.  
- **Sparse**: each learner attempts only a subset of available problems.  
- **Implicit-feedback based**: learner knowledge is inferred from behavior rather than explicit ratings.

Due to privacy, ethical considerations, and dataset licensing, the raw data is **not included** in the public repository.

---

## 3. Knowledge Tracing Model

### 3.1 Model Choice

To capture temporal learning dynamics, the system employs **GRU-based Knowledge Tracing (GRU-KT)**.

Gated Recurrent Units (GRUs) are chosen because they:
- Efficiently model long-term dependencies in sequential data.
- Are robust to vanishing gradient issues.
- Are well-suited for sparse and noisy educational datasets.

---

### 3.2 Input Representation

At each time step *t*, the model receives:
- The problem attempted by the learner.
- The correctness outcome (0 or 1).

These inputs are embedded and fed sequentially into the GRU network.

---

### 3.3 Latent Knowledge State

The GRU hidden state represents the learner’s **latent knowledge state**, dynamically updated after each interaction.

This state captures:
- Mastered concepts.
- Weak or unstable knowledge areas.
- Learning progression trends.

---

### 3.4 Output Prediction

At each step, the model outputs a probability distribution representing the likelihood that the learner would correctly solve each available problem.

These probabilities are later used for recommendation and learning analytics.

---

## 4. Recommendation Strategy

### 4.1 Top-N Recommendation

Based on the learner’s most recent knowledge state:
1. The model predicts correctness probabilities for all candidate problems.
2. Problems already attempted by the learner are excluded.
3. The **Top-N problems** with the highest predicted relevance are selected.

---

### 4.2 Probability Normalization for Interpretability

Raw prediction probabilities may be numerically small and difficult to interpret.  
To improve usability and explainability, probabilities are **normalized within the Top-N set**:

\[
p_i^{norm} = \frac{p_i}{\sum_{j=1}^{N} p_j}
\]

This ensures:
- Relative relevance is clear to human users.
- Percentages sum to 100% within recommendations.
- Ranking quality remains unaffected.

This design choice improves **user experience and interpretability** without altering the model’s predictive behavior.

---

## 5. Learning Curve Analytics

To provide transparent insights into learner progress, two complementary metrics are computed.

### 5.1 Exponential Moving Average (EMA)

EMA reflects **short-term learning stability**, giving higher weight to recent interactions.

- Sensitive to recent mistakes.
- Useful for detecting sudden performance drops.
- Acts as an estimate of current mastery.

---

### 5.2 Cumulative Accuracy

Cumulative accuracy represents **long-term learning performance**, calculated as the running mean of correctness.

- Smooth and stable.
- Reflects overall learning trajectory.
- Less sensitive to short-term fluctuations.

---

### 5.3 Learning Drop Detection

Sharp drops in EMA are identified when:
- The difference between consecutive EMA values exceeds a predefined negative threshold.

These drops often indicate:
- Conceptual misunderstanding.
- Increased difficulty.
- Cognitive overload.

Such points are flagged as **risk indicators**.

---

## 6. Knowledge State Heatmap

To visualize learner mastery across problems, a **Knowledge State Heatmap** is constructed.

### 6.1 Construction

- For each problem, the average correctness across attempts is computed.
- Problems are ranked by mastery level.
- Color encoding is applied:
  - **Green** → High mastery  
  - **Yellow** → Partial mastery  
  - **Red** → Weak or unstable knowledge  

---

### 6.2 Interpretation

The heatmap provides:
- An interpretable snapshot of learner strengths and weaknesses.
- Clear identification of problem clusters requiring reinforcement.
- A foundation for targeted intervention and learning plan generation.

---

## 7. Personalized Learning Plan

The system combines:
- Weak mastery problems (from the heatmap).
- Sudden learning drop points (from EMA analysis).

to generate a **personalized learning focus list**, where:
- Problems causing both low mastery and sharp drops are marked as **high priority**.
- Others are marked as **medium priority**.

This bridges predictive modeling with actionable educational guidance.

---

## 8. System Design Philosophy

The system is intentionally designed as a **research-grade prototype**, emphasizing:
- Interpretability over black-box predictions.
- Educational relevance over pure accuracy.
- Transparency for instructors, learners, and researchers.

Rather than replacing educators, the system aims to **augment decision-making** in personalized learning environments.

---

## 9. Summary

This methodology integrates:
- Deep sequential modeling via **GRU-based Knowledge Tracing**.
- Personalized **Top-N recommendation**.
- Interpretable **learning analytics**.
- Visual **knowledge diagnostics**.

By combining prediction and explanation, the proposed system bridges the gap between **machine learning models and practical educational insights**, making it suitable for academic research, experimentation, and future extension.
