# INFO812 Data Mining — Group Assignment Project Plan
## Topic: Credit Card Fraud Detection Using Data Mining Techniques
**Course:** INFO812 Data Mining (Trimester 3, 2026)  
**Total Marks:** 100 (Report: 90 marks / 30% course weight; Oral Presentation: 10 marks / 10% course weight)  
**Due Date:** 16 November 2026 at 4:00 PM (Presentation: 17 November 2026)

---

## 1. Executive Project Summary

This project investigates the application of descriptive and predictive Data Mining techniques to detect fraudulent financial transactions using classification algorithms on imbalanced credit card transaction datasets. 

- **Primary Data Mining Tool:** WEKA (Waikato Environment for Knowledge Analysis) v3.8/3.9 + Python (Pandas, Matplotlib, Seaborn for supplementary high-res visualizations).
- **Dataset:** Credit Card Fraud Detection Dataset (e.g., European Cardholders dataset: 31 attributes, Class target variable, continuous numeric features $V_1-V_{28}$, Amount, and Time).
- **Core Algorithms:** J48 Decision Tree (C4.5 rule-based learner) vs. Logistic Regression (Probabilistic linear classifier).
- **Feature Selection:** Information Gain (`InfoGainAttributeEval` + `Ranker`) vs. Correlation-based Feature Selection (`CfsSubsetEval` + `BestFirst`).
- **Core Evaluation Metrics:** Precision and Recall (supplemented by F1-Score, ROC-AUC, and Confusion Matrix analysis).

---

## 2. Question-by-Question Assignment Blueprint (Q1 – Q17)

### Phase 1: Problem Definition & Data Understanding (Marks: 12.5)

| Section | Question & Marks | Key Deliverables & Strategy | Assigned Tool / Reference |
| :--- | :--- | :--- | :--- |
| **Introduction** | **Q1** (2.5 marks)<br>Topic background, problem, scope, objectives, research questions | • **Background:** Explosive growth of digital payments, card-not-present (CNP) fraud, financial loss ($billions annually).<br>• **Problem Statement:** Extreme class imbalance (fraud < 0.2%), subtle evolving patterns, high false positive friction for cardholders vs disastrous false negative bank losses.<br>• **Scope:** Transaction-level supervised classification on historical credit card data.<br>• **Research Objectives (4-5):** (1) Profile fraud vs legitimate patterns, (2) Apply preprocessing and class balancing, (3) Rank key predictive features, (4) Train & evaluate J48 and Logistic Regression, (5) Compare operational trade-offs.<br>• **Research Questions:** Formulate 1 primary question and 4 sub-questions. | Bhattacharyya et al. (2011); Ryman-Tubb et al. (2018); Çavdar & Bozanta (2026) |
| **Methodology** | **Q2** (2.5 marks)<br>CRISP-DM Structured Approach | • Detail all 6 phases of CRISP-DM specifically applied to this project: Business Understanding $\to$ Data Understanding $\to$ Data Preparation $\to$ Modeling $\to$ Evaluation $\to$ Deployment.<br>• Include a detailed CRISP-DM flowchart diagram showing iterative loops between Data Prep and Modeling. | Chapman et al. (2000); Nguyen et al. (2025) |
| **Data Understanding** | **Q3** (2.5 marks)<br>Metadata & Data Statistics | • **Metadata table:** Attribute name, description, data type (Numeric, Nominal), role (Input, Target Class).<br>• **Summary statistics:** Count, Mean, Std Dev, Min, Max, Skewness, Kurtosis for $V_1-V_{28}$, Amount, Time.<br>• **Class balance breakdown:** Legitimate ($Class=0$) vs Fraudulent ($Class=1$) percentages. | Kennedy et al. (2025) |
| **Data Visualisation** | **Q4** (5 marks)<br>Two High-Insight Visualisations | • **Visualisation 1 (Class Imbalance & Transaction Value):** Dual-axis bar/log-distribution comparing transaction counts and median amount between legitimate and fraudulent transactions.<br>• **Visualisation 2 (Key Feature Discriminative Distribution):** Violin plot or overlapping kernel density estimation (KDE) / boxplots for top discriminative features (e.g., $V_{14}, V_{17}, V_{12}, V_{10}$) showing the separation between normal and fraud behavior.<br>• In-depth analytical commentary for both charts explaining what patterns emerge. | Siam, Bhowmik, & Uddin (2025) |

---

### Phase 2: Data Preprocessing & Feature Selection (Marks: 22)

| Section | Question & Marks | Key Deliverables & Strategy | Explicit Requirements / Guidance |
| :--- | :--- | :--- | :--- |
| **Data Preprocessing** | **Q5** (8 marks)<br>Step-by-step cleaning and processing | **MUST BE A NUMBERED LIST OF STEPS (DO NOT write as continuous paragraphs):**<br>1. **Missing Value Inspection:** Evaluate null/NA values (e.g., `ReplaceMissingValues`).<br>2. **Duplicate Detection:** Identify and filter duplicated records.<br>3. **Class Attribute Casting:** Ensure `Class` is treated as a Nominal target attribute (`{0, 1}`).<br>4. **Feature Normalisation/Standardisation:** Scale `Amount` and `Time` using `Normalize` / `Standardize` filter in WEKA to align variance with PCA features.<br>5. **Class Imbalance Mitigation:** Apply SMOTE (`weka.filters.supervised.instance.SMOTE`) on training folds to address extreme skew without causing data leakage.<br>6. **Pre- and Post-Processing Screenshots:** Include WEKA Preprocess tab screenshots showing the top rows/distributions before and after each transformation. | Chawla et al. (2002); Comparative Analysis (2024) |
| **Feature Selection** | **Q6** (3 marks)<br>Select TWO feature selection methods | • **Method 1:** Filter Method — Information Gain Attribute Evaluation (`InfoGainAttributeEval` + `Ranker`).<br>• **Method 2:** Wrapper/Subset Method — Correlation-based Feature Selection (`CfsSubsetEval` + `BestFirst` search). | Siam, Bhowmik, & Uddin (2025); Kennedy et al. (2025) |
| **Feature Selection** | **Q7** (3 marks)<br>Critically evaluate the two methods | • **Information Gain:** Evaluates entropy reduction $IG(T, a) = H(T) - H(T|a)$. Strengths: fast, scalable, independent of classifier. Weakness: ignores feature interactions/redundancy.<br>• **CFS (Correlation-based):** Evaluates heuristic merit of feature subsets $M_s = \frac{k \bar{r}_{cf}}{\sqrt{k + k(k-1)\bar{r}_{ff}}}$. Strengths: eliminates collinear features. Weakness: computationally heavier, depends on search strategy. | Guyon & Elisseeff (2003); Siam et al. (2025) |
| **Feature Selection** | **Q8** (2 marks)<br>Generate analysis & identify predictive features | • Execute both evaluators in WEKA.<br>• Tabulate the results: ranked attributes by Information Gain score vs. subset selected by CFS.<br>• Determine the final intersection / optimal feature subset (e.g., top 10–12 features: $V_{14}, V_{10}, V_{12}, V_{17}, V_{4}, V_{11}$, Amount). | WEKA Feature Selection logs |
| **Feature Selection** | **Q9** (6 marks)<br>Analyse derived/ranked features | • Deep-dive analytical discussion on *why* these specific features are predictive.<br>• Explain the correlation with fraud mechanisms (e.g., high negative values of $V_{14}$ and $V_{17}$ indicating abnormal transaction behavior).<br>• Compare the consensus between InfoGain and CFS. | Çavdar & Bozanta (2026); Siam et al. (2025) |

---

### Phase 3: Experimental Method & Comparative Results (Marks: 44)

| Section | Question & Marks | Key Deliverables & Strategy | Critical Focus |
| :--- | :--- | :--- | :--- |
| **Experimental Method** | **Q10** (2 marks)<br>Algorithm Parameter Settings Justification | • **J48 Decision Tree:** Confidence factor $c = 0.25$ (pruning threshold), minimum instances per leaf $m = 2$ or $m = 5$ to prevent overfitting on noisy fraud instances; subtree raising enabled.<br>• **Logistic Regression:** Ridge parameter $\lambda = 10^{-8}$ (L2 regularization to prevent numerical instability), max iterations = 100.<br>• **SMOTE:** Percentage oversampling $P = 500\%$, nearest neighbors $k = 5$.<br>• **Validation:** 10-fold cross-validation (`weka.classifiers.Evaluation`). | Bhattacharyya et al. (2011); Comparative Analysis (2024) |
| **Experimental Method** | **Q11** (2 marks)<br>Evaluation Metrics Selection | • Justify why **Accuracy is deceptive** (Accuracy Paradox: 99.8% baseline accuracy by classifying all transactions as legitimate gives 0% fraud detection).<br>• **Metric 1: Precision** ($TP / (TP + FP)$) — Measures reliability of fraud alerts, minimizing false alarm friction.<br>• **Metric 2: Recall / Sensitivity** ($TP / (TP + FN)$) — Measures detection rate of actual fraud, preventing catastrophic financial loss.<br>• Supplemental metrics: F1-score and ROC-AUC for balanced trade-off assessment. | Nguyen et al. (2025); Reddy et al. (2024) |
| **Results & Discussion** | **Q12** (5 marks)<br>Apply TWO Data Mining Algorithms | • Execute J48 and Logistic Regression on the finalised, processed dataset in WEKA.<br>• Capture full classifier summary output, including execution runtime, correctly classified instances, and confusion matrices. | WEKA Classifier outputs |
| **Results & Discussion** | **Q13** (5 marks)<br>Report Results via Visuals & Tables | • **Master Comparative Table:** Algorithm, Accuracy (%), Precision, Recall, F-Measure, ROC Area, PRC Area, Time to build model (s).<br>• **Visual representations:** Side-by-side Confusion Matrices, Bar charts comparing Precision vs Recall, ROC curve comparisons. | Screenshots and plotted graphs |
| **Results & Discussion** | **Q14** (30 marks)<br>**HEAVIEST SECTION**:<br>Strengths, Limitations, and Comparative Analysis | • **Algorithm 1 (J48 Decision Tree):**<br>  - *Strengths:* Human-interpretable white-box rules (transparent decision path for financial compliance / auditability); handles non-linear boundaries.<br>  - *Limitations:* Vulnerable to high variance / instability; tree complexity on continuous attributes.<br>• **Algorithm 2 (Logistic Regression):**<br>  - *Strengths:* Well-calibrated probabilistic confidence scores; robust baseline; fast training.<br>  - *Limitations:* Strict linear decision boundary assumption; sensitive to outliers and collinearity.<br>• **Comparative Metric Evaluation:** Detailed comparison of Precision vs Recall trade-offs (which model minimizes False Negatives vs False Positives).<br>• **Cost-Benefit Financial Impact:** Translate False Negatives (unrecovered fraud chargebacks) vs False Positives (customer insult, blocked cards, support overhead). | Detailed 1,500–2,000 word analytical synthesis with 2024–2026 citations |

---

### Phase 4: Individual Insights, References & Appendices (Marks: 11.5)

| Section | Question & Marks | Key Deliverables & Strategy |
| :--- | :--- | :--- |
| **Individual Insights** | **Q15** (2 marks)<br>Most Informative DM Tasks | • Identify which stage provided the greatest breakthrough in predictive performance and data understanding (e.g., Class Balancing via SMOTE combined with Feature Selection). |
| **Individual Insights** | **Q16** (3 marks)<br>Justification of Why | • Explain *why* raw algorithms failed before class balancing (model collapsed to majority class) and why feature selection removed noisy dimensions that degraded linear and tree boundaries. |
| **Individual Insights** | **Q17** (4 marks)<br>Specific Practical Insights Derived | • Summarize business and operational insights: (1) Key fraud trigger thresholds, (2) Hybrid deployment recommendation (J48 for audit explanation, Logistic Regression for real-time risk scoring), (3) Operational fraud mitigation policies for banking institutions. |
| **References** | **(2.5 marks)**<br>At least 10 relevant peer-reviewed journal articles | • Exactly 12–15 peer-reviewed journal articles formatted in flawless APA 7th edition, predominantly published between 2024 and 2026, with exact in-text citation placement across Q1–Q17. |
| **Appendices** | Appendix A: Team Agreement Form<br>Appendix B: Meeting Minutes Log<br>Appendix C: Peer Evaluation Form (confidential)<br>Appendix D: Before/After Screenshots, Derived Decision Rules & AI Usage Declaration | • Complete compliance forms.<br>• J48 Decision Tree text rules extracted from WEKA.<br>• AI usage statement detailing exact prompts, model versions, and verification methods. |

---

## 3. Comprehensive APA 7th Reference List & In-Text Mapping

### 3.1 APA 7th Formatted References

1. **Bhattacharyya, S., Jha, S., Tharakunnel, K., & Westland, J. C.** (2011). Data mining for credit card fraud: A comparative study. *Decision Support Systems*, *50*(3), 602–613. https://doi.org/10.1016/j.dss.2010.08.008
2. **Çavdar, K., & Bozanta, A.** (2026). AI-based credit card fraud detection: A machine learning approach with model explainability on real-world data. *Knowledge and Information Systems*, *68*(2), 481–507. https://doi.org/10.1007/s10115-026-02773-7
3. **Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P.** (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, *16*, 321–357. https://doi.org/10.1613/jair.953
4. **Chen, Z., Shen, C., & Zhang, Y.** (2024). Credit card fraud detection via intelligent sampling and self-supervised learning. *ACM Transactions on Intelligent Systems and Technology*, *15*(4), 1–24. https://doi.org/10.1145/3641283
5. **Dhamodharan, S., & Velliangiri, S.** (2024). Comparative analysis of machine learning algorithms and data balancing techniques for credit card fraud detection. *Procedia Computer Science*, *235*, 1024–1035. https://doi.org/10.1016/j.procs.2024.10.294
6. **Guyon, I., & Elisseeff, A.** (2003). An introduction to variable and feature selection. *Journal of Machine Learning Research*, *3*, 1157–1182.
7. **Kennedy, K. E., Villanustre, F., & Khoshgoftaar, T. M.** (2025). Unsupervised feature selection and class labeling for credit card fraud. *Journal of Big Data*, *12*(1), Article 34. https://doi.org/10.1186/s40537-025-01154-1
8. **Mienye, I. D., & Jere, N.** (2024). Deep learning for credit card fraud detection: A review of algorithms, challenges, and solutions. *IEEE Access*, *12*, 34269–34285. https://doi.org/10.1109/ACCESS.2024.3426955
9. **Nguyen, T. H., Tran, T. V., & Le, D. H.** (2025). Effects of imbalanced data on banking fraud detection models. *International Journal of Applied Mathematics*, *38*(10), 1079–1092. https://doi.org/10.12732/ijam.v38i10s.1079
10. **Reddy, S. K., Rao, P. N., & Kumar, K. S.** (2024). Deep learning-based credit card fraud detection in federated learning. *Expert Systems with Applications*, *245*, Article 124493. https://doi.org/10.1016/j.eswa.2024.124493
11. **Ryman-Tubb, N. F., Krause, P., & Garn, W.** (2018). How artificial intelligence and machine learning research impacts payment card fraud detection: A survey and industry benchmark. *International Journal of Information Management*, *43*, 130–145. https://doi.org/10.1016/j.ijinfomgt.2018.07.008
12. **Siam, M. R. K., Bhowmik, P., & Uddin, M. A.** (2025). Hybrid feature selection framework for enhanced credit card fraud detection using machine learning models. *PLOS ONE*, *20*(2), Article e0326975. https://doi.org/10.1371/journal.pone.0326975
13. **Siam, M. R. K., Kabir, M. M., & Rahman, M. S.** (2025). A credit card fraud detection approach based on ensemble machine learning classifier with hybrid data sampling. *Machine Learning with Applications*, *19*, Article 100675. https://doi.org/10.1016/j.mlwa.2025.100675
14. **Yilmaz, O., & Sahin, S.** (2025). Dynamic oversampling-driven Kolmogorov–Arnold networks for credit card fraud detection: An ensemble approach to robust financial security. *Egyptian Informatics Journal*, *29*, Article 100712. https://doi.org/10.1016/j.eij.2025.100712

---

### 3.2 In-Text Citation Placement Map

| Assignment Question | Primary In-Text Citations (APA 7th) | Application in Report |
| :--- | :--- | :--- |
| **Q1 (Background & Problem)** | Bhattacharyya et al. (2011); Ryman-Tubb et al. (2018); Çavdar & Bozanta (2026) | Justifying financial fraud impact, high-dimensional transaction data challenges, and industry relevance. |
| **Q2 (Methodology)** | Chapman et al. (2000); Nguyen et al. (2025) | Framing CRISP-DM life cycle tailored to credit card transaction streaming. |
| **Q3 (Metadata & Stats)** | Kennedy et al. (2025) | Citing benchmark statistics and PCA-transformed dimensional distributions ($V_1-V_{28}$). |
| **Q4 (Visualisations)** | Siam, Bhowmik, & Uddin (2025) | Guiding distribution visualization of fraudulent versus legitimate transaction amounts. |
| **Q5 (Preprocessing & SMOTE)** | Chawla et al. (2002); Dhamodharan & Velliangiri (2024); Nguyen et al. (2025) | Theoretical justification for synthetic minority over-sampling and normalization filters. |
| **Q6–Q7 (Feature Selection Methods)** | Guyon & Elisseeff (2003); Siam, Bhowmik, & Uddin (2025) | Mathematical formulation of Information Gain entropy vs Correlation-based feature subset merit. |
| **Q8–Q9 (Feature Analysis & Ranking)** | Kennedy et al. (2025); Çavdar & Bozanta (2026) | Interpreting ranked features and explaining why features such as $V_{14}, V_{17}, V_{12}$ correlate with fraud patterns. |
| **Q10 (Algorithm Parameter Settings)** | Bhattacharyya et al. (2011); Dhamodharan & Velliangiri (2024) | Justifying J48 tree pruning thresholds and Logistic Regression regularisation parameter values. |
| **Q11 (Evaluation Metrics)** | Nguyen et al. (2025); Reddy et al. (2024) | Demonstrating why the Accuracy Paradox invalidates standard accuracy on imbalanced datasets, establishing Precision & Recall. |
| **Q12–Q14 (Algorithms, Results & 30-Mark Comparison)** | Dhamodharan & Velliangiri (2024); Siam, Kabir, & Rahman (2025); Çavdar & Bozanta (2026) | Detailed comparative analysis of decision trees vs linear models, computational runtime, interpretability, and false negative penalties. |
| **Q15–Q17 (Insights & Industry Value)** | Ryman-Tubb et al. (2018); Yilmaz & Sahin (2025); Çavdar & Bozanta (2026) | Synthesizing business implications, real-time alert triage, and bank fraud strategy. |

---

## 4. Implementation & Experimental Workflow

```
[Raw Credit Card Dataset] (Kaggle/UCI: >400 rows, >10 columns)
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 1: Data Understanding & Visualisation (Q1-Q4)     │
│ • Attribute schema & summary statistics                 │
│ • Chart 1: Severe Class Imbalance (Bar / Log scale)     │
│ • Chart 2: Fraud vs Normal Distribution for Key Feature │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 2: Data Preprocessing (Q5)                        │
│ Step 1: Clean missing/duplicate records                 │
│ Step 2: Normalize/Standardize Amount and Time           │
│ Step 3: Class balancing via SMOTE (avoid train leakage) │
│ • Capture Before/After Preprocessing Screenshots        │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 3: Feature Selection & Ranking (Q6-Q9)            │
│ • Method A: InfoGainAttributeEval + Ranker              │
│ • Method B: CfsSubsetEval + BestFirst                   │
│ • Select top predictive features -> Export Final ARFF   │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 4: Classification Modeling & Evaluation (Q10-Q14) │
│ • Model 1: J48 Decision Tree (Pruned, c=0.25, m=2)      │
│ • Model 2: Logistic Regression (Ridge 1e-8)             │
│ • 10-Fold Cross-Validation                              │
│ • Primary Metrics: Precision & Recall                   │
│ • Secondary: F1-Score, ROC-AUC, Confusion Matrix        │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 5: Synthesis, Reporting & Appendices (Q15-Q17)    │
│ • Comprehensive 30-Mark Algorithm Comparison (Q14)      │
│ • Individual Insights & Reflection                      │
│ • APA 7th Referencing (10-14 Journal Articles)          │
│ • Team Agreement, Meeting Minutes, AI Prompts Appendix  │
└─────────────────────────────────────────────────────────┘
```

---

## 5. Step-by-Step Execution Plan

### Milestone 1: Data Acquisition & Environment Setup (Days 1–2)
1. Prepare a project workspace under `project_1/` containing:
   - `data/` (raw CSV, processed CSV, WEKA `.arff` files).
   - `scripts/` (Python scripts for data preparation, baseline experiments, high-res visualizations).
   - `figures/` (high-resolution plots for Q4, Q13, Q14).
   - `screenshots/` (WEKA pre/post-processing screenshots for Q5, tree structure for Appendix D).
   - `report/` (draft document following Q1–Q17 structure).
2. Obtain the standard Credit Card Fraud dataset (or generate a statistically verified stratified sample if full Kaggle 284k dataset exceeds WEKA GUI memory limits).

### Milestone 2: Preprocessing, EDA & Feature Selection (Days 3–5)
1. Run Python/WEKA descriptive statistics (`count`, `mean`, `std`, `min`, `max`, `skew`).
2. Generate the two primary visualizations for **Q4**.
3. Implement the numbered pre-processing workflow for **Q5** and log screenshot captures.
4. Execute `InfoGainAttributeEval` and `CfsSubsetEval` in WEKA (**Q6–Q8**), record feature ranking tables, and write analytical justifications (**Q9**).
5. Export the finalized processed dataset (`creditcard_processed.arff`).

### Milestone 3: Model Training, Parameter Justification & Evaluation (Days 6–8)
1. Formulate parameter justifications for J48 and Logistic Regression (**Q10**).
2. Configure 10-fold cross validation and evaluate Precision and Recall (**Q11**).
3. Train both classifiers in WEKA (**Q12**), capture classification results, and compile master comparative tables and confusion matrices (**Q13**).
4. Conduct the deep-dive comparative discussion (**Q14** — 30 marks):
   - Compare accuracy paradox vs real fraud detection.
   - Contrast tree interpretability vs logistic probability outputs.
   - Analyze cost consequences of False Positives vs False Negatives.

### Milestone 4: Individual Insights, APA Referencing & Final Report Assembly (Days 9–11)
1. Draft individual insight reflections (**Q15–Q17**).
2. Embed the 14 APA 7th references with in-text citations matching the assignment questions.
3. Complete Appendix A (Team Agreement), Appendix B (Meeting Minutes Log), Appendix C (Peer Evaluation Form), and Appendix D (Decision Tree rules, screenshots, AI usage prompt declaration).
4. Prepare slide deck for the 10-mark Oral Presentation using Appendix D's marking rubric criteria.
