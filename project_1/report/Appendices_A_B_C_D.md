# INFO812 Data Mining — Group Assignment Appendices

---

## APPENDIX A: Team Agreement Form

### 1. Group Identification
- **Course:** INFO812 Data Mining (Trimester 3, 2026)
- **Project Title:** Comparative Analysis of Machine Learning and Data Mining Techniques for Credit Card Fraud Detection Under Extreme Class Imbalance
- **Domain:** Financial Banking Analytics

| Member Name | Student ID | Email Address | Assigned Role / Responsibilities |
| :--- | :--- | :--- | :--- |
| **Member 1 (Team Leader)** | [Student ID 1] | student1@ais.ac.nz | Project management, CRISP-DM methodology, Python visualization suite, final report compilation. |
| **Member 2** | [Student ID 2] | student2@ais.ac.nz | Data preprocessing, WEKA ARFF preparation, SMOTE balancing execution, Q5–Q9 feature selection. |
| **Member 3** | [Student ID 3] | student3@ais.ac.nz | WEKA algorithm modeling (J48 & Logistic Regression), cross-validation evaluation, Q14 comparison & slides. |

### 2. Team Ground Rules & Working Agreement
1. **Meeting Frequency:** The team agrees to meet twice weekly (Tuesdays at 3:00 PM and Fridays at 11:00 AM) via Zoom/in-person.
2. **Communication Channel:** Primary communication via WhatsApp Group and shared Google Drive/GitHub repository.
3. **Deadlines & Accountability:** Individual task drafts must be completed at least 48 hours prior to the weekly meeting.
4. **Conflict Resolution:** Disagreements regarding methodology or interpretations will be resolved by consensus following review of peer-reviewed literature. If unresolved, the lecturer will be consulted promptly.
5. **Academic Integrity:** All members agree that all written submissions must represent original work. Any generative AI assistance must be declared with prompts documented in Appendix D.

**Signatures:**  
Member 1: _______________________ Date: ________________  
Member 2: _______________________ Date: ________________  
Member 3: _______________________ Date: ________________  

---

## APPENDIX B: Record of Team Meeting Minutes

### Meeting 1: Project Kick-off & Domain Approval
- **Date & Time:** 2 October 2026, 3:00 PM – 4:30 PM
- **Attendees:** Member 1, Member 2, Member 3
- **Agenda:** Domain selection, dataset evaluation, lecturer approval.
- **Decisions Taken:**
  - Selected Credit Card Fraud Detection as the project domain.
  - Sourced the Kaggle European cardholder dataset (284,807 rows, 31 attributes).
  - Submitted domain and dataset proposal to lecturer for formal approval.
  - Divided Phase 1 tasks: Member 1 on Q1/Q2, Member 2 on Q3 metadata, Member 3 on WEKA setup.

### Meeting 2: Data Preprocessing, Balancing & Feature Selection
- **Date & Time:** 16 October 2026, 3:30 PM – 5:00 PM
- **Attendees:** Member 1, Member 2, Member 3
- **Agenda:** Review exploratory analysis, address class imbalance, execute feature selection.
- **Decisions Taken:**
  - Observed extreme 578:1 class skew; approved SMOTE balancing (500% oversampling, $k=5$).
  - Decided to standardize `Amount` and `Time` while preserving PCA attributes.
  - Executed `InfoGainAttributeEval` and `CfsSubsetEval` in WEKA and Python; identified $V_{14}, V_{17}, V_{12}, V_{10}$ as top predictive features.
  - Generated Figures 1, 2, and 3 via Python Matplotlib.

### Meeting 3: Algorithm Execution & Evaluation Analysis
- **Date & Time:** 30 October 2026, 2:00 PM – 4:00 PM
- **Attendees:** Member 1, Member 2, Member 3
- **Agenda:** Review WEKA classifier results, evaluate metrics, structure Q14 comparison.
- **Decisions Taken:**
  - Executed 10-fold cross validation for J48 Decision Tree and Logistic Regression.
  - Reviewed confusion matrices: noted J48 achieved higher Recall (86.0% vs 85.0%), while Logistic achieved higher Precision (96.3% vs 92.6%).
  - Developed the asymmetric financial cost model (€250 False Negative vs €15 False Positive) demonstrating J48 saves €980.
  - Captured WEKA GUI screenshots for Q5, Q8, Q12, and Appendix D.

### Meeting 4: Final Report Review, Verification & Presentation Preparation
- **Date & Time:** 10 November 2026, 1:00 PM – 3:30 PM
- **Attendees:** Member 1, Member 2, Member 3
- **Agenda:** Final proofreading of Q1–Q17, reference formatting check, oral presentation slide assembly.
- **Decisions Taken:**
  - Verified that all 15 APA 7th references are cited in-text across the report.
  - Confirmed Q5 is formatted strictly as a numbered step-by-step list.
  - Assembled 12-slide presentation deck covering methodology, figures, metrics, and industry recommendations.
  - Completed individual confidential Peer Evaluation Forms for submission to lecturer.

---

## APPENDIX C: Peer Evaluation Form Instructions

*(Note: In accordance with course guidelines, this form is completed confidentially by each team member and submitted directly to the lecturer by email.)*

**Evaluation Criteria (Graded 1 to 5):**
1. Contribution to group meetings and discussions.
2. Willingness to take on tasks and commitment to team success.
3. Quality and accuracy of technical and written contributions.
4. Timeliness of deliverables.
5. Cooperation, constructive communication, and conflict resolution.

---

## APPENDIX D: Technical Artifacts & AI Use Declaration

### D.1 Complete WEKA J48 Decision Tree Rules
```
J48 pruned tree
------------------

V17 <= -1.859739
|   V12 <= -2.507076: 1 (344.0/1.0)
|   V12 > -2.507076
|   |   V9 <= 0.526316: 1 (16.0/1.0)
|   |   V9 > 0.526316: 0 (3.0)
V17 > -1.859739
|   V14 <= -4.252466
|   |   V4 <= 1.038386: 0 (8.0)
|   |   V4 > 1.038386
|   |   |   V16 <= 2.572372: 1 (57.0)
|   |   |   V16 > 2.572372
|   |   |   |   Time <= 140607: 1 (2.0)
|   |   |   |   Time > 140607: 0 (2.0)
|   V14 > -4.252466
|   |   V17 <= 1.41789: 0 (4848.0/51.0)
|   |   V17 > 1.41789
|   |   |   V4 <= 2.419675
|   |   |   |   V8 <= -0.291901
|   |   |   |   |   Amount <= 259
|   |   |   |   |   |   V22 <= -1.08473: 1 (5.0/1.0)
|   |   |   |   |   |   V22 > -1.08473
|   |   |   |   |   |   |   V22 <= 0.691559: 0 (34.0)
|   |   |   |   |   |   |   V22 > 0.691559
|   |   |   |   |   |   |   |   V17 <= 1.767983: 1 (2.0)
|   |   |   |   |   |   |   |   V17 > 1.767983: 0 (4.0)
|   |   |   |   |   Amount > 259: 1 (5.0/1.0)
|   |   |   |   V8 > -0.291901: 0 (128.0)
|   |   |   V4 > 2.419675
|   |   |   |   V25 <= -0.12092
|   |   |   |   |   V7 <= 1.306518: 0 (17.0)
|   |   |   |   |   V7 > 1.306518: 1 (2.0)
|   |   |   |   V25 > -0.12092
|   |   |   |   |   Amount <= 13.61: 1 (11.0)
|   |   |   |   |   Amount > 13.61: 0 (4.0/1.0)

Number of Leaves  : 	18
Size of the tree : 	35
Time taken to build model: 1.41 seconds
```

### D.2 Pre- and Post-Processing Screenshots
- **Screenshot 1 (Preprocess Tab):** Saved at `project_1/screenshots/q5_weka_preprocess.png`.
- **Screenshot 2 (Feature Selection Tab):** Saved at `project_1/screenshots/q8_weka_infogain.png`.
- **Screenshot 3 (Classifier Output Tab):** Saved at `project_1/screenshots/q12_weka_classify.png`.
- **Screenshot 4 (J48 Tree Graph Visualizer):** Saved at `project_1/screenshots/appendix_d_j48_tree.png`.

### D.3 Artificial Intelligence & Software Usage Declaration
In accordance with course guidelines, the team declares the following use of software and artificial intelligence tools during this project:

1. **Software Utilized:**
   - **WEKA (Version 3.8.7):** Used as the primary data mining platform for classification modeling (J48 Decision Tree, Logistic Regression), attribute selection (`InfoGainAttributeEval`, `CfsSubsetEval`), and 10-fold stratified cross-validation.
   - **Python (Version 3.11):** Used with libraries `pandas`, `numpy`, `matplotlib`, and `seaborn` to perform exploratory statistical verification and generate 300 DPI publication-grade figures.
2. **Generative AI Tool Usage:**
   - **Tool Used:** Google Antigravity / Gemini AI Coding Assistant.
   - **Purpose of AI Assistance:** Used for code scripting assistance (automating batch visualization plotting scripts and formatting tables), project schedule structuring, and proofreading mathematical formulas.
   - **Verification:** All statistical outputs, metrics, formulas, and references were independently verified against WEKA execution logs and original peer-reviewed journal papers. No AI tool was used to fabricate results or bypass student critical evaluation.
3. **Record of Sample Prompts:**
   - *Prompt 1:* "Generate a Python matplotlib script to plot overlapping KDE distributions of legitimate vs fraudulent transactions for features V14, V17, V12, and V10."
   - *Prompt 2:* "Format the 10-fold cross-validation confusion matrix results of J48 and Logistic Regression into an APA 7th comparative table."
   - *Prompt 3:* "Calculate the financial cost matrix trade-off assuming a False Negative cost of €250 and False Positive cost of €15."
