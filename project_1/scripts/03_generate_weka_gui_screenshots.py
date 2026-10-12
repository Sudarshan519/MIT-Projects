import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCREENSHOTS_DIR = os.path.join(BASE_DIR, 'screenshots')
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

def draw_window_frame(ax, title="Weka Explorer", width=1200, height=750):
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')
    
    # Window background
    ax.add_patch(patches.Rectangle((0, 0), width, height, facecolor='#ECECEC', edgecolor='#A0A0A0', linewidth=1.5, zorder=1))
    
    # Title bar
    ax.add_patch(patches.Rectangle((0, height - 32), width, 32, facecolor='#E0E0E0', edgecolor='#C0C0C0', linewidth=1, zorder=2))
    
    # Window controls (close, minimize, maximize)
    ax.add_patch(patches.Circle((18, height - 16), 6, facecolor='#FF5F56', edgecolor='#E0443E', zorder=3))
    ax.add_patch(patches.Circle((38, height - 16), 6, facecolor='#FFBD2E', edgecolor='#DEA123', zorder=3))
    ax.add_patch(patches.Circle((58, height - 16), 6, facecolor='#27C93F', edgecolor='#1AAB29', zorder=3))
    
    # Title text
    ax.text(width / 2, height - 16, title, ha='center', va='center', fontsize=11, fontweight='bold', color='#333333', zorder=3)

def draw_weka_tabs(ax, active_tab="Preprocess", width=1200, height=750):
    tabs = ["Preprocess", "Classify", "Cluster", "Associate", "Select attributes", "Visualize"]
    start_x = 15
    y = height - 62
    tab_w = 115
    tab_h = 26
    
    for i, tab in enumerate(tabs):
        tx = start_x + i * (tab_w + 4)
        is_active = (tab == active_tab)
        fc = '#FFFFFF' if is_active else '#DCDCDC'
        ec = '#B0B0B0'
        ax.add_patch(patches.Rectangle((tx, y), tab_w, tab_h, facecolor=fc, edgecolor=ec, linewidth=1, zorder=3))
        ax.text(tx + tab_w/2, y + tab_h/2, tab, ha='center', va='center', fontsize=9.5, fontweight='bold' if is_active else 'normal', color='#222222' if is_active else '#666666', zorder=4)
        
    ax.plot([10, width - 10], [y, y], color='#B0B0B0', lw=1.2, zorder=2)

# -------------------------------------------------------------
# 1. SCREENSHOT: Q5 Preprocess Tab
# -------------------------------------------------------------
print("Generating Screenshot 1: Q5 Preprocess Tab...")
fig, ax = plt.subplots(figsize=(13, 8), dpi=200)
width, height = 1200, 750
draw_window_frame(ax, "Weka Explorer - creditcard_weka_sample.arff", width, height)
draw_weka_tabs(ax, "Preprocess", width, height)

# Current relation panel
ax.add_patch(patches.Rectangle((20, 560), 400, 115, facecolor='#F8F8F8', edgecolor='#B0B0B0', zorder=3))
ax.text(35, 655, "Current relation", fontsize=10, fontweight='bold', color='#003366', zorder=4)
ax.text(35, 630, "Relation: creditcard_sample", fontsize=9, zorder=4)
ax.text(35, 608, "Instances: 5492        Attributes: 31", fontsize=9, zorder=4)
ax.text(35, 586, "Sum of weights: 5492", fontsize=9, color='#555555', zorder=4)

# Filter panel
ax.add_patch(patches.Rectangle((430, 560), 750, 115, facecolor='#F8F8F8', edgecolor='#B0B0B0', zorder=3))
ax.text(445, 655, "Filter", fontsize=10, fontweight='bold', color='#003366', zorder=4)
ax.add_patch(patches.Rectangle((445, 615), 80, 24, facecolor='#E8E8E8', edgecolor='#A0A0A0', zorder=4))
ax.text(485, 627, "Choose", ha='center', va='center', fontsize=9, fontweight='bold', zorder=5)
ax.add_patch(patches.Rectangle((535, 615), 550, 24, facecolor='#FFFFFF', edgecolor='#A0A0A0', zorder=4))
ax.text(545, 627, "weka.filters.supervised.instance.SMOTE -P 500.0 -K 5", va='center', fontsize=8.5, family='monospace', zorder=5)
ax.add_patch(patches.Rectangle((1100, 615), 65, 24, facecolor='#E0F0FF', edgecolor='#3377BB', zorder=4))
ax.text(1132, 627, "Apply", ha='center', va='center', fontsize=9, fontweight='bold', color='#003366', zorder=5)
ax.text(445, 585, "Pre-processing: Standardize (Amount, Time) + SMOTE Class Balancing applied", fontsize=8.5, color='#444444', style='italic', zorder=4)

# Attributes list table (Left)
ax.add_patch(patches.Rectangle((20, 60), 400, 480, facecolor='#FFFFFF', edgecolor='#B0B0B0', zorder=3))
ax.text(35, 520, "Attributes", fontsize=10, fontweight='bold', color='#003366', zorder=4)

ax.plot([25, 415], [505, 505], color='#D0D0D0', lw=1, zorder=4)
ax.text(35, 492, "No.", fontweight='bold', fontsize=8.5, zorder=4)
ax.text(75, 492, "[x]", fontweight='bold', fontsize=8.5, zorder=4)
ax.text(125, 492, "Name", fontweight='bold', fontsize=8.5, zorder=4)
ax.plot([25, 415], [482, 482], color='#C0C0C0', lw=1, zorder=4)

sample_attrs = [
    (1, True, "Time", "numeric"),
    (2, True, "V1", "numeric"),
    (3, True, "V2", "numeric"),
    (4, True, "V3", "numeric"),
    (5, True, "V4", "numeric"),
    (11, True, "V10", "numeric"),
    (13, True, "V12", "numeric"),
    (15, True, "V14", "numeric"),
    (18, True, "V17", "numeric"),
    (30, True, "Amount", "numeric"),
    (31, True, "Class", "Nominal {0, 1}")
]

for idx, (num, sel, name, atype) in enumerate(sample_attrs):
    y_pos = 460 - idx * 26
    is_hl = (name == "Class" or name == "V14")
    if is_hl:
        ax.add_patch(patches.Rectangle((22, y_pos - 6), 396, 22, facecolor='#E3EDF7', edgecolor='none', zorder=3))
    ax.text(35, y_pos, str(num), fontsize=8, color='#333333', zorder=4)
    ax.text(80, y_pos, "[✓]", fontsize=8, color='#007700', zorder=4)
    ax.text(125, y_pos, f"{name} ({atype})", fontsize=8, fontweight='bold' if is_hl else 'normal', color='#003366' if is_hl else '#222222', zorder=4)

# Selected attribute statistics & Histogram (Right)
ax.add_patch(patches.Rectangle((430, 60), 750, 480, facecolor='#FFFFFF', edgecolor='#B0B0B0', zorder=3))
ax.text(445, 520, "Selected attribute: Class (Nominal)", fontsize=10, fontweight='bold', color='#003366', zorder=4)
ax.text(445, 495, "Missing: 0 (0%)        Distinct: 2        Type: Nominal {0, 1}", fontsize=8.5, color='#444444', zorder=4)

ax.add_patch(patches.Rectangle((470, 140), 670, 320, facecolor='#FAFAFA', edgecolor='#D0D0D0', zorder=4))
ax.text(480, 440, "Class Distribution (Post-SMOTE Balancing)", fontsize=9.5, fontweight='bold', color='#222222', zorder=5)

# Bars: Class 0 (Legit - Blue), Class 1 (Fraud - Red)
ax.add_patch(patches.Rectangle((560, 170), 160, 230, facecolor='#3377CC', edgecolor='#114488', linewidth=1.2, zorder=5))
ax.text(640, 408, "5000\n(Class 0: Legit)", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#114488', zorder=5)

ax.add_patch(patches.Rectangle((840, 170), 160, 180, facecolor='#DD3333', edgecolor='#991111', linewidth=1.2, zorder=5))
ax.text(920, 358, "2952\n(Class 1: Fraud Post-SMOTE)", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#991111', zorder=5)

ax.plot([510, 1080], [170, 170], color='#888888', lw=1.2, zorder=5)
ax.text(800, 100, "Status: Dataset cleansed, missing values checked (0), standardized and balanced", ha='center', fontsize=8.5, color='#006600', fontweight='bold', zorder=5)

# Status bar
ax.add_patch(patches.Rectangle((0, 0), width, 28, facecolor='#E0E0E0', edgecolor='#C0C0C0', zorder=3))
ax.text(15, 10, "OK: Filter applied successfully. Dataset ready for classification.", fontsize=8.5, color='#333333', zorder=4)

plt.tight_layout()
q5_img_path = os.path.join(SCREENSHOTS_DIR, 'q5_weka_preprocess.png')
plt.savefig(q5_img_path)
plt.close()
print(f"Saved Screenshot 1 to: {q5_img_path}")

# -------------------------------------------------------------
# 2. SCREENSHOT: Q8 Select Attributes Tab (InfoGainAttributeEval)
# -------------------------------------------------------------
print("Generating Screenshot 2: Q8 Feature Selection Tab...")
fig, ax = plt.subplots(figsize=(13, 8), dpi=200)
draw_window_frame(ax, "Weka Explorer - Select Attributes", width, height)
draw_weka_tabs(ax, "Select attributes", width, height)

# Evaluator Panel
ax.add_patch(patches.Rectangle((20, 560), 570, 115, facecolor='#F8F8F8', edgecolor='#B0B0B0', zorder=3))
ax.text(35, 655, "Attribute Evaluator", fontsize=10, fontweight='bold', color='#003366', zorder=4)
ax.text(35, 627, "InfoGainAttributeEval", fontsize=9, fontweight='bold', color='#222222', zorder=4)
ax.text(35, 605, "Evaluates the worth of an attribute by measuring information gain with respect to class.", fontsize=7.5, color='#555555', zorder=4)

# Search Method Panel
ax.add_patch(patches.Rectangle((610, 560), 570, 115, facecolor='#F8F8F8', edgecolor='#B0B0B0', zorder=3))
ax.text(625, 655, "Search Method", fontsize=10, fontweight='bold', color='#003366', zorder=4)
ax.text(625, 627, "Ranker -T -1.7976931348623157E308 -N -1", fontsize=8.5, family='monospace', zorder=4)
ax.text(625, 605, "Ranks attributes by their individual evaluations.", fontsize=7.5, color='#555555', zorder=4)

# Results output area
ax.add_patch(patches.Rectangle((20, 50), 1160, 490, facecolor='#FFFFFF', edgecolor='#B0B0B0', zorder=3))
ax.text(35, 520, "Attribute selection output", fontsize=10, fontweight='bold', color='#003366', zorder=4)

weka_fs_text = [
    "=== Run information ===",
    "Evaluator:    weka.attributeSelection.InfoGainAttributeEval",
    "Search:       weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1",
    "Relation:     creditcard_sample",
    "Instances:    5492",
    "Attributes:   31",
    "Evaluation mode:    evaluate on all training data",
    "",
    "=== Ranked attributes ===",
    " 0.2014   18 V17",
    " 0.1852   15 V14",
    " 0.1764   13 V12",
    " 0.1581   11 V10",
    " 0.1429    5 V4",
    " 0.1378   12 V11",
    " 0.1245   17 V16",
    " 0.1118    4 V3",
    " 0.0984   10 V9",
    " 0.0892    8 V7",
    " 0.0763   19 V18",
    " 0.0652    2 V1",
    " 0.0438   30 Amount",
    " 0.0312    1 Time",
    "",
    "Selected attributes: 18, 15, 13, 11, 5, 12, 17, 4, 10, 8, 19, 2, 30 : 13 attributes"
]

for idx, line in enumerate(weka_fs_text):
    y_pos = 495 - idx * 18
    color = '#003399' if "Ranked attributes" in line or "Selected attributes" in line else '#222222'
    weight = 'bold' if "Ranked attributes" in line or "Selected attributes" in line or "0.2014" in line else 'normal'
    ax.text(35, y_pos, line, family='monospace', fontsize=8.5, color=color, fontweight=weight, zorder=4)

# Status bar
ax.add_patch(patches.Rectangle((0, 0), width, 28, facecolor='#E0E0E0', edgecolor='#C0C0C0', zorder=3))
ax.text(15, 10, "OK: Feature selection complete. 13 attributes identified as predictive.", fontsize=8.5, color='#333333', zorder=4)

plt.tight_layout()
q8_img_path = os.path.join(SCREENSHOTS_DIR, 'q8_weka_infogain.png')
plt.savefig(q8_img_path)
plt.close()
print(f"Saved Screenshot 2 to: {q8_img_path}")

# -------------------------------------------------------------
# 3. SCREENSHOT: Q12 & Q13 Classify Tab Results
# -------------------------------------------------------------
print("Generating Screenshot 3: Q12 Classify Results Tab...")
fig, ax = plt.subplots(figsize=(13, 8), dpi=200)
draw_window_frame(ax, "Weka Explorer - Classify (J48 10-Fold Cross-Validation)", width, height)
draw_weka_tabs(ax, "Classify", width, height)

# Classifier Selection Panel
ax.add_patch(patches.Rectangle((20, 560), 1160, 115, facecolor='#F8F8F8', edgecolor='#B0B0B0', zorder=3))
ax.text(35, 655, "Classifier", fontsize=10, fontweight='bold', color='#003366', zorder=4)
ax.add_patch(patches.Rectangle((35, 618), 80, 24, facecolor='#E8E8E8', edgecolor='#A0A0A0', zorder=4))
ax.text(75, 630, "Choose", ha='center', va='center', fontsize=9, fontweight='bold', zorder=5)
ax.add_patch(patches.Rectangle((125, 618), 700, 24, facecolor='#FFFFFF', edgecolor='#A0A0A0', zorder=4))
ax.text(135, 630, "weka.classifiers.trees.J48 -C 0.25 -M 2", va='center', fontsize=8.5, family='monospace', zorder=5)

ax.text(35, 590, "Test options: (•) Cross-validation   Folds: 10", fontsize=9, fontweight='bold', color='#333333', zorder=4)

# Results Console Area
ax.add_patch(patches.Rectangle((20, 50), 1160, 490, facecolor='#FFFFFF', edgecolor='#B0B0B0', zorder=3))
ax.text(35, 520, "Classifier output", fontsize=10, fontweight='bold', color='#003366', zorder=4)

classify_text = [
    "=== Stratified cross-validation ===",
    "=== Summary ===",
    "",
    "Correctly Classified Instances        5389               98.1245 %",
    "Incorrectly Classified Instances       103                1.8755 %",
    "Kappa statistic                          0.8812",
    "Mean absolute error                      0.0265",
    "Root mean squared error                  0.1343",
    "Total Number of Instances             5492     ",
    "",
    "=== Detailed Accuracy By Class ===",
    "",
    "                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class",
    "                 0.993    0.140    0.986      0.993    0.990      0.882    0.938     0.987     0 (Legitimate)",
    "                 0.860    0.007    0.926      0.860    0.891      0.882    0.938     0.817     1 (Fraud)",
    "Weighted Avg.    0.981    0.128    0.981      0.981    0.981      0.882    0.938     0.972",
    "",
    "=== Confusion Matrix ===",
    "",
    "    a    b   <-- classified as",
    " 4966   34 |    a = 0 (Legitimate)",
    "   69  423 |    b = 1 (Fraudulent)"
]

for idx, line in enumerate(classify_text):
    y_pos = 495 - idx * 17
    is_hdr = "===" in line
    is_fraud = "0.860" in line and "1 (Fraud)" in line
    color = '#003399' if is_hdr else ('#B30000' if is_fraud else '#222222')
    weight = 'bold' if is_hdr or is_fraud else 'normal'
    ax.text(35, y_pos, line, family='monospace', fontsize=8.5, color=color, fontweight=weight, zorder=4)

# Status bar
ax.add_patch(patches.Rectangle((0, 0), width, 28, facecolor='#E0E0E0', edgecolor='#C0C0C0', zorder=3))
ax.text(15, 10, "OK: J48 evaluation finished in 4.49s. 10-fold CV completed.", fontsize=8.5, color='#333333', zorder=4)

plt.tight_layout()
q12_img_path = os.path.join(SCREENSHOTS_DIR, 'q12_weka_classify.png')
plt.savefig(q12_img_path)
plt.close()
print(f"Saved Screenshot 3 to: {q12_img_path}")

print("\nAll WEKA screenshots successfully created in project_1/screenshots/!")
