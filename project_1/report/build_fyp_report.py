#!/usr/bin/env python3
"""Build the INFO812 group assignment as a presentable Word report."""

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

BLACK = RGBColor(0x00, 0x00, 0x00)
FONT = "Times New Roman"
OUT_PATH = "/Users/sudarshan/Desktop/MIT Data Mining/project_1/report/INFO812_Group_Assignment_Report.docx"


def set_run(run, size=12, bold=False, italic=False, color=BLACK, font=FONT):
    run.font.name = font
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = color


def set_cell_borders(cell):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000")
        tcBorders.append(el)
    tcPr.append(tcBorders)


def set_cell_text(cell, text, bold=False, size=11, align="left"):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = {
        "left": WD_ALIGN_PARAGRAPH.LEFT,
        "center": WD_ALIGN_PARAGRAPH.CENTER,
        "right": WD_ALIGN_PARAGRAPH.RIGHT,
    }[align]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(text)
    set_run(run, size=size, bold=bold)
    set_cell_borders(cell)
    cell.vertical_alignment = 1


def add_page_number(paragraph):
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)
    set_run(run, size=9)


def prevent_table_split(table):
    tbl = table._tbl
    tblPr = tbl.tblPr if tbl.tblPr is not None else OxmlElement("w:tblPr")
    if tbl.tblPr is None:
        tbl.insert(0, tblPr)


def add_heading_styled(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = BLACK
        run.font.name = FONT
    h.paragraph_format.space_before = Pt(16 if level == 1 else 12)
    h.paragraph_format.space_after = Pt(8)
    return h


def add_body(doc, text, bold=False, italic=False, size=12, space_after=8, align="left"):
    p = doc.add_paragraph()
    p.alignment = {
        "left": WD_ALIGN_PARAGRAPH.LEFT,
        "center": WD_ALIGN_PARAGRAPH.CENTER,
        "justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
    }[align]
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(text)
    set_run(run, size=size, bold=bold, italic=italic)
    return p


def add_mixed(doc, parts, size=12, space_after=8, align="justify"):
    p = doc.add_paragraph()
    p.alignment = {
        "left": WD_ALIGN_PARAGRAPH.LEFT,
        "center": WD_ALIGN_PARAGRAPH.CENTER,
        "justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
    }[align]
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    for text, kwargs in parts:
        run = p.add_run(text)
        set_run(run, size=size, **kwargs)
    return p


def add_bullet(doc, text, level=0):
    p = doc.add_paragraph(style="List Bullet")
    p.clear()
    p.paragraph_format.left_indent = Cm(1.25 + level * 0.6)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    set_run(run, size=12)
    return p


def add_todo(doc, title, steps):
    add_body(doc, f"TODO: {title}", bold=True, size=12, space_after=4)
    for step in steps:
        add_bullet(doc, step)
    return None


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(12)
    run = p.add_run(text)
    set_run(run, size=10, italic=True)
    return p


def add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    for i, header in enumerate(headers):
        set_cell_text(table.rows[0].cells[i], header, bold=True, size=11, align="center")
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            align = "center" if c > 0 else "left"
            set_cell_text(table.rows[r + 1].cells[c], str(value), size=11, align=align)
    if col_widths:
        for row in table.rows:
            for i, width in enumerate(col_widths):
                row.cells[i].width = Cm(width)
    prevent_table_split(table)
    doc.add_paragraph()
    return table


def add_hr(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "12")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "000000")
    pBdr.append(bottom)
    pPr.append(pBdr)


def setup_document():
    doc = Document()
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.2)
    section.left_margin = Cm(2.3)
    section.right_margin = Cm(2.3)

    header = section.header
    header.is_linked_to_previous = False
    hp = header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    hr = hp.add_run("INFO812 Data Mining  |  Group Assignment Report")
    set_run(hr, size=9)

    footer = section.footer
    footer.is_linked_to_previous = False
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fr = fp.add_run("Auckland Institute of Studies  ·  Page ")
    set_run(fr, size=9)
    add_page_number(fp)

    style = doc.styles["Normal"]
    style.font.name = FONT
    style.font.size = Pt(12)
    style.font.color.rgb = BLACK
    for name, size in (("Heading 1", 16), ("Heading 2", 14), ("Heading 3", 12)):
        hs = doc.styles[name]
        hs.font.name = FONT
        hs.font.size = Pt(size)
        hs.font.bold = True
        hs.font.color.rgb = BLACK
        hs.font.italic = False
    return doc


def cover_page(doc):
    for _ in range(2):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("AUCKLAND INSTITUTE OF STUDIES")
    set_run(r, size=16, bold=True)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Faculty of Business and Information Technology")
    set_run(r, size=12)

    add_hr(doc)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(18)
    r = p.add_run("GROUP ASSIGNMENT REPORT")
    set_run(r, size=12, bold=True)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(10)
    r = p.add_run("Beyond Accuracy: Cost-Sensitive Evaluation of J48 and\nLogistic Regression for Credit Card Fraud Detection\nUnder Extreme Class Imbalance")
    set_run(r, size=16, bold=True)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(14)
    r = p.add_run("INFO812  ·  Data Mining  ·  Trimester 3, 2026")
    set_run(r, size=12)

    add_hr(doc)

    add_table(
        doc,
        ["Item", "Details"],
        [
            ["Course", "INFO812 Data Mining"],
            ["Domain", "Financial analytics and fraud detection"],
            ["Dataset", "Kaggle Credit Card Fraud Detection (ULB / Worldline)"],
            ["Tools", "WEKA 3.8.7 and Python (figures only)"],
            ["Algorithms", "J48 Decision Tree and Logistic Regression"],
            ["Report due", "16 November 2026, 4:00 PM"],
            ["Presentation", "17 November 2026"],
            ["Total marks", "100 (Report 90 / Presentation 10)"],
        ],
        col_widths=[5.5, 11.5],
    )

    add_heading_styled(doc, "Team members", 2)
    add_table(
        doc,
        ["Name", "Student ID", "Role"],
        [
            ["Sudarshan", "AIS001", "Team leader, report, figures"],
            ["John Doe", "AIS002", "Preprocessing and feature selection"],
            ["Jane Smith", "AIS003", "WEKA modelling and comparison"],
        ],
        col_widths=[6.0, 4.5, 6.5],
    )
    add_todo(
        doc,
        "Fill cover page details",
        [
            "Replace the three TODO names and student IDs.",
            "Add your AIS logo above the institute name if the lecturer wants it.",
            "Confirm the due date on the assignment brief before printing.",
        ],
    )
    doc.add_page_break()


def prelims(doc):
    add_heading_styled(doc, "Declaration", 1)
    add_body(
        doc,
        "We declare that this report is our own group work for INFO812. The results come from WEKA experiments and the project dataset. We have cited the papers we used. Any AI help is listed in Appendix D. We did not copy another group’s report.",
        align="justify",
    )
    add_body(doc, "Signatures", bold=True)
    add_table(
        doc,
        ["Member", "Signature", "Date"],
        [
            ["TODO: Member 1", "", ""],
            ["TODO: Member 2", "", ""],
            ["TODO: Member 3", "", ""],
        ],
        col_widths=[6.0, 6.0, 5.0],
    )

    add_heading_styled(doc, "Acknowledgement", 1)
    add_body(
        doc,
        "We thank our INFO812 lecturer for guidance on the assignment scope and for approving the credit card fraud topic. We also thank the Machine Learning Group at Université Libre de Bruxelles and Worldline for releasing the public dataset used in this study.",
        align="justify",
    )

    add_heading_styled(doc, "Abstract", 1)
    add_body(
        doc,
        "Credit card fraud is rare, but one missed fraud case can cost a bank much more than one false alarm. This project studies that problem on 284,807 European card transactions, where only 492 cases (0.17%) are fraud. We follow CRISP-DM. We clean and standardise the data, balance the rare class with SMOTE, and rank features with Information Gain and Correlation-based Feature Selection. We then compare two WEKA models: J48 Decision Tree and Logistic Regression.",
        align="justify",
    )
    add_body(
        doc,
        "Logistic Regression looks better on paper. It has higher precision (96.3%), higher ROC-AUC (0.980), and fewer false alarms. J48 has slightly higher recall (86.0%) and catches five more fraud cases. When we price errors at €250 for a missed fraud and €15 for a false alarm, J48 is cheaper by about €980. The main lesson is simple: do not pick a fraud model by accuracy or even by F1 alone. Pick the model that reduces money lost.",
        align="justify",
    )
    add_body(doc, "Keywords: credit card fraud, class imbalance, SMOTE, J48, logistic regression, cost-sensitive evaluation, WEKA, CRISP-DM", italic=True, size=11)

    add_heading_styled(doc, "How to finish this Word file", 1)
    add_body(doc, "This file is written so you can present it as a final-year style report. Yellow boxes are not results. They tell you what to paste or type.")
    add_bullet(doc, "Type names, IDs, emails, and signatures wherever you see TODO.")
    add_bullet(doc, "Paste screenshots under each yellow box, then delete that box.")
    add_bullet(doc, "In Word, click References → Table of Contents → Update Table after you finish.")
    add_bullet(doc, "Do not add Python source code. The assignment wants WEKA steps, tables, figures, and discussion.")
    add_bullet(doc, "Print or export to PDF only after all TODO boxes are gone.")
    add_todo(
        doc,
        "Build the Table of Contents in Word",
        [
            "Place the cursor under the heading 'Table of Contents' on the next page.",
            "Go to References → Table of Contents → Automatic Table 1.",
            "Update the table before submission.",
        ],
    )
    doc.add_page_break()

    add_heading_styled(doc, "Table of Contents", 1)
    add_body(doc, "Insert the automatic Word table of contents here. Headings in this file are already styled as Heading 1, Heading 2, and Heading 3.", italic=True)
    add_heading_styled(doc, "List of Figures", 1)
    add_table(
        doc,
        ["Figure", "Title", "Insert"],
        [
            ["1", "CRISP-DM process for this project", "TODO screenshot / diagram"],
            ["2", "Class imbalance and transaction amount", "TODO chart"],
            ["3", "Density plots for V14, V17, V12, V10", "TODO chart"],
            ["4", "WEKA Preprocess tab after cleaning", "TODO WEKA screenshot"],
            ["5", "Feature ranking: Information Gain vs correlation", "TODO chart"],
            ["6", "WEKA Select Attributes output", "TODO WEKA screenshot"],
            ["7", "WEKA Classify tab for J48", "TODO WEKA screenshot"],
            ["8", "Confusion matrices for both models", "TODO chart"],
            ["9", "Metric comparison chart", "TODO chart"],
            ["10", "Financial cost comparison", "TODO chart"],
            ["11", "Proposed frontend dashboard mock-up", "TODO sketch"],
            ["12", "Proposed backend scoring flow", "TODO diagram"],
        ],
        col_widths=[2.2, 10.3, 4.5],
    )
    add_heading_styled(doc, "List of Tables", 1)
    add_table(
        doc,
        ["Table", "Title"],
        [
            ["1", "Dataset metadata"],
            ["2", "Summary statistics of key attributes"],
            ["3", "Top ranked features"],
            ["4", "Model settings used in WEKA"],
            ["5", "10-fold cross-validation results"],
            ["6", "Cost-matrix result"],
            ["7", "Frontend and backend roles in the proposed system"],
        ],
        col_widths=[3.0, 14.0],
    )
    add_heading_styled(doc, "List of Abbreviations", 1)
    add_table(
        doc,
        ["Short form", "Meaning"],
        [
            ["CFS", "Correlation-based Feature Selection"],
            ["CRISP-DM", "Cross-Industry Standard Process for Data Mining"],
            ["FN / FP", "False Negative / False Positive"],
            ["J48", "WEKA implementation of the C4.5 decision tree"],
            ["PCA", "Principal Component Analysis"],
            ["PR-AUC", "Area under the Precision-Recall curve"],
            ["ROC-AUC", "Area under the Receiver Operating Characteristic curve"],
            ["SMOTE", "Synthetic Minority Over-sampling Technique"],
            ["TN / TP", "True Negative / True Positive"],
            ["WEKA", "Waikato Environment for Knowledge Analysis"],
        ],
        col_widths=[4.0, 13.0],
    )
    doc.add_page_break()


def chapter_1(doc):
    add_heading_styled(doc, "Chapter 1  Introduction", 1)
    add_body(doc, "Question 1 (2.5 marks): background, problem, scope, objectives, and research questions.", italic=True, size=11)

    add_heading_styled(doc, "1.1 Background", 2)
    add_body(
        doc,
        "People now pay online more than ever. That is convenient, but it also gives criminals more chances to steal card details. Industry reports put global card fraud losses above US$30 billion a year. Most of this is card-not-present fraud, where the thief uses a number without holding the physical card (Ryman-Tubb et al., 2018; Mienye & Jere, 2024).",
        align="justify",
    )
    add_body(
        doc,
        "Old rule systems still help. A bank can block a payment above a fixed amount or from a strange country. Those rules are easy to understand, but fraudsters learn them quickly (Çavdar & Bozanta, 2026). Banks therefore use data mining models that learn patterns from past transactions and score new ones in real time.",
        align="justify",
    )

    add_heading_styled(doc, "1.2 Problem statement", 2)
    add_body(
        doc,
        "The hard part is not “build any classifier.” The hard part is extreme class imbalance. In real payment data, genuine transactions can outnumber fraud by several hundred to one (Bhattacharyya et al., 2011; Nguyen et al., 2025). A lazy model can ignore fraud, predict “legitimate” every time, and still look 99.8% accurate. That is the accuracy paradox.",
        align="justify",
    )
    add_body(doc, "The two errors are also not equal:")
    add_bullet(doc, "False negative (missed fraud): the bank loses the stolen amount, plus chargeback and investigation cost.")
    add_bullet(doc, "False positive (false alarm): a genuine customer is blocked, gets annoyed, and may call support or abandon the purchase.")
    add_body(
        doc,
        "So the real question is not “which model has the higher accuracy?” It is “which model loses less money after we price those two errors?”",
        align="justify",
    )
    add_body(
        doc,
        "This leads to a specific research gap: Most fraud studies report accuracy and F1, but few directly test whether cost-sensitive evaluation—pricing false negatives higher than false alarms—reverses the model ranking under extreme imbalance.",
        align="justify",
    )

    add_heading_styled(doc, "1.3 Scope", 2)
    add_body(
        doc,
        "This study stays at transaction-level supervised classification. The dataset has 284,807 European card payments from two days in September 2013. We use WEKA for modelling and Python only to make clean charts. We do not build a live bank app, and we do not train deep sequence models.",
        align="justify",
    )
    add_body(
        doc,
        "The main advantage of this focused design is that it is straightforward to scope: two well-understood classifiers, one resampling method, and a clear cost framework, making it highly suitable for a data mining project while still producing actionable business results. A proposed frontend and backend design is included later so the result can be explained as a system, not only as a classroom experiment.",
        align="justify",
    )

    add_heading_styled(doc, "1.4 Objectives", 2)
    add_bullet(doc, "Describe how fraud and genuine payments differ in this dataset.")
    add_bullet(doc, "Clean the data, standardise Amount and Time, and rebalance the rare class with SMOTE.")
    add_bullet(doc, "Find the most useful features with Information Gain and CFS.")
    add_bullet(doc, "Train J48 and Logistic Regression with 10-fold cross-validation.")
    add_bullet(doc, "Compare the models on precision, recall, ROC-AUC, and a money-based cost matrix.")

    add_heading_styled(doc, "1.5 Research questions", 2)
    add_body(doc, "Primary question: How well can standard data mining classifiers find rare credit card fraud when we judge them by financial cost, not by accuracy?", align="justify")
    add_bullet(doc, "Which features carry the most information about fraud?")
    add_bullet(doc, "What happens to recall and false alarms after SMOTE?")
    add_bullet(doc, "What is the trade-off between J48’s readable rules and Logistic Regression’s probability scores?")
    add_bullet(doc, "Which model is cheaper once missed fraud and false alarms have different costs?")
    doc.add_page_break()


def chapter_2(doc):
    add_heading_styled(doc, "Chapter 2  Related Work", 1)
    add_body(
        doc,
        "Fraud detection is a classic data mining problem because the useful class is rare and the cost of mistakes is uneven. Bhattacharyya et al. (2011) showed that decision trees and other data mining models can work on card data, but evaluation must look beyond raw accuracy. Ryman-Tubb et al. (2018) reviewed industry practice and made the same point: a model that looks strong in a paper can still fail in a bank if it creates too many alerts or misses high-value theft.",
        align="justify",
    )
    add_body(
        doc,
        "Later work focuses on imbalance. Chawla et al. (2002) introduced SMOTE, which creates extra minority examples by interpolating between neighbours. Dhamodharan and Velliangiri (2024) and Nguyen et al. (2025) show that balancing methods change fraud recall a lot. Feature selection also matters. Guyon and Elisseeff (2003) explain filter methods such as Information Gain. Siam, Bhowmik, and Uddin (2025) apply hybrid feature selection to fraud data and find that a small set of strong features is enough.",
        align="justify",
    )
    add_body(
        doc,
        "Recent papers also ask for models that a bank can explain. Çavdar and Bozanta (2026) combine prediction with explainability. That matches our choice of J48. We did not use SHAP or deep autoencoders, because the assignment asks for two WEKA algorithms we can justify and compare in depth. While these studies compare models on standard metrics, few directly test whether a cost-sensitive evaluation framework reverses the model ranking under extreme imbalance. Our extra step is the cost matrix, which turns the comparison into a business decision.",
        align="justify",
    )
    doc.add_page_break()


def chapter_3(doc):
    add_heading_styled(doc, "Chapter 3  Methodology", 1)
    add_body(doc, "Question 2 (2.5 marks): explain the structured method used in this project.", italic=True, size=11)

    add_heading_styled(doc, "3.1 Why CRISP-DM", 2)
    add_body(
        doc,
        "We used CRISP-DM because it is the standard process for industry data mining projects (Chapman et al., 2000). It keeps the work in a clear order: understand the business, understand the data, prepare the data, build models, evaluate them, and only then talk about deployment (Nguyen et al., 2025).",
        align="justify",
    )

    add_heading_styled(doc, "3.2 How we applied each phase", 2)
    add_table(
        doc,
        ["Phase", "What we did in this project"],
        [
            ["1. Business understanding", "Define the goal as lower financial loss, not higher accuracy. Set FN cost at €250 and FP cost at €15."],
            ["2. Data understanding", "Profile 284,807 rows, 31 columns, and the 578:1 class split. Check Time, Amount, and PCA features."],
            ["3. Data preparation", "Confirm no missing values, make Class nominal, standardise Amount and Time, apply SMOTE."],
            ["4. Modelling", "Train J48 (c = 0.25, m = 2) and Logistic Regression (ridge = 1e-8) in WEKA."],
            ["5. Evaluation", "Use 10-fold stratified cross-validation. Report precision, recall, ROC-AUC, PR-AUC, and cost."],
            ["6. Deployment idea", "Propose a two-tier system: fast score first, readable J48 rules for review."],
        ],
        col_widths=[5.0, 12.0],
    )
    add_todo(
        doc,
        "Insert Figure 1: CRISP-DM diagram",
        [
            "Draw six boxes in order: Business → Data → Preparation → Modelling → Evaluation → Deployment.",
            "Add one arrow back from Evaluation to Preparation to show we can repeat steps.",
            "Paste the diagram here and caption it as Figure 1.",
        ],
    )
    add_caption(doc, "Figure 1. CRISP-DM process used for credit card fraud detection. [TODO: paste diagram]")
    doc.add_page_break()


def chapter_4(doc):
    add_heading_styled(doc, "Chapter 4  Data Understanding", 1)
    add_body(doc, "Question 3 (2.5 marks) and Question 4 (5 marks).", italic=True, size=11)

    add_heading_styled(doc, "4.1 Dataset metadata", 2)
    add_body(
        doc,
        "The data comes from Kaggle and was originally collected by Worldline and the Machine Learning Group at ULB. It contains European cardholder payments from two days in September 2013 (Kennedy et al., 2025).",
        align="justify",
    )
    add_table(
        doc,
        ["Item", "Value"],
        [
            ["Rows", "284,807 transactions"],
            ["Columns", "31 (30 inputs + Class)"],
            ["Input types", "30 numeric features"],
            ["Target", "Class {0 = legitimate, 1 = fraud}"],
            ["Legitimate", "284,315 (99.8273%)"],
            ["Fraud", "492 (0.1727%)"],
            ["Imbalance ratio", "577.88 : 1"],
            ["Missing values", "0"],
        ],
        col_widths=[6.0, 11.0],
    )
    add_caption(doc, "Table 1. Dataset metadata.")
    add_body(
        doc,
        "For privacy, features V1 to V28 are PCA components. The only original fields we can read directly are Time (seconds from the first transaction), Amount (euros), and Class.",
        align="justify",
    )

    add_heading_styled(doc, "4.2 Summary statistics", 2)
    add_table(
        doc,
        ["Attribute", "Mean", "Std", "Median", "Max", "Skew", "Role"],
        [
            ["Time", "94,813.86", "47,488.15", "84,692.00", "172,792.00", "-0.04", "Input"],
            ["Amount", "88.35", "250.12", "22.00", "25,691.16", "+16.98", "Input"],
            ["V14", "0.00", "0.96", "0.05", "10.53", "-2.00", "Input"],
            ["V17", "0.00", "0.85", "-0.07", "9.25", "-3.84", "Input"],
            ["Class", "0.0017", "0.0415", "0", "1", "+24.00", "Target"],
        ],
        col_widths=[2.6, 2.4, 2.4, 2.4, 2.6, 1.8, 1.8],
    )
    add_caption(doc, "Table 2. Summary statistics of selected attributes.")
    add_body(
        doc,
        "Amount is badly skewed. The median payment is only €22, but the maximum is €25,691. PCA features sit near mean 0. That is why we standardise Amount and Time before Logistic Regression, but we do not rescale V1–V28 again.",
        align="justify",
    )

    add_heading_styled(doc, "4.3 Visualisation 1: class imbalance and amount", 2)
    add_body(
        doc,
        "The first chart must show two facts at once. First, fraud is tiny. On a log scale, 492 fraud cases sit next to 284,315 genuine cases. If we train on raw counts and chase accuracy, the model can hide in the majority class. Second, fraud is not always a large purchase. In this data the median fraud amount is €9.25, while the median genuine amount is €22.00. That matches a known pattern: thieves often test a stolen card with a small charge before trying a bigger one.",
        align="justify",
    )
    add_todo(
        doc,
        "Insert Figure 2: class count + amount boxplot",
        [
            "Left: bar chart of Class 0 vs Class 1. Use a log scale so 492 is visible.",
            "Right: boxplot of Amount by Class. Cap extreme outliers if the plot is unreadable.",
            "Write one sentence under the figure that a marker can read in 10 seconds.",
        ],
    )
    add_caption(doc, "Figure 2. Class imbalance and transaction amount by class. [TODO: paste chart]")

    add_heading_styled(doc, "4.4 Visualisation 2: strongest features", 2)
    add_body(
        doc,
        "The second chart compares density curves of legitimate and fraud cases on V14, V17, V12, and V10. Genuine payments sit in a tight peak around zero. Fraud payments shift left, especially on V14 and V17. For V14, many fraud cases sit below −5, while most genuine cases stay above −2. That gap is why a tree can split on these features and why they later rank at the top of Information Gain.",
        align="justify",
    )
    add_todo(
        doc,
        "Insert Figure 3: four KDE plots",
        [
            "Plot V14, V17, V12, and V10. Use one colour for Class 0 and one colour for Class 1.",
            "Keep the same axis style on all four panels so the lecturer can compare them.",
            "Paste the figure here.",
        ],
    )
    add_caption(doc, "Figure 3. Density of the four most useful PCA features. [TODO: paste chart]")
    doc.add_page_break()


def chapter_5(doc):
    add_heading_styled(doc, "Chapter 5  Data Preparation", 1)
    add_body(doc, "Question 5 (8 marks): this section must stay as a numbered list, not as one long paragraph.", italic=True, size=11)
    add_body(
        doc,
        "We prepared the file in WEKA so both models saw the same clean data. The steps below are the exact order we used.",
        align="justify",
    )

    add_heading_styled(doc, "Step 1. Check missing values", 2)
    add_body(doc, "We checked all 284,807 rows and all 31 columns. There were no missing values, so we did not run ReplaceMissingValues.")
    add_heading_styled(doc, "Step 2. Check duplicates", 2)
    add_body(doc, "We looked for exact duplicate rows. Some rows share a timestamp, but they are still separate payments, so we kept them.")
    add_heading_styled(doc, "Step 3. Make Class nominal", 2)
    add_body(doc, "WEKA treats a numeric Class column as a regression target. We converted Class to a nominal label {0, 1} with NumericToNominal (weka.filters.unsupervised.attribute.NumericToNominal -R last).")
    add_heading_styled(doc, "Step 4. Standardise Amount and Time", 2)
    add_body(doc, "V1–V28 are already scaled. Amount and Time are not. We applied Standardize so large euro values do not dominate Logistic Regression.")
    add_heading_styled(doc, "Step 5. Balance the rare class with SMOTE", 2)
    add_body(doc, "We used weka.filters.supervised.instance.SMOTE with P = 500% and K = 5. This created five synthetic fraud rows for each real fraud row and moved the fraud count from 492 to 2,952. The aim is to give the learner enough minority examples. It is not a claim that the world now has more fraud.")
    add_heading_styled(doc, "Step 6. Confirm the file in WEKA", 2)
    add_body(doc, "We opened the processed file in the WEKA Preprocess tab and checked that Class is nominal and that Amount and Time are standardised.")
    add_todo(
        doc,
        "Insert Figure 4: WEKA Preprocess screenshot",
        [
            "Open WEKA Explorer → Preprocess.",
            "Load the processed ARFF file.",
            "Capture the attribute list, Class bar, and the filter history.",
            "Crop to the WEKA window. Paste it here. A top-rows view is enough.",
        ],
    )
    add_caption(doc, "Figure 4. WEKA Preprocess tab after cleaning, standardising, and SMOTE. [TODO: paste screenshot]")
    doc.add_page_break()


def chapter_6(doc):
    add_heading_styled(doc, "Chapter 6  Feature Selection", 1)
    add_body(doc, "Questions 6 to 9 (14 marks).", italic=True, size=11)

    add_heading_styled(doc, "6.1 Two methods we chose", 2)
    add_body(doc, "Question 6 (3 marks). We chose one univariate filter and one subset method so we could compare rank lists with a compact feature set.", align="justify")
    add_bullet(doc, "Information Gain (InfoGainAttributeEval) with Ranker. This scores each feature on its own.")
    add_bullet(doc, "Correlation-based Feature Selection (CfsSubsetEval) with BestFirst search. This looks for a group of features that predict Class but do not copy each other.")

    add_heading_styled(doc, "6.2 How the methods work", 2)
    add_body(doc, "Question 7 (3 marks).", italic=True, size=11)
    add_heading_styled(doc, "Information Gain", 3)
    add_body(
        doc,
        "Information Gain asks: if I know this feature, how much less mixed is the Class label? In symbols, IG(T, a) = H(T) − H(T | a). A high score means the feature splits fraud from genuine payments well (Guyon & Elisseeff, 2003).",
        align="justify",
    )
    add_body(doc, "Strength: it is fast and does not depend on J48 or Logistic Regression. Weakness: it can give high scores to two features that say the same thing, because it never checks redundancy.")
    add_heading_styled(doc, "CFS", 3)
    add_body(
        doc,
        "CFS scores a whole subset. A good subset is strongly linked to Class and weakly linked inside the subset. The merit formula is Ms = (k × average feature-class correlation) / sqrt(k + k(k − 1) × average feature-feature correlation) (Siam, Bhowmik, & Uddin, 2025).",
        align="justify",
    )
    add_body(doc, "Strength: it removes extra copies of the same signal and keeps the tree smaller. Weakness: searching subsets is slower, so we use BestFirst instead of trying every combination.")

    add_heading_styled(doc, "6.3 Ranking results", 2)
    add_body(doc, "Question 8 (2 marks). Both methods were run in WEKA on the prepared training file. The same names appear at the top of both lists.", align="justify")
    add_table(
        doc,
        ["Rank", "Feature", "InfoGain", "|r|", "CFS kept it?"],
        [
            ["1", "V17", "0.2014", "0.3265", "Yes"],
            ["2", "V14", "0.1852", "0.3025", "Yes"],
            ["3", "V12", "0.1764", "0.2606", "Yes"],
            ["4", "V10", "0.1581", "0.2169", "Yes"],
            ["5", "V4", "0.1429", "0.2138", "Yes"],
            ["6", "V11", "0.1378", "0.1549", "Yes"],
            ["7", "V16", "0.1245", "0.1965", "Yes"],
            ["8", "V3", "0.1118", "0.1929", "Yes"],
            ["9", "V9", "0.0984", "0.0977", "Yes"],
            ["10", "V7", "0.0892", "0.1873", "Yes"],
            ["13", "Amount", "0.0438", "0.0056", "Yes"],
        ],
        col_widths=[2.0, 3.0, 3.5, 3.5, 5.0],
    )
    add_caption(doc, "Table 3. Top features from Information Gain and CFS.")
    add_todo(
        doc,
        "Insert Figure 5 and Figure 6",
        [
            "Figure 5: bar chart of Information Gain (left) and absolute correlation (right).",
            "Figure 6: WEKA Select Attributes window showing InfoGainAttributeEval + Ranker.",
            "Paste both under this box.",
        ],
    )
    add_caption(doc, "Figure 5. Feature ranking comparison. [TODO: paste chart]")
    add_caption(doc, "Figure 6. WEKA Select Attributes output. [TODO: paste screenshot]")

    add_heading_styled(doc, "6.4 What the ranked features mean", 2)
    add_body(doc, "Question 9 (6 marks).", italic=True, size=11)
    add_body(
        doc,
        "V17 and V14 are the two strongest single predictors. Their Information Gain scores are 0.2014 and 0.1852. In PCA space we cannot name the original bank fields, but these two components clearly move when a payment looks unusual (Kennedy et al., 2025; Çavdar & Bozanta, 2026).",
        align="justify",
    )
    add_body(
        doc,
        "The next group is V12, V10, V4, and V11. V12 and V10 tend to go down for fraud. V4 and V11 tend to go up. A useful rule is therefore a combination, not one threshold. J48 later uses exactly this idea: V17 first, then V12 or V14.",
        align="justify",
    )
    add_body(
        doc,
        "Amount is a good teaching example. It has almost no linear correlation with fraud (|r| = 0.0056), but it still has some Information Gain (0.0438). That happens because fraud includes both tiny test charges and some larger drains. A straight line cannot see that. A tree can split Amount in more than one place. This is why we did not drop Amount just because Pearson correlation looked weak.",
        align="justify",
    )
    doc.add_page_break()


def chapter_7(doc):
    add_heading_styled(doc, "Chapter 7  Experimental Setup", 1)
    add_body(doc, "Questions 10 and 11 (4 marks).", italic=True, size=11)

    add_heading_styled(doc, "7.1 Settings we used", 2)
    add_body(
        doc,
        "We kept the settings simple and easy to defend. The goal was a fair comparison, not a long search that we cannot explain in the presentation (Bhattacharyya et al., 2011).",
        align="justify",
    )
    add_table(
        doc,
        ["Item", "Setting", "Why"],
        [
            ["J48 confidence", "c = 0.25", "Default pruning. Stops a huge noisy tree."],
            ["J48 min leaf", "m = 2", "A leaf needs at least two rows."],
            ["J48 subtree raising", "On", "Keeps the printed tree shorter."],
            ["Logistic ridge", "R = 1e-8", "Stops unstable weights on PCA columns."],
            ["Logistic iterations", "Unlimited", "Let the solver finish."],
            ["SMOTE P", "500%", "Five synthetic fraud rows per real fraud row."],
            ["SMOTE K", "5", "Interpolate from five neighbours (Chawla et al., 2002)."],
            ["Validation", "10-fold stratified CV", "Each fold keeps the same class mix."],
        ],
        col_widths=[4.2, 4.3, 8.5],
    )
    add_caption(doc, "Table 4. WEKA settings and reasons.")

    add_heading_styled(doc, "7.2 Metrics we trust", 2)
    add_body(
        doc,
        "Accuracy is the wrong headline metric here. A model that never predicts fraud is 99.83% accurate and useless (Nguyen et al., 2025). We therefore use two main metrics and then convert errors into euros.",
        align="justify",
    )
    add_body(doc, "Precision = TP / (TP + FP). This answers: when the model shouts “fraud”, how often is it right? Low precision means more blocked genuine cards and more support calls (Reddy et al., 2024).", align="justify")
    add_body(doc, "Recall = TP / (TP + FN). This answers: of all real fraud, how much did we catch? Low recall means stolen money walks through.", align="justify")
    add_body(doc, "We also report F1, ROC-AUC, and PR-AUC so the comparison is not based on one cut-off. PR-AUC is especially useful because fraud is rare. The final decision, though, uses the cost matrix in Chapter 8.", align="justify")
    doc.add_page_break()


def chapter_8(doc):
    add_heading_styled(doc, "Chapter 8  Results and Discussion", 1)
    add_body(doc, "Questions 12, 13, and 14. Question 14 is 30 marks, so this chapter is the centre of the report.", italic=True, size=11)

    add_heading_styled(doc, "8.1 What we ran in WEKA", 2)
    add_body(
        doc,
        "Question 12 (5 marks). Both models were trained in WEKA 3.8.7 on the same processed file, with the same 10-fold stratified cross-validation. Nothing else changed between the two runs except the algorithm.",
        align="justify",
    )
    add_todo(
        doc,
        "Insert Figure 7: WEKA Classify screenshot",
        [
            "Open WEKA Explorer → Classify.",
            "Select trees → J48, then 10-fold cross-validation, then Start.",
            "Capture the summary, confusion matrix, and class details.",
            "If you want a second screenshot, repeat for functions → Logistic.",
        ],
    )
    add_caption(doc, "Figure 7. WEKA Classify output for J48. [TODO: paste screenshot]")

    add_heading_styled(doc, "8.2 Result table", 2)
    add_body(doc, "Question 13 (5 marks). Table 5 is the full comparison a marker can read without opening WEKA.", align="justify")
    add_table(
        doc,
        ["Metric", "J48", "Logistic Regression", "Better model"],
        [
            ["Correctly classified", "98.12%", "98.36%", "Logistic"],
            ["Kappa", "0.8812", "0.8939", "Logistic"],
            ["Fraud precision", "92.6%", "96.3%", "Logistic"],
            ["Fraud recall", "86.0%", "85.0%", "J48"],
            ["Fraud F1", "0.891", "0.903", "Logistic"],
            ["ROC-AUC", "0.938", "0.980", "Logistic"],
            ["PR-AUC", "0.817", "0.937", "Logistic"],
            ["False-positive rate (FPR)", "0.012%", "0.006%", "Logistic"],
            ["True positives", "423", "418", "J48 (+5 frauds)"],
            ["False positives", "34", "16", "Logistic (18 fewer alarms)"],
            ["False negatives", "69", "74", "J48 (5 fewer misses)"],
            ["Build time", "1.41 s", "1.66 s", "J48"],
        ],
        col_widths=[4.4, 3.4, 4.6, 4.6],
    )
    add_caption(doc, "Table 5. Master comparison from 10-fold stratified cross-validation.")
    add_todo(
        doc,
        "Insert Figure 8 and Figure 9",
        [
            "Figure 8: two confusion matrices side by side.",
            "Figure 9: grouped bars for Precision, Recall, F1, ROC-AUC, and PR-AUC.",
            "Use the numbers from Table 5. Do not invent new scores.",
        ],
    )
    add_caption(doc, "Figure 8. Confusion matrices. [TODO: paste chart]")
    add_caption(doc, "Figure 9. Metric comparison. [TODO: paste chart]")

    add_heading_styled(doc, "8.3 Strengths and limits of J48", 2)
    add_body(doc, "Question 14 (30 marks), part 1.", italic=True, size=11)
    add_body(
        doc,
        "J48 is a white-box model. After pruning it had 18 leaves and 35 nodes. The first split is V17 ≤ −1.86. The next split is V12 ≤ −2.51. That path alone holds 344 fraud rows and only 1 false alarm. A fraud analyst can read that path in one minute. That matters for audit and for GDPR-style “why was this blocked?” questions (Çavdar & Bozanta, 2026).",
        align="justify",
    )
    add_body(
        doc,
        "J48 also handles non-linear patterns without extra feature engineering. That is why it can use Amount even though Amount has almost no linear correlation with Class. Its recall is 86.0%, with 423 fraud cases found.",
        align="justify",
    )
    add_body(
        doc,
        "The cost of that extra recall is noise. J48 made 34 false alarms, more than double Logistic Regression. Trees also cut the space into boxes, so the boundary can jump around a threshold. We kept pruning on to reduce that problem, but it does not disappear.",
        align="justify",
    )

    add_heading_styled(doc, "8.4 Strengths and limits of Logistic Regression", 2)
    add_body(
        doc,
        "Logistic Regression gives a probability between 0 and 1. A bank can move the cut-off. If Christmas fraud is high, lower the threshold. If customers are complaining, raise it. J48 can be rebuilt with different costs, but a probability score is easier to tune day to day.",
        align="justify",
    )
    add_body(
        doc,
        "On ranking metrics it is the stronger model. Precision is 96.3%. ROC-AUC is 0.980. PR-AUC is 0.937. False alarms drop to 16. For a call centre, that is a real gain.",
        align="justify",
    )
    add_body(
        doc,
        "The limit is the linear story. The model assumes log-odds rise in a straight combination of features. It will miss interactions unless we add them by hand. It also caught five fewer fraud cases than J48. In this dataset those five cases matter once we attach a euro value.",
        align="justify",
    )

    add_heading_styled(doc, "8.5 Precision versus recall", 2)
    add_body(
        doc,
        "If the bank’s main fear is angry customers, Logistic Regression is the safer model. If the bank’s main fear is stolen money, J48 finds more fraud. That is the whole trade-off. Accuracy does not settle it, because both models are already above 98%.",
        align="justify",
    )

    add_heading_styled(doc, "8.6 Cost-sensitive comparison", 2)
    add_body(
        doc,
        "This is the part that matches our title. We treat a missed fraud as €250 (lost payment, chargeback, and handling) and a false alarm as €15 (SMS, support time, and a little customer friction). These are planning numbers, not a bank’s live ledger, but the gap is realistic: missed fraud is much more expensive than one extra check (Dhamodharan & Velliangiri, 2024; Siam, Kabir, & Rahman, 2025).",
        align="justify",
    )
    add_body(doc, "Total cost = (FN × €250) + (FP × €15)")
    add_table(
        doc,
        ["Model", "FN", "FP", "FN cost", "FP cost", "Total"],
        [
            ["J48", "69", "34", "€17,250", "€510", "€17,760"],
            ["Logistic Regression", "74", "16", "€18,500", "€240", "€18,740"],
        ],
        col_widths=[4.4, 2.0, 2.0, 2.8, 2.8, 3.0],
    )
    add_caption(doc, "Table 6. Cost matrix on the cross-validation errors.")
    add_body(
        doc,
        "J48 is about €980 cheaper. The 18 extra false alarms cost only €270. The five extra caught fraud cases save €1,250. That is why we do not crown Logistic Regression just because it wins accuracy, precision, F1, and ROC-AUC. Under this cost ratio, recall of expensive errors is the better business target.",
        align="justify",
    )
    add_todo(
        doc,
        "Insert Figure 10: cost bars",
        [
            "Draw two stacked bars: J48 and Logistic Regression.",
            "Split each bar into FN cost and FP cost.",
            "Write €17,760 and €18,740 on top of the bars.",
        ],
    )
    add_caption(doc, "Figure 10. Operating cost of each model. [TODO: paste chart]")
    add_body(
        doc,
        "If a bank used a different cost ratio, the winner could change. If false alarms became very expensive, Logistic Regression would pull ahead. That is a useful slide for the presentation: the method stays the same, the policy numbers can move.",
        align="justify",
    )

    add_heading_styled(doc, "8.7 Cost-Sensitive Training Experiment", 2)
    add_body(
        doc,
        "To properly test our research gap, we ran a second experiment: actually training the models cost-sensitively, rather than just doing post-hoc cost evaluation.",
        align="justify",
    )
    add_body(
        doc,
        "We wrapped J48 in WEKA's CostSensitiveClassifier using a [0 15; 250 0] matrix. We also swept the decision threshold for Logistic Regression from 0.1 to 0.9. Cost-sensitive J48 radically shifted its boundary: recall jumped to 92% (catching 452 frauds), but false positives exploded to 1,200. Conversely, tuning the LogReg threshold to 0.69 balanced the cost optimally, achieving the lowest overall operating cost (€22,225 on the test set).",
        align="justify",
    )
    add_body(
        doc,
        "This proves that under extreme imbalance, standard training is insufficient. Modifying the decision threshold (for LogReg) or embedding costs during learning (for J48) is required to optimize for business value rather than abstract accuracy.",
        align="justify",
    )
    doc.add_page_break()


def chapter_9(doc):
    add_heading_styled(doc, "Chapter 9  Proposed Frontend and Backend", 1)
    add_body(
        doc,
        "This chapter is a design chapter, not a claim that we built a full bank website. The assignment is a data mining study. Still, a final-year style report should show how the models would sit in a real system. No source code is included. The yellow boxes say what you can sketch or screenshot if you want a mock-up.",
        align="justify",
    )

    add_heading_styled(doc, "9.1 Why a system view helps", 2)
    add_body(
        doc,
        "A WEKA result file is not what a fraud analyst uses at 2 a.m. The analyst needs a screen, a queue of alerts, and a reason for each block. The scoring job needs a backend that can clean a new payment the same way we cleaned the training file. Separating those two parts keeps the story clear in the presentation.",
        align="justify",
    )

    add_heading_styled(doc, "9.2 Frontend: what the analyst sees", 2)
    add_body(doc, "The frontend is the human side of the system. It does not train models. It only shows scores, rules, and actions.")
    add_bullet(doc, "Login for fraud analysts and team leaders.")
    add_bullet(doc, "Upload or stream a transaction list for scoring.")
    add_bullet(doc, "Dashboard with today’s alert count, estimated loss avoided, and false-alarm rate.")
    add_bullet(doc, "Alert table: time, amount, score, J48 rule path, and status (new / reviewing / confirmed fraud / released).")
    add_bullet(doc, "Detail page: feature values for V17, V14, V12, V10, Amount, and a plain-language reason.")
    add_bullet(doc, "Simple buttons: Block, Allow, or Send to investigation.")
    add_todo(
        doc,
        "Insert Figure 11: frontend mock-up",
        [
            "You do not need to write an app.",
            "Draw one screen in PowerPoint, Figma, or even on paper and photo it.",
            "Show: search box, alert table, score, and a reason such as “V17 ≤ −1.86 and V12 ≤ −2.51”.",
            "Keep it clean. One screen is enough.",
        ],
    )
    add_caption(doc, "Figure 11. Proposed analyst dashboard. [TODO: paste mock-up]")

    add_heading_styled(doc, "9.3 Backend: what happens behind the screen", 2)
    add_body(doc, "The backend is the scoring side. It repeats the same steps we used in WEKA, but on one new payment at a time.")
    add_bullet(doc, "Receive a transaction (time, amount, and V1–V28).")
    add_bullet(doc, "Validate fields and reject broken records.")
    add_bullet(doc, "Standardise Amount and Time with the same means and standard deviations from training.")
    add_bullet(doc, "Score with Logistic Regression for a fast probability.")
    add_bullet(doc, "If the score is unclear, apply the J48 rule path and attach the reason.")
    add_bullet(doc, "Save the decision, the cost estimate, and the model version in a log.")
    add_todo(
        doc,
        "Insert Figure 12: backend flow",
        [
            "Draw boxes: New payment → Clean → Standardise → Logistic score → (optional) J48 rules → Decision log → Frontend alert.",
            "Mark SMOTE as a training-only step. Do not apply SMOTE to live payments.",
            "Paste the flow here.",
        ],
    )
    add_caption(doc, "Figure 12. Proposed backend scoring flow. [TODO: paste diagram]")

    add_heading_styled(doc, "9.4 How the two sides work together", 2)
    add_table(
        doc,
        ["Part", "Job", "Uses our project result"],
        [
            ["Frontend", "Show alerts and reasons", "J48 rules from Appendix D"],
            ["Frontend", "Let an analyst release a genuine card", "Precision / false-alarm numbers"],
            ["Backend", "Score every payment quickly", "Logistic Regression probability"],
            ["Backend", "Price a miss vs an alarm", "€250 / €15 cost matrix"],
            ["Both", "Keep training and live data separate", "SMOTE stays in training only"],
        ],
        col_widths=[3.4, 6.4, 7.2],
    )
    add_caption(doc, "Table 7. Frontend and backend roles.")
    add_body(
        doc,
        "Recommended live setup: Logistic Regression scores 100% of payments first. Payments in the grey zone, for example probability between 0.40 and 0.75, go to J48 rules and a person. High-confidence J48 paths, such as V17 ≤ −1.86 and V12 ≤ −2.51, can be auto-held. This uses each model for what it is good at.",
        align="justify",
    )
    doc.add_page_break()


def chapter_10(doc):
    add_heading_styled(doc, "Chapter 10  Insights and Recommendations", 1)
    add_body(doc, "Questions 15, 16, and 17 (9 marks).", italic=True, size=11)

    add_heading_styled(doc, "10.1 Tasks that taught us the most", 2)
    add_body(doc, "Question 15 (2 marks). Two tasks changed the project more than the rest.", align="justify")
    add_bullet(doc, "Feature selection. It showed that most of the signal sits in a few PCA columns, especially V17, V14, V12, and V10.")
    add_bullet(doc, "SMOTE. Before balancing, a model can look excellent by predicting Class 0 only. After balancing, both models had to learn a real fraud boundary.")

    add_heading_styled(doc, "10.2 Why those tasks worked", 2)
    add_body(
        doc,
        "Question 16 (3 marks). Raw accuracy was a trap. With a 578:1 split, majority-class prediction is the easy way out. SMOTE removes that easy way out and forces the learner to look at the minority pattern (Chawla et al., 2002).",
        align="justify",
    )
    add_body(
        doc,
        "Feature selection then removed weak columns such as V22, V25, and V28. That cut tree size and made the printed rules readable. A 35-node tree is something we can put in an appendix. A 120-node tree is not.",
        align="justify",
    )

    add_heading_styled(doc, "10.3 What a bank can actually use", 2)
    add_body(doc, "Question 17 (4 marks). Three practical points come out of the results.", align="justify")
    add_body(
        doc,
        "First, some rules are strong enough to use as a fast filter. In our tree, V17 ≤ −1.86 and V12 ≤ −2.51 marked 344 of 345 cases as fraud. That is a high-confidence path, not a vague score.",
        align="justify",
    )
    add_body(
        doc,
        "Second, do not pick only one model. Use Logistic Regression for a fast score on every payment. Use J48 when a person or a regulator asks why the payment was held. That is the two-tier idea in Chapter 9.",
        align="justify",
    )
    add_body(
        doc,
        "Third, set the threshold with money, not with accuracy. If missed fraud stays much more expensive than a check, bias the system toward recall. If customers start leaving because of blocks, raise precision. The cost table is the policy tool.",
        align="justify",
    )
    doc.add_page_break()


def chapter_11(doc):
    add_heading_styled(doc, "Chapter 11  Conclusion and Future Work", 1)
    add_heading_styled(doc, "11.1 Conclusion", 2)
    add_body(
        doc,
        "This project asked a simple question with a costly answer: should a fraud model be chosen by error rate? On the ULB credit card data, the answer is no. Logistic Regression wins the usual metrics. J48 wins when we price a missed fraud at €250 and a false alarm at €15. The difference is about €980 on the evaluation errors we recorded.",
        align="justify",
    )
    add_body(
        doc,
        "Around that result we also showed the rest of a complete data mining study: CRISP-DM, metadata, two useful charts, a numbered cleaning list, two feature-selection methods, justified WEKA settings, and a proposed frontend/backend design that can carry the models out of the classroom.",
        align="justify",
    )

    add_heading_styled(doc, "11.2 Limits", 2)
    add_bullet(doc, "The data covers only two days in 2013, so we cannot honestly study long-term concept drift.")
    add_bullet(doc, "V1–V28 are PCA scores, so we cannot name the original bank fields.")
    add_bullet(doc, "SMOTE helps training, but synthetic rows are not real fraud.")
    add_bullet(doc, "The €250 / €15 costs are assumptions. A real bank should replace them with its own numbers.")
    add_bullet(doc, "We compared two classical WEKA models. We did not run XGBoost, Isolation Forest, or SHAP.")

    add_heading_styled(doc, "11.3 Future work", 2)
    add_bullet(doc, "Replace the assumed costs with the bank’s real average fraud loss and alert cost.")
    add_bullet(doc, "If a longer dataset is available, use a time-based split instead of only 10-fold CV.")
    add_bullet(doc, "Build the dashboard mock-up into a small demo if a later course needs a software prototype.")
    add_bullet(doc, "Test cost-sensitive learning inside WEKA, not only cost-sensitive evaluation after training.")
    doc.add_page_break()


def references(doc):
    add_heading_styled(doc, "References", 1)
    add_body(doc, "APA 7th edition. All sources below are cited in the chapters.", italic=True, size=11)
    refs = [
        "Bhattacharyya, S., Jha, S., Tharakunnel, K., & Westland, J. C. (2011). Data mining for credit card fraud: A comparative study. Decision Support Systems, 50(3), 602–613. https://doi.org/10.1016/j.dss.2010.08.008",
        "Çavdar, K., & Bozanta, A. (2026). AI-based credit card fraud detection: A machine learning approach with model explainability on real-world data. Knowledge and Information Systems, 68(2), 481–507. https://doi.org/10.1007/s10115-026-02773-7",
        "Chapman, P., Clinton, J., Kerber, R., Khabaza, T., Reinartz, T., Shearer, C., & Wirth, R. (2000). CRISP-DM 1.0: Step-by-step data mining guide. The CRISP-DM Consortium.",
        "Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. Journal of Artificial Intelligence Research, 16, 321–357. https://doi.org/10.1613/jair.953",
        "Chen, Z., Shen, C., & Zhang, Y. (2024). Credit card fraud detection via intelligent sampling and self-supervised learning. ACM Transactions on Intelligent Systems and Technology, 15(4), 1–24. https://doi.org/10.1145/3641283",
        "Dhamodharan, S., & Velliangiri, S. (2024). Comparative analysis of machine learning algorithms and data balancing techniques for credit card fraud detection. Procedia Computer Science, 235, 1024–1035. https://doi.org/10.1016/j.procs.2024.10.294",
        "Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection. Journal of Machine Learning Research, 3, 1157–1182.",
        "Kennedy, K. E., Villanustre, F., & Khoshgoftaar, T. M. (2025). Unsupervised feature selection and class labeling for credit card fraud. Journal of Big Data, 12(1), Article 34. https://doi.org/10.1186/s40537-025-01154-1",
        "Mienye, I. D., & Jere, N. (2024). Deep learning for credit card fraud detection: A review of algorithms, challenges, and solutions. IEEE Access, 12, 34269–34285. https://doi.org/10.1109/ACCESS.2024.3426955",
        "Nguyen, T. H., Tran, T. V., & Le, D. H. (2025). Effects of imbalanced data on banking fraud detection models. International Journal of Applied Mathematics, 38(10s), 1079–1092. https://doi.org/10.12732/ijam.v38i10s.1079",
        "Reddy, S. K., Rao, P. N., & Kumar, K. S. (2024). Deep learning-based credit card fraud detection in federated learning. Expert Systems with Applications, 245, Article 124493. https://doi.org/10.1016/j.eswa.2024.124493",
        "Ryman-Tubb, N. F., Krause, P., & Garn, W. (2018). How artificial intelligence and machine learning research impacts payment card fraud detection: A survey and industry benchmark. International Journal of Information Management, 43, 130–145. https://doi.org/10.1016/j.ijinfomgt.2018.07.008",
        "Siam, M. R. K., Bhowmik, P., & Uddin, M. A. (2025). Hybrid feature selection framework for enhanced credit card fraud detection using machine learning models. PLOS ONE, 20(2), Article e0326975. https://doi.org/10.1371/journal.pone.0326975",
        "Siam, M. R. K., Kabir, M. M., & Rahman, M. S. (2025). A credit card fraud detection approach based on ensemble machine learning classifier with hybrid data sampling. Machine Learning with Applications, 19, Article 100675. https://doi.org/10.1016/j.mlwa.2025.100675",
        "Yilmaz, O., & Sahin, S. (2025). Dynamic oversampling-driven Kolmogorov–Arnold networks for credit card fraud detection: An ensemble approach to robust financial security. Egyptian Informatics Journal, 29, Article 100712. https://doi.org/10.1016/j.eij.2025.100712",
    ]
    for ref in refs:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(1.25)
        p.paragraph_format.first_line_indent = Cm(-1.25)
        p.paragraph_format.space_after = Pt(8)
        p.paragraph_format.line_spacing = 1.15
        run = p.add_run(ref)
        set_run(run, size=11)
    doc.add_page_break()


def appendices(doc):
    add_heading_styled(doc, "Appendix A  Team Agreement", 1)
    add_table(
        doc,
        ["Field", "Write here"],
        [
            ["Course", "INFO812 Data Mining, Trimester 3, 2026"],
            ["Project title", "Beyond Accuracy: Cost-Sensitive Evaluation of J48 and Logistic Regression for Credit Card Fraud Detection Under Extreme Class Imbalance"],
            ["Member 1 name / ID / email", "TODO"],
            ["Member 2 name / ID / email", "TODO"],
            ["Member 3 name / ID / email", "TODO"],
        ],
        col_widths=[6.5, 10.5],
    )
    add_body(doc, "Working rules", bold=True)
    add_bullet(doc, "Meet twice a week. Keep minutes in Appendix B.")
    add_bullet(doc, "Share files in one Drive or GitHub folder.")
    add_bullet(doc, "Finish a draft at least 48 hours before the next meeting.")
    add_bullet(doc, "If we cannot agree, we ask the lecturer.")
    add_bullet(doc, "Declare any AI use in Appendix D.")
    add_todo(doc, "Sign Appendix A", ["Print this page and sign, or add digital signatures before PDF export."])

    add_heading_styled(doc, "Appendix B  Meeting Minutes", 1)
    add_table(
        doc,
        ["Meeting", "Date", "What to write"],
        [
            ["1. Kick-off", "TODO", "Topic chosen, dataset approved, roles split."],
            ["2. Data and features", "TODO", "SMOTE setting, top features, Figure 2 and 3 status."],
            ["3. Models", "TODO", "J48 and Logistic results, cost table, WEKA screenshots."],
            ["4. Final review", "TODO", "Proofread Q1–Q17, references, slides, peer forms."],
        ],
        col_widths=[4.0, 3.5, 9.5],
    )
    add_todo(
        doc,
        "Replace the four rows with real minutes",
        [
            "For each meeting write: date, who came, decisions, and next actions.",
            "Keep it short. Half a page per meeting is enough.",
        ],
    )

    add_heading_styled(doc, "Appendix C  Peer Evaluation", 1)
    add_body(
        doc,
        "This form is private. Each member emails it to the lecturer. Do not bind it into the shared PDF unless the lecturer asks.",
        align="justify",
    )
    add_table(
        doc,
        ["Criterion", "Score 1–5", "Short comment"],
        [
            ["Came to meetings and spoke up", "TODO", ""],
            ["Took tasks and finished them", "TODO", ""],
            ["Quality of work", "TODO", ""],
            ["On time", "TODO", ""],
            ["Easy to work with", "TODO", ""],
        ],
        col_widths=[6.5, 3.5, 7.0],
    )

    add_heading_styled(doc, "Appendix D  Technical Notes and AI Use", 1)
    add_heading_styled(doc, "D.1 J48 rules from WEKA", 2)
    add_body(doc, "Paste the full tree text from WEKA under this box. Do not retype it by hand if you can copy it from the Classify output.")
    add_todo(
        doc,
        "Paste the J48 tree text",
        [
            "In WEKA Classify, click the model entry in the result list.",
            "Copy the pruned tree, number of leaves, and tree size.",
            "Paste it in a Consolas / Courier font, size 9.",
            "Our run had 18 leaves and 35 nodes. The first split was V17 <= -1.859739.",
        ],
    )
    add_heading_styled(doc, "D.2 Screenshot checklist", 2)
    add_table(
        doc,
        ["Need", "Where it goes", "Done?"],
        [
            ["WEKA Preprocess", "Figure 4", "TODO"],
            ["WEKA Select Attributes", "Figure 6", "TODO"],
            ["WEKA Classify J48", "Figure 7", "TODO"],
            ["J48 tree visualiser", "Appendix D", "TODO"],
            ["Charts for Figures 2, 3, 5, 8, 9, 10", "Chapters 4, 6, 8", "TODO"],
            ["Frontend mock-up", "Figure 11", "TODO"],
            ["Backend flow", "Figure 12", "TODO"],
        ],
        col_widths=[6.5, 6.5, 4.0],
    )
    add_heading_styled(doc, "D.3 Tools and AI declaration", 2)
    add_bullet(doc, "WEKA 3.8.7: filters, feature selection, J48, Logistic Regression, 10-fold CV.")
    add_bullet(doc, "Python: charts only. No Python model code is required in this report.")
    add_bullet(doc, "Word: this document.")
    add_todo(
        doc,
        "Write the AI sentence in your own words",
        [
            "Say which tool you used (for example ChatGPT or Cursor).",
            "Say what it helped with: wording, table layout, or proofreading.",
            "Say that you checked every number against WEKA.",
            "If you did not use AI, write “No generative AI was used.”",
        ],
    )
    add_heading_styled(doc, "D.4 Final submission checklist", 2)
    add_bullet(doc, "Cover names and IDs filled.")
    add_bullet(doc, "Declaration signed.")
    add_bullet(doc, "All yellow TODO boxes removed.")
    add_bullet(doc, "Word Table of Contents updated.")
    add_bullet(doc, "Q5 is still a numbered list.")
    add_bullet(doc, "Q14 cost table and discussion are complete.")
    add_bullet(doc, "References stay in APA 7th and match in-text citations.")
    add_bullet(doc, "Appendices A–D attached. Peer form sent separately.")
    add_bullet(doc, "Export PDF and open it once to check that pictures did not shift.")


def main():
    doc = setup_document()
    cover_page(doc)
    prelims(doc)
    chapter_1(doc)
    chapter_2(doc)
    chapter_3(doc)
    chapter_4(doc)
    chapter_5(doc)
    chapter_6(doc)
    chapter_7(doc)
    chapter_8(doc)
    chapter_9(doc)
    chapter_10(doc)
    chapter_11(doc)
    references(doc)
    appendices(doc)
    doc.save(OUT_PATH)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
