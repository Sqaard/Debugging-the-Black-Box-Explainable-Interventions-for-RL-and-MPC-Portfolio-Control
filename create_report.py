from __future__ import annotations

import csv
import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "XRL_MPC_project" / "artifacts"
RUN_ID = "xrl_mpc_real_market_20261124"
STAGE9 = RUN / "stage9" / RUN_ID
STAGE8 = RUN / "stage8" / RUN_ID
STAGE1 = RUN / "stage1" / RUN_ID
STAGE6 = RUN / "stage6" / RUN_ID
OUT = ROOT / "XRL_MPC_project" / "Explainable_RL_and_MPC_Report.docx"


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fmt_pct(x):
    return f"{float(x) * 100:.2f}%"


def fmt_num(x):
    return f"{float(x):.3f}"


def set_cell_shading(cell, fill: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_text(cell, text: str, bold: bool = False):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run(str(text))
    run.bold = bold
    run.font.name = "Times New Roman"
    run.font.size = Pt(9)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_table(doc, headers, rows, widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for i, header in enumerate(headers):
        set_cell_text(table.rows[0].cells[i], header, bold=True)
        set_cell_shading(table.rows[0].cells[i], "D9EAF7")
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            set_cell_text(cells[i], value)
    if widths:
        for row in table.rows:
            for i, width in enumerate(widths):
                row.cells[i].width = Inches(width)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return table


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(8)
    run = p.add_run(text)
    run.italic = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(9)


def add_bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    p.add_run(text)


metrics = read_csv(STAGE9 / "final_metrics.csv")
controls = read_csv(STAGE8 / "matched_random_and_negative_controls.csv")
quality = json.loads((STAGE1 / "primitive_quality.json").read_text(encoding="utf-8"))
effect = json.loads((STAGE8 / "control_adjusted_effect.json").read_text(encoding="utf-8"))
adapter = json.loads((STAGE6 / "repair_adapter_spec.json").read_text(encoding="utf-8"))

metric_by_name = {row["model"]: row for row in metrics}
opposite = next(row for row in controls if row["control_type"] == "negative_opposite")

doc = Document()
section = doc.sections[0]
section.top_margin = Inches(0.8)
section.bottom_margin = Inches(0.8)
section.left_margin = Inches(0.85)
section.right_margin = Inches(0.85)

styles = doc.styles
styles["Normal"].font.name = "Times New Roman"
styles["Normal"].font.size = Pt(10.5)
styles["Normal"].paragraph_format.space_after = Pt(6)
styles["Normal"].paragraph_format.line_spacing = 1.08
for name in ["Heading 1", "Heading 2", "Heading 3"]:
    styles[name].font.name = "Times New Roman"
    styles[name].font.color.rgb = RGBColor(31, 78, 121)
styles["Heading 1"].font.size = Pt(15)
styles["Heading 2"].font.size = Pt(12)

title = doc.add_paragraph(style="Title")
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title.paragraph_format.space_after = Pt(8)
title_run = title.add_run("Explainable Reinforcement Learning and Model Predictive Control for Stress Aware Portfolio Decisions")
title_run.font.name = "Times New Roman"
title_run.font.size = Pt(18)
title_run.bold = True

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
subtitle_run = subtitle.add_run("Machine Learning II Final Report")
subtitle_run.font.name = "Times New Roman"
subtitle_run.font.size = Pt(12)
subtitle_run.italic = True

add_table(doc, ["Field", "Value"], [
    ("Student ID", "____________________"),
    ("Name", "Ivan Pavliuk"),
    ("Major", "Financial Modelling"),
    ("Course Instructor", "____________________"),
    ("School Department Institute", "____________________"),
    ("Date", "November 24, 2026"),
], widths=[2.3, 4.4])

doc.add_heading("Abstract", level=1)
doc.add_paragraph(
    "This report compares a tabular reinforcement learning portfolio controller with a transparent receding horizon model predictive controller on a real United States market stress panel. The state contains SPY returns and volatility, VIX and VIX3M, MOVE, high yield and investment grade bond ETFs, and FRED or ALFRED stress variables. The RL controller is trained only on 2010-2017 data. A validation period from 2018-2021 selects a single primitive intervention, while the frozen test from 2022-01-03 to 2023-02-28 is used once for the before and after comparison. On the frozen test, baseline RL has Sharpe -0.653 and maximum drawdown -21.71%; baseline MPC has Sharpe 0.322 and maximum drawdown -6.62%. A train-only behavior vocabulary identifies a stress overexposure primitive. Replacing its action with cash preservation improves RL to Sharpe -0.460 and maximum drawdown -14.95%, with a control-adjusted Sharpe lift of +0.214 versus twenty matched-random repairs. Robust MPC reduces the frozen-test maximum drawdown to -7.84% and reaches Sharpe 1.99 on the outer 2023-2026 holdout. The results support explainability as a debugging and control interface, while the outer holdout also shows that a local RL repair is not a universal improvement."
)
doc.add_paragraph("Keywords: reinforcement learning; model predictive control; explainable AI; portfolio control; stress regime; counterfactual intervention")

doc.add_heading("1 Introduction", level=1)
doc.add_paragraph(
    "Financial decisions are sequential, costly, and exposed to distribution shift. A portfolio controller can be profitable in a calm period and fail when volatility and credit conditions change. This makes explainability a model validation problem rather than a presentation feature. The useful question is whether an explanation identifies a behavior that can be changed under a frozen policy and whether the change survives controls. This report uses that idea to compare two course methods: reinforcement learning, which adapts through a value function, and model predictive control, which solves an explicit optimization problem at every decision time."
)
doc.add_paragraph(
    "The project has three experimental stages. Stage A measures both controllers before any intervention. Stage B discovers behavior primitives, labels their market mechanisms, and uses validation data to select one low-dose repair. Stage C repeats the frozen-test comparison and evaluates matched-random and opposite-direction negative controls. The primary contribution is therefore a controlled XAI loop: describe behavior, intervene on the described primitive, and test whether the result is better than an equally sized arbitrary edit."
)

doc.add_heading("2 Problem Formulation", level=1)
doc.add_paragraph(
    "The portfolio is represented as a Markov decision process. At day t, the state s_t contains a causal window of market features, the current allocation, and a discrete stress representation. The action a_t is one of six allocation vectors over SPY, TLT, HYG, and residual cash. The action is applied to the return from t to t+1. The daily reward used by Q-learning is"
)
eq = doc.add_paragraph()
eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
eq_run = eq.add_run("r_t = log(1 + w(a_t)' R_(t+1)) - 0.0006 ||w(a_t) - w(a_(t-1))||_1")
eq_run.italic = True
doc.add_paragraph(
    "where the second term is a small turnover cost. The state does not use future returns. Rolling percentiles for volatility and VIX are estimated from past observations only. The frozen test is isolated from model fitting and repair selection."
)
doc.add_paragraph(
    "The MPC controller uses the same action grid and solves a short-horizon discrete approximation to a mean-variance problem. For each candidate action, it forms a rolling mean vector mu_t and covariance matrix Sigma_t from the previous sixty observations and scores"
)
eq = doc.add_paragraph()
eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
eq_run = eq.add_run("J(w) = H w' mu_t - 8 H w' Sigma_t w - 0.0006 ||w - w_(t-1)||_1")
eq_run.italic = True
doc.add_paragraph(
    "The robust MPC variant adds an explicit equity concentration constraint during stress and a VIX-linked uncertainty penalty. This makes the risk control visible in the objective and constraints, which is the main explainability advantage of MPC."
)

doc.add_heading("3 Methods and Algorithms", level=1)
doc.add_heading("3.1 Reinforcement learning challenger", level=2)
doc.add_paragraph(
    "The RL controller is tabular Q-learning over 18 stress states formed by three SPY trend bins, three VIX percentile bins, and a credit-relative-return flag. Six actions range from cash preservation to full SPY exposure. The Q-table is updated for 160 passes through the training period with epsilon-greedy exploration, discount factor 0.985, and learning rate 0.08. This small state space makes the action logic auditable and provides an interpretable baseline for the intervention experiment."
)
doc.add_heading("3.2 Behavior primitives and intervention", level=2)
doc.add_paragraph(
    "The primitive layer clusters train and validation behavior windows using six KMeans primitives. Each row contains the state, selected action, stress flag, twenty-day SPY return, VIX percentile, and credit-relative return. Labels are assigned from convergent descriptive evidence: stress frequency, mean action, and mean return. The vocabulary uses all six codes, has perplexity 5.336, median run length 2 days, and state-to-primitive NMI 0.640. These are descriptive alignment metrics, not proof of causal structure."
)
doc.add_paragraph(
    "The repair is selected on validation data only. Among stress overexposure primitives and replacement actions 0, 1, and 2, primitive 0 mapped to cash preservation gives the highest validation selection score: Sharpe lift 0.306 plus a small drawdown-relief term. The frozen policy and Q-table remain unchanged. Only the primitive-to-action adapter is edited."
)
doc.add_heading("3.3 Controls", level=2)
doc.add_paragraph(
    "The intervention is compared with twenty matched-random remappings that use the same replacement action but a randomly selected non-target primitive. An opposite-direction negative control maps the target primitive to the full-equity action. The reported causal-style quantity is the target frozen-test Sharpe minus the mean matched-random Sharpe. This is a research control-adjusted effect, not a product certification."
)

doc.add_heading("4 Software Architecture and Implementation", level=1)
doc.add_paragraph(
    "The implementation is a single reproducible Python program. It loads the real market files, constructs causal features, fits Q-learning, runs MPC, discovers primitives, and writes machine-readable artifacts. Stage 0 records the data contract and splits. Stage 0.1 records model outputs. Stages 1-3 record primitive discovery and labels. Stage 4 locks the frozen test. Stages 5-6 record the control surface and repair adapter. Stages 7-9 record post-intervention results, controls, and the final verdict."
)
doc.add_paragraph(
    "The project was cross-checked against the remote directory `/home/vanya/projects/trustbench_stress_vanya/data/data_RLagent_for_Joseph/data/_dow_extended` on host `tsinghua-quant`. The local copy is used for the reported run so that the course submission contains all inputs needed for reproduction."
)

doc.add_heading("5 Data Description and Experimental Design", level=1)
doc.add_paragraph(
    "The main panel contains daily adjusted prices for SPY, TLT, HYG, and LQD, plus VIX, VIX3M, and MOVE. Derived features include one-day returns, twenty-day SPY momentum and volatility, VIX term structure, HYG minus LQD relative performance, and five-day changes in STLFSI4 and the 10Y-2Y spread. The market data are real US observations from 2010-01-04 through 2026-07-07. FRED and ALFRED values are merged by observation date and forward-filled only after availability; missing early stress values are treated as zero for the diagnostic feature."
)
add_table(doc, ["Split", "Dates", "Purpose"], [
    ("Train", "2010-01-04 to 2017-12-29", "Q-learning and primitive fit"),
    ("Validation", "2018-01-02 to 2021-12-31", "Repair selection"),
    ("Frozen test", "2022-01-03 to 2023-02-28", "Used once for Stage A and Stage C"),
    ("Outer holdout", "2023-03-01 to 2026-07-07", "Post-test robustness check"),
], widths=[1.35, 2.2, 3.15])
add_caption(doc, "Table 1. Time splits and their roles in the experiment")

doc.add_heading("6 Experimental Results", level=1)
doc.add_heading("6.1 Stage A baseline comparison", level=2)
rows = []
for row in metrics:
    if row["model"] in ["RL validation", "MPC validation", "RL frozen test", "MPC frozen test"]:
        rows.append([row["model"], fmt_pct(row["annual_return"]), fmt_num(row["sharpe"]), fmt_pct(row["max_drawdown"]), fmt_pct(row["turnover"])])
add_table(doc, ["Model", "Annual return", "Sharpe", "Max drawdown", "Turnover"], rows, widths=[1.7, 1.3, 0.9, 1.25, 1.0])
add_caption(doc, "Table 2. Stage A performance before intervention")
doc.add_paragraph(
    "On validation, RL is more profitable than the baseline MPC but trades more and carries greater drawdown. The frozen test reverses the ranking: RL loses 11.51% annualized with Sharpe -0.653, while MPC earns 3.05% annualized with Sharpe 0.322 and a much smaller drawdown. This is a direct example of why validation return alone is not an adequate model-selection criterion in a drifting market."
)

doc.add_heading("6.2 Stage B diagnosis and repair", level=2)
add_table(doc, ["Diagnostic", "Value"], [
    ("Primitive vocabulary", "K = 6; utilization = 1.000"),
    ("Primitive perplexity", f"{quality['perplexity']:.3f}"),
    ("Median primitive run length", f"{quality['median_run_length']:.1f} trading days"),
    ("State to primitive NMI", f"{quality['state_primitive_nmi']:.3f}"),
    ("Selected target", f"Primitive {adapter['target_primitive']} stress overexposure"),
    ("Replacement", f"Action {adapter['replacement_action']} {adapter['replacement_name']}"),
], widths=[2.8, 3.4])
add_caption(doc, "Table 3. Stage B behavior vocabulary and selected intervention")
doc.add_paragraph(
    "The selected primitive is a stress overexposure mode: its action distribution is more aggressive than the other stress primitives. The validation selector maps it to cash preservation rather than changing the reward or retraining the policy. This makes the intervention a small behavioral edit with a clear operational meaning."
)

doc.add_heading("6.3 Stage C post-intervention results", level=2)
rows = []
for row in metrics:
    if row["model"] in ["RL frozen test", "RL repaired frozen test", "MPC frozen test", "MPC robust frozen test"]:
        rows.append([row["model"], fmt_pct(row["annual_return"]), fmt_num(row["sharpe"]), fmt_pct(row["max_drawdown"]), fmt_pct(row["turnover"])])
add_table(doc, ["Model", "Annual return", "Sharpe", "Max drawdown", "Turnover"], rows, widths=[1.7, 1.3, 0.9, 1.25, 1.0])
add_caption(doc, "Table 4. Frozen-test Stage C comparison")
doc.add_paragraph(
    f"The RL repair improves the frozen-test Sharpe from {effect['baseline_sharpe']:.3f} to {effect['repaired_sharpe']:.3f}, changes turnover from {fmt_pct(metric_by_name['RL frozen test']['turnover'])} to {fmt_pct(metric_by_name['RL repaired frozen test']['turnover'])}, and improves maximum drawdown from {fmt_pct(effect['baseline_max_dd'])} to {fmt_pct(effect['repaired_max_dd'])}. Annualized return improves from {fmt_pct(metric_by_name['RL frozen test']['annual_return'])} to {fmt_pct(metric_by_name['RL repaired frozen test']['annual_return'])}, but remains negative, so this is a risk-control and debugging result rather than evidence of a profitable strategy. The matched-random mean Sharpe is {effect['random_control_mean_sharpe']:.3f}; the control-adjusted lift is {effect['control_adjusted_sharpe_lift']:+.3f}. The opposite-direction negative control has Sharpe {float(opposite['sharpe']):.3f}, which is weaker than the targeted repair."
)
figure = STAGE9 / "stage_A_to_C_wealth.png"
if figure.exists():
    doc.add_picture(str(figure), width=Inches(6.5))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_caption(doc, "Figure 1. Frozen-test wealth trajectories for Stage A and Stage C")

doc.add_heading("6.4 Outer holdout", level=2)
outer_rows = []
for row in metrics:
    if "outer holdout" in row["model"]:
        outer_rows.append([row["model"], fmt_pct(row["annual_return"]), fmt_num(row["sharpe"]), fmt_pct(row["max_drawdown"]), fmt_pct(row["turnover"])])
add_table(doc, ["Model", "Annual return", "Sharpe", "Max drawdown", "Turnover"], outer_rows, widths=[1.7, 1.3, 0.9, 1.25, 1.0])
add_caption(doc, "Table 5. Exploratory outer holdout after the frozen test")
doc.add_paragraph(
    "The outer holdout changes the interpretation. Robust MPC reaches Sharpe 1.990 with maximum drawdown -7.44%, whereas baseline MPC reaches Sharpe 0.728 with maximum drawdown -11.10%. The RL repair does not generalize as a universal improvement: repaired RL has Sharpe 0.800 versus 1.451 for baseline RL. This is exactly the type of drift alert that the XAI layer is intended to expose."
)

doc.add_heading("7 Discussion and Limitations", level=1)
add_bullet(doc, "The experiment uses a compact tabular RL challenger rather than the full CHRL PPO stack. This keeps the course submission reproducible and makes the intervention semantics explicit, but it limits claims about deep policy representations.")
add_bullet(doc, "The primitive labels are convergent descriptive labels, not mechanistic proof. Hidden PPO activations and TCAV-style probes are outside this run.")
add_bullet(doc, "The frozen test is a single historical episode and the outer holdout was inspected after the main experiment. Multiple markets, purged folds, block bootstrap, and a full Deflated Sharpe calculation are required before any investment claim.")
add_bullet(doc, "MPC is transparent by construction, but its forecasts are rolling sample estimates. The robust constraints improve risk in the holdout while sacrificing frozen-test return, illustrating the return-risk tradeoff.")
add_bullet(doc, "The customer behavior and marketing datasets described in the research statement are not used here. This report is restricted to real US market and macro data, so it avoids treating synthetic client behavior as financial evidence.")

doc.add_heading("8 Conclusion", level=1)
doc.add_paragraph(
    "The before and after experiment demonstrates a practical XAI workflow for sequential portfolio decisions. MPC provides a readable controller and strong drawdown protection under the reported stress split. RL provides an adaptable challenger but is vulnerable to regime drift. A primitive-based intervention selected on validation improves the frozen-test RL behavior relative to matched-random edits, yet the outer holdout shows that the repair is local rather than universal. The defensible conclusion is therefore methodological: explanations are useful when they identify an editable behavior, specify a low-dose intervention, and survive negative controls."
)

doc.add_heading("References", level=1)
references = [
    "Atrey, A., Clary, K., and Jensen, D. (2020). Exploratory not explanatory: Counterfactual analysis of saliency maps for deep reinforcement learning. ICLR.",
    "Bailey, D. H., and Lopez de Prado, M. (2014). The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting and non-normality. Journal of Portfolio Management, 40(5), 94-107.",
    "Boyd, S., and Vandenberghe, L. (2004). Convex Optimization. Cambridge University Press.",
    "Doshi-Velez, F., and Kim, B. (2017). Towards a rigorous science of interpretable machine learning. arXiv:1702.08608.",
    "Fama, E. F., and French, K. R. (1993). Common risk factors in the returns on stocks and bonds. Journal of Financial Economics, 33(1), 3-56.",
    "Milani, S., Topin, N., Veloso, M., and Fang, F. (2024). Explainable reinforcement learning: A survey and comparative review. ACM Computing Surveys, 56(7).",
    "van den Oord, A., Vinyals, O., and Kavukcuoglu, K. (2017). Neural discrete representation learning. NeurIPS.",
]
for reference in references:
    p = doc.add_paragraph(style="List Number")
    p.paragraph_format.left_indent = Inches(0.25)
    p.add_run(reference)

doc.save(OUT)
print(OUT)
