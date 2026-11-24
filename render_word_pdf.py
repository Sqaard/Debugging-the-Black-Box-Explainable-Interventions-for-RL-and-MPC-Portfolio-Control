from pathlib import Path
import fitz

pdf = Path(__file__).with_name('Explainable_RL_and_MPC_Report_word.pdf')
out = Path(__file__).with_name('rendered_report_word')
out.mkdir(exist_ok=True)
doc = fitz.open(pdf)
for i, page in enumerate(doc, 1):
    pix = page.get_pixmap(matrix=fitz.Matrix(1.35, 1.35), alpha=False)
    pix.save(out / f'page-{i}.png')
print('pages', len(doc), 'out', out)
