from fpdf import FPDF

inp = 'hw4p3.py'  # your Python file
out = 'hw4p3.pdf'    # desired PDF filename

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

with open(inp, 'r') as f:
    txt = f.read()
    pdf.multi_cell(0, 10, txt)

pdf.output(out)
print("PDF is Saved Successfully")