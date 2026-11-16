# PBL-3 Reports Pack

## University submission report

| File | Description |
|---|---|
| `PBL3_Project_Report.docx` | University-format PBL-3 report |
| `PBL3_Project_Report.pdf` | Same content as DOCX |

Structure: Cover → Abstract (2 paragraphs) → List of Tables → List of Figures → TOC → Chapter 1 → Chapter 3 → Chapter 4 → Chapter 5 → IEEE References → Appendix.

Formatting: A4, 0.7 in margins, Times New Roman 12 pt, 1.15 spacing, justified body text.

## Internal companions (not part of the submission body)

| File | Description |
|---|---|
| `Technical_Audit.md` | Critical technical audit |
| `Feature_Verification_Matrix.md` | Feature verification matrix |
| `Architecture_Documentation.md` | Architecture notes |
| `assets/` | Screenshots and generated diagrams |
| `_evidence/` | pytest / build logs |
| `generate_pbl3_report.py` | Regenerates DOCX/PDF |

```bash
.venv/bin/python -m pip install python-docx fpdf2 matplotlib pillow
.venv/bin/python reports/generate_pbl3_report.py
```
