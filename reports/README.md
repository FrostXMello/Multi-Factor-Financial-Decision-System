# PBL-3 Reports Pack

Generated from repository audit at commit documented in `_evidence/audit_meta.txt`.

## Deliverables

| File | Description |
|---|---|
| `PBL3_Project_Report.docx` | Academic project report (placeholders for university/team) |
| `PBL3_Project_Report.pdf` | PDF version of the academic report |
| `Technical_Audit.md` | Critical technical audit with severity register |
| `Feature_Verification_Matrix.md` | Feature-by-feature verification status |
| `Architecture_Documentation.md` | Architecture, data flows, databases |
| `assets/` | Screenshots of the deployed frontend |
| `_evidence/` | pytest / frontend build logs from the audit run |
| `generate_pbl3_report.py` | Regenerates DOCX/PDF (requires `python-docx` and `fpdf2`) |

## Regenerate DOCX/PDF

```bash
.venv/bin/python -m pip install python-docx fpdf2
.venv/bin/python reports/generate_pbl3_report.py
```

Do not treat Backtesting UI profit figures as empirical market results (see Technical Audit).
