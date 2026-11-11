# Sleeping Minds verification pipeline (BMET3921 Team 17)

Verifies the six-stage pipeline on 18 Sleep-EDF Expanded recordings (PhysioNet).

Run:
pip install yasa mne neurokit2 scikit-learn scipy matplotlib
python3 verify_pipeline.py --n 20

Outputs: vv_report.html, vv_results.json and figures in vv_figures/.
Data downloads automatically from PhysioNet. No anxiety-labelled data is included.# sleeping-minds-verification
BMET3921 Team 17 verification pipeline for sleep-derived anxiety indicators

## Solution pipeline (run_pipeline.py)

The analysis pipeline for the clinical go/no-go study (Stages 1–5 and clinician reports).
Research software only: not a certified medical device.

Install:
pip install mne yasa neurokit2 scikit-learn scipy statsmodels matplotlib edfio joblib

Run on a labelled cohort:
python3 run_pipeline.py analyse --edf-dir data/ --clinical clinical.csv --out results/

Inputs: one EDF per person (<id>.edf), optional clinician hypnogram, and clinical.csv with
columns id, anxiety (1/0), age, sex, ahi, med_class, cognitive_status.

Demo on simulated data (no real data needed):
python3 run_pipeline.py demo

verify_pipeline.py is the verification code that tests the pipeline's methods on public
Sleep-EDF recordings (report Appendix 12).
