# Sleeping Minds verification pipeline (BMET3921 Team 17)

Verifies the six-stage pipeline on 18 Sleep-EDF Expanded recordings (PhysioNet).

Run:
pip install yasa mne neurokit2 scikit-learn scipy matplotlib
python3 verify_pipeline.py --n 20

Outputs: vv_report.html, vv_results.json and figures in vv_figures/.
Data downloads automatically from PhysioNet. No anxiety-labelled data is included.# sleeping-minds-verification
BMET3921 Team 17 verification pipeline for sleep-derived anxiety indicators
