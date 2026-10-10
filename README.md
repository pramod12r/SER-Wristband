# SER-Wristband
# Outdoor-noise robustness of speech emotion recognition, and wrist-only stress sensing: pilot experiments

Code and results for two small pilot experiments on public data:

- **Part A.** How much does outdoor noise degrade speech emotion recognition (SER), and how much of the loss does noise-aware training recover? (RAVDESS speech with ESC-50 outdoor noise added.)
- **Part B.** How well can stress be told from baseline using wrist-worn sensors alone (electrodermal activity, pulse-derived heart rate variability, skin temperature, motion)? (WESAD.)

Both experiments use acted speech or laboratory stress protocols. They measure effect sizes on public benchmarks. They are not field results.

## Contents

| File | Purpose |
|---|---|
| `[notebook name].ipynb` | Google Colab notebook that runs Part A and Part B end to end |
| `part_b_wesad_cell.py` | Stand-alone version of Part B (one Colab cell or a Python script) |
| `results/pilot_ser_results.csv` | Part A, eight-class results, per fold |
| `results/pilot_ser_4class.csv` | Part A, four-class results, per fold |
| `results/pilot_wesad_results.csv` | Part B summary table |
| `results/uar_vs_snr.png` | Part A figure: UAR against SNR |



## Data (not included)

Download the datasets from their original sources and check each licence before use. The notebook downloads RAVDESS and ESC-50 automatically. WESAD must be downloaded separately.

| Dataset | Used for | Source |
|---|---|---|
| RAVDESS, speech audio (1,440 utterances, 24 actors) | Part A speech | Zenodo record 1188976 |
| ESC-50 (environmental sound clips) | Part A noise | github.com/karoldvl/ESC-50 |
| WESAD (wrist signals, 15 subjects) | Part B | UCI Machine Learning Repository and the authors' page at the University of Siegen |

## How to run

1. Open the notebook in Google Colab and choose **Runtime > Change runtime type > T4 GPU**.
2. Run the Part A cells in order. Feature extraction takes roughly 30 to 60 minutes and is cached in `features.npz`.
3. For Part B, download WESAD, set `WESAD_DIR` to the folder that contains `S2`, `S3`, and so on, and run the Part B cell.

Python packages: `torch`, `transformers`, `librosa`, `soundfile`, `scikit-learn`, `scipy`, `pandas`, `matplotlib`
## Method summary

### Part A: speech under outdoor noise

- **Speech.** RAVDESS speech, 1,440 utterances, 8 emotion classes (neutral, calm, happy, sad, angry, fearful, disgust, surprised).
- **Noise.** ESC-50 clips from eight outdoor-type classes: engine, car horn, siren, wind, train, helicopter, airplane, rain. Clips are trimmed of silence, and clips shorter than 1 s are dropped. Noise from ESC-50 folds 1 to 4 is used for training and fold 5 for testing, so test noise is never seen in training (254 training clips, 64 test clips).
- **Mixing.** Noise is added at 20, 10, 5 and 0 dB SNR.
- **Features.** (a) Mean and standard deviation of 40 MFCCs. (b) Frozen `facebook/wav2vec2-base` embeddings, layer 6, mean-pooled.
- **Classifier.** Standardisation followed by logistic regression with balanced class weights.
- **Evaluation.** Actor-independent 4-fold cross-validation (`GroupKFold` by actor). Metric: unweighted average recall (UAR, the same as balanced accuracy). Two training conditions: clean speech only, and clean plus noisy speech from the training noise pool.
- **Four-class proxy.** Neutral and calm are merged into neutral; happy and surprised into positive; sad is low-arousal negative; angry, fearful and disgust are high-arousal negative.

### Part B: wrist-only stress sensing

- **Signals.** Empatica E4 wrist data from WESAD: blood volume pulse (64 Hz), electrodermal activity (4 Hz), skin temperature (4 Hz), accelerometer (32 Hz).
- **Windows.** 60 s windows with 30 s step. Baseline (label 1) against stress (label 2). A window is kept if at least 80% of its samples carry the majority label.
- **Features.** HRV from band-pass-filtered pulse peaks (mean inter-beat interval, SDNN, RMSSD); EDA mean, standard deviation, slope and maximum; skin temperature mean, standard deviation and slope; accelerometer magnitude mean and standard deviation.
- **Classifier and evaluation.** Random forest (300 trees, balanced class weights), leave-one-subject-out. 15 subjects, 883 windows, 35% stress.

## Results

### Part A: UAR (%), mean over 4 actor-independent folds

| Classes | Training | Clean | 20 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|---|
| 8 (chance 12.5%) | clean only | 73.2 | 66.3 | 52.5 | 40.9 | 27.1 |
| 8 | noise-augmented | 68.7 | 68.4 | 63.0 | 54.7 | 42.4 |
| 4 (chance 25%) | clean only | 74.0 | 70.1 | 60.6 | 47.1 | 35.2 |
| 4 | noise-augmented | 66.9 | 66.9 | 65.1 | 58.9 | 50.7 |

These rows use frozen wav2vec 2.0 features. The MFCC baseline reached 39.7% on clean eight-class speech and fell to 18.1% at 0 dB when trained on clean speech. Fold-to-fold standard deviation is up to 7.8 points for the eight-class wav2vec 2.0 results, since each fold has six actors.

### Part B: baseline versus stress, leave-one-subject-out (chance 50%)

| Wrist feature set | UAR (%) | SD (%) |
|---|---|---|
| EDA only | 72.4 | 15.2 |
| EDA + skin temperature | 86.1 | 13.8 |
| EDA + skin temperature + HRV + motion | 88.1 | 14.5 |

## Limitations

- RAVDESS is acted, studio-recorded speech, and the noise is added artificially. Accuracy on real outdoor speech will differ.
- Features are frozen, and the classifier is linear. Fine-tuning would probably raise all numbers.
- WESAD is a seated laboratory stress protocol, with no outdoor heat or movement. Wrist signals also change with exertion, posture, caffeine, illness and ambient temperature, so the Part B result shows physiological arousal, not emotional state alone.
- Results are means over a small number of folds or subjects, with large between-fold and between-subject variation.

## References

- Livingstone SR, Russo FA. The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). PLoS ONE 2018; 13(5): e0196391.
- Piczak KJ. ESC: Dataset for environmental sound classification. Proceedings of the 23rd ACM International Conference on Multimedia; 2015: 1015-1018.
- Schmidt P, Reiss A, Duerichen R, Marberger C, Van Laerhoven K. Introducing WESAD, a multimodal dataset for wearable stress and affect detection. Proceedings of the 20th ACM International Conference on Multimodal Interaction (ICMI); 2018: 400-408.
- Baevski A, Zhou H, Mohamed A, Auli M. wav2vec 2.0: a framework for self-supervised learning of speech representations. Advances in Neural Information Processing Systems 2020; 33: 12449-12460.

## Licence
The datasets keep their own licences and are not redistributed here.

## Citation and contact

[Dr. Pramod Reddy Ayiluri, Associate Professor, Department of CSE, VJIT, Hyderabad,INDIA] If you use this code, please cite the repository .
