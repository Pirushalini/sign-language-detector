# Notice about third-party data

This project does not contain or redistribute any trained model.

For the reference templates it can read a public hand-landmark dataset:

- Dataset: "BSL Numbers and Alphabet Hand Position for MediaPipe" on Kaggle, by erentatepe.
- It comes from the GitHub project `bsl_numbers_and_alphabet_recognition` by datMaul, a final-year
  project at Brunel University.
- The Kaggle license is listed as "Unknown" and the GitHub repository has no license file.

Because of this:

1. The two CSV files are used for local study only. They are kept in `data/external/` and
   `.gitignore` excludes that folder and the generated `data/templates_external.npz`.
   Do not commit or publish them.
2. No code from that repository was copied. Only the CSV landmark data is read, and the feature
   preparation, matcher, calibration tool and interface in this project were written from scratch.
3. The pickled `.pkl` models in that dataset are deliberately not loaded. Unpickling files from the
   internet can run arbitrary code, and loading a trained model would also go against the
   "no model training" rule of this project.
4. If you publish your work, cite the dataset and its authors, and contact them about permission.
   A safer option is to publish only the code and ask readers to download the data themselves.

The alphabet chart in `assets/alphabet_chart.png` is the "British Sign Language Fingerspelling Chart" from
british-sign.co.uk (marked "Free to print", copyright british-sign.co.uk). It is used here only as an
on-screen reference for personal study. If you publish this project, check their terms first, or replace
the image with your own chart. The calibration highlight boxes assume a 7 x 4 grid; the position is set by
`CHART_BOX` in `utils.py`.
