# Data Description

This project uses synthetic input signals generated directly in Python.

No external dataset is required.

- Input distribution: Uniform [-1, 1]
- Input generation: NumPy random generator
- Input sampling interval: 10 ms
- Reproducibility: Fixed random seeds

The same input sequence is reused across coupling strengths within each random seed to ensure fair comparisons.