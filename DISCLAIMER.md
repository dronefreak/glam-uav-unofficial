# Disclaimer

This is an **unofficial, third-party reimplementation** of *UAV Imagery Real-Time Semantic Segmentation with Global–Local Information Attention*
(Zikang Zhang, Gongquan Li, Sensors 2025, 25(6), 1786, doi:10.3390/s25061786).

- **Not affiliated with or endorsed by the paper's authors.**
- Reconstructed from the paper text and figures. **No official code or trained weights
  were used or are provided.**
- **Numbers will not match the paper.** There are no trained checkpoints here; anything you
  measure is on random initialisation. Where the paper underspecifies the architecture,
  reasonable choices were made (see `README.md` → "What is faithful / what is a guess").
- The reported TRT-FP16 latency / parameter counts in this repo are from random-init models
  and are **indicative only**.
- If the original authors release code, **defer to it.** Issues and PRs with corrections,
  especially from the authors, are welcome. Open an issue and I will take this down or
  redirect it on request.

Reimplementation code: Apache-2.0 (see `LICENSE`). The original paper is the authors' work
(CC BY 4.0 (Sensors / MDPI, open access).).
