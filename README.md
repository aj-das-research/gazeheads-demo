# Gaze-Steered Region Reports — interactive demo

Hover a region of a medical scan and read what the model reports when its
**gaze heads** are steered onto that region. The model is never told the region
in words, and never sees the radiologist's box — that box is drawn only so a
viewer can check the answer.

**Live page:** https://AJ_OWNER.github.io/AJ_REPO/

Companion demo for our study of gaze heads in medical vision–language models.
Every answer is **precomputed** on an HPC cluster; the page is static and runs
no model in your browser.

## What it shows

- **3 models** — Lingshu-7B, MedGemma-4B, Qwen3-VL-2B
- **6 steering strengths** × **25 cases**, every combination decoded ahead of time
- Per-region log-probability of the radiologist's phrase, with the region that
  actually contains their box marked
- The unsteered answer to the same question, for comparison

Raising the steering strength past 8 is worth trying: localisation stops
improving, while the generated text collapses into repetition. The value
inherited from prior work is 10,000, far beyond that point.

## Data

Scans are from **SLAKE** (Liu et al., 2021), *SLAKE: A Semantically-Labeled
Knowledge-Enhanced Dataset for Medical Visual Question Answering*, used here
with attribution. Please consult the SLAKE authors' terms for redistribution
conditions.

**No credentialed data appears here.** MS-CXR, VinDr-CXR and Chest ImaGenome are
PhysioNet collections under a Data Use Agreement; their images and phrases are
excluded by construction, and the build script refuses any non-SLAKE run.

## Rebuilding

`index.html` is generated, not hand-edited:

```bash
python tools/build_demo.py          # in the main research repository
```

## Licence

Demo code: MIT. Scan images remain under their original SLAKE terms.
