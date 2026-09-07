# Gaze-Steered Region Reports — interactive demo

Hover a region of a medical scan and read what the model reports when its
**gaze heads** are steered onto that region. The model is never told the region
in words, and never sees the radiologist's box — that box is drawn only so a
viewer can check the answer.

**Two versions:**

| | | |
|---|---|---|
| **Live** | https://aj-das-research.github.io/gazeheads-demo/live/ | Qwen3-VL-2B runs *in your browser* on WebGPU. Your cursor steers its gaze heads while it writes. Needs recent Chrome/Edge on a desktop; downloads ~1.2 GB once. |
| **Static** | https://aj-das-research.github.io/gazeheads-demo/ | Precomputed hover grid, 3 models × 6 steering strengths. Works anywhere. |

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

## How the live version works

The language model decodes on your GPU through transformers.js. The ONNX build
carries two extra inputs — a head mask and a per-position sign — whose product is
added to the attention scores of the selected heads before softmax. Steering is
+δ on image tokens under the cursor's spotlight and −δ on the rest; nothing is
retrained. The image's vision features are precomputed (`live/cases/*.embeds.bin`)
by `tools/build_webdemo.py` in the research repository; the browser never runs a
vision encoder.

The browser model omits Qwen3-VL's DeepStack injections (the export has no
inputs for them), so it is slightly weaker than the same model on a cluster.

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

## Credits and licence

- **Model weights (live demo):** `baulab/Qwen3-VL-2B-Instruct-GazeHeads-AllLayers-ONNX`,
  Apache-2.0, published by Rohit Gandikota and David Bau alongside *Gaze Heads:
  How VLMs Look at What They Describe* (arXiv:2606.14703). Their work on comic
  strips is what this study extends to medical imaging; the browser-steering
  design — steering inputs baked into the ONNX graph, vision features supplied
  precomputed — follows their demo. The page code here is our own.
- **Scans:** SLAKE (Liu et al., 2021), under the SLAKE authors' terms.
- **Demo code:** MIT.
