# EmbeddingGemma 2 quantization ladder: quality, speed and size (2026-10-08)

The MLX counterpart of Unsloth's GGUF ladder for `google/embeddinggemma-2`. Unsloth's
card publishes no per-quant quality numbers; every Krill build below has measured
quality, speed and size. Recipes and `krill quantize` commands are in
[`docs/EMBEDDINGGEMMA2.md`](../EMBEDDINGGEMMA2.md) ("Quantized builds").

**Summary**

- Nine of twelve builds pass the publish gate (below): mixed-mxfp8, 8bit, 6bit, 6bit-dyn,
  5bit, 4bit-g32, 4bit-dyn, 4bit-dyn-text and nvfp4. Plain 4bit (g64), mxfp4 and 3bit do not.
- Quantization is lossless for retrieval down to 6 bit (all deltas are inside the noise of
  the test sets) and costs under 1 nDCG point at 5 bit.
- At 4 bit the *placement* of precision matters more than the format: plain 4-bit loses 3.9 SciFact
  points, while keeping the text projection, the text attention, the per-layer-input block and the last two
  layers at 8-bit (+31 MB, +7%) loses 0.6.
- Against Unsloth at equal size, text only: roughly level at 4 bit (see "Unsloth comparison"); Unsloth's
  K-quants are ahead on cosine at 5 bit and equal on retrieval. Krill's builds also carry the vision and
  audio towers, which Unsloth ships as a separate 0.55-0.98 GB `mmproj`.

## Gate (Claude's proposal, not Sourav's)

A variant is publishable if, compared with the bf16 reference:

- **8-bit class** (mixed-mxfp8, 8bit, 6bit, 6bit-dyn): SciFact and Hindi nDCG@10 each drop at most
  1.0 point, and Flickr image Recall@1 drops at most 1.0 point (the worse of the two directions).
- **4/5-bit class** (everything else): the same three drops at most 3.0 points, and text fidelity
  mean (cosine to the Google fp32 reference) at least 0.97.

A failing variant stays in the tables, marked "not published". Image-text and audio-text Recall@1 are
reported for every variant; the gate uses image only (audio has 60 clips, so one clip is 1.7 points).

## Protocol

- Every quality number is computed by Krill's release server over HTTP at fp32 compute (the default),
  one variant at a time, `task: "SearchQuery"` for queries and `"Document"` for documents. Quality does not
  depend on machine load; speed does, see below.
- **Fidelity:** cosine to Google's fp32 sentence-transformers reference on the repo fixtures
  (`Tests/KrillEngineTests/Fixtures/eg2_reference.json`: 21 strings raw and with `Document`, = 42 vectors;
  `eg2_mm/`: image 8, mixed 12, audio 6, video 4).
- **Ranking agreement:** on the same 21 strings, Spearman correlation of the pairwise-similarity matrix
  against Krill's bf16-weights fp32-compute run, and the number of strings (of 21) whose top-1 neighbour is the same.
- **SciFact:** MTEB `mteb/scifact` test, 300 queries, 5,183 documents, nDCG@10, 768 d (256 d is the
  first 256 components re-normalised). Document text is `title. text`.
- **Hindi / Kannada:** MTEB `IndicQARetrieval`, `hi` split (1,544 queries, 261 passages) and `kn` split (1,517
  queries, 257 passages), nDCG@10. The corpora are small, which compresses the range between good and
  bad models; treat differences under about 0.5 point as noise.
- **Image-text:** the first 200 images of the Flickr30k 1K test split (`nlphuji/flickr_1k_test_image_text_retrieval`) with
  their 1,000 captions. Image to text counts a hit if any of an image's 5 captions is in the top k; text to image is
  each caption against the 200 images. Captions are `SearchQuery`, images are sent without a task.
- **Audio-text:** 60 clips and captions from the Clotho v2 test split (`CLAPv2/Clotho`, two parquet batches, 127 MB;
  the 5 captions of a clip are concatenated into one text). Both directions. n = 60, so Recall@1 moves in steps of
  1.7 points: it separates working from broken (3 bit: 7 / 2), not close neighbours.
- **Sampling error:** no confidence intervals were computed. SciFact has 300 queries; a difference of 1 nDCG point is
  within the noise of a single run of this test. The gate is therefore a coarse filter, and "PASS" means "no
  measurable loss on these sets", not "equal".
- **Harness check:** Google sentence-transformers on MPS (bf16) gives SciFact 86.72 and Hindi 72.62, Krill bf16 gives
  86.92 and 72.78, so the retrieval harness is not the source of any gap.

## Hardware and versions

| | |
|---|---|
| Machine | Apple M4 Pro, 24 GB RAM, macOS 26.6.2 |
| Krill | release build of branch commit `this PR's head` (base `5b19423`, "feat(embeddings): mxfp8 and nvfp4 builds of EmbeddingGemma 2 (#333)") |
| Ollama | 0.40.0 (private `ollama serve` on 127.0.0.1:11499 with its own model directory; the user's Ollama was not touched) |
| llama.cpp | master `24e4183` built with Metal (Homebrew 0.5.0 cannot load the Unsloth GGUFs) |
| PyTorch / transformers / sentence-transformers | 2.14.1 / 5.19.0 / 6.1.0 (MPS) |
| Unsloth GGUFs | `unsloth/embeddinggemma-2-GGUF`: Q8_0, UD-Q6_K_XL, UD-Q5_K_XL, UD-Q4_K_XL (text models) |
| Datasets | `mteb/scifact`, `mteb/IndicQARetrieval` (hi, kn), `nlphuji/flickr_1k_test_image_text_retrieval`, `CLAPv2/Clotho` (2 batches) |

## The ladder (quality)

`bf16` is the unmodified `google/embeddinggemma-2` weights run through Krill (never uploaded; Google serves it).
Sizes are the `model.safetensors` file in decimal MB. "6 / 8 text" means 6-bit with the text transformer layers and
the text projection at 8-bit; the 4-bit dynamic builds are described under "Dynamic variants".

| variant | bits | model.safetensors MB | text cos min / mean | image | mixed | audio | video | Spearman raw / Doc | top-1 raw / Doc |
|---|---|---|---|---|---|---|---|---|---|
| bf16 | 16 | 1489 | 1.000 / 1.000 | 0.999 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.999 / 1.000 | 1.000 / 1.000 | 21 / 21 |
| mixed-mxfp8 | 8 (mxfp8 mixed) | 1007 | 0.994 / 0.998 | 0.992 / 0.996 | 0.996 / 0.997 | 0.996 / 0.997 | 0.996 / 0.997 | 0.998 / 0.997 | 21 / 21 |
| 8bit | 8 | 806 | 1.000 / 1.000 | 0.999 / 1.000 | 0.999 / 1.000 | 1.000 / 1.000 | 0.999 / 1.000 | 1.000 / 1.000 | 21 / 20 |
| 6bit | 6 | 624 | 0.997 / 0.999 | 0.998 / 0.999 | 0.998 / 0.999 | 0.998 / 0.999 | 0.998 / 0.998 | 0.998 / 0.998 | 21 / 20 |
| 6bit-dyn | 6 / 8 text | 657 | 0.999 / 1.000 | 0.999 / 0.999 | 0.999 / 0.999 | 0.999 / 1.000 | 0.998 / 0.999 | 1.000 / 0.999 | 21 / 20 |
| 5bit | 5 | 533 | 0.989 / 0.995 | 0.994 / 0.995 | 0.993 / 0.995 | 0.994 / 0.995 | 0.995 / 0.995 | 0.992 / 0.993 | 20 / 20 |
| 4bit | 4 (g64) | 442 | 0.968 / 0.981 | 0.974 / 0.984 | 0.976 / 0.981 | 0.982 / 0.985 | 0.978 / 0.980 | 0.976 / 0.973 | 16 / 20 |
| 4bit-g32 | 4 (g32) | 488 | 0.974 / 0.985 | 0.976 / 0.986 | 0.977 / 0.984 | 0.985 / 0.988 | 0.984 / 0.986 | 0.982 / 0.987 | 17 / 21 |
| 4bit-dyn | 4 / 8 attn+ple+proj+last2 | 473 | 0.983 / 0.993 | 0.981 / 0.992 | 0.986 / 0.992 | 0.992 / 0.994 | 0.989 / 0.991 | 0.992 / 0.986 | 19 / 19 |
| 4bit-dyn-text | 4 / 8 text layers | 507 | 0.993 / 0.998 | 0.986 / 0.995 | 0.993 / 0.996 | 0.997 / 0.998 | 0.994 / 0.995 | 0.996 / 0.998 | 20 / 21 |
| nvfp4 | 4 (nvfp4) | 442 | 0.952 / 0.981 | 0.970 / 0.983 | 0.972 / 0.982 | 0.984 / 0.985 | 0.981 / 0.981 | 0.973 / 0.968 | 20 / 18 |
| mxfp4 | 4 (mxfp4) | 419 | 0.954 / 0.973 | 0.954 / 0.973 | 0.925 / 0.962 | 0.903 / 0.944 | 0.973 / 0.976 | 0.950 / 0.963 | 18 / 19 |
| 3bit | 3 | 351 | 0.880 / 0.923 | 0.855 / 0.907 | 0.692 / 0.853 | 0.543 / 0.621 | 0.894 / 0.906 | 0.895 / 0.859 | 18 / 18 |

| variant | SciFact nDCG@10 768d | 256d | Hindi | Kannada | Flickr i2t R@1 / R@5 | Flickr t2i R@1 / R@5 | Clotho a2t R@1 / R@5 | Clotho t2a R@1 / R@5 | drop SciFact / Hindi / image R@1 | gate |
|---|---|---|---|---|---|---|---|---|---|---|
| bf16 | 86.92 | 84.44 | 72.78 | 73.25 | 99.5 / 100.0 | 96.1 / 99.7 | 25.0 / 61.7 | 30.0 / 58.3 | - | reference (not uploaded) |
| mixed-mxfp8 | 86.80 | 84.31 | 72.58 | 73.18 | 99.5 / 100.0 | 96.1 / 99.6 | 23.3 / 50.0 | 26.7 / 58.3 | 0.12 / 0.20 / 0.0 | PASS |
| 8bit | 86.95 | 84.32 | 72.81 | 73.25 | 99.5 / 100.0 | 96.1 / 99.7 | 23.3 / 58.3 | 30.0 / 55.0 | -0.03 / -0.03 / 0.0 | PASS |
| 6bit | 86.98 | 84.60 | 72.81 | 73.19 | 99.5 / 100.0 | 96.4 / 99.7 | 23.3 / 65.0 | 26.7 / 58.3 | -0.06 / -0.03 / 0.0 | PASS |
| 6bit-dyn | 86.70 | 84.08 | 72.74 | 73.37 | 99.5 / 100.0 | 96.1 / 99.7 | 23.3 / 63.3 | 26.7 / 58.3 | 0.22 / 0.04 / 0.0 | PASS |
| 5bit | 86.51 | 83.88 | 72.59 | 73.25 | 99.5 / 100.0 | 96.1 / 99.6 | 26.7 / 58.3 | 28.3 / 58.3 | 0.41 / 0.19 / 0.0 | PASS |
| 4bit | 83.03 | 81.80 | 71.42 | 72.30 | 99.0 / 100.0 | 94.8 / 99.5 | 25.0 / 60.0 | 21.7 / 60.0 | 3.89 / 1.37 / 1.3 | not published |
| 4bit-g32 | 83.93 | 82.76 | 72.04 | 72.68 | 99.5 / 100.0 | 95.1 / 99.3 | 21.7 / 55.0 | 20.0 / 60.0 | 2.99 / 0.75 / 1.0 | PASS |
| 4bit-dyn | 86.30 | 82.94 | 72.03 | 73.00 | 99.5 / 100.0 | 95.3 / 99.5 | 23.3 / 65.0 | 28.3 / 61.7 | 0.62 / 0.76 / 0.8 | PASS |
| 4bit-dyn-text | 85.32 | 83.20 | 72.38 | 73.06 | 99.5 / 100.0 | 95.7 / 99.5 | 25.0 / 66.7 | 28.3 / 58.3 | 1.60 / 0.40 / 0.4 | PASS |
| nvfp4 | 84.43 | 82.88 | 72.44 | 72.04 | 98.5 / 99.5 | 95.3 / 99.7 | 21.7 / 60.0 | 15.0 / 58.3 | 2.49 / 0.34 / 1.0 | PASS |
| mxfp4 | 82.71 | 80.66 | 71.31 | 72.30 | 99.0 / 100.0 | 95.3 / 99.1 | 21.7 / 48.3 | 16.7 / 50.0 | 4.21 / 1.47 / 0.8 | not published |
| 3bit | 75.62 | 74.23 | 70.26 | 67.12 | 97.0 / 99.0 | 90.1 / 98.2 | 6.7 / 16.7 | 1.7 / 16.7 | 11.30 / 2.52 / 6.0 | not published |


Notes:

- `mixed-mxfp8` and `nvfp4` are the existing builds (rebuilt today, same numbers as the 17-row experiment table).
- 3 bit works in the loader but is not usable (audio fidelity 0.62, Clotho Recall@1 7 / 2, SciFact -11).
- mxfp4 (group 32, shared power-of-two scale) is worse than nvfp4 (group 16, fp8 scale) everywhere and the worst on
  audio (min cosine 0.90); nvfp4 stays the 4-bit float format.
- Plain 4-bit affine (g64) is about as good as nvfp4 on cosine (text mean 0.981 both) but worse on SciFact (83.0
  vs 84.4); group 32 (+46 MB) fixes most of it and passes.

## Dynamic variants (what was protected and why)

Method: build plain 4-bit (and 6-bit) affine g64, then for each tensor class re-quantize only that class at 8-bit
(`--protect 're:...'`, regex patterns added to `krill quantize` for this) and measure the fixture fidelity (text, image,
mixed, audio, video) through the real server. Deltas are against the plain build; "gain/MB" is (delta text mean + delta
all-modality mean) per MB added. The fixture is 42 text vectors and 30 multimodal vectors, so deltas under about 0.001
are noise.

4-bit base (plain: text 0.9677 min / 0.9806 mean, 442 MB):

```
plain text mean 0.9806 min 0.9677 | all-mod mean 0.9822 worst-min 0.9677 | 442 MB
class            dTextMean  dTextMin  dAllMean dWorstMin      dMB gain/MB*1e3
text_proj          +0.0032   +0.0031   +0.0031   +0.0031      0.2       6.27
text_last2         +0.0030   +0.0019   +0.0035   +0.0019      5.8       1.13
text_ple_block     +0.0026   +0.0043   +0.0019   +0.0043      6.3       0.71
text_o_proj        +0.0026   +0.0013   +0.0021   +0.0013      7.3       0.64
text_global        +0.0034   +0.0014   +0.0043   +0.0014     12.6       0.61
text_first2        +0.0015   +0.0019   +0.0009   +0.0019      5.2       0.46
text_attn          +0.0054   +0.0060   +0.0042   +0.0060     21.0       0.46
text_all_layers    +0.0143   +0.0222   +0.0115   +0.0156     65.0       0.40
text_down_proj     +0.0021   -0.0008   +0.0023   -0.0008     12.6       0.35
text_mlp           +0.0059   -0.0032   +0.0056   -0.0032     37.7       0.30
ple_proj           +0.0006   +0.0021   +0.0002   +0.0021      3.1       0.26
patch_embedder     +0.0000   +0.0000   +0.0002   +0.0000      0.3       0.25
audio_front        +0.0000   +0.0000   +0.0001   +0.0000      1.3       0.08
vision_attn        +0.0000   +0.0000   +0.0007   +0.0000     18.9       0.04
embed_tokens       +0.0010   +0.0027   +0.0001   +0.0027     67.1       0.02
vision_mlp         +0.0000   +0.0000   +0.0006   +0.0000     56.6       0.01
audio_ff           +0.0000   +0.0000   +0.0000   +0.0000    100.7       0.00
audio_attn_conv    +0.0000   +0.0000   -0.0001   +0.0000     50.3      -0.00
mm_proj            +0.0000   +0.0000   -0.0000   +0.0000      0.6      -0.01
```

6-bit base (plain: text 0.9973 min / 0.9988 mean, 624 MB):

```
plain text mean 0.9988 min 0.9973 | all-mod mean 0.9986 worst-min 0.9973 | 624 MB
class            dTextMean  dTextMin  dAllMean dWorstMin      dMB gain/MB*1e3
text_proj          +0.0002   +0.0001   +0.0002   +0.0001      0.1       0.31
text_last2         +0.0002   +0.0001   +0.0002   +0.0001      2.9       0.14
text_ple_block     +0.0002   +0.0005   +0.0001   +0.0002      3.1       0.08
text_o_proj        +0.0001   +0.0006   +0.0002   +0.0003      3.7       0.08
text_first2        +0.0001   +0.0005   +0.0001   +0.0005      2.6       0.08
text_global        +0.0002   +0.0004   +0.0002   +0.0003      6.3       0.07
text_down_proj     +0.0002   +0.0008   +0.0002   +0.0006      6.3       0.06
text_attn          +0.0003   +0.0010   +0.0003   +0.0003     10.5       0.06
text_all_layers    +0.0008   +0.0015   +0.0007   +0.0007     32.5       0.05
text_mlp           +0.0004   +0.0011   +0.0003   +0.0006     18.9       0.03
ple_proj           +0.0000   -0.0003   +0.0000   -0.0003      1.6       0.03
vision_attn        +0.0000   +0.0000   +0.0001   +0.0000      9.4       0.01
vision_mlp         +0.0000   +0.0000   +0.0001   +0.0000     28.3       0.00
embed_tokens       +0.0001   +0.0001   +0.0000   +0.0001     33.6       0.00
audio_attn_conv    +0.0000   +0.0000   +0.0000   +0.0000     25.2       0.00
audio_ff           +0.0000   +0.0000   +0.0000   +0.0000     50.3       0.00
patch_embedder     +0.0000   +0.0000   -0.0000   +0.0000      0.1      -0.01
audio_front        +0.0000   +0.0000   -0.0000   +0.0000      0.7      -0.01
mm_proj            +0.0000   +0.0000   -0.0000   +0.0000      0.3      -0.02
```

Reading it:

- At 4 bit the **text tower is the only part that matters**. Vision, audio, the multimodal projectors and the 134M-parameter
  embedding table (30% of the file) show no measurable gain from 8-bit: they stay at 4 bit.
- The best value per MB is the text projection (0.2 MB, +0.003), then the last two layers, the per-layer-input
  (`ple_block`) tensors, `o_proj`, and the four global-attention layers. All text attention at 8-bit is +0.0054 text mean
  for 21 MB. All text layers at 8-bit is +0.014 text mean and +0.022 text min for 65 MB.
- `down_proj` and the MLP raise the mean but *lower* the text min (-0.0008, -0.0032), so they are not chosen for the
  small recipe.
- At 6 bit nothing is worth protecting: the best class gains 0.0003 for 10 MB, and the whole text stack at 8-bit gains
  0.0008 for 32 MB. `6bit-dyn` is therefore published only because it passes the gate; it is not recommended over `6bit`.

Chosen sets (all re-quantized at 8-bit affine g64, everything else at the base precision):

| Build | Protected at 8-bit | Plain build | Dynamic build | Size delta |
|---|---|---|---|---|
| `4bit-dyn` | text projection, text self-attention, text `ple_block`, last two text layers | 4bit (442 MB) | 473 MB | +31 MB |
| `4bit-dyn-text` | text projection + every text transformer layer | 4bit (442 MB) | 507 MB | +65 MB |
| `6bit-dyn` | text projection + every text transformer layer | 6bit (624 MB) | 657 MB | +33 MB |

Dynamic against the plain build of the same bits:

| build | MB | text cos min / mean | SciFact | Hindi | Kannada | Flickr t2i R@1 | verdict |
|---|---|---|---|---|---|---|---|
| 4bit | 442 | 0.968 / 0.981 | 83.03 | 71.42 | 72.30 | 94.8 | not published |
| 4bit-dyn | 473 | 0.983 / 0.993 | 86.30 | 72.03 | 73.00 | 95.3 | PASS |
| 4bit-dyn-text | 507 | 0.993 / 0.998 | 85.32 | 72.38 | 73.06 | 95.7 | PASS |
| 6bit | 624 | 0.997 / 0.999 | 86.98 | 72.81 | 73.19 | 96.4 | PASS |
| 6bit-dyn | 657 | 0.999 / 1.000 | 86.70 | 72.74 | 73.37 | 96.1 | PASS |

## Unsloth comparison (text quality at equal size)

Unsloth's GGUFs hold the text tower only (176-310 MB; the vision and audio towers are in separate `mmproj` files of
0.55-0.98 GB). To compare at equal size, text-only copies of the Krill builds were made by dropping the vision and audio
tensors from `model.safetensors` (the loader binds the text tensors strictly and loads the towers lazily, so the result
loads and serves text). These are comparison copies and are not uploaded.

**Engine note.** Ollama 0.40.0's bundled llama.cpp cannot import the Unsloth GGUFs (`unknown model architecture:
'gemma-embedding2'`; Ollama's own `embeddinggemma-2` uses its own engine), and Homebrew's llama.cpp 0.5.0 (build 11146) has
the same error. They were run with `llama-server --embedding` built from llama.cpp master (commit 24e4183), Metal, with
the literal prompt prefixes from `config_sentence_transformers.json` applied by the client. Fidelity of Unsloth Q8_0 to the
Google reference is 0.9999, so the engine and prefixes are right.

| contender | text-only MB | text cos min / mean | Spearman raw / Doc | top-1 raw / Doc | SciFact nDCG@10 | Hindi | Kannada |
|---|---|---|---|---|---|---|---|
| Krill nvfp4 (text-only) | 153 | 0.9523 / 0.9809 | 0.973 / 0.968 | 20 / 18 | 84.43 | 72.44 | 72.04 |
| Krill 4bit (text-only) | 153 | 0.9677 / 0.9806 | 0.976 / 0.973 | 16 / 20 | 83.03 | 71.42 | 72.30 |
| Krill 4bit-g32 (text-only) | 170 | 0.9736 / 0.9849 | 0.982 / 0.987 | 17 / 21 | 83.93 | 72.04 | 72.68 |
| Unsloth UD-Q4_K_XL | 176 | 0.9899 / 0.9943 | 0.992 / 0.991 | 20 / 20 | 85.42 | 72.28 | 73.15 |
| Krill 4bit-dyn (text-only) | 183 | 0.9825 / 0.9927 | 0.992 / 0.986 | 19 / 19 | 86.30 | 72.03 | 73.00 |
| Krill 5bit (text-only) | 187 | 0.9887 / 0.9953 | 0.992 / 0.993 | 20 / 20 | 86.51 | 72.59 | 73.25 |
| Unsloth UD-Q5_K_XL | 210 | 0.9966 / 0.9985 | 0.997 / 0.997 | 20 / 21 | 86.92 | 72.43 | 73.39 |
| Krill 4bit-dyn-text (text-only) | 218 | 0.9928 / 0.9980 | 0.996 / 0.998 | 20 / 21 | 85.32 | 72.38 | 73.06 |
| Krill 6bit (text-only) | 220 | 0.9973 / 0.9988 | 0.998 / 0.998 | 21 / 20 | 86.98 | 72.81 | 73.19 |
| Unsloth UD-Q6_K_XL | 249 | 0.9984 / 0.9994 | 0.999 / 0.999 | 20 / 20 | 86.86 | 72.82 | 73.30 |
| Krill 8bit (text-only) | 288 | 0.9997 / 0.9999 | 1.000 / 1.000 | 21 / 20 | 86.95 | 72.81 | 73.25 |
| Unsloth Q8_0 | 310 | 0.9999 / 0.9999 | 1.000 / 1.000 | 21 / 21 | 86.80 | 72.76 | 73.18 |

Other contenders on the same text tests (fidelity here is cosine to the Google fp32 reference):

| contender | file MB (text only?) | text min / mean | Spearman raw/Doc | top-1 raw/Doc | SciFact | Hindi | Kannada |
|---|---|---|---|---|---|---|---|
| Google ST (MPS bf16) | 1489 | 0.9999 / 1.0000 | 1.000 / 1.000 | 21 / 21 | 86.72 | 72.62 | - |
| Ollama embeddinggemma-2 (nvfp4) | 1300 | 0.9786 / 0.9885 | 0.982 / 0.983 | 20 / 16 | 84.15 | 72.08 | 72.40 |
| Unsloth Q8_0 | 310 | 0.9999 / 0.9999 | 1.000 / 1.000 | 21 / 21 | 86.80 | 72.76 | 73.18 |
| Unsloth UD-Q6_K_XL | 249 | 0.9984 / 0.9994 | 0.999 / 0.999 | 20 / 20 | 86.86 | 72.82 | 73.30 |
| Unsloth UD-Q5_K_XL | 210 | 0.9966 / 0.9985 | 0.997 / 0.997 | 20 / 21 | 86.92 | 72.43 | 73.39 |
| Unsloth UD-Q4_K_XL | 176 | 0.9899 / 0.9943 | 0.992 / 0.991 | 20 / 20 | 85.42 | 72.28 | 73.15 |

Reading it:

- Unsloth's mixed K-quants are strong: UD-Q4_K_XL (176 MB) has text cosine 0.9943 mean, ahead of the plain Krill 4-bit
  builds. The dynamic Krill 4-bit (183 MB) reaches 0.9927 and a higher SciFact (86.30 vs 85.42); Hindi and Kannada are level within noise (72.03 / 73.00 vs 72.28 / 73.15).
- At 5 bit Unsloth is better on cosine at the same size class (UD-Q5_K_XL 0.9985 at 210 MB vs Krill 5bit 0.9953 at 187 MB; Krill `4bit-dyn-text` at 218 MB, 0.9980, is level with it); retrieval is equal within noise (86.92 vs 86.51 and 85.32 SciFact).
- Ollama's own `embeddinggemma-2` is nvfp4 (1.3 GB with the towers); its text quality (cosine 0.9885, SciFact 84.15) matches
  Krill's nvfp4 (0.9809, 84.43).

## Speed and memory

Protocol (quiet machine): before each measured run the harness requires 1-minute load < 3.0, `memory_pressure` >= 50%
free, swap used < 1 GB, no `pmset -g therm` limit and AC power, and otherwise waits (20 s poll). If the machine is still
not quiet after 2 hours of waiting it measures anyway and flags the numbers. One contender at a time; the server is
started and stopped for every repetition (so cold start is measured 3 times); 10 discarded warm-up requests; 3
repetitions per metric, reporting the median of the three with the min-max range in brackets.

- single query: 50 requests of one 11-word query, p50 / p95 in ms.
- batch of 32 at ~256 tokens (mean 277) and ~1k tokens (mean 1,086): documents per second, median of 5 / 4 timed batches
  after 2 warm-up batches; inputs are windows of the SciFact corpus with the `Document` prefix.
- one 640x480 PNG image (Krill only; Ollama's and llama.cpp's text paths do not take images here): p50 over 12 requests.
- cold first request: process launch (or model load, for in-process ST) to the first completed embedding.
- peak memory: Krill `KRILL_EMBED_LOG_MEM` MLX peak; sentence-transformers `torch.mps.driver_allocated_memory()`;
  Ollama and llama-server resident set of the runner process (`ps rss`; `ollama ps` output is in the raw JSON).
- Krill rows use fp32 compute (the default) unless marked `bf16` (bfloat16 compute); the Krill binary is the release build of this
  branch; Google sentence-transformers runs in process on MPS in bf16.

**Every speed run is flagged as NOT quiet.** The machine never met the protocol: swap used stayed at 5.9-8.3 GB (other
processes; it did not fall below 1 GB during a 2 hour wait that started 12:42, after which the harness measured anyway, as
specified), the free-memory reading was 32-89% (below the 50% threshold during most Krill runs), and the 1-minute load
peaked at 2.6 for the Krill and Ollama runs and at 3.1-8.5 for the llama.cpp runs (another session's work on the
machine). AC power and no thermal limit held throughout. The three repetitions of a contender were taken minutes apart, and the ranges
show the effect: most single-query medians are tight (within 0.3 ms) but some rows carry an outlier repetition
(for example `6bit` single query 9.2 (9.2-12.9), `unsloth-UD-Q5_K_XL` 16.3 (11.5-17.2)). Read differences under about 10% as noise,
and read the ordering between engine families (Krill ~9-10 ms, llama.cpp ~17 ms, Ollama and
sentence-transformers ~25 ms) as real. The per-run gate readings are in the scratch logs (`bench-logs/gate-*.log`).
Memory: Krill is the MLX peak over the whole run (single queries, both batch sizes and the image); `bf16` rows differ by compute dtype
(fp32 compute doubles activations); sentence-transformers is the MPS driver allocation after the run; Ollama and
llama.cpp are the resident set of the server process.

| contender | weights MB | single p50 ms | single p95 ms | batch-32 ~256 tok, docs/s | batch-32 ~1k tok, docs/s | one image p50 ms | cold first request ms | peak memory MB | quiet machine |
|---|---|---|---|---|---|---|---|---|---|
| krill-bf16-fp32 | 1488.9 | 10.6 (10.4-10.6) | 14.7 (14.5-15.5) | 38 (33-39) | 9 (8-9) | 368 (362-375) | 1115 (994-1358) | 3234 (3234-3234) | NO - flagged |
| krill-bf16-bf16 | 1488.9 | 9.2 (9.2-9.3) | 9.4 (9.4-9.6) | 47 (46-47) | 10 (10-10) | 367 (365-377) | 1078 (1073-1788) | 1934 (1934-1934) | NO - flagged |
| krill-mixed-mxfp8-fp32 | 1007.4 | 10.4 (10.4-10.5) | 11.6 (10.6-13.1) | 38 (38-39) | 8 (8-8) | 393 (391-394) | 1030 (1007-1047) | 2835 (2835-2835) | NO - flagged |
| krill-8bit-fp32 | 806.2 | 9.1 (8.8-9.1) | 13.2 (13.2-13.2) | 35 (33-36) | 7 (7-7) | 408 (405-408) | 1079 (1054-1085) | 2498 (2498-2498) | NO - flagged |
| krill-6bit-fp32 | 624.2 | 9.2 (9.2-12.9) | 13.1 (12.8-15.5) | 34 (34-34) | 7 (7-7) | 409 (409-409) | 999 (988-1038) | 2433 (2433-2433) | NO - flagged |
| krill-6bit-dyn-fp32 | 656.8 | 9.1 (9.1-9.1) | 13.2 (13.0-13.3) | 35 (34-35) | 7 (7-7) | 412 (407-414) | 1016 (984-1040) | 2464 (2464-2464) | NO - flagged |
| krill-5bit-fp32 | 533.1 | 8.9 (8.9-9.1) | 12.4 (12.1-13.2) | 34 (32-35) | 7 (6-7) | 417 (415-530) | 1063 (1018-1077) | 2401 (2401-2401) | NO - flagged |
| krill-4bit-g32-fp32 | 487.6 | 8.8 (8.6-9.1) | 12.4 (12.3-12.5) | 24 (22-30) | 6 (5-7) | 459 (395-573) | 1073 (1037-1073) | 2401 (2401-2401) | NO - flagged |
| krill-4bit-dyn-fp32 | 472.7 | 8.9 (8.9-9.1) | 12.9 (11.9-13.0) | 34 (33-35) | 7 (6-7) | 407 (396-465) | 1044 (1035-1077) | 2398 (2398-2398) | NO - flagged |
| krill-4bit-dyn-text-fp32 | 507.3 | 9.0 (9.0-9.1) | 12.7 (12.6-13.0) | 26 (22-27) | 5 (5-6) | 494 (426-548) | 1082 (1065-1083) | 2431 (2431-2431) | NO - flagged |
| krill-nvfp4-fp32 | 442.0 | 8.6 (8.6-8.7) | 9.0 (8.9-9.4) | 34 (32-36) | 7 (7-7) | 403 (403-429) | 1058 (1057-1069) | 2353 (2353-2353) | NO - flagged |
| st-mps-bf16 | 1488.9 | 26.5 (26.0-27.0) | 28.9 (28.7-29.7) | 25 (22-26) | 5 (5-5) | - | 2912 (2348-3699) | 16117 (14483-18113) | NO - flagged |
| ollama-own | 1300 | 25.5 (25.5-28.1) | 37.8 (36.9-37.9) | 31 (31-32) | 8 (8-8) | - | 766 (757-1487) | 1470 (1461-1481) | NO - flagged |
| unsloth-Q8_0 | 309.9 | 16.8 (16.2-16.9) | 18.3 (18.3-18.8) | 33 (33-33) | 8 (7-8) | - | 877 (872-883) | 1824 (1819-1828) | NO - flagged |
| unsloth-UD-Q6_K_XL | 248.8 | 17.1 (16.7-17.3) | 18.3 (18.3-18.4) | 32 (32-32) | 7 (7-7) | - | 869 (866-880) | 1767 (1764-1768) | NO - flagged |
| unsloth-UD-Q5_K_XL | 210.1 | 16.3 (11.5-17.2) | 18.0 (13.3-18.3) | 31 (30-31) | 7 (7-7) | - | 880 (651-882) | 1730 (1728-1731) | NO - flagged |
| unsloth-UD-Q4_K_XL | 175.7 | 17.1 (16.4-17.3) | 18.2 (18.0-18.7) | 32 (32-32) | 7 (7-7) | - | 875 (651-879) | 1697 (1697-1697) | NO - flagged |

## Caveats

- Test sets are small (see "Sampling error"); the gate separates broken from fine, not neighbours.
- Hindi and Kannada are one passage-retrieval task each (IndicQARetrieval) with a 260-passage corpus; they say little about
  long-document retrieval in those languages. No Sanskrit retrieval set was available.
- Audio-text uses 60 Clotho clips with 5 captions concatenated; Recall@1 around 25% is a property of that protocol (hard,
  concatenated captions), not of the model.
- Fidelity fixtures contain 42 text and 30 multimodal vectors; per-tensor sensitivity deltas below 0.001 are noise.
- The Krill bf16 reference runs with fp32 compute; bf16 compute rows are listed separately in the speed table.
- Unsloth and Ollama are text only here: image and audio were not measured on them.
- The 256-dimension numbers are the first 256 components re-normalised (Matryoshka); they are reported for all variants.
- The speed numbers were measured with the benchmark binary built from this branch; the serving path is unchanged from
  `main` (5b19423): this PR only changes `krill quantize`, the aliases, and docs.

## Commands

```
# build the Krill release binary
make release

# every variant (bf16 source folder = google/embeddinggemma-2, --dtype bf16)
krill quantize <bf16-dir> --bits 8 --group-size 64 --dtype bf16 --output-dir 8bit
krill quantize <bf16-dir> --bits 6 --group-size 64 --dtype bf16 --output-dir 6bit
krill quantize <bf16-dir> --bits 5 --group-size 64 --dtype bf16 --output-dir 5bit
krill quantize <bf16-dir> --bits 4 --group-size 64 --dtype bf16 --output-dir 4bit
krill quantize <bf16-dir> --bits 4 --group-size 32 --dtype bf16 --output-dir 4bit-g32
krill quantize <bf16-dir> --bits 3 --group-size 64 --dtype bf16 --output-dir 3bit
krill quantize <bf16-dir> --mode mxfp4 --dtype bf16 --output-dir mxfp4
krill quantize <bf16-dir> --mode nvfp4 --dtype bf16 --output-dir nvfp4
krill quantize <bf16-dir> --mode mxfp8 --dtype bf16 --skip language_model.layers --skip self_attn --skip patch_embedder --output-dir mixed-mxfp8
# dynamic (see EMBEDDINGGEMMA2.md for the patterns)
krill quantize <bf16-dir> --bits 4 --group-size 64 --dtype bf16 --protect-bits 8 --protect-group-size 64 \
  --protect 're:^language_model\.embedding_projection$' --protect 're:^language_model\.layers\.\d+\.self_attn\.' \
  --protect 're:^language_model\.layers\.\d+\.ple_block\.' --protect 're:^language_model\.layers\.2[23]\.' --output-dir 4bit-dyn

# quality (per variant, release server, one at a time)
python scripts/evalq.py <variant> float32
# speed
python scripts/bench_speed.py krill:<variant>:float32 3
```

The harness scripts (`evalq.py`, `lib2.py`, `bench_speed.py`, `aggregate.py`, ...) live in the scratch directory of the run, not in
the repository; they use only the Krill HTTP API, sentence-transformers, `llama-server` and the Ollama HTTP API as described above.
