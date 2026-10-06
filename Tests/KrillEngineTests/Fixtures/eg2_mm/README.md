# EmbeddingGemma 2 multimodal fixtures (Milestone 2)

Reference vectors for `google/embeddinggemma-2` from **sentence-transformers 6.1.0 +
transformers 5.19.0, fp32, CPU** (torch 2.14), one input per `encode` call (batch size 1,
so no padding is involved), `normalize_embeddings=True`. Everything here is synthetic
(generated, seeded); nothing is copied from the web.

Repo copy: `Tests/KrillEngineTests/Fixtures/eg2_mm/` (about 2.3 MB). Scratch (same files, plus the
venv, scripts and debug dumps): `/private/tmp/claude-501/-Users-sourav/50cbce9a-3e53-4e5c-8d19-610463c2c451/scratchpad/eg2-m2/`.
Weights (reuse, do not delete): `.../scratchpad/eg2-weights`.

## Layout

```
eg2_mm/
  images/   8 files: 6 aspect ratios + RGBA (partial alpha) + grayscale
  audio/    a3.wav (2.1 s), a8.wav (5.8 s), a20.wav (23.3 s)   16 kHz mono PCM16, macOS `say`
  video/    v3.mp4 (3 s), v40.mp4 (40 s)   320x240, 10 fps, h264, no audio, ffmpeg lavfi
  reference_image.json   reference_mixed.json   reference_audio.json   reference_video.json
```

Each `reference_*.json` is `{"model","dtype","device","sentence_transformers","transformers","cases":[...]}`
and every case has:

| key | meaning |
|---|---|
| `name`, `modality` | `image` / `mixed` / `audio` / `video` |
| `parts` | ordered `{"type":"text","text"}` / `{"type":"image"\|"audio"\|"video","file"}` (files are relative to the matching subfolder) |
| `task` | the sentence-transformers prompt name used (`Document`, `SearchQuery`) or null |
| `input_ids_rle` | the **exact token ids the model saw** (from `model.preprocess`), run-length coded as `[id, count]` pairs (placeholder runs are long) |
| `n_tokens` | `len(input_ids)` (== `usage.prompt_tokens` Krill must report) |
| `embedding` | 768 floats, L2-normalised, 8 significant digits |
| image only: `grid` | `{patches, pW, pH, soft_tokens}` of the resized patch grid |
| audio: `samples`, `audio_soft_tokens` | clip length in samples, number of `<audio>` placeholders |
| video: `video_soft_tokens_total` | total `<|video|>` placeholders over all frames |

## How each reference was made (`reference.py <image|mixed|audio|video> out.json`)

```python
m = SentenceTransformer(W, model_kwargs={"torch_dtype": torch.float32}, device="cpu")
# image : m.encode(PIL.Image.open(f), normalize_embeddings=True)                         (ST modality "image")
# mixed : m.encode([{"role":"user","content":[{"type":"text","text":..},{"type":"image","image":PIL}, ..]}],
#                  prompt_name=task, normalize_embeddings=True)      (ST modality "message"; the prompt is prepended
#                  as a system message, i.e. as a string prefix; touching text parts are one string)
# audio : m.encode({"array": float32 mono 16 kHz, "sampling_rate": 16000}, normalize_embeddings=True)
# video : m.encode("<path>.mp4", normalize_embeddings=True)        # needs `torchcodec`; processor defaults:
#                  fps=1, max_frames=32 (uniform), 140 soft tokens/frame budget, no timestamps
# ids   : m.preprocess([same input], prompt=m.prompts[task])["input_ids"]
```

Media generation: `gen_media.py` (images, PIL + numpy, seeded) and `gen_av.sh` (`say` + ffmpeg).
Mixed cases: `text_then_image`, `image_then_text`, `text_image_text`, `two_images_caption`,
`document_task_text_image` (task `Document`), `searchquery_task_image_text` (task `SearchQuery`).

## Facts the audio / video work should know (all measured with these files)

* **Audio is NOT capped at 280 soft tokens** in the sentence-transformers path: a 23.3 s clip produced
  583 `<audio>` placeholders (25 tokens per second; `audio_seq_length: 280` is only applied by
  `_compute_audio_num_tokens`, a helper for serving frameworks, not by `replace_audio_token`). a3.wav -> 52, a8.wav -> 146.
  Only the 8,192-token context limits it. Block layout `<boa> <audio>xN <eoa>`.
* **Video**: 3 s -> 3 frames, 40 s -> 32 frames (uniform down-sample of 40 at 1 fps); every frame of these
  320x240 clips is 130 soft tokens (<= the 140 budget). v3: 390 soft / 398 total ids; v40: 4160 soft / 4226 total.
  Layout per frame `<boi> <|video|>xN <eoi>`, concatenated, then mean pooled with everything else.
* The image fixtures already pin: `<bos>` first, `<eos>` last, `<boi> <image>xN <eoi>` blocks, no separator
  tokens between parts, task prefix tokenised jointly with the first text part.
* JPEG decoding differs between Apple ImageIO and libjpeg (chroma upsampling): about 0.2-0.4/255 mean abs,
  up to 40-66 levels on 3-6% of pixels at chroma edges. That costs ~0.0002-0.0006 cosine on the JPEG
  fixtures; PNG fixtures match to 0.99998-1.0000.

## Milestone 2b additions (audio / video decoders, mixed audio / video)

Extra media (synthetic, ffmpeg): `audio/a3_44k_stereo.flac` (a3 at 44.1 kHz stereo, tests resampling + downmix),
`audio/a8.mp3` (LAME tag: delay 576, padding 956), `audio/a8.m4a` (AAC), `audio/a8_m4a_ffmpeg16k.wav` (the m4a
decoded by ffmpeg, 94,208 samples: proves the m4a token-count gap is the decoder), `video/v_ntsc.mp4`
(480x270, 30000/1001 fps, 4.004 s), `video/v_audio.mp4` (256x192, 12 fps, WITH an audio track; the reference ignores it).

| file | content |
|---|---|
| `reference_audio_extra.json` | flac / mp3 / m4a. Non-16 kHz-PCM files were decoded + resampled to 16 kHz mono float32 by ffmpeg (swr) and passed to ST as arrays: that is the "reference decoder" |
| `reference_video_extra.json` | v_ntsc, v_audio (paths, torchcodec) |
| `reference_mixed_av.json` | `text_then_audio`, `audio_then_text_searchquery`, `document_task_text_audio`, `text_video_text`, `document_task_video_text`, `text_audio_image` (ST "message" modality: audio -> `<|audio|>`, video -> `<|video|>`) |
| `reference_video_frames.json` | 12 reference frames (torchcodec, exact seek) of 4 videos as 12x16 block means: pins the decoder's pixels without shipping frames |

Facts measured with these files: audio is NOT capped at 280 tokens (see above); an MP3's LAME/Xing delay + padding
are trimmed by ffmpeg but not by AVAudioFile; ffmpeg keeps the AAC priming (94,208 samples) where AVFoundation applies
the edit list (93,520); a video's frame indices come from `int(i * fps)` with `num_sampled = max(1, int(duration))`
and a `np.linspace(.., dtype=int)` cut to 32; AVFoundation's BGRA frames differ from torchcodec's by 2.9/255 mean
(chroma upsampling) while its 4:2:0 planes converted with nearest chroma + BT.601 differ by 0.19/255 (max 1).
