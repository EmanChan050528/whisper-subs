# Roadmap

Status: active

## Now
- [ ] Publish a zip as a GitHub release ([4.2](docs/DEVELOPMENT.md#42-packaging-packagingwhisper-subsspec))
  - Under the 2 GB asset limit
- [ ] Alignment check in the eval, so a line-shift regression shows up ([3.1](docs/DEVELOPMENT.md#31-evaluation-harness-build-this-before-tuning-anything))

## Next
- [ ] Test on a stream archive with a long BGM-only waiting screen ([3.4](docs/DEVELOPMENT.md#34-hallucination-guards-filterspy))
- [ ] Glossary: infer the name from the parent folder ([3.5](docs/DEVELOPMENT.md#35-glossary-per-channel-or-series-glossarypy))
- [ ] Glossary: see and edit one from the window, without finding the file
- [ ] Reuse jp-subs' human-reference English in the eval
- [ ] Overlapping speech: record as a known limitation, measure on the two-speaker sample ([3.7](docs/DEVELOPMENT.md#37-overlapping-speech-keep-this-scope-small))

## Later
- [ ] Diarization with pyannote (stretch; needs a Hugging Face token)
- [ ] Share `prompt` and `segment` with jp-subs from one source
- [ ] Korean, reusing jp-subs' readiness probe
- [ ] `.vtt` and `.ass` output
- [ ] Burn in subtitles with ffmpeg
