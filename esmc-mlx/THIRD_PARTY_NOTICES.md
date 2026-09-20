# Third-party attribution and source provenance

This package provides Apple inference for fine-tuned DecoderTCR checkpoints,
adapting the ESM-C encoder and checkpoint layout from the sources below.
Frameworks and weights are installed separately. Converted weights retain their
source identity and license snapshots. Custom checkpoints require a user-supplied
model ID and are labeled unvalidated, without public-release attribution.

| Material | Exact source | Treatment |
|---|---|---|
| ESM-C MLX encoder architecture | MLX-LM PR #1484 head `c26b9af872158d822a8c95589708eedd3b9c0831`, `mlx_lm/models/esmc.py` | Adapted minimal encoder operations, names and architecture; MIT Apple notice retained in `licenses/`. |
| Published checkpoint split/fusion mapping | Biohub/esm `43b4548b86762edfa747b07d5f440aad3c33acee`, `esm/models/esmc/checkpoint_layout.py` | Adapted explicit name tables and ordered concatenation in `weights.py`; stricter schema, rejection, hashing and atomic bundle publication added. Biohub MIT notice retained. |
| Official tokenizer vocabulary and policy | Biohub/ESMC-300M `e4bac860f0c502cd80f3aeac3d0ce6524c6627cb`, `tokenizer.json` | Vocabulary and behavior implemented locally and tested against the pinned serialized tokenizer; no Transformers tokenizer implementation copied. |
| DecoderTCR architecture, tensor names and alphabet | Biohub/DecoderTCR `3e3d9889d26d79635f674940ddab039c4dc6f9f9`, `src/esmc/`, `src/esm/data.py`, `src/DecoderTCR/constants.py` | Separate strict checkpoint adapter and ESM-1b vocabulary compatibility. Biohub and bundled backbone MIT notices retained. |
| ESM-1b alphabet vocabulary | Meta ESM, bundled by DecoderTCR; reference license from facebookresearch/esm `2b369911bb5b4b0dda914521b9475cad1656b2ac` | MIT Meta/Facebook notice retained. Tokenization code independently implemented rather than copying the historical Transformers-derived splitter. |
| Historical backbone reference weights | Biohub/ESMC-300M `e4bac860f0c502cd80f3aeac3d0ce6524c6627cb` | Used for port-development comparisons, not a DecoderTCR workflow choice. Model card declares MIT and `other`, linking Biohub third-party notices; exact model card, metadata, Biohub license and dependency notices retained. |
| DecoderTCR 300M weights | Biohub/DecoderTCR `803fed3bbf3dcc40ed481f7d50a20f668db0ef01`, `DecoderTCR-ESMC-V0.3/300M.ckpt` | Local-only conversion. Release MIT metadata and Biohub license retained. |

Optional reference-comparison environments have separate dependencies and
upstream notices. Model weights are not bundled or relicensed.
