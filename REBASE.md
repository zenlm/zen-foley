# Commercial Rebase — Zen Foley

## Current base (RESTRICTED)
Tencent HunyuanVideo-Foley, under the TENCENT HUNYUAN COMMUNITY LICENSE
(commercial use capped at 100M MAU; territorial/use restrictions). Not
unconditionally commercial.

## Target base (commercial-clean)
**MMAudio** (https://github.com/hkchengrex/MMAudio) — MIT-licensed code, permits
unconditional commercial use. NOTE: MMAudio's released weights were trained on
VGGSound/AudioSet/etc.; for a fully clean product, retrain on commercially
cleared foley/audio data.

## Steps
1. Replace `hunyuanvideo_foley/` stack with the MMAudio (MIT) architecture.
2. Retrain / fine-tune on commercially-licensed audio.
3. Publish weights to HF `zenlm/zen-foley` under Apache-2.0/MIT.
4. Swap LICENSE -> Apache-2.0 + MMAudio (MIT) attribution in NOTICE.
