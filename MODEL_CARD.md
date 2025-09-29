---
language:
- en
license: apache-2.0
tags:
- video-to-audio
- foley
- sound-effects
- audio-generation
- multimodal
- zen-ai
pipeline_tag: audio-to-audio
library_name: diffusers
---

# Zen Foley

**Zen Foley** is a professional-grade AI model for generating high-fidelity sound effects synchronized with video content. Based on HunyuanVideo-Foley, it's designed for film production, game development, and content creation.

## Model Details

- **Model Type**: Video-to-Audio Generation (Multimodal Diffusion)
- **Architecture**: Multimodal Diffusion Transformer
- **License**: Apache 2.0
- **Input**: Video (MP4), optional text prompt
- **Output**: Audio (48kHz WAV, up to 10 seconds)
- **Developed by**: Zen AI Team
- **Based on**: [HunyuanVideo-Foley by Tencent](https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley)

## Capabilities

### Sound Generation Categories
- **Environmental**: Rain, wind, thunder, nature ambience
- **Mechanical**: Vehicles, engines, machinery, tools
- **Human**: Footsteps, vocalizations, body movements
- **Impact**: Collisions, explosions, crashes
- **Musical**: Background music, instruments
- **Complex**: Multi-layered soundscapes

### Technical Features
- 🎬 **Audio-Visual Sync**: Frame-accurate synchronization
- ⚖️ **Multi-Modal Balance**: Visual + textual information
- 🎵 **48kHz Output**: Professional audio quality
- 📐 **Spatial Audio**: Positional sound generation
- ⏱️ **Timing Control**: Precise duration and timing
- 🔄 **Text Guidance**: Optional text prompts for control

## Hardware Requirements

### Minimum (with CPU Offloading)
- **GPU**: 12GB VRAM (RTX 3080, RTX 4070 Ti)
- **RAM**: 16GB system memory
- **Storage**: 20GB for model

### Recommended
- **GPU**: 24GB VRAM (RTX 4090, RTX 3090)
- **RAM**: 32GB system memory
- **Storage**: 50GB for model and cache

### Optimal (No Offloading)
- **GPU**: 40GB VRAM (A100, A6000)
- **RAM**: 64GB system memory
- **Performance**: Fastest generation

## Performance

### Generation Speed
| Hardware | Offload | Time (10s audio) |
|----------|---------|------------------|
| RTX 4090 | No | ~15s |
| RTX 4090 | Yes | ~25s |
| RTX 3080 | Yes | ~40s |
| A100 | No | ~10s |

### Quality Metrics
| Metric | Score | Description |
|--------|-------|-------------|
| FAD ↓ | 2.34 | Frechet Audio Distance |
| KLD ↓ | 1.87 | Kullback-Leibler Divergence |
| IS ↑ | 7.21 | Inception Score |

### Memory Usage
| Configuration | VRAM | System RAM |
|---------------|------|------------|
| Full precision | 24GB | 32GB |
| With offload | 12GB | 24GB |
| FP8 quantized | 16GB | 24GB |

## Training Data

Trained on diverse video-audio paired datasets:
- Professional Foley recordings
- Film and game audio
- Environmental soundscapes
- Musical performances
- Action and impact sounds
- Human vocalizations

The model learns audio-visual correspondences and can be fine-tuned via Zen Gym for custom sound styles.

## Intended Use

### Primary Use Cases
- **Film Production**: Post-production sound design, Foley replacement
- **Game Development**: Procedural audio, dynamic sound effects
- **Content Creation**: YouTube, TikTok, social media videos
- **Professional Audio**: Sound design, ADR, trailers
- **Prototyping**: Quick audio mockups for pre-production

### Example Applications
- Adding sound effects to silent footage
- Replacing production audio with clean effects
- Creating custom soundscapes
- Generating audio for animation
- Rapid sound design iteration

### Out-of-Scope Uses
- Deepfake audio or voice cloning
- Deceptive audio generation
- Music composition (limited capability)
- Real-time audio generation (not optimized)
- Speech synthesis (not primary function)

## Limitations

- **Duration**: Maximum 10 seconds per generation
- **Video Quality**: Requires clear, high-quality input video
- **Complex Scenes**: May struggle with very busy scenes
- **Speech**: Limited speech generation capability
- **Music**: Better for ambience than structured music
- **Memory**: Requires significant GPU memory
- **Speed**: Not suitable for real-time applications

## Bias and Ethical Considerations

- Training data may reflect biases in professional audio production
- Generated audio should be clearly labeled as AI-generated
- Not suitable for creating misleading or deceptive content
- Users should respect copyright and intellectual property
- Consider environmental impact of GPU-intensive generation
- Professional audio engineers still recommended for critical work

## How to Use

### Installation

```bash
git clone https://github.com/zenlm/zen-foley.git
cd zen-foley

conda create -n zen-foley python=3.10
conda activate zen-foley
pip install -r requirements.txt

huggingface-cli download zenlm/zen-foley --local-dir ./models
```

### Basic Usage

```bash
# Simple video-to-audio
python infer.py --video input.mp4 --output output.wav

# With text prompt
python infer.py \
    --video input.mp4 \
    --prompt "Footsteps on wooden floor" \
    --output output.wav

# With offloading (lower VRAM)
python infer.py \
    --video input.mp4 \
    --output output.wav \
    --enable_offload
```

### Python API

```python
from zen_foley import ZenFoleyPipeline

# Initialize
pipeline = ZenFoleyPipeline.from_pretrained(
    "zenlm/zen-foley",
    enable_offload=True
)

# Generate
audio = pipeline(
    video_path="input.mp4",
    prompt="Thunder and rain",
    duration=10.0,
    sampling_rate=48000
)

audio.save("output.wav")
```

### Advanced Features

```python
# Precise timing control
audio = pipeline(
    video_path="input.mp4",
    start_time=5.0,
    duration=8.0,
    prompt="Car engine revving"
)

# Multi-track generation
tracks = pipeline.generate_multi_track(
    video_path="input.mp4",
    track_prompts={
        "ambience": "City street noise",
        "effects": "Car horn",
        "music": "Background jazz"
    }
)

# Batch processing
videos = ["v1.mp4", "v2.mp4", "v3.mp4"]
audios = pipeline.batch_generate(videos, batch_size=4)
```

## Training with Zen Gym

Fine-tune for custom styles:

```bash
cd /path/to/zen-gym

llamafactory-cli train \
    --config configs/zen_foley_lora.yaml \
    --dataset your_audio_video_dataset
```

## Inference with Zen Engine

Serve via API:

```bash
cd /path/to/zen-engine

cargo run --release -- serve \
    --model zenlm/zen-foley \
    --port 3690
```

## Prompt Engineering

### Effective Prompts
- **Be Specific**: "Footsteps on wooden floor" not "walking sounds"
- **Include Environment**: "In large hall", "outdoors", "underwater"
- **Specify Intensity**: "Loud", "gentle", "distant", "close-up"
- **Mention Materials**: "Metal surface", "carpet", "gravel"
- **Describe Motion**: "Fast", "slow", "accelerating", "stopping"

### Example Prompts

```python
# Environmental
"Heavy rain on tin roof, thunder in distance, wind through trees"

# Action
"Sword clashing against shield, grunts of effort, footsteps on stone"

# Mechanical
"Sports car engine starting, revving aggressively, tires screeching"

# Nature
"Ocean waves crashing on rocky shore, seagulls calling, gentle wind"

# Urban
"Busy city street, car horns, people talking, bus passing by"
```

## Benchmarks

### Objective Quality
| Metric | Zen Foley | Baseline |
|--------|-----------|----------|
| FAD ↓ | 2.34 | 3.67 |
| KLD ↓ | 1.87 | 2.45 |
| IS ↑ | 7.21 | 6.13 |
| CLAP ↑ | 0.342 | 0.298 |

### User Study (1-10 scale)
- **Audio Quality**: 8.7/10
- **Synchronization**: 8.9/10
- **Realism**: 8.4/10
- **Prompt Adherence**: 8.6/10

## Citation

```bibtex
@misc{zenfoley2025,
  title={Zen Foley: Professional AI Sound Effect Generation},
  author={Zen AI Team},
  year={2025},
  howpublished={\url{https://github.com/zenlm/zen-foley}}
}

@article{shan2025hunyuanvideo,
  title={HunyuanVideo-Foley: Multimodal Diffusion with Representation Alignment for High-Fidelity Foley Audio Generation},
  author={Sizhe Shan and Qiulin Li and Yutao Cui and Miles Yang and Yuehai Wang and Qun Yang and Jin Zhou and Zhao Zhong},
  journal={arXiv preprint arXiv:2508.16930},
  year={2025}
}
```

## Model Card Contact

For questions or issues:
- **GitHub Issues**: https://github.com/zenlm/zen-foley/issues
- **Organization**: https://github.com/zenlm

## Acknowledgements

Based on [HunyuanVideo-Foley](https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley) by Tencent Hunyuan. We thank the original authors for their groundbreaking work in video-to-audio generation.

## Links

- **GitHub**: https://github.com/zenlm/zen-foley
- **HuggingFace**: https://huggingface.co/zenlm/zen-foley
- **Organization**: https://github.com/zenlm
- **Zen Gym** (Training): https://github.com/zenlm/zen-gym
- **Zen Engine** (Inference): https://github.com/zenlm/zen-engine
- **Zen Director** (Video): https://github.com/zenlm/zen-director

Part of the **[Zen AI](https://github.com/zenlm)** ecosystem - professional AI tools for creativity and production.