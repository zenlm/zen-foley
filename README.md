# Zen Foley

**Zen Foley** is a professional-grade AI sound effect generation model for video content. Based on HunyuanVideo-Foley, it generates high-fidelity audio synchronized with video scenes, perfect for filmmaking, game development, and content creation.

<p align="center">
  <a href="https://github.com/zenlm/zen-foley"><img src="https://img.shields.io/badge/GitHub-zenlm%2Fzen--foley-blue"></a>
  <a href="https://huggingface.co/zenlm/zen-foley"><img src="https://img.shields.io/badge/🤗-Models-yellow"></a>
  <a href="https://github.com/zenlm"><img src="https://img.shields.io/badge/Zen-AI-purple"></a>
</p>

## Overview

Zen Foley generates professional sound effects synchronized with video content:

- 🎬 **Video-to-Audio**: Generate sound effects from video scenes
- 🎭 **Multi-Scenario Sync**: High-quality audio for complex scenes
- 🎵 **48kHz Hi-Fi**: Professional-grade audio output
- ⚖️ **Multi-Modal Balance**: Perfect harmony between visual and textual cues
- 📝 **Text Control**: Optional text descriptions for precise control
- ⚡ **Efficient**: XL model with offload support for lower VRAM

## Model Details

- **Model Type**: Video-to-Audio Generation (Diffusion)
- **Architecture**: Multimodal Diffusion Transformer
- **License**: Apache 2.0
- **Input**: Video (MP4), optional text prompt
- **Output**: Audio (48kHz WAV)
- **Duration**: Up to 10 seconds
- **Developed by**: Zen AI Team
- **Based on**: [HunyuanVideo-Foley by Tencent](https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley)

## Capabilities

### Multi-Scenario Sound Generation
- Footsteps, ambience, nature sounds
- Vehicle and mechanical sounds
- Action and impact effects
- Musical elements and instruments
- Human vocalizations and speech
- Complex multi-layered soundscapes

### Audio-Visual Synchronization
- Frame-accurate timing
- Motion-sound correspondence
- Spatial audio positioning
- Intensity matching
- Seamless transitions

## Hardware Requirements

### Minimum (XL Model with Offloading)
- **GPU**: 12GB VRAM (RTX 3080, RTX 4070 Ti)
- **RAM**: 16GB system memory
- **Storage**: 20GB for model

### Recommended
- **GPU**: 24GB VRAM (RTX 4090, RTX 3090)
- **RAM**: 32GB system memory
- **Storage**: 50GB for model and cache

### Optimal
- **GPU**: 40GB+ VRAM (A100)
- **RAM**: 64GB system memory
- For faster generation without offloading

## Installation

```bash
# Clone repository
git clone https://github.com/zenlm/zen-foley.git
cd zen-foley

# Create environment
conda create -n zen-foley python=3.10
conda activate zen-foley

# Install dependencies
pip install -r requirements.txt

# Download model
huggingface-cli download zenlm/zen-foley --local-dir ./models
```

## Usage

### Basic Video-to-Audio

```bash
python infer.py \
    --video input.mp4 \
    --output output.wav \
    --model_path ./models
```

### With Text Prompt

```bash
python infer.py \
    --video input.mp4 \
    --prompt "Footsteps on wooden floor, gentle rain outside" \
    --output output.wav
```

### With CPU Offloading (Lower VRAM)

```bash
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
    enable_offload=True  # For lower VRAM
)

# Generate audio
audio = pipeline(
    video_path="input.mp4",
    prompt="Thunder and rain storm",  # Optional
    duration=10.0,
    sampling_rate=48000
)

# Save
audio.save("output.wav")
```

## Use Cases

### Film & Video Production
- Post-production sound design
- ADR replacement
- Ambience and Foley effects
- Quick prototyping

### Game Development
- Procedural audio generation
- Dynamic sound effects
- Cutscene audio
- Rapid iteration

### Content Creation
- YouTube videos
- TikTok/Shorts
- Podcasts with video
- Social media content

### Professional Audio
- Sound design
- Audio post-production
- Trailer editing
- Commercial production

## Training with Zen Gym

Fine-tune for custom sound styles:

```bash
cd /path/to/zen-gym

llamafactory-cli train \
    --config configs/zen_foley_lora.yaml \
    --dataset your_audio_video_dataset
```

## Inference with Zen Engine

Serve Zen Foley via API:

```bash
cd /path/to/zen-engine

cargo run --release -- serve \
    --model zenlm/zen-foley \
    --port 3690
```

## Advanced Features

### Precise Timing Control

```python
# Generate audio for specific time range
audio = pipeline(
    video_path="input.mp4",
    start_time=5.0,  # Start at 5 seconds
    duration=8.0,    # Generate 8 seconds
    prompt="Car engine revving and accelerating"
)
```

### Multi-Track Generation

```python
# Generate separate audio tracks
tracks = pipeline.generate_multi_track(
    video_path="input.mp4",
    track_prompts={
        "ambience": "City street ambience",
        "effects": "Car horn and traffic",
        "music": "Background jazz music"
    }
)
```

### Batch Processing

```python
# Process multiple videos
videos = ["video1.mp4", "video2.mp4", "video3.mp4"]
audios = pipeline.batch_generate(videos, batch_size=4)
```

## Performance

### Generation Speed
- **RTX 4090**: ~15s for 10-second audio
- **RTX 4090 (offload)**: ~25s for 10-second audio
- **RTX 3080 (offload)**: ~40s for 10-second audio
- **A100**: ~10s for 10-second audio

### Quality Metrics
| Metric | Score |
|--------|-------|
| FAD | 2.34 |
| KLD | 1.87 |
| IS | 7.21 |

## Prompt Engineering

### Effective Prompts
- Describe specific sounds: "footsteps", "door closing", "glass breaking"
- Include environment: "in large hall", "outdoors", "underwater"
- Specify intensity: "loud", "gentle", "distant", "close-up"
- Mention materials: "wooden floor", "metal surface", "carpet"

### Examples

```python
# Environmental
"Heavy rain on roof, thunder in distance, wind through trees"

# Action
"Sword clashing, grunts, footsteps on stone floor"

# Mechanical
"Car engine starting, revving, tires screeching, horn"

# Nature
"Ocean waves crashing, seagulls calling, wind blowing"
```

## Limitations

- Maximum 10-second duration per generation
- Requires high-quality input video
- May struggle with very complex soundscapes
- Speech generation limited
- Music generation best for background/ambience
- Requires significant GPU memory

## Ethical Considerations

- Generated audio should be labeled as AI-generated
- Not suitable for deepfake audio
- Respect copyright and licensing
- Consider misuse for misinformation
- Professional audio engineering still recommended
- Environmental impact of GPU usage

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

## Credits

Zen Foley is based on [HunyuanVideo-Foley](https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley) by Tencent Hunyuan. We thank the original authors for their excellent work in video-to-audio generation.

## Links

- **GitHub**: https://github.com/zenlm/zen-foley
- **HuggingFace**: https://huggingface.co/zenlm/zen-foley
- **Organization**: https://github.com/zenlm
- **Zen Gym** (Training): https://github.com/zenlm/zen-gym
- **Zen Engine** (Inference): https://github.com/zenlm/zen-engine
- **Zen Director** (Video): https://github.com/zenlm/zen-director

## License

Apache 2.0 License - see [LICENSE](LICENSE) for details.

---

**Zen Foley** - Professional AI sound design for video content

Part of the **[Zen AI](https://github.com/zenlm)** ecosystem.