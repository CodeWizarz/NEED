# VM Pipeline Status Report
## Date: Thu Mar 19 2026 (Updated)

---

## ✅ COMPLETED THIS SESSION

### 1. AlpaSim Pipeline — FULLY OPERATIONAL
- **Docker image**: `alpasim-base:0.1.5` (24.3GB) — packages copied to system Python path
- **VaVAM-B model**: Downloaded (1.7GB) + tokenizers (274MB) from valeoai/VideoActionModel
- **5 microservices running** successfully end-to-end
- **GPU**: ~9GB used during sim, 14GB free
- **Simulation completed**: 3 episodes, generating MP4, metrics.parquet, rollout.asl

### 2. Data Collection — 3 Episodes Complete
| Episode | Collisions | Collision Rate | Progress | Avg Dist to GT |
|---------|-----------|---------------|----------|----------------|
| ep_0000 | 28/40 | 70% | 86% | 1.26m |
| ep_0001 | 16/40 | **40%** (best) | 88% | 1.01m |
| ep_0002 | 28/40 | 70% | 89% | 1.37m |

**Total: 120 steps | 72 collisions | 48 safe | 9 offroad**

### 3. Vision Model Fine-tuning — Complete
- **Architecture**: CNN (3 layers) → LSTM (256) → Dense (256) → Dense (40) → Reshape(20,2)
- **Training**: 123 samples from 3 episodes
- **Epochs**: 23 (early stopping)
- **Loss**: train=4.184, val=2.216
- **Output**: 20-step trajectories (steering + speed per step)
- **Saved**: `/tmp/alpamayo_v2/model.keras`

### 4. PhysicalAI Dataset — ACCESS CONFIRMED
- USDZ files downloadable with `repo_type="dataset"`
- Labels accessible (behavior, layout, lighting, weather, vrus)
- 2GB per scene, sample scenes at 1.1-1.2GB
- PhysicalAI NuRec dataset: `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec`

### 5. Demo Outputs Created
| File | Size | Description |
|------|------|-------------|
| `alpasim_pipeline_final.mp4` | 8.5MB | Full pipeline demo with camera footage + metrics overlay |
| `pipeline_complete.png` | 900KB | Comprehensive dashboard (dark theme, all metrics) |
| `pipeline_final_dashboard.png` | 825KB | 3-episode comparison dashboard |
| `trajectory_final.png` | 38KB | Trajectory comparison across episodes |
| `training_metrics.png` | 108KB | Per-metric bar charts |

---

## INFRASTRUCTURE STATE

| Component | Status |
|-----------|--------|
| VM | Running (136.119.37.171) |
| GPU | 14GB free of 23GB |
| Disk | 149GB free of 485GB |
| AlpaSim containers | Stopped (ran 3 episodes successfully) |
| Docker image | `alpasim-base:0.1.5` (24.3GB) |
| VAM model | Loaded (318M + 38M params) |
| Vision model | Trained (`/tmp/alpamayo_v2/model.keras`, 6.8MB) |
| HF_TOKEN | `<REDACTED>` |

---

## DEMOS (Local: `/Users/Balu/Documents/NEED/`)

**Videos:**
- `alpasim_pipeline_final.mp4` — Main demo video (8.5MB)
- `alpasim_final_demo.mp4` — Enhanced demo (10.5MB)
- `alpasim_demo_ep0.mp4` — Episode 0 raw footage (1.2MB)

**Images:**
- `pipeline_complete.png` — Full dark-theme dashboard
- `pipeline_final_dashboard.png` — 3-episode comparison
- `trajectory_final.png` — Trajectory overlays

---

## NEXT STEPS

1. **Replace VAM with Alpamayo-R1-10B** — Use Alpamayo from `/tmp/alpamayo_model/` as planner
2. **Download more PhysicalAI scenes** — 2GB each for diverse conditions
3. **Continuous loop automation** — Auto-run: simulate → collect → train → eval → repeat
4. **Improve vision model** — Use actual camera frames instead of synthetic, more episodes
5. **Restart simulation** — Run more episodes for broader failure coverage
