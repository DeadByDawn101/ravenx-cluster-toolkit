<div align="center">

# RavenX Cluster Toolkit

### Heterogeneous Mac + Linux MLX Clusters — One Command

> **The missing toolkit for distributed ML on Apple Silicon + NVIDIA.**
> JACCL + mlx.distributed + CUDA — unified training and inference across every GPU you own.

[![License](https://img.shields.io/badge/license-Apache_2.0-green)](LICENSE)
[![MLX](https://img.shields.io/badge/MLX-distributed-blueviolet)](https://ml-explore.github.io/mlx/build/html/usage/distributed.html)
[![Star Platinum](https://img.shields.io/badge/Star_Platinum_v3-OPERATIONAL-gold)](https://github.com/DeadByDawn101/RavenX-JACCL-MLX)

**Built by [RavenX AI Labs](https://github.com/DeadByDawn101) — San Jose, CA**

</div>

---

## Current Cluster: Star Platinum v3

| Node | Machine | Memory | Backend | Status |
|------|---------|--------|---------|--------|
| 0 | M4 Max 128GB MacBook Pro | 128GB | MLX Metal | ✅ Active |
| 1 | M4 Max 128GB MacBook Pro | 128GB | MLX Metal | ✅ Active |
| — | ~~M3 Ultra 96GB Studio~~ | ~~96GB~~ | — | ⚠️ Selling |
| — | Linux 2× RTX 3090 + 5080 | 72GB VRAM | MLX CUDA | 🔜 Assembly |

**Active cluster:** 2× M4 Max 128GB = **256GB unified** over Thunderbolt 5
**Interconnect:** 80 Gbps TB5 (120 Gbps cable installed)
**Backend:** Ring (TCP/TB5), RDMA enabled on both nodes

### Why Sell the M3 Ultra 96GB Studio

The M3 Ultra 96GB is the weakest link in the cluster:
- **96GB limits shard size** — can't hold its share of 100B+ models
- **Not M4 architecture** — no TB5 Bandwidth Boost, older Neural Engine
- **Resale value is STRONG right now** (~$5,500-6,500 on eBay/Swappa)
- **M5 Ultra arriving** — resale drops significantly when announced
- **128GB minimum per node** — our cluster design requires uniform memory
- **Better use of capital** — $6K toward Linux rig completion (Threadripper + 256GB + 2× RTX 3090 + RTX 5080)

**Decision: Sell NOW while the market is hot. Reinvest into Linux rig.**

The 2× M4 Max 128GB cluster (256GB) outperforms 3× mixed nodes (128+128+96 = 352GB) because uniform sharding is more efficient than heterogeneous memory management.

---

## The Problem

Apple shipped JACCL and `mlx.distributed` at WWDC 2026. Linux got `mlx[cuda12]`. The pieces exist. But nobody has put them together into a toolkit that makes heterogeneous Mac + Linux clusters accessible to everyone.

You have two MacBook Pros and a Linux box with RTX 3090s. You should be able to train and run 100B+ models across all of them with one command. Today, that requires stitching together JACCL configs, SSH keys, hostfiles, environment matching, and device routing manually.

**This toolkit does it for you.**

## What It Does

```bash
# Setup: configure all nodes automatically
ravenx-cluster setup

# Train: distributed LoRA across Mac + Linux
ravenx-cluster train \
  --model mlx-community/Qwen3-70B-4bit \
  --data ./training_data \
  --nodes all

# Infer: run 100B+ models across the cluster
ravenx-cluster serve \
  --model mlx-community/Llama-4-100B-4bit \
  --port 8080

# Benchmark: measure your cluster
ravenx-cluster bench
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              ravenx-cluster-toolkit                  │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ Mac Node │  │ Mac Node │  │   Linux Node     │  │
│  │ M4 Max   │  │ M4 Max   │  │ 2x RTX 3090     │  │
│  │ 128GB    │  │ 128GB    │  │ RTX 5080        │  │
│  │ Metal    │  │ Metal    │  │ CUDA (mlx)      │  │
│  └────┬─────┘  └────┬─────┘  └───────┬──────────┘  │
│       │              │                │              │
│       └──────┬───────┘                │              │
│              │                        │              │
│        JACCL (TB5 RDMA)         Ethernet/RoCE        │
│        50-60 Gbps               10-40 Gbps           │
│        sub-50µs                                      │
│              │                        │              │
│              └────────┬───────────────┘              │
│                       │                              │
│              ┌────────┴────────┐                     │
│              │ mlx.distributed │                     │
│              │   all_sum       │                     │
│              │   all_gather    │                     │
│              │   send/recv     │                     │
│              └─────────────────┘                     │
│                       │                              │
│              ┌────────┴────────┐                     │
│              │ Device Router   │                     │
│              │ Metal ↔ CUDA    │                     │
│              └─────────────────┘                     │
└─────────────────────────────────────────────────────┘
```

## Requirements

- macOS 26.2+ on Mac nodes (Tahoe with RDMA support)
- CUDA 12+ on Linux nodes
- Python 3.9+
- MLX 0.32.0+ (both `mlx` and `mlx[cuda12]`)
- Thunderbolt 5 cables between Mac nodes
- 10GbE or Thunderbolt between Mac ↔ Linux
- SSH passwordless authentication between all nodes
- Same Python environment on all nodes

## Quick Start

```bash
# Install
pip install ravenx-cluster-toolkit

# Or from source
git clone https://github.com/DeadByDawn101/ravenx-cluster-toolkit
cd ravenx-cluster-toolkit
pip install -e .

# Discover your nodes
ravenx-cluster discover

# Auto-configure networking
ravenx-cluster setup --auto

# Run a test
ravenx-cluster test
```

## Features

### Auto-Discovery
Scans Thunderbolt Bridge interfaces and SSH-reachable hosts. Detects GPU type (Metal/CUDA), memory, and capabilities per node.

### Device-Class Routing
Automatically routes workloads by device:
- **Metal nodes**: MLX inference, LoRA training, KV cache
- **CUDA nodes**: Large batch training, Flash Attention, vLLM serving
- **Mixed**: Tensor parallel across both — same `mx.distributed.all_sum()`

### One-Command Training
```bash
# Distributed LoRA fine-tuning
ravenx-cluster train \
  --model mlx-community/gemma-4-e4b-it-4bit \
  --adapter-path ./adapters \
  --data ./data \
  --iters 1000 \
  --strategy data-parallel
```

### One-Command Inference
```bash
# Serve a model too large for one machine
ravenx-cluster serve \
  --model mlx-community/Llama-4-Maverick-100B-4bit \
  --strategy tensor-parallel \
  --port 8080
```

### Benchmarking
```bash
# Full cluster benchmark
ravenx-cluster bench --all

# Current output (Star Platinum v3):
# Node          | Type  | Memory | Bandwidth | Status
# M4 Max 128GB  | Metal | 128 GB | 80 Gbps   | ✅ Active
# M4 Max 128GB  | Metal | 128 GB | 80 Gbps   | ✅ Active
# TOTAL         |       | 256 GB |           | OPERATIONAL

# Target output (with Linux rig):
# Node          | Type  | Memory | Bandwidth | Status
# M4 Max 128GB  | Metal | 128 GB | 80 Gbps   | ✅ Active
# M4 Max 128GB  | Metal | 128 GB | 80 Gbps   | ✅ Active
# Linux 3090x2  | CUDA  | 48 GB  | 10 Gbps   | ✅ Active
# Linux 5080    | CUDA  | 24 GB  | 10 Gbps   | ✅ Active
# TOTAL         |       | 328 GB |           | OPERATIONAL
```

## Companion Projects

| Repo | What |
|------|------|
| [RavenX-JACCL-MLX](https://github.com/DeadByDawn101/RavenX-JACCL-MLX) | Production JACCL cluster: always-on, daemon, training, master guide |
| [star-platinum-cluster](https://github.com/DeadByDawn101/star-platinum-cluster) | Hardware docs + topology for our reference cluster |
| [exo fork](https://github.com/DeadByDawn101/exo) | Original heterogeneous cluster research (archived) |
| [exo-mlxring-loader](https://github.com/DeadByDawn101/exo-mlxring-loader) | MLX ring loader for exo (archived) |
| [turboquant-mlx](https://github.com/DeadByDawn101/turboquant-mlx) | KV cache compression for distributed inference |
| [unsloth-mlx](https://github.com/DeadByDawn101/unsloth-mlx) | Training framework with Gemma 4 support |

## Cluster Evolution

| Phase | Config | Memory | Transport | Speed |
|-------|--------|--------|-----------|-------|
| v1 (Mar 2026) | 4× Macs (exo) | 184GB | TCP/WiFi | ~1 Gbps |
| v2 (May 2026) | 5 nodes (exo) | 200GB+ | TCP/Ethernet | ~2 Gbps |
| **v3 (Aug 2026)** | **2× M4 Max (JACCL)** | **256GB** | **TB5 RDMA** | **80 Gbps** |
| v4 (target) | 2× M4 Max + Linux | 328GB | TB5 + 10GbE | Mixed |

## The Story

We started building heterogeneous Apple Silicon + NVIDIA clusters in March 2026 — custom exo fork, tinygrad CUDA engines, OdinLink RDMA protocols, 800K nack storm patches. In June, Apple shipped JACCL and `mlx[cuda12]` at WWDC 2026, replacing our custom stack with first-party solutions.

We went from patching TCP nack storms to 80 Gbps RDMA in one cable swap. The exo era is over. The JACCL era has begun.

This toolkit takes what we learned building Star Platinum and makes it accessible to everyone. No custom drivers. No reverse engineering. Just Apple's public APIs, NVIDIA's freely available CUDA runtime, and MLX connecting them.

**The sovereign AI cluster is here. No cloud. No vendor lock-in. Just your hardware, your data, your models.**

## License

Apache 2.0

---

<div align="center">

*"We were building this before Apple shipped it. Now we're making it accessible to everyone."*

**RavenX AI Labs LLC — San Jose, CA — Founded June 2026**

GitHub: [@DeadByDawn101](https://github.com/DeadByDawn101) | X: [@RavenXllm](https://x.com/RavenXllm)

</div>
