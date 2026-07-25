<div align="center">

# RavenX Cluster Toolkit

### Heterogeneous Mac + Linux MLX Clusters — One Command

> **The missing toolkit for distributed ML on Apple Silicon + NVIDIA.**
> JACCL + mlx.distributed + CUDA — unified training and inference across every GPU you own.

[![License](https://img.shields.io/badge/license-Apache_2.0-green)](LICENSE)
[![MLX](https://img.shields.io/badge/MLX-distributed-blueviolet)](https://ml-explore.github.io/mlx/build/html/usage/distributed.html)
[![Star Platinum](https://img.shields.io/badge/Star_Platinum-cluster-gold)](https://github.com/DeadByDawn101/star-platinum-cluster)

**Built by [RavenX AI Labs](https://github.com/DeadByDawn101) — San Jose, CA**

</div>

---

## The Problem

Apple shipped JACCL and `mlx.distributed` at WWDC 2026. Linux got `mlx[cuda12]`. The pieces exist. But nobody has put them together into a toolkit that makes heterogeneous Mac + Linux clusters accessible to everyone.

You have a Mac Studio, a MacBook Pro, and a Linux box with a 3090. You should be able to train and run 100B+ models across all of them with one command. Today, that requires stitching together JACCL configs, SSH keys, hostfiles, environment matching, and device routing manually.

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
│  │ M4 Max   │  │ M3 Ultra │  │ 2x RTX 3090     │  │
│  │ 128GB    │  │ 96GB     │  │ RTX 5080        │  │
│  │ Metal    │  │ Metal    │  │ CUDA (mlx)      │  │
│  └────┬─────┘  └────┬─────┘  └───────┬──────────┘  │
│       │              │                │              │
│       └──────┬───────┘                │              │
│              │                        │              │
│        JACCL (TB5 RDMA)         Ethernet/TB          │
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
│              │ Auto-placement  │                     │
│              └─────────────────┘                     │
│                                                      │
└─────────────────────────────────────────────────────┘
```

## Requirements

- macOS 26.2+ (JACCL / RDMA support)
- Thunderbolt 5 cables (fully connected mesh for Mac nodes)
- Linux node: `pip install mlx[cuda12]`
- SSH keys configured between all nodes
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

# Output:
# Node          | Type  | Memory | Bandwidth | TFLOPS
# M4 Max 128GB  | Metal | 128 GB | 60 Gbps   | 45
# M3 Ultra 96GB | Metal | 96 GB  | 60 Gbps   | 38
# Linux 3090x2  | CUDA  | 48 GB  | 10 Gbps   | 70
# TOTAL         |       | 272 GB |           | 153
```

## Companion Projects

| Repo | What |
|------|------|
| [star-platinum-cluster](https://github.com/DeadByDawn101/star-platinum-cluster) | Hardware docs + topology for our reference cluster |
| [exo fork](https://github.com/DeadByDawn101/exo) | Original heterogeneous cluster research (ravenx/cuda-linux branch) |
| [turboquant-mlx](https://github.com/DeadByDawn101/turboquant-mlx) | KV cache compression for distributed inference |
| [unsloth-mlx](https://github.com/DeadByDawn101/unsloth-mlx) | Training framework with Gemma 4 support |
| [OdinLink-Five](https://github.com/DeadByDawn101/OdinLink-Five) | TB5 RDMA research (pre-JACCL) |

## The Story

We started building heterogeneous Apple Silicon + NVIDIA clusters in March 2026 — custom exo fork, tinygrad CUDA engines, OdinLink RDMA protocols. In June, Apple shipped JACCL and `mlx[cuda12]` at WWDC 2026, replacing our custom stack with first-party solutions.

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
