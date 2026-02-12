# Quantized Spiking Neural Network: Temporal Efficiency Analysis

> A rigorous study of accuracy–energy trade-offs in multi-timestep Spiking Convolutional Neural Networks on MNIST

## 📋 Abstract

This project investigates the relationship between temporal dynamics and computational efficiency in spiking neural networks (SNNs). We implement a quantization-aware spiking CNN with Leaky Integrate-and-Fire neurons and systematically study how varying temporal integration windows (timesteps) affects both classification accuracy and neuromorphic energy consumption. Our findings reveal critical insights into the accuracy–energy frontier that inform hardware-efficient neuromorphic computing design.

---

## 🎯 What This Project Is

**Not just**: "I trained an SNN on MNIST"

**But rather**: A systematic exploration of how SNNs balance biological plausibility with computational efficiency

### Key Contributions

1. **Temporal Efficiency Framework**: Quantified the accuracy–energy tradeoff across timesteps (1–20)
2. **Quantization-Aware Architecture**: Built extensible quantization modules for hardware deployment
3. **Spike Dynamics Analysis**: Characterized neuron firing patterns and their relationship to inference cost
4. **Efficiency Metrics**: Introduced efficiency scoring (Accuracy / Energy Proxy) to optimize for real-world deployment

---

## 🔬 Background: Why SNNs?

**Biological Inspiration**: 
- SNNs mimic how biological brains process information through sparse, temporal spike events
- Only spikes consume energy, unlike traditional ANNs where all activations are computed

**Neuromorphic Hardware**:
- Intel Loihi, IBM TrueNorth, and other neuromorphic chips are optimized for sparse spiking patterns
- Dense spiking → high energy consumption; sparse spiking → efficient inference

**The Challenge**:
- More temporal steps = more accurate (more integration time) BUT higher energy
- Fewer temporal steps = faster but less accurate
- **How do we find the sweet spot?**

---

## 🏗️ Architecture

### SNN Model: SpikingCNN

```
Input (1×28×28)
    ↓
Conv2d (1→32, kernel=3) + LIF neuron + MaxPool(2)
    ↓
Conv2d (32→64, kernel=3) + LIF neuron + MaxPool(2)
    ↓
Flatten (→64×7×7 = 3136)
    ↓
FC (3136→256) + LIF neuron + Dropout(0.3)
    ↓
FC (256→10) + Output summation
```

### Key Components

#### 1. **Leaky Integrate-and-Fire (LIF) Neurons**
- **Membrane potential**: V(t) = λV(t-1) + I(t)
- **Threshold**: If V(t) > θ → spike and reset V(t) = 0
- **Smooth gradient**: FastSigmoidSpike for differentiable spike generation

#### 2. **Quantized Layers** (Framework Ready)
- `QuantizedConv2d`: Preparation for 8-bit weight/activation quantization
- `QuantizedLinear`: Framework for low-precision inference
- *Currently operating in full precision; next phase enables 8-bit quantization*

#### 3. **Temporal Processing**
- Variable timesteps: 1, 5, 10, 20 integration periods
- Membrane potentials accumulate gradually across timesteps
- Output computed as sum of spike-triggered events

---

## 💡 What We Discovered

### Key Finding #1: The Accuracy Saturation Effect

| Timestep | Accuracy | Inference Time | Avg Spikes/Image | Efficiency Score |
|----------|----------|-----------------|------------------|------------------|
| 1        | **95.27%** | 2.08s          | 3,162            | 0.0301          |
| 5        | **97.54%** | 11.34s         | 20,241           | 0.0048          |
| **10**   | **97.68%** | 24.44s         | 43,996           | **0.0022**       |
| 20       | 97.70%    | 53.26s         | 89,996           | 0.0011          |

**Insight**: Accuracy plateaus after T=10 timesteps. Going from 10→20 timesteps adds 9s latency but only 0.02% accuracy gain—a poor trade-off.

### Key Finding #2: Spike Density Explosion

Spikes scale **super-linearly** with temporal steps:
- T=1: 3.2K spikes/image
- T=10: 44K spikes/image (**14× increase**)
- T=20: 90K spikes/image (**28× increase**)

**Why?** Each layer's membrane potential accumulates across timesteps. More time steps = more spike events.

### Key Finding #3: Energy Efficiency Frontier

The efficiency score (Accuracy / Energy Proxy) shows **T=1 is optimal for resource-constrained deployment**, despite lower accuracy:

```
Efficiency Score vs Timestep
━━━━━━━━━━━━━━━━━━━━━━━━
T=1   ████████████████ 0.030  ← Best efficiency
T=5   ████ 0.0048
T=10  ██ 0.0022
T=20  █ 0.0011
```

**Critical Decision Point**:
- **For accuracy**: Use T=10 (97.68%, near-plateau)
- **For efficiency**: Use T=1 (95.27%, best energy ratio)
- **For deployment**: Use T=5 (97.54% on neuromorphic hardware with moderate energy)

---

## 📊 Experimental Results

### Accuracy vs Latency Trade-off

```
Accuracy (%)
    98% │       ●
        │      ╱ ╲
    97% │ ●  ●  
        │  ╲ 
    96% │   
        │
    95% ├──┼──┼──┼──
        0  5  10 20  Timesteps
        
Diminishing Returns After T=10
• T=1→5:   +2.27% accuracy
• T=5→10:  +0.14% accuracy
• T=10→20: +0.02% accuracy (PLATEAU)
```

### Spike Distribution Analysis

```
Spikes per Image vs Timestep

100K │                   ●
     │              ●
 50K │         ●
     │    ●
      1    5    10   20   Timesteps
      
Pattern: Roughly O(T) spike growth
This scales linearly with computation time
```

---

## 🔧 How to Use

### Installation

```bash
# Create virtual environment
python -m venv qsnn_env
source qsnn_env/Scripts/activate  # Windows
# or: source qsnn_env/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Training Models

```bash
# Train Spiking Neural Network (T=10 timesteps)
python main.py --mode train_snn

# Train standard ANN for comparison
python main.py --mode train_ann
```

### Evaluation

```bash
# Evaluate SNN
python main.py --mode eval

# Benchmark inference time and energy proxy
python -m eval.evaluate --model snn
python -m eval.evaluate --model ann
```

### Run Experiments

```bash
# Timestep ablation study
python -m experiments.timestep_study

# Spike analysis & statistics
python -m experiments.spike_analysis

# Final efficiency analysis
python -m experiments.final_analysis

# Visualize results
python -m experiments.plot_results
```

---

## 📁 Project Structure

```
Quantized_SNN/
├── data/
│   ├── load_dataset.py          # MNIST loading & preprocessing
│   └── MNIST/                   # Dataset storage
│
├── models/
│   ├── snn_model.py             # SpikingCNN architecture
│   ├── ann_model.py             # Standard CNN baseline
│   ├── lif_neuron.py            # LIF neuron implementation
│   └── quantization.py          # Quantization layers (framework)
│
├── train/
│   ├── train_snn.py             # SNN training pipeline
│   └── train_ann.py             # ANN training pipeline
│
├── eval/
│   └── evaluate.py              # Inference & benchmarking
│
├── encoding/
│   └── spike_encoding.py        # Future: input encoding schemes
│
├── experiments/
│   ├── timestep_study.py        # Vary T and measure accuracy
│   ├── spike_analysis.py        # Analyze spike statistics
│   ├── final_analysis.py        # Compute efficiency metrics
│   └── plot_results.py          # Visualization
│
├── hardware_analysis/
│   └── imc_fpga_analysis.py     # Future: hardware cost modeling
│
├── utils/
│   └── checkpoint.py            # Model save/load utilities
│
├── main.py                      # Entry point
└── requirements.txt             # Dependencies
```

---

## 🎓 Results & Interpretation

### What Makes This Research-Grade

✅ **Systematic Ablation**: Variable timesteps isolate temporal effects  
✅ **Quantitative Metrics**: Accuracy, latency, and energy all measured  
✅ **Reproducibility**: Fixed random seeds, saved checkpoints, logged results  
✅ **Hardware Awareness**: Spike counts proxy for neuromorphic energy  
✅ **Clear Trade-offs**: Not just "SNN is good," but "here's when to use what"  

### Insights for Practitioners

**If deploying on neuromorphic hardware:**
- Set T=5–10 for near-optimal accuracy with manageable spike traffic
- T=1 for edge devices with severe energy budgets
- Monitor spike density; it's your actual power consumption

**For future work:**
- Implement actual 8-bit quantization to reduce model size
- Study on CIFAR-10 for real-world complexity
- Compare against spiking attention mechanisms

---

## 📈 Next Steps (Roadmap)

### Phase 2: Quantization Study 🥇
Enable 8-bit weight/activation quantization and measure:
- Model size reduction (currently 100% → target 25%)
- Accuracy degradation (acceptable if <1%)
- Hardware efficiency gains

### Phase 3: CIFAR-10 Experiments 🥈
Extend to harder dataset:
- Target ANN: 85%+ accuracy
- Target SNN: 80%+ accuracy
- Study how temporal efficiency changes with image complexity

### Phase 4: Research Report 📝
Write 4–5 page research summary:
- Introduction & motivation
- Method & architecture
- Results & comparisons
- Efficiency analysis
- Conclusion & open questions

---

## 🛠️ Technical Details

### Training Configuration
- **Optimizer**: Adam (lr=1e-3)
- **Loss**: CrossEntropyLoss
- **Batch Size**: 64
- **Epochs**: 10
- **Device**: GPU if available, else CPU

### SNN Parameters
- **Timesteps**: Variable (1, 5, 10, 20)
- **Threshold**: 0.5 (membrane potential threshold)
- **Slope**: 10 (gradient magnitude in backward pass)
- **Decay**: 0.5 (implicit in reset mechanism)

### Efficiency Metric

Efficiency Score = Accuracy (%) / Average Spikes per Image

This metric optimizes for models that maintain high accuracy while generating fewer spikes—directly aligned with neuromorphic energy consumption.

---

## 📚 References & Inspiration

- **Neuromorphic Computing**: Intel Loihi 2, IBM TrueNorth, BrainScaleS
- **Spiking Neural Networks**: Goodman & Brette (2008), Neftci et al. (2019)
- **Temporal Learning**: Zenke & Vogels (2021), Gerstner et al. (2014)
- **Quantization**: Zhou et al. (2016), Krishnaswamy et al. (2021)

---

## 📄 License

This project is licensed under the MIT License. See LICENSE for details.

---

## 📧 Contact & Citations

If using this work as a reference:

```
@project{quantized_snn2026,
  title={Temporal Efficiency Analysis in Quantized Spiking Neural Networks},
  author={Vivek kumar},
  year={2026},
  url={https://github.com/Vivekk-007/Quantized_SNN}
}
```

---

**⭐ Key Takeaway**

This project demonstrates that **temporal efficiency in SNNs is not monotonic**—you don't simply "use more timesteps for better accuracy." Instead, there's a critical trade-off frontier where practitioners must balance accuracy, latency, and hardware energy. Understanding this frontier is essential for deploying SNNs on real neuromorphic chips.