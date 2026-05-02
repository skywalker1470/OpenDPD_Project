# Neural Digital Pre-Distortion (DPD)
Power amplifier linearization using LSTM and MLP neural networks, trained on real PA measurement data from the [dpdOpen](https://github.com/pa-predistortion/dpdOpen) dataset. Includes FPGA inference pipeline via ONNX export and hls4ml HLS conversion targeting Xilinx Artix-7.

## Results
Evaluated on the DPA_200MHz dataset (10-carrier 64QAM, 200 MHz bandwidth, Doherty PA).

| Metric | Without DPD | With DPD | Improvement |
|--------|------------|----------|-------------|
| EVM | 218.3 % | 6.1 % | −97.2 % |
| NMSE | +6.8 dB | −24.3 dB | +31.1 dB |

![Results](results_my_dpd/dpd_results.png)

## FPGA Inference Pipeline
The trained LSTM model is exported to ONNX and converted to fixed-point HLS C++ using hls4ml, targeting a Xilinx Artix-7 (xc7a35tcpg236-1) at 100 MHz.

| Implementation | Latency | EVM |
|----------------|---------|-----|
| CPU (PyTorch FP32) | ~200 us | 6.1 % |
| Quantized (INT8) | ~80 us | ~6.3 % |
| FPGA HLS (C-sim) | ~2 us | ~6.5 % |

## Project Structure

.
├── backbones/
│   └── my_dpd_backbone.py       # MyMLPDPD and MyLSTMDPD model definitions
├── datasets/
│   └── DPA_200MHz/              # Real PA measurement data (from dpdOpen)
├── fpga_inference/
│   ├── export_onnx.py           # Export trained .pth -> ONNX
│   ├── quantize_model.py        # INT8 post-training quantization
│   ├── hls4ml_convert.py        # ONNX -> HLS C++ via hls4ml (Artix-7)
│   ├── latency_benchmark.py     # CPU vs quantized latency comparison
|
├── run_my_dpd.py                # Training and evaluation script
├── results_my_dpd/              # Output plots and saved model weights
└── README.md

## Setup
```bash

git clone https://github.com/pa-predistortion/dpdOpen.git
cd dpdOpen
pip install -e .

cp my_dpd_backbone.py backbones/
cp run_my_dpd.py .

pip install onnx onnxruntime
pip install hls4ml[profiling]
```

## Usage

### Training
```bash

python run_my_dpd.py --dataset DPA_200MHz --backbone my_lstm_dpd --hidden 32 --epochs 100


python run_my_dpd.py --dataset DPA_200MHz --backbone my_mlp_dpd --hidden 64 --epochs 100


python run_my_dpd.py --dataset DPA_200MHz --backbone my_lstm_dpd --device cuda --epochs 200
```

Outputs are saved to `results_my_dpd/`: the result plot, best checkpoint (`_best.pth`), and final weights (`_final.pth`).

### FPGA Inference Pipeline
Run in order after training is complete:

```bash

python fpga_inference/export_onnx.py

python fpga_inference/quantize_model.py


python fpga_inference/latency_benchmark.py

python fpga_inference/hls4ml_convert.py
```

HLS project and synthesis report are written to `fpga_inference/hls_project/`.

## Models
**MyLSTMDPD** — Bidirectional LSTM with single-head scaled dot-product attention. Two stacked LSTM layers capture long-range PA memory effects; attention aggregates across the sequence before the output projection.

**MyMLPDPD** — Feedforward network with residual blocks. Flattens a sliding window of I/Q samples and passes them through stacked ResBlocks with LayerNorm and GELU activations.

Both models use the **indirect learning architecture**: the DPD is trained to approximate the inverse PA mapping (PA output → PA input), so that when composed with the real PA, the combined response is approximately linear.

## Key Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `DPA_200MHz` | Dataset folder under `datasets/` |
| `--backbone` | `my_lstm_dpd` | Model: `my_lstm_dpd` or `my_mlp_dpd` |
| `--hidden` | `32` | Hidden layer size |
| `--seq_len` | `15` | Memory depth (samples) |
| `--epochs` | `100` | Training epochs |
| `--lr` | `5e-4` | Learning rate |
| `--device` | auto | `cpu` or `cuda` |

## Dependencies
- Python 3.8+
- PyTorch
- NumPy, SciPy, Matplotlib
- opendpd (`pip install -e .` from dpdOpen repo)
- onnx, onnxruntime
- hls4ml (`pip install hls4ml[profiling]`)
- Vitis HLS / Vivado HLS (for full synthesis; C-simulation works without a board)