"""
Phase 4 — Deployment (NOT YET IMPLEMENTED).

This package will contain:
  - export_onnx.py:   Export trained PyTorch model to ONNX (opset 17)
  - quantise_trt.py:  INT8 calibration + TensorRT engine build for Jetson Orin NX
  - benchmark.py:     Latency / throughput benchmark on target hardware

Target hardware:
  - Primary: NVIDIA Jetson Orin NX (TensorRT 8.x, FP16 / INT8)
  - Optional: Cortex-M / DSP MCU via TFLM or CMSIS-NN (INT8, ~50-100k params)
"""
