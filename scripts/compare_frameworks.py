"""
Framework Comparison Benchmark: PyTorch vs. TensorFlow / Keras

Purpose for Interview / Technical Defense:
This script demonstrates a practical, empirical comparison between PyTorch and TensorFlow 
for computer vision feature extraction on the home-design product catalog.

Comparison Criteria:
1. Feature Extractor Architecture & Output Dimension
2. Model Initialization Time (Cold Start)
3. Inference Latency (Batch size = 1, CPU, averaged over 20 runs)
4. L2 Normalization Property & Output Vector Consistency
5. Architectural Trade-offs & Production Suitability
"""

import time
import numpy as np
from pathlib import Path


def benchmark_pytorch(input_np: np.ndarray, num_runs: int = 20):
    import torch
    import torchvision.models as tv_models
    import torch.nn as nn

    print("\n[1/2] Benchmarking PyTorch ResNet-18 Feature Extractor...")
    
    t0 = time.perf_counter()
    try:
        from torchvision.models import ResNet18_Weights
        pt_model = tv_models.resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception:
        pt_model = tv_models.resnet18(pretrained=True)
    
    # Strip classifier
    pt_model.fc = nn.Identity()
    pt_model.eval()
    init_time_ms = (time.perf_counter() - t0) * 1000

    # Convert CHW numpy input to Torch tensor
    # input_np shape: (1, 3, 224, 224)
    tensor_input = torch.from_numpy(input_np.astype(np.float32))

    # Warm-up run
    with torch.no_grad():
        _ = pt_model(tensor_input)

    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            t_start = time.perf_counter()
            features = pt_model(tensor_input).squeeze(0).cpu().numpy()
            norm = np.linalg.norm(features)
            if norm > 0:
                normalized = features / norm
            else:
                normalized = features
            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1000)

    avg_latency_ms = float(np.mean(latencies))
    std_latency_ms = float(np.std(latencies))
    output_dim = len(normalized)
    vector_norm = float(np.linalg.norm(normalized))

    return {
        "framework": "PyTorch (TorchVision)",
        "model_name": "ResNet-18 (fc=Identity)",
        "init_time_ms": round(init_time_ms, 2),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "std_latency_ms": round(std_latency_ms, 2),
        "output_dim": output_dim,
        "l2_norm": round(vector_norm, 4)
    }


def benchmark_tensorflow(input_np: np.ndarray, num_runs: int = 20):
    print("\n[2/2] Benchmarking TensorFlow / Keras MobileNetV2 Feature Extractor...")
    try:
        import tensorflow as tf
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
    except ImportError:
        print("TensorFlow not installed. Skipping TF benchmark.")
        return None

    t0 = time.perf_counter()
    # Load MobileNetV2 without top classification head and with global average pooling
    tf_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3)
    )
    init_time_ms = (time.perf_counter() - t0) * 1000

    # Convert CHW numpy input to HWC for TensorFlow: (1, 224, 224, 3)
    tf_input = np.transpose(input_np, (0, 2, 3, 1)).astype(np.float32)

    # Warm-up run
    _ = tf_model(tf_input, training=False)

    latencies = []
    for _ in range(num_runs):
        t_start = time.perf_counter()
        features = tf_model(tf_input, training=False).numpy().flatten()
        norm = np.linalg.norm(features)
        if norm > 0:
            normalized = features / norm
        else:
            normalized = features
        t_end = time.perf_counter()
        latencies.append((t_end - t_start) * 1000)

    avg_latency_ms = float(np.mean(latencies))
    std_latency_ms = float(np.std(latencies))
    output_dim = len(normalized)
    vector_norm = float(np.linalg.norm(normalized))

    return {
        "framework": "TensorFlow / Keras",
        "model_name": "MobileNetV2 (pooling=avg)",
        "init_time_ms": round(init_time_ms, 2),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "std_latency_ms": round(std_latency_ms, 2),
        "output_dim": output_dim,
        "l2_norm": round(vector_norm, 4)
    }


def main():
    print("=" * 70)
    print("CYNCLY INTERVIEW DEFENSE: PYTORCH vs. TENSORFLOW BENCHMARK")
    print("=" * 70)

    # Create a synthetic normalized test input tensor: (1, 3, 224, 224)
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

    pt_results = benchmark_pytorch(dummy_input)
    tf_results = benchmark_tensorflow(dummy_input)

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS SUMMARY TABLE")
    print("=" * 70)
    headers = ["Metric", "PyTorch (ResNet-18)", "TensorFlow (MobileNetV2)"]
    print(f"{headers[0]:<25} | {headers[1]:<22} | {headers[2]:<22}")
    print("-" * 75)

    if tf_results:
        print(f"{'Initialization Time':<25} | {pt_results['init_time_ms']} ms{'':<15} | {tf_results['init_time_ms']} ms")
        print(f"{'Avg Inference (CPU)':<25} | {pt_results['avg_latency_ms']} ? {pt_results['std_latency_ms']} ms | {tf_results['avg_latency_ms']} ? {tf_results['std_latency_ms']} ms")
        print(f"{'Embedding Dimension':<25} | {pt_results['output_dim']} dims{'':<13} | {tf_results['output_dim']} dims")
        print(f"{'Vector L2-Norm':<25} | {pt_results['l2_norm']}{'':<17} | {tf_results['l2_norm']}")
    else:
        print(f"{'Initialization Time':<25} | {pt_results['init_time_ms']} ms{'':<15} | N/A")
        print(f"{'Avg Inference (CPU)':<25} | {pt_results['avg_latency_ms']} ? {pt_results['std_latency_ms']} ms | N/A")
        print(f"{'Embedding Dimension':<25} | {pt_results['output_dim']} dims{'':<13} | N/A")

    print("=" * 75)
    print("""
KEY TAKEAWAYS FOR THE INTERVIEW:
1. Why PyTorch for Primary API:
   - ResNet-18 produces a compact 512-dim embedding which matches pgvector storage and indexing efficiency.
   - Clean, lightweight eager-mode execution with no session overhead.
2. Why compare with TensorFlow:
   - Evaluated MobileNetV2 in Keras to analyze mobile-optimized edge inference vs server-side feature extraction.
   - Proves hands-on proficiency across both industry-standard ML frameworks.
""")


if __name__ == "__main__":
    main()
