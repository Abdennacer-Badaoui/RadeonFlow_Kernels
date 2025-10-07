#!/usr/bin/env python3
"""Demo comparing two methods of using RadeonFlow GEMM kernel"""

import torch
import ctypes
import os

# Matrix dimensions (must be supported - see gemm_launcher.cpp)
M, N, K = 1024, 1536, 7168
QUANT_SIZE = 128

print("=" * 70)
print(f"RadeonFlow FP8 GEMM Demo: ({M}x{K}) @ ({K}x{N}) = ({M}x{N})")
print("=" * 70)

# Setup device
device = torch.device("cuda")
torch.manual_seed(42)

# ============================================================================
# PREPARE INPUTS (shared by both methods)
# ============================================================================
print("\n[1] Preparing inputs...")

# Create inputs - kernel expects A:(K,M), B:(K,N)
A_fp32 = torch.randn(M, K, device=device)
B_fp32 = torch.randn(K, N, device=device)

# Convert to FP8
A_fp8 = A_fp32.to(torch.float8_e4m3fnuz)
B_fp8 = B_fp32.to(torch.float8_e4m3fnuz)

# Create scale factors (uniform scaling)
A_scale = torch.ones(K // QUANT_SIZE, M, device=device, dtype=torch.float32)
B_scale = torch.ones(K // QUANT_SIZE, N // QUANT_SIZE, device=device, dtype=torch.float32)

print(f"  ✓ A_fp8: {A_fp8.shape}")
print(f"  ✓ B_fp8: {B_fp8.shape}")
print(f"  ✓ A_scale: {A_scale.shape}")
print(f"  ✓ B_scale: {B_scale.shape}")


# ============================================================================
# METHOD 1: Normal build (ctypes + libgemm.so)
# ============================================================================
print("\n" + "=" * 70)
print("[2] METHOD 1: Using normal build (libgemm.so)")
print("=" * 70)

try:
    # Allocate output
    C_method1 = torch.zeros(M, N, device=device, dtype=torch.bfloat16)
    
    # Load the kernel library
    lib_path = os.path.join(os.path.dirname(__file__), 'build/libgemm.so')
    lib = ctypes.CDLL(lib_path)
    lib.run.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
    lib.run.restype = None
    
    stream = torch.cuda.current_stream().cuda_stream
    
    # Warm-up
    print("  Warming up...")
    for _ in range(10):
        lib.run(A_fp8.data_ptr(), B_fp8.data_ptr(), A_scale.data_ptr(),
                B_scale.data_ptr(), C_method1.data_ptr(), M, N, K, None, stream)
    torch.cuda.synchronize()
    
    # Benchmark with multiple runs
    print("  Benchmarking (1000 iterations)...")
    num_iters = 1000
    times = []
    
    for run in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(num_iters):
            lib.run(A_fp8.data_ptr(), B_fp8.data_ptr(), A_scale.data_ptr(),
                    B_scale.data_ptr(), C_method1.data_ptr(), M, N, K, None, stream)
        end.record()
        torch.cuda.synchronize()
        
        times.append(start.elapsed_time(end) / num_iters)
    
    # Calculate statistics
    time_method1_min = min(times)
    time_method1_max = max(times)
    time_method1_avg = sum(times) / len(times)
    time_method1_std = (sum((t - time_method1_avg) ** 2 for t in times) / len(times)) ** 0.5

    print(f"  ✓ Executed successfully!")
    print(f"  Performance: {time_method1_avg:.3f} ms/iter")
    
    method1_success = True
    
except Exception as e:
    print(f"  ✗ Method 1 failed: {e}")
    print(f"  Make sure you built the project: cd build && cmake .. && make -j")
    method1_success = False
    C_method1 = None


# ============================================================================
# METHOD 2: Kernel from Hugging Face Hub
# ============================================================================
print("\n" + "=" * 70)
print("[3] METHOD 2: Using kernel from Hugging Face Hub")
print("=" * 70)

try:
    from kernels import get_kernel
    
    # Download and load kernel
    print("  Downloading kernel from Hub...")
    gemm_hub = get_kernel("kernels-community/gemm")
    print("  ✓ Kernel loaded from Hub")
    
    # Allocate output
    C_method2 = torch.zeros(M, N, device=device, dtype=torch.bfloat16)
    
    # Warm-up
    print("  Warming up...")
    for _ in range(10):
        gemm_hub.gemm(A_fp8, B_fp8, A_scale, B_scale, C_method2)
    torch.cuda.synchronize()
    
    # Benchmark with multiple runs
    print("  Benchmarking (1000 iterations)...")
    num_iters = 1000
    times = []
    
    for run in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(num_iters):
            gemm_hub.gemm(A_fp8, B_fp8, A_scale, B_scale, C_method2)
        end.record()
        torch.cuda.synchronize()
        
        times.append(start.elapsed_time(end) / num_iters)
    
    # Calculate statistics
    time_method2_min = min(times)
    time_method2_max = max(times)
    time_method2_avg = sum(times) / len(times)
    time_method2_std = (sum((t - time_method2_avg) ** 2 for t in times) / len(times)) ** 0.5
        
    print(f"  ✓ Executed successfully!")
    print(f"  Performance: {time_method2_avg:.3f} ms/iter")
    
    method2_success = True
    
except ImportError:
    print(f"  ✗ 'kernels' package not found. Install with: pip install kernels")
    method2_success = False
    C_method2 = None
except Exception as e:
    print(f"  ✗ Method 2 failed: {e}")
    import traceback
    traceback.print_exc()
    method2_success = False
    C_method2 = None


# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "=" * 70)
print("[4] COMPARISON")
print("=" * 70)

if method1_success and method2_success:
    # Compare outputs
    max_diff = (C_method1 - C_method2).abs().max().item()
    print(f"\nResults match: {'✓' if max_diff < 1e-3 else '✗'} (max diff: {max_diff:.6f})")
    
    # Compare performance
    print(f"\nPerformance comparison:")
    print(f"  Method 1 (libgemm.so):  {time_method1_avg:.3f} ms/iter")
    print(f"  Method 2 (Hub kernel):  {time_method2_avg:.3f} ms/iter")
    
    speedup = time_method2_avg / time_method1_avg
    if speedup > 1.05:
        print(f"  → Method 1 is {speedup:.2f}x faster")
    elif speedup < 0.95:
        print(f"  → Method 2 is {1/speedup:.2f}x faster")
    else:
        print(f"  → Similar performance")

elif method1_success:
    print("\n  Only Method 1 succeeded")
    print(f"\nSample output (top-left 4x4):")
    print(C_method1[:4, :4])

elif method2_success:
    print("\n  Only Method 2 succeeded")
    print(f"\nSample output (top-left 4x4):")
    print(C_method2[:4, :4])

else:
    print("\n  ✗ Both methods failed")

print("\n" + "=" * 70)
print("Demo completed!")
print("=" * 70)
