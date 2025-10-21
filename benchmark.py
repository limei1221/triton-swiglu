import torch
import triton
import triton.testing
import torch.nn.functional as F
import matplotlib.pyplot as plt

from triton_swiglu_fused import SwiGLU
# from triton_swiglu_fused_bak import SwiGLU
# from triton_swiglu_fused_bak2 import SwiGLU


swiglu = SwiGLU.apply

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M'],
        x_vals=[128, 512, 1024, 2048, 4096, 8192],
        line_arg='provider',
        line_vals=['torch', 'triton'],
        line_names=['Torch', 'Triton'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='ms',
        plot_name='SwiGLU_Performance',
        args={'K': 1024, 'N': 4096, 'dtype': torch.float16, 'device': 'cuda'},
    )
)
def benchmark(M, K, N, dtype, device, provider, mode='forward'):
    x = torch.randn(M, K, device=device, dtype=dtype)
    w_proj = torch.randn(K, N, device=device, dtype=dtype)
    v_proj = torch.randn(K, N, device=device, dtype=dtype)

    dy = .1 * torch.randn(M, N, device=device, dtype=dtype)
    x.requires_grad_(True)
    w_proj.requires_grad_(True)
    v_proj.requires_grad_(True)

    quantiles = [0.5, 0.2, 0.8]

    def y_fwd():

        if provider == "triton":
            return swiglu(x, w_proj, v_proj)  # noqa: F811, E704

        if provider == "torch":
            return F.silu(x @ w_proj) * (x @ v_proj)  # noqa: F811, E704

     # forward pass
    if mode == 'forward':
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
    # backward pass
    if mode == 'backward':
        y = y_fwd()
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: y.backward(dy, retain_graph=True), quantiles=quantiles,
                                                     grad_to_none=[x], rep=500)

    return ms, min_ms, max_ms


if __name__ == "__main__":
    benchmark.run(mode='forward', print_data=True, show_plots=True)
    plt.savefig('SwiGLU_forward_Performance.png', dpi=300, bbox_inches='tight')
    benchmark.run(mode='backward', print_data=True, show_plots=True)
    plt.savefig('SwiGLU_backward_Performance.png', dpi=300, bbox_inches='tight')
