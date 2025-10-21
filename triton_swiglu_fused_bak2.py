import torch
import torch.nn.functional as F

import triton
import triton.language as tl


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'GROUP_SIZE_K': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'GROUP_SIZE_K': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'GROUP_SIZE_K': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'GROUP_SIZE_K': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'GROUP_SIZE_K': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'GROUP_SIZE_K': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'GROUP_SIZE_K': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'GROUP_SIZE_K': 8}, num_stages=5,
                      num_warps=2),
    ]


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def _swiglu_forward_fused(
        x_ptr, w_ptr, v_ptr, y_ptr, a_ptr, b_ptr,
        M, N, K,
        stride_xm, stride_xk,
        stride_wk, stride_wn,
        stride_vk, stride_vn,
        stride_ym, stride_yn,
        stride_am, stride_an,
        stride_bm, stride_bn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        GROUP_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    # grouped scheduling
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
    v_ptrs = v_ptr + (offs_k[:, None] * stride_vk + offs_n[None, :] * stride_vn)

    # accumulators
    a = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    b = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask_k = offs_k < K - k * BLOCK_SIZE_K

        x_block = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w_block = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        v_block = tl.load(v_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        a += tl.dot(x_block, w_block, out_dtype=tl.float32)
        b += tl.dot(x_block, v_block, out_dtype=tl.float32)

        # advance pointers
        x_ptrs += BLOCK_SIZE_K * stride_xk
        w_ptrs += BLOCK_SIZE_K * stride_wk
        v_ptrs += BLOCK_SIZE_K * stride_vk

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_an)
    b_ptrs = b_ptr + (offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn)
    tl.store(a_ptrs, a.to(tl.float16), mask=mask_m[:, None] & mask_n[None, :])
    tl.store(b_ptrs, b.to(tl.float16), mask=mask_m[:, None] & mask_n[None, :])

    silu_a = a * tl.sigmoid(a)
    y = (silu_a * b).to(tl.float16)

    y_ptrs = y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    tl.store(y_ptrs, y, mask=mask_m[:, None] & mask_n[None, :])


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def _swiglu_backward_dw_dv(
        x_ptr, a_ptr, b_ptr, dy_ptr,
        dw_ptr, dv_ptr,
        M, N, K,
        stride_xm, stride_xk,
        stride_am, stride_an,
        stride_bm, stride_bn,
        stride_dym, stride_dyn,
        stride_dwk, stride_dwn,
        stride_dvk, stride_dvn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        GROUP_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    # grouped scheduling
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_K * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_k = group_id * GROUP_SIZE_K
    group_size_k = min(num_pid_k - first_pid_k, GROUP_SIZE_K)
    pid_k = first_pid_k + ((pid % num_pid_in_group) % group_size_k)
    pid_n = (pid % num_pid_in_group) // group_size_k

    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_m = tl.arange(0, BLOCK_SIZE_M)

    mask_k = offs_k < K
    mask_n = offs_n < N

    dw_acc = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
    dv_acc = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)

    for m0 in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
        cur_m = m0 * BLOCK_SIZE_M + offs_m
        mask_m = cur_m < M

        x_ptrs = x_ptr + (cur_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        a_ptrs = a_ptr + (cur_m[:, None] * stride_am + offs_n[None, :] * stride_an)
        b_ptrs = b_ptr + (cur_m[:, None] * stride_bm + offs_n[None, :] * stride_bn)
        dy_ptrs = dy_ptr + (cur_m[:, None] * stride_dym + offs_n[None, :] * stride_dyn)

        x_block = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        a_block = tl.load(a_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
        b_block = tl.load(b_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
        dy_block = tl.load(dy_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

        sig_a = tl.sigmoid(a_block)
        silu_a = a_block * sig_a
        d_silu = sig_a * (1 + a_block * (1 - sig_a))

        da_block_fp32 = dy_block * b_block * d_silu
        db_block_fp32 = dy_block * silu_a

        # cast to match multiplicands (use fp16 inputs, fp32 accumulation)
        da_block = da_block_fp32.to(x_block.dtype)
        db_block = db_block_fp32.to(x_block.dtype)

        # dw += x^T @ da, dv += x^T @ db
        dw_acc += tl.dot(tl.trans(x_block), da_block, out_dtype=tl.float32)
        dv_acc += tl.dot(tl.trans(x_block), db_block, out_dtype=tl.float32)

    dw_ptrs = dw_ptr + (offs_k[:, None] * stride_dwk + offs_n[None, :] * stride_dwn)
    dv_ptrs = dv_ptr + (offs_k[:, None] * stride_dvk + offs_n[None, :] * stride_dvn)
    tl.store(dw_ptrs, dw_acc.to(tl.float16), mask=mask_k[:, None] & mask_n[None, :])
    tl.store(dv_ptrs, dv_acc.to(tl.float16), mask=mask_k[:, None] & mask_n[None, :])


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def _swiglu_backward_dx(
        a_ptr, b_ptr, dy_ptr,
        w_ptr, v_ptr, dx_ptr,
        M, N, K,
        stride_am, stride_an,
        stride_bm, stride_bn,
        stride_dym, stride_dyn,
        stride_wk, stride_wn,
        stride_vk, stride_vn,
        stride_dxm, stride_dxk,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        GROUP_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    # grouped scheduling
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_k
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_k = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    mask_m = offs_m < M
    mask_k = offs_k < K

    dx_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for n0 in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        cur_n = n0 * BLOCK_SIZE_N + offs_n
        mask_n = cur_n < N

        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + cur_n[None, :] * stride_an)
        b_ptrs = b_ptr + (offs_m[:, None] * stride_bm + cur_n[None, :] * stride_bn)
        dy_ptrs = dy_ptr + (offs_m[:, None] * stride_dym + cur_n[None, :] * stride_dyn)

        # load weights as transposed tiles: (N_blk, K_blk)
        w_t_ptrs = w_ptr + (cur_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)
        v_t_ptrs = v_ptr + (cur_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)

        a_block = tl.load(a_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
        b_block = tl.load(b_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
        dy_block = tl.load(dy_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

        w_t_block = tl.load(w_t_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        v_t_block = tl.load(v_t_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)

        sig_a = tl.sigmoid(a_block)
        silu_a = a_block * sig_a
        d_silu = sig_a * (1 + a_block * (1 - sig_a))

        da_block_fp32 = dy_block * b_block * d_silu
        db_block_fp32 = dy_block * silu_a

        # cast to match multiplicands (use fp16 inputs, fp32 accumulation)
        da_block = da_block_fp32.to(w_t_block.dtype)
        db_block = db_block_fp32.to(v_t_block.dtype)

        # dx += da @ w^T + db @ v^T
        dx_acc += tl.dot(da_block, w_t_block, out_dtype=tl.float32)
        dx_acc += tl.dot(db_block, v_t_block, out_dtype=tl.float32)

    dx_ptrs = dx_ptr + (offs_m[:, None] * stride_dxm + offs_k[None, :] * stride_dxk)
    tl.store(dx_ptrs, dx_acc.to(tl.float16), mask=mask_m[:, None] & mask_k[None, :])

class SwiGLU(torch.autograd.Function):
    """Triton-fused SwiGLU with custom backward.
    Forward: y = silu(x @ w) * (x @ v)
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, v: torch.Tensor):
        M, K = x.shape
        _, N = w.shape

        y = torch.empty((M, N), device=x.device, dtype=x.dtype)
        a = torch.empty((M, N), device=x.device, dtype=x.dtype)
        b = torch.empty((M, N), device=x.device, dtype=x.dtype)

        x = x.contiguous()
        w = w.contiguous()
        v = v.contiguous()

        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        )

        _swiglu_forward_fused[grid](
            x, w, v, y, a, b,
            M, N, K,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            v.stride(0), v.stride(1),
            y.stride(0), y.stride(1),
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
        )

        ctx.save_for_backward(x, w, v, a, b)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, w, v, a, b = ctx.saved_tensors

        M, K = x.shape
        _, N = w.shape

        x = x.contiguous()
        w = w.contiguous()
        v = v.contiguous()
        a = a.contiguous()
        b = b.contiguous()
        dy = dy.contiguous()

        # allocate outputs
        dx = torch.empty((M, K), device=x.device, dtype=x.dtype)
        dw = torch.empty((K, N), device=w.device, dtype=w.dtype)
        dv = torch.empty((K, N), device=v.device, dtype=v.dtype)

        grid_dw_dv = lambda META: (
            triton.cdiv(K, META['BLOCK_SIZE_K']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        )

        _swiglu_backward_dw_dv[grid_dw_dv](
            x, a, b, dy,
            dw, dv,
            M, N, K,
            x.stride(0), x.stride(1),
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            dy.stride(0), dy.stride(1),
            dw.stride(0), dw.stride(1),
            dv.stride(0), dv.stride(1),
        )

        grid_dx = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(K, META['BLOCK_SIZE_K']),
        )

        _swiglu_backward_dx[grid_dx](
            a, b, dy,
            w, v, dx,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            dy.stride(0), dy.stride(1),
            w.stride(0), w.stride(1),
            v.stride(0), v.stride(1),
            dx.stride(0), dx.stride(1),
        )

        return dx.to(x.dtype), dw.to(w.dtype), dv.to(v.dtype)


swiglu = SwiGLU.apply


def test_swiglu(M, K, N, dtype=torch.float16, device='cuda'):
    x = torch.rand((M, K), device=device, dtype=dtype) - 0.5
    w = torch.rand((K, N), device=device, dtype=dtype) - 0.5
    v = torch.rand((K, N), device=device, dtype=dtype) - 0.5

    dy = .1 * torch.rand((M, N), device=device, dtype=dtype)

    x.requires_grad_(True)
    w.requires_grad_(True)
    v.requires_grad_(True)

    # forward pass
    y_tri = swiglu(x, w, v)
    y_ref = F.silu(x @ w) * (x @ v)

    # assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0), "forward mismatch"
    assert torch.allclose(y_tri, y_ref, atol=0.02, rtol=0.01), "forward mismatch"

    # backward pass (triton)
    y_tri.backward(dy, retain_graph=True)
    dx_tri, dw_tri, dv_tri = [_.grad.clone() for _ in [x, w, v]]
    x.grad, w.grad, v.grad = None, None, None
    # backward pass (torch)
    y_ref.backward(dy, retain_graph=True)
    dx_ref, dw_ref, dv_ref = [_.grad.clone() for _ in [x, w, v]]

    assert torch.allclose(dx_tri, dx_ref, atol=0.02, rtol=0.01), "dx mismatch"
    assert torch.allclose(dw_tri, dw_ref, atol=0.02, rtol=0.01), "dw mismatch"
    assert torch.allclose(dv_tri, dv_ref, atol=0.02, rtol=0.01), "dv mismatch"


if __name__ == "__main__":
    M, K, N = 2048, 1024, 4096
    dtype = torch.float16
    device = 'cuda'
    test_swiglu(M, K, N, dtype, device)
