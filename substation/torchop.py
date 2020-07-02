""" Wrapper for DaCe pytorch implementations. """

import torch
import torch.nn
from torch.nn.init import xavier_uniform_
from dace.codegen.compiler import CompiledSDFG
import numpy as np

from substation.dtypes import torch_dtype

class DaceMHA(torch.nn.Module):
    def __init__(self, N: int, P: int, H: int, csdfg_fwd: CompiledSDFG,
                 csdfg_bwd: CompiledSDFG):
        """
        Initialize a DaCe-implemented multi-head attention PyTorch module.
        :param P: Head dimension.
        :param H: Number of heads.
        :param csdfg_fwd: Compiled SDFG for forward pass.
        :param csdfg_bwd: Compiled SDFG for backpropagation.
        """
        super().__init__()

        class DaceMHAFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, Q, K, V, WQ, WK, WV, WO, scaler):
                B, SN, N = Q.shape
                _, SM, _ = K.shape
                output = torch.empty([B, SN, N], device=Q.device)
                csdfg_fwd(gK=K.contiguous(),
                          gQ=Q.contiguous(),
                          gV=V.contiguous(),
                          gOUT=output,
                          gWK=WK.contiguous(),
                          gWQ=WQ.contiguous(),
                          gWV=WV.contiguous(),
                          gWO=WO.contiguous(),
                          scaler=scaler,
                          B=B, P=P, H=H, N=H*P, SN=SN,
                          SM=SM)
                ctx.save_for_backward(Q, K, V, output)
                return output

            @staticmethod
            def backward(ctx, grads):
                # ctx.saved_tensors
                output = torch.empty_like(K, device='cuda')
                gK, gQ, gV, gWK, gWQ, gWV = [torch.empty(grads.shape,
                                                         grads.dtype)
                                             for _ in range(6)]
                csdfg_bwd(output=output, gK=gK, gQ=gQ, gV=gV, gWK=gWK, gWQ=gWQ,
                          gWV=gWV)
                return gK, gQ, gV, gWK, gWQ, gWV

        self.P = P
        self.H = H
        self.N = N

        self.op = DaceMHAFunction
        self.WK = torch.nn.Parameter(torch.Tensor(P, H, P*H))
        self.WQ = torch.nn.Parameter(torch.Tensor(P, H, P*H))
        self.WV = torch.nn.Parameter(torch.Tensor(P, H, P*H))
        self.WO = torch.nn.Parameter(torch.Tensor(P, H, P*H))
        self.scaler = torch.tensor([P ** -0.5], dtype=torch_dtype)

        # Initialize parameters
        xavier_uniform_(self.WK)
        xavier_uniform_(self.WQ)
        xavier_uniform_(self.WV)
        xavier_uniform_(self.WO)

    def forward(self, query, key, value):
        # Q [SN, B, N] -> [B, SN, N]
        # K/V [SM, B, N] -> [B, SM, N]
        # WQ/WK/WV [H * P, N] -> [H, P, N] -> [P, H, N]
        # WO [N, H * P] -> [N, H, P] -> [P, H, N]

        result = self.op.apply(query.transpose(0, 1),
                               key.transpose(0, 1),
                               value.transpose(0, 1),
                               self.WQ, self.WK, self.WV, self.WO, self.scaler)

        # OUT [B, SN, N] -> [SN, B, N]

        return result.transpose(0, 1)


if __name__ == '__main__':
    B = 2
    H = 16
    P = 64
    N = P*H
    SM, SN = 512, 512
    K, Q, V = (torch.randn([SM, B, N], requires_grad=True).cuda(),
               torch.randn([SN, B, N], requires_grad=True).cuda(),
               torch.randn([SM, B, N], requires_grad=True).cuda())
    from substation.attention import create_attn_forward_and_compile
    csdfg = create_attn_forward_and_compile(copy_to_device=False)

    op = DaceMHA(N, P, H, csdfg, None).cuda()
    res_dace = op.forward(Q, K, V)

    attn = torch.nn.MultiheadAttention(N, H, bias=False).cuda()
    attn.in_proj_weight.data = torch.cat((
        op.WQ.transpose(0, 1).reshape(N, N),
        op.WK.transpose(0, 1).reshape(N, N),
        op.WV.transpose(0, 1).reshape(N, N)), dim=0)
    attn.out_proj.weight.data = op.WO.transpose(0, 2).reshape(N, N)
    attn.train()
    res_torch, _ = attn.forward(Q, K, V, need_weights=False)

    result = torch.allclose(res_torch, res_dace, rtol=1e-04, atol=1e-06)
    print('Result:', result)
    if not result:
        exit(1)
