import numpy as np

class Conv2D:
    """Cross-correlation of (N,C,H,W) input with F kernels of shape (C,k,k)."""

    def __init__(self, in_ch, out_ch, k, stride=1, pad=0, rng=None):
        rng = rng or np.random.default_rng()
        # He initialisation (good for ReLU networks)
        self.W = rng.standard_normal((out_ch, in_ch, k, k)) * np.sqrt(2.0 / (in_ch * k * k))
        self.b = np.zeros(out_ch)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.k, self.stride, self.pad = k, stride, pad


    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        k, s, p = self.k, self.stride, self.pad
        xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
        H_out = (H + 2 * p - k) // s + 1
        W_out = (W + 2 * p - k) // s + 1
        out = np.zeros((N, self.W.shape[0], H_out, W_out))
        for i in range(H_out):
            for j in range(W_out):
                patch = xp[:, :, i * s:i * s + k, j * s:j * s + k]            # (N, C, k, k)
                # dot product of every kernel with this patch -> (N, F)
                out[:, :, i, j] = np.tensordot(patch, self.W, axes=([1, 2, 3], [1, 2, 3])) + self.b
        return out

    def backward(self, dout):
        x = self.x
        N, C, H, W = x.shape
        k, s, p = self.k, self.stride, self.pad
        xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
        dxp = np.zeros_like(xp)
        self.dW = np.zeros_like(self.W)
        self.db = dout.sum(axis=(0, 2, 3))
        H_out, W_out = dout.shape[2:]
        for i in range(H_out):
            for j in range(W_out):
                patch = xp[:, :, i * s:i * s + k, j * s:j * s + k]            # (N, C, k, k)
                g = dout[:, :, i, j]                                          # (N, F)
                # gradient w.r.t. kernels: upstream gradient x patch
                self.dW += np.tensordot(g, patch, axes=([0], [0]))            # (F, C, k, k)
                # gradient w.r.t. input: upstream gradient x kernels
                dxp[:, :, i * s:i * s + k, j * s:j * s + k] += np.tensordot(g, self.W, axes=([1], [0]))
        return dxp[:, :, p:p + H, p:p + W]                                    # strip the padding

    def params(self):
        return [(self.W, self.dW), (self.b, self.db)]