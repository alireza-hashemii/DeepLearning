import numpy as np


class Conv2D:
    def __init__(self, c_in, c_out, k , rng=None ,  stride=1 , pad=0):
        rng = rng or np.random.default_rng()
        self.x_ = c_in
        self.k = k 
        self.stride = stride
        self.pad = pad
        self.W = rng.standard_normal((k, k, c_in, c_out)) * np.sqrt(2.0 / (c_in * k * k))
        self.b = np.zeros(c_out)


    def forward(self, x):
        xp = np.pad(x, ((self.pad, self.pad),(self.pad, self.pad),(0, 0),(0, 0)))
        H, W , C, B = xp.shape
        H_out = (H + 2 * self.pad - self.k) // self.stride + 1
        W_out = (W + 2 * self.pad - self.k) // self.stride + 1
        out = np.zeros((H_out, W_out, C, B))
        k , s = self.k , self.stride

        for i in range(H_out):
            for j in range(W_out):
                patch = xp[i*s: i*s + k, j*s: j*s+k , :, :]
                out[i , j, :, :] = np.tensordot(patch, self.W,axes=([0,1,2], [0,1,2])) + self.b
        return out




    def backward():
        pass


