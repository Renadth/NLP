"""Lab 2 starter: scaled dot-product attention and multi-head attention."""

import math
import torch


def attention(q, k, v, mask=None):
    # Q × Kᵀ
    scores = q @ k.transpose(-2, -1)

    # Scale by √d_k
    scores = scores / math.sqrt(k.size(-1))

    # Apply mask
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    # Softmax
    weights = torch.softmax(scores, dim=-1)

    # Weights × V
    output = weights @ v

    return output

    def forward(self, q, k, v, mask=None):

        # Project inputs
        Q = np.matmul(q, self.Wq)
        K = np.matmul(k, self.Wk)
        V = np.matmul(v, self.Wv)

        # Split into multiple heads
        batch_size = q.shape[0]

        Q = Q.reshape(batch_size, -1, self.num_heads, self.d_k)
        K = K.reshape(batch_size, -1, self.num_heads, self.d_k)
        V = V.reshape(batch_size, -1, self.num_heads, self.d_k)

        # Move heads before sequence length
        Q = np.transpose(Q, (0, 2, 1, 3))
        K = np.transpose(K, (0, 2, 1, 3))
        V = np.transpose(V, (0, 2, 1, 3))

        # Apply attention
        output, weights = attention(Q, K, V, mask)

        # Combine heads
        output = np.transpose(output, (0, 2, 1, 3))
        output = output.reshape(batch_size, -1, self.d_model)

        # Final output projection
        output = mha(x)
        return output, weights