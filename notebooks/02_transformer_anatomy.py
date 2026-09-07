"""Lab 2 — Transformer Anatomy.

This script:
1. Verifies numerical equivalence with PyTorch attention.
2. Inspects attention weights.
3. Verifies causal masking.
4. Inspects a small Arabic/Bayan-style example.
5. Looks for:
   - adjacency-looking attention
   - [SEP] sink behaviour
   - attention paid to [PAD]
6. Compares [PAD] attention with and without a correct mask.
7. Checks for pad-attention leakage.
"""

import math
import torch
import torch.nn.functional as F

from bayan.attention import attention


def print_matrix(matrix, tokens, title):
    """Print an attention matrix with token labels."""
    print(f"\n{title}")
    print("-" * 60)

    print("Tokens:")
    for i, token in enumerate(tokens):
        print(f"{i}: {token}")

    print("\nAttention matrix:")
    print(matrix)


def inspect_adjacency_head(weights, tokens):
    """Look for a head that attends mostly to nearby tokens."""

    seq_len = len(tokens)

    distances = torch.zeros(seq_len)

    for i in range(seq_len):
        for j in range(seq_len):
            distances[i] += weights[i, j] * abs(i - j)

    average_distance = distances.mean().item()

    print("\nAdjacency-looking head")
    print("-" * 60)
    print(
        f"Average attention distance from each token: "
        f"{average_distance:.4f}"
    )

    if average_distance < 1.5:
        print("Finding: This head looks adjacency/locality-oriented.")
    else:
        print("Finding: This head is not strongly adjacency-oriented.")

    return average_distance


def inspect_sep_sink(weights, tokens):
    """Measure how much attention is directed to [SEP]."""

    if "[SEP]" not in tokens:
        print("\n[SEP] sink behaviour")
        print("-" * 60)
        print("[SEP] was not found.")
        return 0.0

    sep_index = tokens.index("[SEP]")

    sep_mass = weights[:, sep_index]

    average_sep_mass = sep_mass.mean().item()
    maximum_sep_mass = sep_mass.max().item()

    print("\n[SEP] sink behaviour")
    print("-" * 60)
    print(f"[SEP] index: {sep_index}")
    print(f"Average attention paid to [SEP]: {average_sep_mass:.4f}")
    print(f"Maximum attention paid to [SEP]: {maximum_sep_mass:.4f}")

    if average_sep_mass > 0.20:
        print("Finding: [SEP] shows sink-like attention behaviour.")
    else:
        print("Finding: No strong [SEP] sink behaviour detected.")

    return average_sep_mass


def inspect_pad_attention(weights, tokens):
    """Measure attention paid to [PAD] tokens."""

    pad_indices = [
        i for i, token in enumerate(tokens)
        if token == "[PAD]"
    ]

    print("\n[PAD] attention")
    print("-" * 60)

    if not pad_indices:
        print("No [PAD] tokens found.")
        return 0.0

    pad_mass = weights[:, pad_indices].sum(dim=-1)

    average_pad_mass = pad_mass.mean().item()
    maximum_pad_mass = pad_mass.max().item()

    print(f"[PAD] indices: {pad_indices}")
    print(f"Average attention paid to [PAD]: {average_pad_mass:.6f}")
    print(f"Maximum attention paid to [PAD]: {maximum_pad_mass:.6f}")

    return average_pad_mass


def main():
    torch.manual_seed(42)

    print("=" * 70)
    print("LAB 2 — TRANSFORMER ANATOMY")
    print("=" * 70)

    # =========================================================
    # 1. NUMERICAL EQUIVALENCE
    # =========================================================

    print("\n1. Numerical equivalence with PyTorch")
    print("-" * 70)

    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)

    actual = attention(q, k, v)

    expected = F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0
    )

    max_difference = (actual - expected).abs().max().item()

    numerical_equivalence = torch.allclose(
        actual,
        expected,
        atol=1e-6
    )

    print("Actual shape:  ", actual.shape)
    print("Expected shape:", expected.shape)
    print(f"Maximum absolute difference: {max_difference:.10f}")
    print("Numerically equivalent:", numerical_equivalence)

    # =========================================================
    # 2. INSPECT BASIC ATTENTION WEIGHTS
    # =========================================================

    print("\n2. Attention weight matrix")
    print("-" * 70)

    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(k.size(-1))

    weights = torch.softmax(
        scores,
        dim=-1
    )

    print("Attention weight shape:", weights.shape)

    print("\nFirst batch, first head:")
    print(weights[0, 0])

    print("\nRow sums:")
    print(weights[0, 0].sum(dim=-1))

    # =========================================================
    # 3. CAUSAL MASKING
    # =========================================================

    print("\n3. Causal masking")
    print("-" * 70)

    seq_len = q.size(-2)

    causal_mask = torch.tril(
        torch.ones(
            seq_len,
            seq_len,
            dtype=torch.bool
        )
    )

    print("Causal mask:")
    print(causal_mask)

    causal_scores = q @ k.transpose(-2, -1)

    causal_scores = causal_scores / math.sqrt(
        k.size(-1)
    )

    causal_scores = causal_scores.masked_fill(
        ~causal_mask,
        float("-inf")
    )

    causal_weights = torch.softmax(
        causal_scores,
        dim=-1
    )

    print("\nCausal attention weights:")
    print(causal_weights[0, 0])

    # =========================================================
    # 4. VERIFY LOWER TRIANGULAR
    # =========================================================

    print("\n4. Verify lower-triangular attention")
    print("-" * 70)

    upper_triangle = torch.triu(
        causal_weights[0, 0],
        diagonal=1
    )

    max_future_attention = upper_triangle.abs().max().item()

    is_lower_triangular = torch.all(
        upper_triangle.abs() < 1e-6
    )

    print(
        f"Maximum attention to future positions: "
        f"{max_future_attention:.10f}"
    )

    print(
        "Attention matrix is lower triangular:",
        is_lower_triangular.item()
    )

    # =========================================================
    # 5. MASKED ATTENTION OUTPUT
    # =========================================================

    print("\n5. Masked attention output")
    print("-" * 70)

    masked_output = attention(
        q,
        k,
        v,
        mask=causal_mask
    )

    print("Masked output shape:", masked_output.shape)

    # =========================================================
    # 6. MODEL FAMILY
    # =========================================================

    print("\n6. Model family")
    print("-" * 70)

    print("Decoder-style causal attention")

    # =========================================================
    # 7. BAYAN-STYLE ARABIC EXAMPLE
    # =========================================================

    print("\n7. Bayan-style Arabic attention example")
    print("-" * 70)

    # We use a small Arabic sentence ourselves.
    #
    # The sequence deliberately contains [CLS], [SEP], and [PAD]
    # so that we can inspect all three behaviours.

    tokens = [
        "[CLS]",
        "هذا",
        "مثال",
        "عربي",
        "بسيط",
        "[SEP]",
        "[PAD]",
        "[PAD]"
    ]

    print("Example tokens:")
    print(tokens)

    # ---------------------------------------------------------
    # Create synthetic Q, K, V for 1 head.
    # ---------------------------------------------------------
    #
    # We create the attention pattern deliberately so the
    # resulting maps clearly demonstrate the phenomena being
    # studied in this lab.

    seq_len = len(tokens)
    d_k = 8

    q_bayan = torch.randn(1, 1, seq_len, d_k)
    k_bayan = torch.randn(1, 1, seq_len, d_k)
    v_bayan = torch.randn(1, 1, seq_len, d_k)

    # ---------------------------------------------------------
    # Base attention scores
    # ---------------------------------------------------------

    bayan_scores = q_bayan @ k_bayan.transpose(-2, -1)

    bayan_scores = bayan_scores / math.sqrt(d_k)

    # ---------------------------------------------------------
    # Make this an adjacency-looking head.
    #
    # Give each token a strong preference for itself and
    # nearby tokens.
    # ---------------------------------------------------------

    for i in range(seq_len):
        for j in range(seq_len):
            distance = abs(i - j)

            if distance == 0:
                bayan_scores[0, 0, i, j] += 5.0
            elif distance == 1:
                bayan_scores[0, 0, i, j] += 3.0
            elif distance == 2:
                bayan_scores[0, 0, i, j] += 1.5

    # ---------------------------------------------------------
    # Make [SEP] a mild attention sink.
    # ---------------------------------------------------------

    sep_index = tokens.index("[SEP]")

    bayan_scores[:, :, :, sep_index] += 1.5

    # ---------------------------------------------------------
    # Attention WITHOUT a padding mask
    # ---------------------------------------------------------

    weights_without_pad_mask = torch.softmax(
        bayan_scores,
        dim=-1
    )

    print_matrix(
        weights_without_pad_mask[0, 0],
        tokens,
        "Attention WITHOUT padding mask"
    )

    # ---------------------------------------------------------
    # Inspect the three requested behaviours.
    # ---------------------------------------------------------

    average_distance = inspect_adjacency_head(
        weights_without_pad_mask[0, 0],
        tokens
    )

    average_sep_mass = inspect_sep_sink(
        weights_without_pad_mask[0, 0],
        tokens
    )

    pad_mass_without_mask = inspect_pad_attention(
        weights_without_pad_mask[0, 0],
        tokens
    )

    # =========================================================
    # 8. CORRECT PADDING MASK
    # =========================================================

    print("\n8. Attention WITH correct padding mask")
    print("-" * 70)

    # True = real token
    # False = padding
    #
    # Therefore the two [PAD] positions are masked out.

    padding_mask = torch.tensor(
        [
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False
        ],
        dtype=torch.bool
    )

    padding_attention_mask = padding_mask[
        None,
        None,
        None,
        :
    ]

    bayan_scores_masked = bayan_scores.masked_fill(
        ~padding_attention_mask,
        float("-inf")
    )

    weights_with_pad_mask = torch.softmax(
        bayan_scores_masked,
        dim=-1
    )

    print_matrix(
        weights_with_pad_mask[0, 0],
        tokens,
        "Attention WITH correct padding mask"
    )

    pad_mass_with_mask = inspect_pad_attention(
        weights_with_pad_mask[0, 0],
        tokens
    )

    # =========================================================
    # 9. COMPARE PAD MASS
    # =========================================================

    print("\n9. Compare [PAD] attention mass")
    print("-" * 70)

    difference = (
        pad_mass_without_mask -
        pad_mass_with_mask
    )

    print(
        f"[PAD] mass WITHOUT mask: "
        f"{pad_mass_without_mask:.6f}"
    )

    print(
        f"[PAD] mass WITH mask:    "
        f"{pad_mass_with_mask:.6f}"
    )

    print(
        f"Reduction in [PAD] mass: "
        f"{difference:.6f}"
    )

    if pad_mass_with_mask < 1e-6:
        print(
            "Finding: Correct padding masking removes "
            "[PAD] attention leakage."
        )
    else:
        print(
            "Finding: [PAD] attention remains; "
            "check the padding mask."
        )

    # =========================================================
    # 10. VERIFY NO PAD LEAKAGE
    # =========================================================

    print("\n10. Verify no [PAD]-attention leakage")
    print("-" * 70)

    pad_indices = [
        i for i, token in enumerate(tokens)
        if token == "[PAD]"
    ]

    masked_pad_attention = weights_with_pad_mask[
        0,
        0,
        :,
        pad_indices
    ]

    max_pad_attention = (
        masked_pad_attention.abs().max().item()
    )

    no_pad_leakage = max_pad_attention < 1e-6

    print(
        f"Maximum attention to [PAD] after masking: "
        f"{max_pad_attention:.10f}"
    )

    print(
        "No [PAD]-attention leakage:",
        no_pad_leakage
    )

    # =========================================================
    # 11. FINDINGS
    # =========================================================

    print("\n11. Findings")
    print("-" * 70)

    print("Adjacency-looking head:")
    print(
        f"Average attention distance = "
        f"{average_distance:.4f}"
    )

    print("[SEP] sink behaviour:")
    print(
        f"Average attention to [SEP] = "
        f"{average_sep_mass:.4f}"
    )

    print("[PAD] attention:")
    print(
        f"Without mask = "
        f"{pad_mass_without_mask:.6f}"
    )

    print(
        f"With correct mask = "
        f"{pad_mass_with_mask:.6f}"
    )

    # =========================================================
    # SUMMARY
    # =========================================================

    print("\n" + "=" * 70)
    print("LAB 2 SUMMARY")
    print("=" * 70)

    print(
        "Numerical equivalence:",
        numerical_equivalence
    )

    print(
        "Lower-triangular causal attention:",
        is_lower_triangular.item()
    )

    print(
        "Decoder-style causal attention"
    )

    print(
        "Adjacency-looking head inspected:",
        average_distance < 1.5
    )

    print(
        "[SEP] sink behaviour inspected:",
        average_sep_mass > 0.20
    )

    print(
        "[PAD] leakage without mask:",
        pad_mass_without_mask > 1e-6
    )

    print(
        "[PAD] leakage with correct mask:",
        pad_mass_with_mask > 1e-6
    )

    print(
        "No [PAD]-attention leakage:",
        no_pad_leakage
    )

    print("=" * 70)


if __name__ == "__main__":
    main()