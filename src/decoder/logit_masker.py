import numpy as np

class LogitMasker:
    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        finite_mask = np.isfinite(logits)
        if not np.any(finite_mask):
            return np.zeros_like(logits)
        max_val = np.max(logits[finite_mask])
        exp_logits = np.where(finite_mask, np.exp(logits - max_val), 0.0)
        total = exp_logits.sum()
        return exp_logits / total if total > 0 else np.zeros_like(logits)

    @staticmethod
    def shannon_entropy(probs: np.ndarray) -> float:
        """Computes Shannon Entropy H(P) = -sum P(x) log2 P(x) in bits."""
        nonzero = probs[probs > 1e-12]
        if len(nonzero) == 0:
            return 0.0
        return float(np.round(-np.sum(nonzero * np.log2(nonzero)), 3))

    @classmethod
    def apply_pda_mask(cls, vocabulary: list, raw_logits: np.ndarray, valid_tokens: list, top_k: int = 0) -> dict:
        raw_probs = cls.softmax(raw_logits)
        raw_entropy = cls.shannon_entropy(raw_probs)
        unconstrained_idx = int(np.argmax(raw_probs))

        masked_logits = raw_logits.copy()

        # 1. Apply PDA Hard Grammar Constraint Mask M_t
        for i, token in enumerate(vocabulary):
            if token not in valid_tokens:
                masked_logits[i] = -np.inf

        # 2. Optional Top-k Filtering among valid grammatical candidates
        if top_k > 0:
            valid_indices = [i for i in range(len(vocabulary)) if np.isfinite(masked_logits[i])]
            if len(valid_indices) > top_k:
                sorted_valid = sorted(valid_indices, key=lambda idx: masked_logits[idx], reverse=True)
                keep_set = set(sorted_valid[:top_k])
                for idx in valid_indices:
                    if idx not in keep_set:
                        masked_logits[idx] = -np.inf

        constrained_probs = cls.softmax(masked_logits)
        constrained_entropy = cls.shannon_entropy(constrained_probs)
        constrained_idx = int(np.argmax(constrained_probs))

        return {
            "tokens": vocabulary,
            "raw_logits": raw_logits,
            "masked_logits": masked_logits,
            "raw_probs": np.round(raw_probs * 100, 1),
            "constrained_probs": np.round(constrained_probs * 100, 1),
            "raw_entropy": raw_entropy,
            "constrained_entropy": constrained_entropy,
            "valid_tokens": valid_tokens,
            "unconstrained_pick": vocabulary[unconstrained_idx],
            "constrained_pick": vocabulary[constrained_idx],
            "intercepted": vocabulary[unconstrained_idx] != vocabulary[constrained_idx]
        }