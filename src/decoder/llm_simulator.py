import numpy as np

class LLMSimulator:
    @staticmethod
    def generate_logits(vocabulary: list, valid_tokens: list, hallucination_bias: bool = True, temperature: float = 1.0) -> np.ndarray:
        """
        Simulates raw pre-softmax LLM logits z_t in R^|V| with controllable temperature T
        and adversarial syntax hallucination spikes.
        """
        raw_logits = np.random.uniform(-1.2, 3.8, len(vocabulary))
        invalid_indices = [i for i, t in enumerate(vocabulary) if t not in valid_tokens]

        if hallucination_bias and invalid_indices:
            bad_idx = np.random.choice(invalid_indices)
            raw_logits[bad_idx] = np.random.uniform(5.0, 8.2)

        scaled_logits = raw_logits / max(temperature, 0.05)
        return np.round(scaled_logits, 2)