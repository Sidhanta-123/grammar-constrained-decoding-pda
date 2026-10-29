from abc import ABC, abstractmethod

class BasePDA(ABC):
    """
    Formal Abstract Base Class representing a Pushdown Automaton:
    M = (Q, Sigma, Gamma, delta, q0, Z0, F)
    """
    def __init__(self, name: str, description: str, states: list, terminals: list,
                 stack_alphabet: list, start_state: str, accepting_states: list, productions: list):
        self.name = name
        self.description = description
        self.Q = states
        self.Sigma = terminals
        self.Gamma = stack_alphabet
        self.q0 = start_state
        self.F = accepting_states
        self.P = productions
        self.Z0 = "Z0"
        self.reset()

    def reset(self):
        self.state = self.q0
        self.stack = []
        self.history = []
        self.trace_log = []
        self.last_transition_edge = None

    def get_stack_str(self) -> str:
        return "".join(reversed(self.stack)) + self.Z0 if self.stack else self.Z0

    def record_trace(self, token: str, old_state: str, new_state: str, old_stack: str, stack_op: str):
        self.history.append(token)
        new_stack = self.get_stack_str()
        self.last_transition_edge = (old_state, new_state, token)
        self.trace_log.append({
            "Step": len(self.history),
            "Token (a)": token,
            "Transition (δ)": f"δ({old_state}, {token}) → {new_state}",
            "Stack Op": stack_op,
            "Formal ID (⊢_M)": f"({old_state}, {token}, {old_stack}) ⊢ ({new_state}, ε, {new_stack})",
            "Stack Top→Base": new_stack
        })

    def get_budget_safe_lookahead(self, tokens_remaining: int) -> list:
        """
        Switches to deterministic stack-draining lookahead when remaining token budget
        approaches the minimum tokens required to empty the stack and reach F.
        """
        if tokens_remaining <= self.min_tokens_to_close() + 4:
            closing = self.get_closing_lookahead()
            if closing:
                return closing
        return self.get_valid_lookahead()

    def verify_custom_sequence(self, tokens: list) -> dict:
        """
        Runs formal language membership verification (w in L(G)) on an arbitrary token list.
        """
        self.reset()
        steps = []
        for idx, tok in enumerate(tokens):
            valid_now = self.get_valid_lookahead()
            old_state = self.state
            old_stack = self.get_stack_str()
            if tok not in valid_now:
                steps.append({
                    "Step": idx + 1,
                    "Token": tok,
                    "Status": "❌ REJECTED (Invalid Transition)",
                    "State": old_state,
                    "Stack": old_stack,
                    "Expected One Of": ", ".join(valid_now) if valid_now else "[End of String]"
                })
                return {
                    "accepted": False,
                    "reason": f"Syntax error at token #{idx + 1} ('{tok}'). State '{old_state}' with stack '{old_stack}' only permits: {valid_now}.",
                    "steps": steps
                }
            self.transition(tok)
            steps.append({
                "Step": idx + 1,
                "Token": tok,
                "Status": "✅ VALID",
                "State": self.state,
                "Stack": self.get_stack_str(),
                "Expected One Of": ", ".join(valid_now)
            })

        if self.is_accepted():
            return {
                "accepted": True,
                "reason": f"String accepted! Reached final state '{self.state}' in F with empty stack ({self.Z0}).",
                "steps": steps
            }
        return {
            "accepted": False,
            "reason": f"Premature end of input. Automaton stopped in state '{self.state}' with stack '{self.get_stack_str()}' (requires {self.min_tokens_to_close()} more token(s) to close).",
            "steps": steps
        }

    @abstractmethod
    def get_valid_lookahead(self) -> list:
        pass

    @abstractmethod
    def get_closing_lookahead(self) -> list:
        pass

    @abstractmethod
    def min_tokens_to_close(self) -> int:
        pass

    @abstractmethod
    def transition(self, token: str) -> bool:
        pass

    @abstractmethod
    def is_accepted(self) -> bool:
        pass