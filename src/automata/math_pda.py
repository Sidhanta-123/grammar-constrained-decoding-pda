from src.automata.base_pda import BasePDA

class MathPDAEngine(BasePDA):
    def __init__(self):
        super().__init__(
            name="Nested Math Expressions",
            description="Enforces balanced (), [], {} delimiters and binary operator placement.",
            states=["q_operand", "q_operator"],
            terminals=["(", ")", "[", "]", "{", "}", "x", "y", "+", "*"],
            stack_alphabet=["(", "[", "{", "Z0"],
            start_state="q_operand",
            accepting_states=["q_operator"],
            productions=[
                "E -> E + T | E * T | T",
                "T -> ( E ) | [ E ] | { E }",
                "T -> x | y"
            ]
        )
        self.matching_close = {"(": ")", "[": "]", "{": "}"}

    def get_valid_lookahead(self) -> list:
        if self.state == "q_operand":
            return ["x", "y", "(", "[", "{"]
        elif self.state == "q_operator":
            valid = ["+", "*"]
            if self.stack:
                valid.append(self.matching_close[self.stack[-1]])
            return valid
        return []

    def min_tokens_to_close(self) -> int:
        if self.state == "q_operand":
            return 1 + len(self.stack)
        return len(self.stack) if self.history else 1

    def get_closing_lookahead(self) -> list:
        if self.state == "q_operand":
            return ["x", "y"]
        elif self.state == "q_operator":
            if self.stack:
                return [self.matching_close[self.stack[-1]]]
            return []
        return []

    def transition(self, token: str) -> bool:
        if token not in self.get_valid_lookahead():
            return False

        old_state = self.state
        old_stack = self.get_stack_str()
        stack_op = "NO_OP"

        if token in ["x", "y"]:
            self.state = "q_operator"
        elif token in ["(", "[", "{"]:
            self.stack.append(token)
            self.state = "q_operand"
            stack_op = f"PUSH({token})"
        elif token in [")", "]", "}"]:
            popped = self.stack.pop()
            self.state = "q_operator"
            stack_op = f"POP({popped})"
        elif token in ["+", "*"]:
            self.state = "q_operand"

        self.record_trace(token, old_state, self.state, old_stack, stack_op)
        return True

    def is_accepted(self) -> bool:
        return len(self.stack) == 0 and self.state == "q_operator" and len(self.history) > 0