from src.automata.base_pda import BasePDA

class JSONPDAEngine(BasePDA):
    def __init__(self):
        super().__init__(
            name="Structured JSON Schema",
            description="Enforces strict key-value pairs, nested objects {}, and arrays [].",
            states=["q_start", "q_obj_key_or_close", "q_obj_key", "q_colon", "q_val", "q_arr_val_or_close", "q_after_val"],
            terminals=["{", "}", "[", "]", '"key"', ":", '"val"', "123", ","],
            stack_alphabet=["{", "[", "Z0"],
            start_state="q_start",
            accepting_states=["q_after_val"],
            productions=[
                "Value -> Object | Array | '\"val\"' | '123'",
                "Object -> '{' '}' | '{' Members '}'",
                "Members -> '\"key\"' ':' Value | Members ',' '\"key\"' ':' Value",
                "Array -> '[' ']' | '[' Elements ']'",
                "Elements -> Value | Elements ',' Value"
            ]
        )

    def get_valid_lookahead(self) -> list:
        if self.state == "q_start":
            return ["{", "["]
        elif self.state == "q_obj_key_or_close":
            return ['"key"', "}"]
        elif self.state == "q_obj_key":
            return ['"key"']
        elif self.state == "q_colon":
            return [":"]
        elif self.state == "q_val":
            return ['"val"', "123", "{", "["]
        elif self.state == "q_arr_val_or_close":
            return ['"val"', "123", "{", "[", "]"]
        elif self.state == "q_after_val":
            if not self.stack:
                return []
            top = self.stack[-1]
            return [",", "}"] if top == "{" else [",", "]"]
        return []

    def min_tokens_to_close(self) -> int:
        if self.state == "q_start":
            return 2
        elif self.state in ["q_obj_key_or_close", "q_arr_val_or_close", "q_after_val"]:
            return len(self.stack)
        elif self.state == "q_val":
            return 1 + len(self.stack)
        elif self.state == "q_colon":
            return 2 + len(self.stack)
        elif self.state == "q_obj_key":
            return 3 + len(self.stack)
        return len(self.stack)

    def get_closing_lookahead(self) -> list:
        if self.state == "q_start":
            return ["{", "["]
        elif self.state == "q_obj_key_or_close":
            return ["}"]
        elif self.state == "q_arr_val_or_close":
            return ["]"]
        elif self.state == "q_obj_key":
            return ['"key"']
        elif self.state == "q_colon":
            return [":"]
        elif self.state == "q_val":
            return ['"val"', "123"]
        elif self.state == "q_after_val":
            if not self.stack:
                return []
            return ["}"] if self.stack[-1] == "{" else ["]"]
        return []

    def transition(self, token: str) -> bool:
        if token not in self.get_valid_lookahead():
            return False

        old_state = self.state
        old_stack = self.get_stack_str()
        stack_op = "NO_OP"

        if self.state == "q_start":
            self.stack.append(token)
            stack_op = f"PUSH({token})"
            self.state = "q_obj_key_or_close" if token == "{" else "q_arr_val_or_close"

        elif self.state in ["q_obj_key_or_close", "q_obj_key"]:
            if token == '"key"':
                self.state = "q_colon"
            elif token == "}":
                self.stack.pop()
                stack_op = "POP({)"
                self.state = "q_after_val"

        elif self.state == "q_colon":
            self.state = "q_val"

        elif self.state in ["q_val", "q_arr_val_or_close"]:
            if token in ['"val"', "123"]:
                self.state = "q_after_val"
            elif token in ["{", "["]:
                self.stack.append(token)
                stack_op = f"PUSH({token})"
                self.state = "q_obj_key_or_close" if token == "{" else "q_arr_val_or_close"
            elif token == "]":
                self.stack.pop()
                stack_op = "POP([)"
                self.state = "q_after_val"

        elif self.state == "q_after_val":
            top = self.stack[-1]
            if token == ",":
                self.state = "q_obj_key" if top == "{" else "q_val"
            elif token in ["}", "]"]:
                popped = self.stack.pop()
                stack_op = f"POP({popped})"
                self.state = "q_after_val"

        self.record_trace(token, old_state, self.state, old_stack, stack_op)
        return True

    def is_accepted(self) -> bool:
        return len(self.stack) == 0 and self.state == "q_after_val" and len(self.history) > 0