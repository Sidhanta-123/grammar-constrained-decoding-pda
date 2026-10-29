from src.automata.base_pda import BasePDA

class SQLPDAEngine(BasePDA):
    def __init__(self):
        super().__init__(
            name="SQL Query Enforcer",
            description="Enforces SELECT-FROM-WHERE syntax with nested boolean parenthesis depth.",
            states=["q_start", "q_select", "q_from", "q_table", "q_where_cond", "q_op", "q_val", "q_post_cond", "q_done"],
            terminals=["SELECT", "col", "FROM", "tbl", "WHERE", "(", ")", "=", "10", "AND", ";"],
            stack_alphabet=["(", "Z0"],
            start_state="q_start",
            accepting_states=["q_done"],
            productions=[
                "Query -> 'SELECT' 'col' 'FROM' 'tbl' 'WHERE' Cond ';'",
                "Cond -> '(' Cond ')' | Cond 'AND' Cond | 'col' '=' '10'"
            ]
        )

    def get_valid_lookahead(self) -> list:
        if self.state == "q_start":
            return ["SELECT"]
        elif self.state == "q_select":
            return ["col"]
        elif self.state == "q_from":
            return ["FROM"]
        elif self.state == "q_table":
            return ["tbl"]
        elif self.state == "q_where_cond":
            return ["(", "col"]
        elif self.state == "q_op":
            return ["="]
        elif self.state == "q_val":
            return ["10"]
        elif self.state == "q_post_cond":
            valid = ["AND"]
            if self.stack:
                valid.append(")")
            else:
                valid.append(";")
            return valid
        return []

    def min_tokens_to_close(self) -> int:
        base_map = {
            "q_start": 8,
            "q_select": 7,
            "q_from": 6,
            "q_table": 5,
            "q_where_cond": 4 + len(self.stack),
            "q_op": 3 + len(self.stack),
            "q_val": 2 + len(self.stack),
            "q_post_cond": 1 + len(self.stack),
            "q_done": 0
        }
        return base_map.get(self.state, 0)

    def get_closing_lookahead(self) -> list:
        if self.state == "q_where_cond":
            return ["col"]
        elif self.state == "q_post_cond":
            return [")"] if self.stack else [";"]
        return self.get_valid_lookahead()

    def transition(self, token: str) -> bool:
        if token not in self.get_valid_lookahead():
            return False

        old_state = self.state
        old_stack = self.get_stack_str()
        stack_op = "NO_OP"

        if self.state == "q_start" and token == "SELECT":
            self.state = "q_select"
        elif self.state == "q_select" and token == "col":
            self.state = "q_from"
        elif self.state == "q_from" and token == "FROM":
            self.state = "q_table"
        elif self.state == "q_table" and token == "tbl":
            self.state = "q_where_cond"
        elif self.state == "q_where_cond":
            if token == "(":
                self.stack.append("(")
                stack_op = "PUSH(()"
            elif token == "col":
                self.state = "q_op"
        elif self.state == "q_op" and token == "=":
            self.state = "q_val"
        elif self.state == "q_val" and token == "10":
            self.state = "q_post_cond"
        elif self.state == "q_post_cond":
            if token == "AND":
                self.state = "q_where_cond"
            elif token == ")":
                self.stack.pop()
                stack_op = "POP(()"
            elif token == ";":
                self.state = "q_done"

        display_tok = "tbl WHERE" if (old_state == "q_table" and token == "tbl") else token
        self.record_trace(display_tok, old_state, self.state, old_stack, stack_op)
        return True

    def is_accepted(self) -> bool:
        return len(self.stack) == 0 and self.state == "q_done"