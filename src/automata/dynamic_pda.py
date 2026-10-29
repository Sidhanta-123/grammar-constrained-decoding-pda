from src.automata.base_pda import BasePDA

class DynamicPDAEngine(BasePDA):
    def __init__(self, name: str, raw_rules: dict, start_symbol: str):
        """
        Dynamically constructs a PDA from CFG production rules.
        raw_rules format: {"S": [["(", "S", ")"], ["x"]]}
        """
        self.rules = raw_rules
        self.start_symbol = start_symbol
        
        # Extract all non-terminals (keys) and terminals (everything else)
        self.non_terminals = set(self.rules.keys())
        self.extracted_terminals = set()
        for rhs_list in self.rules.values():
            for rhs in rhs_list:
                for symbol in rhs:
                    if symbol not in self.non_terminals and symbol != "ε":
                        self.extracted_terminals.add(symbol)
                        
        super().__init__(
            name=name,
            description="Dynamically compiled from Context-Free Grammar production rules.",
            states=["q_loop"],
            terminals=list(self.extracted_terminals),
            stack_alphabet=list(self.non_terminals.union(self.extracted_terminals)) + ["Z0"],
            start_state="q_loop",
            accepting_states=["q_loop"],
            productions=[f"{k} -> " + " | ".join([" ".join(p) for p in v]) for k, v in self.rules.items()]
        )

    def reset(self):
        super().reset()
        # Standard CFG-to-PDA initialization: Push start symbol onto Z0
        self.stack = [self.start_symbol]

    def get_valid_lookahead(self) -> list:
        """
        Simulates epsilon-expansions to find which terminal tokens are valid next.
        """
        if not self.stack:
            return []
            
        valid_terminals = set()
        simulated_stack = list(self.stack)
        visited_non_terminals = set()
        
        # We need to peek at what terminals can eventually emerge from the top of the stack
        while simulated_stack:
            top = simulated_stack[-1]
            if top in self.Sigma:
                valid_terminals.add(top)
                break
            elif top in self.non_terminals:
                if top in visited_non_terminals:
                    break # Avoid infinite left-recursion loops in lookahead
                visited_non_terminals.add(top)
                # Just take the first production for lookahead simplicity in this prototype
                first_prod = self.rules[top][0]
                simulated_stack.pop()
                for symbol in reversed(first_prod):
                    if symbol != "ε":
                        simulated_stack.append(symbol)
            else:
                break
                
        return list(valid_terminals)

    def min_tokens_to_close(self) -> int:
        return len([s for s in self.stack if s in self.Sigma or s in self.non_terminals])

    def get_closing_lookahead(self) -> list:
        return self.get_valid_lookahead()

    def transition(self, token: str) -> bool:
        if token not in self.Sigma:
            return False

        old_state = self.state
        old_stack = self.get_stack_str()
        stack_op = []

        # Epsilon expansions: Expand non-terminals until a terminal is on top
        while self.stack and self.stack[-1] in self.non_terminals:
            top_nt = self.stack.pop()
            # For this prototype, we pick the production rule that starts with or leads to our target token
            # (A full LL(1) parser would use a precomputed parsing table here)
            chosen_production = None
            for prod in self.rules[top_nt]:
                if prod[0] == token or (prod[0] in self.non_terminals):
                    chosen_production = prod
                    break
            
            if not chosen_production:
                chosen_production = self.rules[top_nt][-1] # Fallback to base case
                
            stack_op.append(f"EXPAND({top_nt} → {' '.join(chosen_production)})")
            
            for symbol in reversed(chosen_production):
                if symbol != "ε":
                    self.stack.append(symbol)

        # Match terminal
        if self.stack and self.stack[-1] == token:
            popped = self.stack.pop()
            stack_op.append(f"MATCH({popped})")
            self.record_trace(token, old_state, self.state, old_stack, " | ".join(stack_op))
            return True
            
        return False

    def is_accepted(self) -> bool:
        return len(self.stack) == 0 and len(self.history) > 0