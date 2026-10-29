def render_stack_html(stack: list, z0: str = "Z0") -> str:
    html = "<div style='display: flex; flex-direction: column; gap: 6px; padding: 12px; background: #181825; border-radius: 8px; text-align: center; border: 1px solid #313244;'>"
    if not stack:
        html += "<div style='padding: 10px; color: #6c7086; font-style: italic;'>[ Empty above Z0 ]</div>"
    else:
        for idx, item in enumerate(reversed(stack)):
            is_top = (idx == 0)
            bg = "#f38ba8" if is_top else "#313244"
            fg = "#11111b" if is_top else "#cdd6f4"
            tag = f"<b>{item}</b> &larr; TOP" if is_top else f"<b>{item}</b>"
            html += f"<div style='padding: 10px; background: {bg}; color: {fg}; border-radius: 5px; font-family: monospace; font-size: 15px;'>{tag}</div>"
    html += f"<div style='padding: 10px; background: #1e1e2e; color: #a6e3a1; border: 1px dashed #a6e3a1; border-radius: 5px; font-family: monospace;'><b>{z0} (Stack Base)</b></div>"
    html += "</div>"
    return html


def render_state_graphviz(states: list, current_state: str, accepting_states: list, is_accepted: bool, last_edge: tuple = None) -> str:
    dot = "digraph PDA {\n  rankdir=LR;\n  bgcolor=\"transparent\";\n"
    dot += "  node [fontname=\"Helvetica\", fontsize=10, style=filled];\n"
    dot += "  edge [fontname=\" monospace\", fontsize=9, color=\"#6c7086\", fontcolor=\"#cdd6f4\"];\n"

    for s in states:
        shape = "doublecircle" if s in accepting_states else "circle"
        if s == current_state:
            color = "#a6e3a1" if is_accepted else "#f9e2af"
            dot += f'  "{s}" [shape={shape}, fillcolor="{color}", color="#11111b", penwidth=2.5, fontcolor="#11111b"];\n'
        else:
            dot += f'  "{s}" [shape={shape}, fillcolor="#313244", color="#cdd6f4", fontcolor="#cdd6f4"];\n'

    if last_edge:
        u, v, tok = last_edge
        safe_tok = tok.replace('"', '\\"')
        dot += f'  "{u}" -> "{v}" [label=" {safe_tok} ", color="#f38ba8", penwidth=2.2, fontcolor="#f38ba8"];\n'

    dot += "}"
    return dot


def render_parse_tree_graphviz(history: list) -> str:
    dot = "digraph ParseTree {\n  bgcolor=\"transparent\";\n"
    dot += "  node [fontname=\"monospace\", fontsize=10, style=filled, shape=box, rounded=true];\n"
    dot += "  edge [color=\"#89b4fa\"];\n"
    dot += '  "root" [label="Derivation Root (S)", fillcolor="#89b4fa", fontcolor="#11111b", shape=oval];\n'

    scope_stack = ["root"]
    openers = {"(", "[", "{"}
    closers = {")", "]", "}"}

    for i, token in enumerate(history):
        node_id = f"tok_{i}"
        safe_label = token.replace('"', '\\"')
        parent = scope_stack[-1]

        if token in openers:
            scope_label = f"Scope {safe_label} (step {i+1})"
            dot += f'  "{node_id}" [label="{scope_label}", fillcolor="#f9e2af", fontcolor="#11111b"];\n'
            dot += f'  "{parent}" -> "{node_id}";\n'
            scope_stack.append(node_id)
        elif token in closers:
            dot += f'  "{node_id}" [label="Close {safe_label}", fillcolor="#f38ba8", fontcolor="#11111b"];\n'
            dot += f'  "{parent}" -> "{node_id}";\n'
            if len(scope_stack) > 1:
                scope_stack.pop()
        else:
            dot += f'  "{node_id}" [label="{safe_label}", fillcolor="#313244", fontcolor="#cdd6f4"];\n'
            dot += f'  "{parent}" -> "{node_id}";\n'

    dot += "}"
    return dot