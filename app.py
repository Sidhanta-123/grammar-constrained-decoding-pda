import streamlit as st
import pandas as pd
import json
from src.automata.math_pda import MathPDAEngine
from src.automata.json_pda import JSONPDAEngine
from src.automata.sql_pda import SQLPDAEngine
from src.decoder.llm_simulator import LLMSimulator
from src.decoder.logit_masker import LogitMasker
from src.ui.components import render_stack_html, render_state_graphviz, render_parse_tree_graphviz
from cli_benchmark import run_constrained_episode, run_unconstrained_episode

st.set_page_config(page_title="PDA Constrained Decoding Workbench", layout="wide")

ENGINES = {
    "Nested Math Expressions": MathPDAEngine,
    "Structured JSON Schema": JSONPDAEngine,
    "SQL Query Enforcer": SQLPDAEngine
}

# --- Sidebar Controls ---
st.sidebar.title("⚙️ Engine Architecture")
grammar_choice = st.sidebar.selectbox("Select Formal Grammar (CFG):", list(ENGINES.keys()))

if "current_grammar" not in st.session_state or st.session_state.current_grammar != grammar_choice:
    st.session_state.current_grammar = grammar_choice
    st.session_state.engine = ENGINES[grammar_choice]()
    st.session_state.unconstrained_history = []
    st.session_state.interceptions = 0
    if "last_step" in st.session_state:
        del st.session_state.last_step

engine = st.session_state.engine

st.sidebar.markdown("---")
st.sidebar.subheader("LLM Inference Controls")
temperature = st.sidebar.slider("Temperature (T)", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
top_k = st.sidebar.slider("Top-k Candidate Limit", min_value=0, max_value=len(engine.Sigma), value=0, step=1)
bias_invalid = st.sidebar.checkbox("Inject Adversarial Hallucinations", value=True)
max_budget = st.sidebar.slider("Max Token Budget", min_value=12, max_value=40, value=25)

if st.sidebar.button("🔄 Reset Automaton", use_container_width=True):
    engine.reset()
    st.session_state.unconstrained_history = []
    st.session_state.interceptions = 0
    if "last_step" in st.session_state:
        del st.session_state.last_step
    st.rerun()

# --- Main Header ---
st.title("🧠 Grammar-Constrained LLM Decoding Workbench")
st.caption(f"**Active Automaton:** `{engine.name}` — *{engine.description}*")

tab_decoder, tab_verifier, tab_bench, tab_formal = st.tabs([
    "🔬 1. Live Decoder & AST",
    "🧪 2. String Verifier & DFA vs. PDA Proof",
    "📊 3. Multi-Grammar Benchmark Lab",
    "📐 4. Formal 7-Tuple & CFG Spec"
])

# =====================================================================
# TAB 1: LIVE DECODER & AST VISUALIZER
# =====================================================================
with tab_decoder:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Active State (q)", engine.state)
    m2.metric("Stack Depth (|Γ|)", len(engine.stack))
    m3.metric("Tokens Generated", f"{len(engine.history)} / {max_budget}")
    m4.metric("Min Tokens to Close", engine.min_tokens_to_close())
    m5.metric("Hallucinations Blocked", st.session_state.interceptions)

    st.markdown("##### Finite Control State Diagram ($Q$) — *Double Circle = Accepting State ($F$)*")
    st.graphviz_chart(render_state_graphviz(
        engine.Q, engine.state, engine.F, engine.is_accepted(), engine.last_transition_edge
    ))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### ❌ Unconstrained LLM Output (Raw Greedy Decoding)")
        st.code(" ".join(st.session_state.unconstrained_history) or "[Waiting for tokens...]", language="text")
    with c2:
        st.markdown("##### ✅ PDA-Constrained Output (Guaranteed $w \in L(G)$)")
        st.code(" ".join(engine.history) or "[Waiting for tokens...]", language="sql" if "SQL" in grammar_choice else "json")

    st.divider()
    col_ctrl, col_stack, col_viz = st.columns([1.1, 0.8, 1.5])

    def execute_step(force_close: bool = False):
        tokens_left = max_budget - len(engine.history)
        if force_close:
            valid_tokens = engine.get_closing_lookahead()
        else:
            valid_tokens = engine.get_budget_safe_lookahead(tokens_left)

        if not valid_tokens:
            return False

        raw_logits = LLMSimulator.generate_logits(engine.Sigma, valid_tokens, bias_invalid, temperature)
        result = LogitMasker.apply_pda_mask(engine.Sigma, raw_logits, valid_tokens, top_k)
        st.session_state.unconstrained_history.append(result["unconstrained_pick"])
        if result["intercepted"]:
            st.session_state.interceptions += 1
        engine.transition(result["constrained_pick"])
        st.session_state.last_step = result
        return True

    with col_ctrl:
        st.subheader("1. Decoder Controller")
        valid_tokens = engine.get_valid_lookahead()
        can_step = len(valid_tokens) > 0 and len(engine.history) < max_budget

        if st.button("🚀 Step Decoder (1 Token)", type="primary", use_container_width=True, disabled=not can_step):
            execute_step(force_close=False)
            st.rerun()

        if st.button("⚡ Auto-Step 5 Tokens", use_container_width=True, disabled=not can_step):
            for _ in range(5):
                if len(engine.history) < max_budget and not engine.is_accepted():
                    execute_step(force_close=False)
            st.rerun()

        if st.button("🏁 Drain Stack & Complete Valid String", use_container_width=True, disabled=engine.is_accepted() or not valid_tokens):
            safety = 0
            while not engine.is_accepted() and safety < 35:
                if not execute_step(force_close=True):
                    break
                safety += 1
            st.rerun()

        st.markdown("---")
        st.markdown("**Manual Token Injection ($\Sigma$):**")
        b_cols = st.columns(3)
        for i, tok in enumerate(engine.Sigma):
            with b_cols[i % 3]:
                if st.button(tok, key=f"tok_{tok}", disabled=(tok not in valid_tokens), use_container_width=True):
                    engine.transition(tok)
                    st.session_state.unconstrained_history.append(tok)
                    st.rerun()

        st.markdown("---")
        if engine.is_accepted():
            st.success("🎉 **STRING ACCEPTED!** ($q \in F$, Stack = $Z_0$)")
        else:
            st.warning(f"⏳ **INCOMPLETE** (Min {engine.min_tokens_to_close()} token(s) to close)")

        if engine.trace_log:
            st.download_button(
                label="📥 Export Formal ID Trace (JSON)",
                data=json.dumps(engine.trace_log, indent=2),
                file_name=f"pda_trace_{grammar_choice.lower().replace(' ', '_')}.json",
                mime="application/json",
                use_container_width=True
            )

    with col_stack:
        st.subheader("2. PDA Stack ($\Gamma$)")
        st.markdown(render_stack_html(engine.stack, engine.Z0), unsafe_allow_html=True)

    with col_viz:
        st.subheader("3. Logit Masking & Entropy Matrix")
        if "last_step" in st.session_state:
            res = st.session_state.last_step
            if res["intercepted"]:
                st.error(f"🛡️ **Blocked `{res['unconstrained_pick']}`** → Forced `{res['constrained_pick']}`")
            else:
                st.success(f"✅ Valid prediction `{res['constrained_pick']}` allowed.")

            e1, e2 = st.columns(2)
            e1.caption(f"**Raw Entropy $H(P)$:** `{res['raw_entropy']} bits`")
            e2.caption(f"**Constrained Entropy $H(P')$:** `{res['constrained_entropy']} bits`")

            df = pd.DataFrame({
                "Token": res["tokens"],
                "Raw Logit": res["raw_logits"],
                "Raw Prob %": res["raw_probs"],
                "PDA Gate": ["ALLOW" if t in res["valid_tokens"] else "MASK (-inf)" for t in res["tokens"]],
                "Final Prob %": res["constrained_probs"]
            })
            st.dataframe(df, use_container_width=True, height=290)
        else:
            st.info("Click **Step Decoder** to inspect logit masking and Shannon entropy reduction.")

    st.divider()
    tree_col, trace_col = st.columns([1.1, 1.3])
    with tree_col:
        st.subheader("🌳 Live Structural Parse Tree (AST)")
        if engine.history:
            st.graphviz_chart(render_parse_tree_graphviz(engine.history))
        else:
            st.caption("Generate tokens to watch the hierarchical parse tree grow.")
    with trace_col:
        st.subheader("📜 Instantaneous Description Trace ($\vdash_M$)")
        if engine.trace_log:
            st.dataframe(pd.DataFrame(engine.trace_log), use_container_width=True, height=300)
        else:
            st.caption("No transitions executed yet.")

# =====================================================================
# TAB 2: CUSTOM STRING VERIFIER & DFA FAILURE PROOF
# =====================================================================
with tab_verifier:
    st.subheader("🧪 Interactive Language Membership Tester ($w \stackrel{?}{\in} L(G)$)")
    st.markdown("Enter space-separated tokens below to test whether the Pushdown Automaton accepts or rejects the string, or click a preset example.")

    presets = {
        "Nested Math Expressions": {
            "Valid Nested": "{ [ ( x + y ) * x ] + y }",
            "Invalid Mismatched Bracket": "{ [ ( x + y ] ) }",
            "Invalid Double Operator": "( x + * y )"
        },
        "Structured JSON Schema": {
            "Valid Nested": '{ "key" : [ 123 , { "key" : "val" } ] }',
            "Invalid Missing Colon": '{ "key" "val" }',
            "Invalid Unclosed Array": '{ "key" : [ 123 , "val" }'
        },
        "SQL Query Enforcer": {
            "Valid Nested": "SELECT col FROM tbl ( col = 10 AND ( col = 10 ) ) ;",
            "Invalid Unbalanced Paren": "SELECT col FROM tbl ( col = 10 ;",
            "Invalid Missing Semicolon": "SELECT col FROM tbl col = 10"
        }
    }

    current_presets = presets[grammar_choice]
    p_cols = st.columns(3)
    selected_preset = None
    for idx, (label, val) in enumerate(current_presets.items()):
        if p_cols[idx].button(f"Load: {label}", use_container_width=True):
            st.session_state.custom_input_str = val

    default_str = st.session_state.get("custom_input_str", list(current_presets.values())[0])
    user_input = st.text_input("Token Sequence (space-separated):", value=default_str)

    if st.button("🔍 Run Formal PDA Verification", type="primary"):
        test_engine = ENGINES[grammar_choice]()
        token_list = [t for t in user_input.strip().split(" ") if t]
        report = test_engine.verify_custom_sequence(token_list)
        if report["accepted"]:
            st.success(f"**ACCEPTED ($w \in L(G)$):** {report['reason']}")
        else:
            st.error(f"**REJECTED ($w \\notin L(G)$):** {report['reason']}")
        st.dataframe(pd.DataFrame(report["steps"]), use_container_width=True)

    st.divider()
    st.subheader("💥 Pumping Lemma / Why Finite Automata (DFA) Fail on Deep Nesting")
    st.markdown("A Deterministic Finite Automaton (DFA) has a fixed number of states $k$ and **no stack memory**. Adjust the nesting depth $n$ below for the language $L = \\{ (^n x )^n \\mid n \\ge 1 \\}$ to compare a $k$-state DFA against our PDA:")
    dfa_k = st.slider("Hypothetical DFA Max State Capacity (k)", min_value=2, max_value=10, value=4)
    test_depth = st.slider("Expression Nesting Depth (n)", min_value=1, max_value=15, value=6)

    nested_tokens = ["("] * test_depth + ["x"] + [")"] * test_depth
    st.code(" ".join(nested_tokens), language="text")

    d_col1, d_col2 = st.columns(2)
    with d_col1:
        if test_depth > dfa_k:
            st.error(f"❌ **DFA CRASHES!** Nesting depth ($n={test_depth}$) exceeds finite state memory ($k={dfa_k}$). By the Pigeonhole Principle, the DFA loops and loses track of how many `)` are required.")
        else:
            st.warning(f"⚠️ DFA survives up to depth $n={test_depth}$, but will fail as soon as $n > {dfa_k}$.")
    with d_col2:
        math_tester = MathPDAEngine()
        res_pda = math_tester.verify_custom_sequence(nested_tokens)
        if res_pda["accepted"]:
            st.success(f"✅ **PDA SUCCEEDS!** Pushed {test_depth} frames onto $\\Gamma^*$ and popped all {test_depth} frames cleanly down to $Z_0$.")

# =====================================================================
# TAB 3: MULTI-GRAMMAR BENCHMARK LAB
# =====================================================================
with tab_bench:
    st.subheader("Statistical Verification Across All Grammars")
    st.markdown("Run Monte Carlo decoding episodes to verify that **Lookahead Budget Pruning** achieves **100.0% syntactic validity** across all Context-Free Grammars.")

    b_col1, b_col2 = st.columns([1, 2])
    with b_col1:
        num_runs = st.number_input("Episodes per Grammar:", min_value=50, max_value=1000, value=250, step=50)
        run_btn = st.button("▶️ Execute Full Benchmark Suite", type="primary", use_container_width=True)

    if run_btn:
        rows = []
        for name, cls in ENGINES.items():
            eng_instance = cls()
            u_ok = sum(run_unconstrained_episode(eng_instance, max_budget, bias_invalid) for _ in range(num_runs))
            c_ok = sum(run_constrained_episode(eng_instance, max_budget, bias_invalid) for _ in range(num_runs))
            rows.append({
                "Grammar": name,
                "Unconstrained Validity (%)": round((u_ok / num_runs) * 100, 1),
                "PDA-Constrained Validity (%)": round((c_ok / num_runs) * 100, 1)
            })
        st.session_state.bench_df = pd.DataFrame(rows)

    if "bench_df" in st.session_state:
        st.dataframe(st.session_state.bench_df, use_container_width=True)
        st.bar_chart(st.session_state.bench_df.set_index("Grammar"))
        st.download_button(
            label="📥 Export Benchmark Report (CSV)",
            data=st.session_state.bench_df.to_csv(index=False).encode("utf-8"),
            file_name="pda_benchmark_results.csv",
            mime="text/csv"
        )

# =====================================================================
# TAB 4: FORMAL 7-TUPLE & CFG SPECIFICATION
# =====================================================================
with tab_formal:
    st.subheader(f"Formal Specification: {engine.name}")
    st.latex(r"M = (Q, \Sigma, \Gamma, \delta, q_0, Z_0, F)")

    f1, f2 = st.columns(2)
    with f1:
        st.markdown("**1. Finite States ($Q$):**")
        st.code(", ".join(engine.Q))
        st.markdown("**2. Input Alphabet ($\Sigma$):**")
        st.code(", ".join(engine.Sigma))
        st.markdown("**3. Stack Alphabet ($\Gamma$):**")
        st.code(", ".join(engine.Gamma))
    with f2:
        st.markdown("**4. Start State ($q_0$) & Initial Stack ($Z_0$):**")
        st.code(f"q0 = {engine.q0}  |  Z0 = {engine.Z0}")
        st.markdown("**5. Accepting States ($F$):**")
        st.code(", ".join(engine.F))
        st.markdown("**6. Context-Free Grammar Productions ($P$):**")
        st.code("\n".join(engine.P), language="ebnf")