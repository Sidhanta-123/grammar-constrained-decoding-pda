from src.automata.math_pda import MathPDAEngine
from src.automata.json_pda import JSONPDAEngine
from src.automata.sql_pda import SQLPDAEngine
from cli_benchmark import run_constrained_episode

def test_math_balanced_brackets():
    pda = MathPDAEngine()
    report = pda.verify_custom_sequence(["{", "[", "(", "x", "+", "y", ")", "]", "}"])
    assert report["accepted"] is True

def test_math_rejects_mismatched():
    pda = MathPDAEngine()
    report = pda.verify_custom_sequence(["(", "x", "]"])
    assert report["accepted"] is False

def test_json_valid_nested():
    pda = JSONPDAEngine()
    report = pda.verify_custom_sequence(["{", '"key"', ":", "[", "123", ",", '"val"', "]", "}"])
    assert report["accepted"] is True

def test_sql_valid_nested():
    pda = SQLPDAEngine()
    report = pda.verify_custom_sequence(["SELECT", "col", "FROM", "tbl", "(", "col", "=", "10", ")", ";"])
    assert report["accepted"] is True

def test_100_percent_constrained_convergence():
    for cls in [MathPDAEngine, JSONPDAEngine, SQLPDAEngine]:
        eng = cls()
        for _ in range(50):
            assert run_constrained_episode(eng, max_tokens=25) is True

if __name__ == "__main__":
    test_math_balanced_brackets()
    test_math_rejects_mismatched()
    test_json_valid_nested()
    test_sql_valid_nested()
    test_100_percent_constrained_convergence()
    print("✅ All formal PDA & 100% convergence unit tests passed successfully!")