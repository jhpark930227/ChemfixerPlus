from chemfixerplus.tokenizer import SmilesTokenizer
from chemfixerplus.structure import lexical_ring_states, RING_OPEN, RING_CLOSE, RING_UNMATCHED


def test_roundtrip_extended_tokens():
    tok = SmilesTokenizer()
    s = r"F/C=C/[C@H](Cl)[N+](C)(C)C%12CCCCC%12"
    tokens = tok.tokenize(s)
    assert tok.detokenize(tokens) == s
    assert "%12" in tokens
    assert "[C@H]" in tokens
    assert "[N+]" in tokens


def test_ring_states():
    tok = SmilesTokenizer()
    tokens = tok.tokenize("C1CCCCC1C2CC")
    states = lexical_ring_states(tokens)
    ring_positions = [(i, t, states[i]) for i, t in enumerate(tokens) if t in {"1", "2"}]
    assert ring_positions[0][2] == RING_OPEN
    assert ring_positions[1][2] == RING_CLOSE
    assert ring_positions[2][2] == RING_UNMATCHED


def test_directional_bonds_are_stereo_roles():
    from chemfixerplus.structure import token_roles
    assert "stereo" in token_roles("/")
    assert "stereo" in token_roles("\\")
    assert "bond" in token_roles("/")
    assert "bond" in token_roles("\\")
