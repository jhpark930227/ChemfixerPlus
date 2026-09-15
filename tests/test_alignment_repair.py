from chemfixerplus.alignment import align_tokens, reconstruct_target, spans_from_predictions
from chemfixerplus.repair import assemble_from_replacements, routing_stats, serialize_repair
from chemfixerplus.tokenizer import SmilesTokenizer


def test_figure1_alignment_and_routing():
    tok = SmilesTokenizer()
    x = tok.tokenize("CCC(=O)O)CC1CCCCC")
    y = tok.tokenize("CCC(=O)OCC1CCCCC1")
    aln = align_tokens(x, y)
    assert len(x) == 17
    assert len(aln.spans) == 2
    assert reconstruct_target(x, aln.spans) == y
    rho, enc_len = routing_stats(len(x), aln.spans)
    assert abs(rho - 2 / 17) < 1e-12
    assert enc_len == 20
    ser = serialize_repair(x, aln.spans)
    assert ser.repair_target[0] == "<e1>"
    assert ser.repair_target[-1] == "<e3>"
    recovered = assemble_from_replacements(x, aln.spans, [s.replacement for s in aln.spans])
    assert recovered == y


def test_prediction_span_merging_no_context_absorption():
    # tokens 1 and 2 merge; isolated gap 4 stays zero-width and does not absorb token 3.
    spans = spans_from_predictions(
        [False, True, True, False],
        [False, True, False, False, True],
    )
    assert len(spans) == 2
    assert (spans[0].source_start, spans[0].source_end) == (1, 3)
    assert (spans[1].source_start, spans[1].source_end) == (4, 4)
