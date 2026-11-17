"""A chain the family's jump cannot cross, crossed by a torus from traced field lines (torux.beyond): the search for the
largest asks it of every request, a torus one request found is kept for the others, and one real CNT case."""
import copy
import types

import pytest

from torux import api, ask, beyond


def test_the_largest_asks_every_request_with_one_kept_dict_and_leaves_the_source_alone(monkeypatch):
    seen = []

    def request(src, flux, b, o):
        seen.append(src)
        if flux is None:
            prof = [dict(flux=1e-3 * k / 4, converged=True, bn_off=1e-12) for k in (1, 2, 3, 4)]
            return api.Result("range", b, 0.1, record=dict(range=dict(verified_lower_bound=1e-3), profile=prof))
        if flux <= 1e-3:
            return api.Result("surface", b, 0.1, surface=object(), iota=-0.33, record={})
        return api.Result("undetermined", b, 0.1, record={})
    monkeypatch.setattr(ask, "_request", request)
    monkeypatch.setattr(ask, "size", lambda s: dict(volume=1.0, minor_radius=0.1, mean_cross_section=0.03))
    monkeypatch.setattr(ask, "say", None)
    monkeypatch.setattr(api, "estimate_budget", lambda src, flux: 300.0 if flux is not None else 600.0)
    mine = types.SimpleNamespace(id="coils", request=None)
    ask._largest(mine, 1200.0, {})
    assert not hasattr(mine, "_traced")
    kept = {id(s._traced) for s in seen}
    assert len(seen) >= 3 and len(kept) == 1 and isinstance(seen[0]._traced, dict)


def test_nothing_kept_is_nothing_reused():
    chain = types.SimpleNamespace(p=-2, q=9)
    last = types.SimpleNamespace(flux=2e-3)
    assert beyond.reuse(None, None, [last], chain, None) == (None, None)
    assert beyond.reuse({}, None, [last], chain, None) == (None, None)
    # a chain the tracing found nothing beyond, in an earlier request of the question
    assert beyond.reuse({(-2, 9): None}, None, [last], chain, None) == (None, None)


def test_a_stall_waits_for_the_stage_and_the_outermost_is_kept():
    import numpy as np
    from torux.torus import TorusState

    def torus(flux):
        return TorusState(np.zeros((3, 5)), np.zeros((3, 5)), -0.44, flux, 2, 2, 2)
    keep = {}
    chain, axis = types.SimpleNamespace(p=-2, q=9), object()
    beyond.stall(keep, [torus(1.9e-3), torus(2.0e-3)], chain, axis, 1.0, 0.05)
    beyond.stall(keep, [torus(1.8e-3), torus(1.95e-3)], chain, axis, 1.0, 0.05)
    beyond.stall(keep, [torus(1.0e-4), torus(1.2e-4)], types.SimpleNamespace(p=-1, q=3), axis, 1.0, 0.05)
    assert beyond.pending(keep) == [(-1, 3), (-2, 9)] and beyond.pending(keep, 1.5e-3) == [(-2, 9)]
    assert keep["stalls"][(-2, 9)]["flux"] == 2.0e-3
    keep[(-2, 9)] = None            # traced, and nothing found
    beyond.stall(keep, [torus(2.0e-3), torus(2.1e-3)], chain, axis, 1.0, 0.05)
    assert beyond.pending(keep) == [(-1, 3)] and keep["stalls"][(-2, 9)]["flux"] == 2.0e-3


@pytest.mark.slow
def test_cnt_78_degrees_ratio_5_2_beyond_the_4_9_chain():
    """CNT at 78 degrees and I_IL / I_PF = 5.2: the family from the axis stalls at iota = -4/9 near 2.04e-3 Wb and the
    jump finds no torus beyond (an island from 2.05e-3 to 2.96e-3 Wb by an independent trace, with surfaces beyond it
    to 5.26e-3 Wb). Asked for 4.04e-3 Wb, 0.2.2 answered undetermined. The request records where its family stalled,
    the stage traces lines beyond the chain for a torus, and the same request asked again goes on from it to a
    verified surface."""
    import time
    from test_crossed import _cnt
    src = copy.copy(_cnt(78, 5.2))
    src._traced = {}
    r = api.solve(src, 4.0e-3, budget_s=400)
    assert r.status != "surface" and beyond.pending(src._traced) == [(-2, 9)]
    log = beyond.cross(src, src._traced, deadline=time.time() + 300)
    assert log["outcome"] == "torus" and 2.96e-3 < abs(log["torus"]["flux"]) < 4.0e-3
    r = api.solve(src, 4.0e-3, budget_s=600)
    assert r.status == "surface", (r.status, r.evidence.get("unverified"))
    assert abs(abs(r.surface.flux) / 4.0e-3 - 1) < 1e-12
    tj = [j for j in r.record["jumps"] if j.get("traced")]
    assert tj and (tj[0]["p"], tj[0]["q"]) == (-2, 9) and tj[0]["ok"] and tj[0]["traced"].get("reused")
