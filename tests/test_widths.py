"""Island chain widths by closing in with verified surfaces (torux.widths): which side of a chain a surface lies on,
decided by the chain's crossings and never by iota; the closing in on a made-up chain whose islands and layer are
known, from both sides, from one side, and with an iota profile that turns back at the chain; and the option's
place in torux.solve and torux.many."""
import json
import types

import numpy as np
import pytest

from torux import ask, batch, widths
from torux.torus import TorusState

NFP = 2
K = 0.2                       # flux = K r^2 for the made-up tori (circles of radius r about (1, 0))


def _circle(r, iota, R0=1.0, M=2, N=2):
    rc = np.zeros((M + 1, 2 * N + 1))
    zs = np.zeros((M + 1, 2 * N + 1))
    rc[0, N], rc[1, N], zs[1, N] = R0, r, r
    t = TorusState(rc, zs, iota, K * r * r, M, N, NFP)
    t.converged, t.bn_off = True, 1e-15
    return t


def _ring(r, q, R0=1.0, phase=0.0):
    th = phase + 2 * np.pi * np.arange(q) / q
    return [[R0 + r * np.cos(t), r * np.sin(t)] for t in th]


def _chain(rc=0.07, q=11, p=-1, half=None, found="stepped across", flux=None):
    O = [dict(residue=1e-3, section=_ring(rc, q), plane_phi=0.0)]
    X = [dict(residue=-1e-3, section=_ring(rc, q, phase=np.pi / q), plane_phi=0.0)]
    c = dict(p=p, q=q, iota=p * NFP / q, found_in=[found], O=O, X=X)
    if half is not None:
        c["half_width_Wb"] = half
    if flux is not None:
        c["flux_Wb"] = flux
    return c


def test_a_surface_is_placed_by_the_chains_crossings_and_never_when_too_close_to_tell():
    """A circle inside the ring of crossings is the chain's inner surface, one outside it its outer; one through
    the ring crosses the chain; one closer to every crossing than the margin is not placed at all."""
    c = _chain(0.07)
    lines = widths.chain_lines(c)
    assert len(lines) == 2
    m = widths.line_margin(c)
    assert m == widths.LINE_M and widths.line_margin(_chain(found="edge barrier")) == widths.EDGE_LINE_M
    assert widths.side(_circle(0.05, -0.2), lines, m) == "inner"
    assert widths.side(_circle(0.09, -0.2), lines, m) == "outer"
    assert widths.side(_circle(0.07 + 0.5 * m, -0.2), lines, m) is None
    # crossings either side of one surface: the surface crosses the chain
    mixed = dict(c, X=[dict(residue=-1e-3, section=_ring(0.08, 11), plane_phi=0.0)], O=[
        dict(residue=1e-3, section=_ring(0.06, 11), plane_phi=0.0)])
    assert widths.side(_circle(0.07, -0.2), widths.chain_lines(mixed), m) == "crosses"
    # the polygon's chord error widens the margin: a coarse section counts fewer crossings
    poly, sag = widths.section(_circle(0.07, -0.2), 0.0, n=64)
    assert sag > 1e-5


class _Runner:
    """Requests answered at once from a made-up truth: no surface within the islands and layer [r0 - w, r0 + w],
    and a circle of the flux asked elsewhere, with iota from the profile given."""

    def __init__(self, r0, w, iota, jobs=3, fail=()):
        self.r0, self.w, self.iota, self.jobs, self.fail = r0, w, iota, jobs, fail
        self.out, self.asked = [], []

    def slots(self):
        return self.jobs - len(self.out)

    def start(self, flux, budget):
        self.asked.append(flux)
        r = np.sqrt(flux / K)
        if abs(r - self.r0) <= self.w or any(a <= flux <= b for a, b in self.fail):
            ans = types.SimpleNamespace(status="undetermined", surface=None, record={})
        else:
            ans = types.SimpleNamespace(status="surface", surface=_circle(r, self.iota(r)), record={})
        self.out.append((flux, ans, 1.0))

    def poll(self):
        out, self.out = self.out, []
        return out

    def stop(self):
        self.out = []


def _close(chain, tori, runner, failed=(), bound=None, t=60.0):
    import time
    st = widths.Stage([chain], tori, failed=failed, bound=bound)
    st.run(runner, time.time() + t)
    return st, st.results()[(abs(chain["p"]), chain["q"], 0)]


def test_a_two_sided_chain_is_closed_in_on_from_both_sides_and_never_tighter_than_its_layer():
    """From family tori far either side, the bracket closes to within a percent of the layer's own width, each side
    a verified surface on its own side of the islands and layer."""
    r0, w = 0.07, 0.002
    iota = lambda r: -0.17 - 0.3 * (r - 0.05)
    tori = [_circle(0.04, iota(0.04)), _circle(0.11, iota(0.11))]
    ch = _chain(r0, half=1.45 * 2 * K * r0 * w, flux=K * 0.071 ** 2)
    st, b = _close(ch, tori, _Runner(r0, w, iota))
    lo, hi = K * (r0 - w) ** 2, K * (r0 + w) ** 2
    assert b["inner_Wb"] < lo and b["outer_Wb"] > hi
    assert b["start_Wb"] == [K * 0.04 ** 2, K * 0.11 ** 2]
    assert (lo - b["inner_Wb"]) < 0.03 * (hi - lo) and (b["outer_Wb"] - hi) < 0.03 * (hi - lo)
    assert b["stop"] == dict(inner="closed", outer="closed") and not b["one_sided"]
    assert b["half_width_Wb"] == pytest.approx(0.5 * (b["outer_Wb"] - b["inner_Wb"]))
    assert b["requests"] <= 2 * widths.MAX_SIDE


def test_the_side_is_the_crossings_even_where_iota_turns_back_at_the_chain():
    """iota with its extremum at the chain (the same value both sides, so iota cannot tell the sides apart): the
    bracket still holds the islands and layer, from verified surfaces on the right side."""
    r0, w = 0.07, 0.003
    iota = lambda r: -0.20 + 30.0 * (r - r0) ** 2
    tori = [_circle(0.045, iota(0.045)), _circle(0.095, iota(0.095))]
    ch = _chain(r0, q=10, half=None)
    st, b = _close(ch, tori, _Runner(r0, w, iota))
    lo, hi = K * (r0 - w) ** 2, K * (r0 + w) ** 2
    assert b["inner_Wb"] < lo and b["outer_Wb"] > hi
    assert (lo - b["inner_Wb"]) < 0.05 * (hi - lo) and (b["outer_Wb"] - hi) < 0.05 * (hi - lo)


def test_a_chain_with_nothing_verified_beyond_it_is_one_sided():
    """Nothing verifies beyond the inner edge of the islands: the inner side closes, and the outer edge is None,
    said as such, never made up."""
    r0, w = 0.07, 0.002
    iota = lambda r: -0.17 - 0.3 * (r - 0.05)
    run = _Runner(r0, w, iota, fail=[(K * r0 ** 2, 1.0)])
    ch = _chain(r0, half=1.45 * 2 * K * r0 * w, flux=K * r0 ** 2)
    st, b = _close(ch, [_circle(0.05, iota(0.05))], run, failed=[K * 0.09 ** 2])
    assert b["one_sided"] and b["outer_Wb"] is None and b.get("half_width_Wb") is None
    assert b["inner_Wb"] < K * (r0 - w) ** 2
    assert K * (r0 - w) ** 2 - b["inner_Wb"] < 0.05 * K * ((r0 + w) ** 2 - (r0 - w) ** 2)
    assert b["stop"]["outer"] == "no verified surface beyond the chain"


def test_requests_that_fail_short_of_the_layer_leave_the_bracket_wider_never_wrong():
    """Bands where the solver fails although surfaces exist, on either side: whatever the closing in reaches, its
    bracket holds the islands and layer, from verified surfaces on the right side."""
    r0, w = 0.07, 0.002
    iota = lambda r: -0.17 - 0.3 * (r - 0.05)
    lo, hi = K * (r0 - w) ** 2, K * (r0 + w) ** 2
    for band in ((0.06, 0.065), (0.0675, 0.06799), (0.07201, 0.073), (0.05, 0.1)):
        ch = _chain(r0, half=1.45 * 2 * K * r0 * w, flux=K * 0.07 ** 2)
        run = _Runner(r0, w, iota, fail=[(K * band[0] ** 2, K * band[1] ** 2)])
        st, b = _close(ch, [_circle(0.04, iota(0.04)), _circle(0.11, iota(0.11))], run)
        assert b["inner_Wb"] < lo and b["outer_Wb"] > hi
        assert not any(K * band[0] ** 2 <= e["flux_reached"] <= K * band[1] ** 2 for e in st.log
                       if e.get("flux_reached") is not None)
    assert b["inner_Wb"] < K * 0.05 ** 2 and b["outer_Wb"] > K * 0.1 ** 2


def test_the_deadline_stops_the_closing_in_and_says_so():
    import time
    r0, w = 0.07, 0.002
    iota = lambda r: -0.17 - 0.3 * (r - 0.05)
    st = widths.Stage([_chain(r0)], [_circle(0.04, iota(0.04)), _circle(0.11, iota(0.11))])
    st.run(_Runner(r0, w, iota), time.time() + 1.0)
    b = st.results()[(1, 11, 0)]
    assert b["requests"] == 0 and b["stop"] == dict(inner="time", outer="time")
    assert b["inner_Wb"] == K * 0.04 ** 2


def test_a_bracket_reaches_the_chain_entry_the_words_and_the_table():
    c = _chain(0.07)
    res = {(1, 11, 0): dict(inner_Wb=1e-3, outer_Wb=1.2e-3, one_sided=False, half_width_Wb=1e-4, flux_Wb=1.1e-3,
                            requests=7)}
    widths.attach([c], res)
    assert c["bracket"]["inner_Wb"] == 1e-3
    cell = json.loads(widths.table_cell([c, _chain(0.08, q=13)]))
    assert cell == [dict(p=-1, q=11, occurrence=0, inner_Wb=1e-3, outer_Wb=1.2e-3, one_sided=False,
                         half_width_Wb=1e-4, flux_Wb=1.1e-3, requests=7, disagreement=False)]
    assert widths.table_cell([_chain(0.08)]) is None
    assert "chain widths" in widths.words(dict(chains=1, two_sided=1, requests=7, wall_s=12.0, closed=2))


def test_widths_is_asked_only_with_the_largest_and_off_by_default():
    assert ask.widths_question() is None
    assert ask.widths_question(600, None, largest=True) == (600.0, None)
    assert ask.widths_question(600, 4, largest=True) == (600.0, 4)
    for bad in (dict(widths_s=0, largest=True), dict(widths_s=-1, largest=True), dict(widths_s=True, largest=True),
                dict(widths_jobs=2), dict(widths_s=60, widths_jobs=0, largest=True),
                dict(widths_s=60, flux=1e-3), dict(widths_s=60, minor_radius=0.1)):
        with pytest.raises(ValueError):
            ask.widths_question(**bad)


def test_the_table_has_its_brackets_column_only_when_widths_is_asked(tmp_path):
    assert batch.COLUMNS_WIDTHS == batch.COLUMNS + ("chain_brackets",)
    t = batch._Table(str(tmp_path / "a.csv"), [], "largest")
    t.close()
    assert (tmp_path / "a.csv").read_text().splitlines()[0].split(",") == list(batch.COLUMNS)
    t = batch._Table(str(tmp_path / "b.csv"), [], "largest", widths=True)
    t.close()
    assert (tmp_path / "b.csv").read_text().splitlines()[0].split(",")[-1] == "chain_brackets"
    # and a table of either kind is continued in its own columns
    t = batch._Table(str(tmp_path / "b.csv"), [], "largest")
    assert t.cols == batch.COLUMNS_WIDTHS
    t.close()


def test_the_search_attaches_each_bracket_to_its_chain_and_says_so(monkeypatch):
    """After the largest search, the stage's brackets reach answer.evidence['chains'], its summary
    answer.evidence['search']['widths'] and a line of the words; without widths_s there is none of it."""
    import time as _time
    from torux import api
    surf = _circle(0.10, -0.21)
    entry = dict(flux=float(surf.flux), wall_s=1.0, chains=[dict(_chain(0.07), found_in="stepped across", request=1)])

    def answer(widths_on):
        S = ask._Search(types.SimpleNamespace(id="coils", request=None, nfp=NFP), "largest", None, 100.0, {})
        r = api.Result("surface", 100.0, 0.1, surface=surf, iota=-0.21, record=dict(nfp=NFP))
        r._ask_entry = dict(entry)
        S.log.append(r._ask_entry)
        if widths_on:
            S.widths_asked = (60.0, 1)
            ask._widths_stage(S, r)
        return S.finish(r, [])

    def stage(S, r, secs, jobs=None, verbose=False):
        return ({(1, 11, 0): dict(inner_Wb=1e-3, outer_Wb=1.1e-3, one_sided=False, half_width_Wb=5e-5,
                                  flux_Wb=1.05e-3, requests=3, stop=dict(inner="closed", outer="closed"))},
                dict(budget_s=secs, wall_s=1.0, chains=1, two_sided=1, one_sided=0, requests=3, closed=2))
    monkeypatch.setattr(widths, "stage", stage)
    monkeypatch.setattr(ask, "say", None)
    on = answer(True)
    c = on.evidence["chains"][0]
    assert c["bracket"]["inner_Wb"] == 1e-3 and on.evidence["search"]["widths"]["requests"] == 3
    assert any(w.startswith("chain widths: 1 of 1 chains bracketed") for w in on.evidence["search"]["words"])
    off = answer(False)
    assert "bracket" not in off.evidence["chains"][0] and "widths" not in off.evidence["search"]
    assert not any("chain widths" in w for w in off.evidence["search"]["words"])


@pytest.mark.slow
def test_cnt_64_degrees_ratio_4_5_brackets_hold_the_traced_layers():
    """CNT at 64 degrees, I_IL / I_PF = 4.5, the largest with widths: the brackets of iota = -2/11 and -1/5 hold the
    separatrix layers an independent tracer found (closed-form circular coils, RK4: -2/11 from 8.56742e-4 to
    8.59653e-4 Wb, -1/5 from 2.011599e-3 to 2.077986e-3 Wb, their edges good to 1e-7 and 3e-7 Wb), and are narrower than
    the family tori they started from."""
    import torux
    from test_crossed import _cnt
    a = torux.solve(_cnt(64, 4.5), largest=True, budget_s=900, widths_s=2400)
    got = {(c["p"], c["q"]): c for c in a.evidence["chains"] if c.get("bracket")}
    for pq, (lo, hi), tol in (((-1, 11), (8.56742e-4, 8.59653e-4), 1e-7), ((-1, 10), (2.011599e-3, 2.077986e-3), 3e-7)):
        b = got[pq]["bracket"]
        assert b["inner_Wb"] <= lo + tol and b["outer_Wb"] >= hi - tol, (pq, b)
        assert b["outer_Wb"] - b["inner_Wb"] < b["start_Wb"][1] - b["start_Wb"][0], (pq, b)
    assert a.evidence["search"]["widths"]["surfaces"]


def test_with_widths_the_answer_is_the_largest_surface_the_answer_verified(monkeypatch):
    """A surface the closing in verified beyond the search's largest becomes the answer: its surface, the search's
    flux, the request that answered, and words that say the search itself had reached less, in no time past
    widths_s. A surface the closing in verified inside the answer changes nothing."""
    from torux import api
    small = _circle(0.10, -0.21)
    src = types.SimpleNamespace(id="coils", request=None, nfp=NFP, parts=[])
    monkeypatch.setattr(ask, "_crossed_stage", lambda S, r: None)
    monkeypatch.setattr(ask, "say", None)

    def run(far_surface):
        S = ask._Search(src, "largest", None, 100.0, {})
        f0 = float(small.flux)
        r = api.Result("surface", 100.0, 0.1, surface=small, iota=-0.21, record=dict(nfp=NFP))
        S.log.append(dict(flux=f0, purpose="the search", budget_s=100.0, wall_s=1.0,
                          chains=[dict(_chain(0.07), found_in="stepped across", request=1)]))
        r._ask_entry = S.log[-1]
        S.best[f0], S.size_of[f0] = r, ask.size(small)
        S.widths_asked = (60.0, 1)
        far = api.Result("surface", 300.0, 20.0, surface=far_surface, iota=float(far_surface.iota),
                         record=dict(nfp=NFP))

        def stage(S_, r_, secs, jobs=None, verbose=False):
            S_.widths_largest = (float(far_surface.flux), far, 20.0)
            return ({}, dict(budget_s=secs, wall_s=1.0, chains=0, two_sided=0, one_sided=0, requests=1, closed=0))
        monkeypatch.setattr(widths, "stage", stage)
        return ask._largest_answer(S, f0, stop=None), f0

    big = _circle(0.104, -0.212)
    res, f0 = run(big)
    s = res.evidence["search"]
    assert res.surface is big and s["flux"] == float(big.flux) and s["answered_by"]["request"] == 2
    assert s["requests"][-1]["purpose"].startswith("the closing in on the chains")
    assert s["words"][1].startswith(f"reached: {ask._wb(float(big.flux))}")
    assert any(w.startswith("reached by the closing in on the chains: the search itself had reached "
                            f"{ask._wb(f0)}") and "none past it" in w for w in s["words"])
    res, f0 = run(_circle(0.09, -0.20))
    s = res.evidence["search"]
    assert res.surface is small and s["flux"] == f0 and len(s["requests"]) == 1
    assert not any("reached by the closing in" in w for w in s["words"])


def test_widths_jobs_without_widths_is_an_error_on_the_command_line(capsys):
    from torux import cli
    assert cli.main(["solve", "no-such-coils.json", "--largest", "--widths-jobs", "2"]) == 1
    assert "--widths-jobs is the requests at once of --widths" in capsys.readouterr().err
