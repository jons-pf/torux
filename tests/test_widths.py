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
