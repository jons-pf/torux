"""Every chain's band held against every surface the answer verified (torux.bands): an estimate is cut on a side only
by a surface the chain's crossings put on that side, an estimate whose flux the crossings contradict, or with a surface
inside it that cannot be placed, gives no band, a chain with no crossings gives none when a surface lies inside it, a
bracket moves to a nearer placed surface, and the largest search keeps every surface its requests verified."""
import numpy as np
import pytest

from torux import ask, bands, widths
from torux.torus import TorusState

NFP = 2
K = 0.2                       # flux = K r^2 for the made-up tori (circles of radius r about (1, 0))


def _circle(r, iota=-0.2, R0=1.0, M=2, N=2):
    rc = np.zeros((M + 1, 2 * N + 1))
    zs = np.zeros((M + 1, 2 * N + 1))
    rc[0, N], rc[1, N], zs[1, N] = R0, r, r
    return TorusState(rc, zs, iota, K * r * r, M, N, NFP)


def _ring(r, q, R0=1.0, phase=0.0):
    th = phase + 2 * np.pi * np.arange(q) / q
    return [[R0 + r * np.cos(t), r * np.sin(t)] for t in th]


def _chain(rc=0.07, q=11, flux=None, half=None, found="stepped across", lines=True):
    c = dict(p=-1, q=q, iota=-NFP / q, found_in=[found], O=[], X=[])
    if lines:
        c["O"] = [dict(residue=1e-3, section=_ring(rc, q), plane_phi=0.0)]
        c["X"] = [dict(residue=-1e-3, section=_ring(rc, q, phase=np.pi / q), plane_phi=0.0)]
    if flux is not None:
        c["flux_Wb"] = flux
    if half is not None:
        c["half_width_Wb"] = half
    return c


def _f(r):
    return K * r * r


def _pool(*radii):
    return bands.pool(([_circle(r) for r in radii], "search"))


def test_an_estimate_is_cut_only_on_the_side_of_the_surface_that_cuts_it():
    """The chain's crossings at r = 0.07: a surface at r = 0.0695 lies inside it. An estimate reaching below that
    surface is cut there and nowhere else: its far end stays, and flux_Wb and half_width_Wb are the middle and half
    of what is left, the estimate kept beside them."""
    f, h = _f(0.07), _f(0.07) - _f(0.069)
    c = _chain(flux=f, half=h)
    S = _pool(0.0695, 0.075)
    assert bands.fit_chain(c, S) == "cut"
    lo = _f(0.0695)
    assert c["band_cut_Wb"] == [pytest.approx(lo), None]
    assert c["flux_Wb"] - c["half_width_Wb"] == pytest.approx(lo)
    assert c["flux_Wb"] + c["half_width_Wb"] == pytest.approx(f + h)
    assert c["flux_estimate_Wb"] == f and c["half_width_pendulum_Wb"] == h
    assert c["between_verified_Wb"] == [pytest.approx(lo), pytest.approx(_f(0.075))]
    assert c["placement"]["placed"] == 2 and "unplaced" not in c["placement"]
    # flagged: the band it reported held that surface, and holds none now
    assert c["band_held_verified_Wb"] == [pytest.approx(lo)] and bands.held(c, S) == []
    # fitted again (the same answer saved and fitted twice), the same
    before = dict(c)
    bands.fit_chain(c, S)
    assert c == before


def test_an_estimate_inside_its_placed_surfaces_is_left_as_it_is():
    f, h = _f(0.07), 1e-7
    c = _chain(flux=f, half=h)
    assert bands.fit_chain(c, _pool(0.06, 0.08)) is None
    assert c["flux_Wb"] == f and c["half_width_Wb"] == h
    assert "flux_estimate_Wb" not in c and "band_withdrawn" not in c


def test_an_estimated_flux_beyond_a_surface_the_crossings_place_gives_no_band():
    """The flux interpolated in iota lies outside a surface that the chain's crossings put outside the chain (the
    case of -1/11 at 64 degrees, ratio 4.5, whose band lay wholly beyond its island): no band, the estimate kept."""
    f = _f(0.072)
    c = _chain(flux=f, half=1e-6)
    assert bands.fit_chain(c, _pool(0.06, 0.071)) == "withdrawn"
    assert "flux_Wb" not in c and "half_width_Wb" not in c
    assert c["flux_estimate_Wb"] == f and c["half_width_pendulum_Wb"] == 1e-6
    assert "outside the verified surface" in c["band_withdrawn"]
    assert c["between_verified_Wb"] == [pytest.approx(_f(0.06)), pytest.approx(_f(0.071))]


def test_a_surface_inside_the_estimate_that_cannot_be_placed_withdraws_it():
    """A surface closer to every crossing than the lines' accuracy cannot be placed: it lies on one side of the chain,
    but not one that can be told, so no band is given rather than one cut on a guessed side."""
    m = widths.LINE_M
    f = _f(0.07)
    c = _chain(flux=f, half=_f(0.071) - f)
    S = _pool(0.07 + 0.3 * m, 0.06)
    assert bands.fit_chain(c, S) == "withdrawn"
    assert "do not place" in c["band_withdrawn"]
    (u,) = c["placement"]["unplaced"]
    assert u["flux_Wb"] == pytest.approx(_f(0.07 + 0.3 * m)) and u["nearest_m"] < m


def test_a_chain_without_crossings_gives_no_band_with_a_surface_inside_it():
    f = _f(0.07)
    c = _chain(flux=f, half=_f(0.071) - f, lines=False)
    assert bands.fit_chain(c, _pool(0.0705)) == "withdrawn"
    assert "no crossings" in c["band_withdrawn"]
    c = _chain(flux=f, half=1e-9, lines=False)
    assert bands.fit_chain(c, _pool(0.0705)) is None and c["half_width_Wb"] == 1e-9


def test_a_bracket_moves_to_a_nearer_placed_surface_and_says_so():
    c = _chain(flux=_f(0.07), half=1e-7)
    c["bracket"] = dict(inner_Wb=_f(0.05), outer_Wb=_f(0.09), one_sided=False, inner_surface=0, outer_surface=1)
    bands.fit_chain(c, _pool(0.05, 0.065, 0.09))
    b = c["bracket"]
    assert b["inner_Wb"] == pytest.approx(_f(0.065)) and b["moved_from_Wb"] == {"inner": _f(0.05)}
    assert b["inner_surface"] is None and b["outer_Wb"] == _f(0.09)
    assert b["half_width_Wb"] == pytest.approx(0.5 * (_f(0.09) - _f(0.065)))
    # a surface between the two that cannot be placed: no half width or middle
    c = _chain(flux=_f(0.07), half=1e-9)
    c["bracket"] = dict(inner_Wb=_f(0.05), outer_Wb=_f(0.09), one_sided=False)
    S = _pool(0.05, 0.07 + 0.3 * widths.LINE_M, 0.09)
    bands.fit_chain(c, S)
    b = c["bracket"]
    assert b["unplaced_Wb"] and b["half_width_Wb"] is None and b["inner_Wb"] is None and b["outer_Wb"] is None
    assert b["withdrawn_Wb"] == [_f(0.05), _f(0.09)] and "do not place" in b["withdrawn"]
    assert c["band_held_verified_Wb"] == [pytest.approx(_f(0.07 + 0.3 * widths.LINE_M))]
    assert bands.held(c, S) == []


def test_the_fit_summary_counts_what_it_did():
    cs = [_chain(flux=_f(0.07), half=_f(0.0702) - _f(0.07)), _chain(flux=_f(0.072), half=1e-9),
          _chain(q=13, rc=0.05, flux=_f(0.05), half=1e-9)]
    out = bands.fit(cs, _pool(0.06, 0.0701, 0.071))
    assert out["cut"] == 1 and out["withdrawn"] == 1 and out["chains"] == 3 and out["surfaces"] == 3
    assert out["held"] == 1 and out["still_held"] == 0


def test_the_search_keeps_every_surface_its_requests_verified():
    """A verified surface at the flux asked, one cut short by its deadline, and a family's verified lower bound are
    kept; an unverified range and a failure are not."""
    from torux.api import Result
    T = _circle(0.07)
    ok = Result("surface", 1.0, 0.0, surface=T)
    late = Result("surface", 1.0, 0.0, surface=T, record=dict(refine_stop="deadline"))
    rng = Result("range", 1.0, 0.0, surface=T, record=dict(range=dict(verified=True, verified_lower_bound=0.1)))
    bad = Result("range_unverified", 1.0, 0.0, surface=T, record=dict(range=dict(verified=True)))
    none = Result("undetermined", 1.0, 0.0)
    assert [ask._verified_torus(r) is not None for r in (ok, late, rng, bad, none)] == [True, True, True, False,
                                                                                       False]
