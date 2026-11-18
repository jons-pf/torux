"""The island chains a family of surfaces steps across (torux.crossed), and where every chain of a search lies against
the verified surfaces: the rationals between the family's tori, the placement of closed lines between tori, the
records in the answer, the words, and one real CNT family."""
import time
import types

import numpy as np
import pytest

from torux import api, ask, chainlist, crossed
from torux.torus import TorusState

NFP = 2


def _circle(r, flux, iota, R0=1.0, M=2, N=2):
    """A torus whose sections are circles of radius r about (R0, 0)."""
    rc = np.zeros((M + 1, 2 * N + 1))
    zs = np.zeros((M + 1, 2 * N + 1))
    rc[0, N], rc[1, N], zs[1, N] = R0, r, r
    t = TorusState(rc, zs, iota, flux, M, N, NFP)
    t.converged, t.bn_off = True, 1e-15
    return t


def _ring(r, q, R0=1.0, phase=0.0):
    """q crossings of the plane phi = 0 on the circle of radius r about (R0, 0)."""
    th = phase + 2 * np.pi * np.arange(q) / q
    return [[R0 + r * np.cos(t), r * np.sin(t)] for t in th]


def test_the_rationals_between_two_tori_of_the_family_are_its_crossings():
    """Every reduced p/q with q <= QMAX strictly between the rotations per field period of two neighbouring tori,
    with the index of the inner one, negative iota as torux gives it for CNT."""
    tori = [_circle(0.01 * k, 1e-4 * k, io) for k, io in enumerate((-0.16, -0.17, -0.19, -0.21), start=1)]
    assert crossed.crossings(tori, NFP, 20) == [(-1, 12, 0), (-1, 11, 1), (-1, 10, 2)]
    # q up to 21 adds 2/21 (iota -4/21 = -0.1905, between -0.19 and -0.21)
    assert (-2, 21, 2) in crossed.crossings(tori, NFP, 21)
    # a profile that turns back crosses its rational twice
    back = [_circle(0.01 * k, 1e-4 * k, io) for k, io in enumerate((-0.17, -0.19, -0.17), start=1)]
    assert [c for c in crossed.crossings(back, NFP, 20) if c[:2] == (-1, 11)] == [(-1, 11, 0), (-1, 11, 1)]


def test_a_closed_line_is_placed_between_the_tori_its_crossings_lie_between():
    """Closed lines whose every crossing lies between two tori are between them; one crossing outside is not; and
    place gives the largest torus inside every crossing and the least outside them all."""
    t1, t2, t3 = _circle(0.05, 1e-3, -0.2), _circle(0.10, 4e-3, -0.21), _circle(0.15, 9e-3, -0.22)
    ring = dict(section=_ring(0.07, 11), plane_phi=0.0)
    assert crossed.between([ring], t1, t2) and not crossed.between([ring], t2, t3)
    assert not crossed.between([dict(section=_ring(0.07, 10) + [[1.12, 0.0]], plane_phi=0.0)], t1, t2)
    inner, outer = crossed.place([ring], [t3, t1, t2])
    assert (inner, outer) == (t1, t2)
    beyond = dict(section=_ring(0.2, 13), plane_phi=0.0)
    assert crossed.place([beyond], [t1, t2, t3]) == (t3, None)
    assert crossed.place([dict(section=[], plane_phi=0.0)], [t1]) == (None, None)


def _crossed_record(p=-1, q=11, inner=(3.4e-4, -0.1721), outer=(9.6e-4, -0.1836), lines=True, ok=True):
    rec = dict(p=p, q=q, iota_r=p * NFP / q, kind="crossed", family_Wb=[inner[0], outer[0]],
               family_iota=[inner[1], outer[1]], shear=(outer[1] - inner[1]) / (outer[0] - inner[0]),
               flux_r=8.66e-4, flux_from="interpolated", inner_Wb=inner[0], inner_iota=inner[1],
               outer_Wb=outer[0], outer_iota=outer[1], half_width_est=2.2e-6)
    if lines:
        rec["lines"] = [dict(residue=1.4e-5, res=1e-14, section=_ring(0.07, q), plane_phi=0.0),
                        dict(residue=-1.4e-5, res=1e-14, section=_ring(0.07, q, phase=np.pi / q), plane_phi=0.0)]
        rec["between_verified"] = ok
    return rec


def test_a_stepped_across_chain_is_in_the_answer_with_its_place_between_verified_tori():
    """The record of a stepped-across chain becomes a chain 'stepped across' with its flux between the family tori
    either side, the verified surfaces holding it, its lines and their residues, and its half width from the shear
    between the tori either side. Merged with the same chain where another request's family stalled, the
    stepped-across numbers are kept, and the family's flux is said to be extrapolated."""
    got = chainlist.of_record(dict(nfp=NFP, crossed=[_crossed_record()]))
    assert len(got) == 1
    c = got[0]
    assert (c["p"], c["q"], c["found_in"], c["flux_from"]) == (-1, 11, "stepped across", "between family tori")
    assert c["flux_Wb"] == 8.66e-4 and c["family_Wb"] == [3.4e-4, 9.6e-4]
    assert c["between_verified_Wb"] == [3.4e-4, 9.6e-4] and c["O_residue"] == 1.4e-5
    assert c["half_width_cap_Wb"] == pytest.approx(3.1e-4) and c["half_width_Wb"] == 2.2e-6
    # a pendulum estimate wider than the verified tori either side leave about the chain's flux is cut to that
    wide = chainlist.of_record(dict(nfp=NFP, crossed=[dict(_crossed_record(), half_width_est=5e-4)]))[0]
    assert wide["half_width_pendulum_Wb"] == 5e-4 and wide["half_width_Wb"] == pytest.approx(9.6e-4 - 8.66e-4)
    assert len(c["O"]) == 1 and len(c["X"]) == 1 and len(c["O"][0]["section"]) == 11
    # lines found outside the verified tori: no interval claimed, and the record says so
    bad = chainlist.of_record(dict(nfp=NFP, crossed=[_crossed_record(ok=False)]))[0]
    assert "between_verified_Wb" not in bad and bad["lines_outside_verified"]
    # no line found: the interval from the rotations of the two verified tori either side (Poincare-Birkhoff)
    none = chainlist.of_record(dict(nfp=NFP, crossed=[_crossed_record(lines=False)]))[0]
    assert none["between_verified_Wb"] == [3.4e-4, 9.6e-4] and none["O"] == [] and none["X"] == []
    fam = dict(p=-1, q=11, iota_r=-2 / 11, flux_r=-9.9e-4, shear=-7.0, half_width_est=4e-6,
               lines=[dict(residue=1.4e-5, section=_ring(0.07, 11))],
               shear_at=dict(index=4, n_tori=5, flux=-9.0e-4, iota=-0.1806))
    f = chainlist.of_record(dict(nfp=NFP, chains=[fam]))[0]
    assert f["flux_from"] == "extrapolated" and f["family_Wb"] == [9.0e-4]
    m = chainlist.merge([(1, [dict(f, request=1)]), (3, [dict(c, request=3)])])
    assert len(m) == 1 and m[0]["found_in"] == ["family", "stepped across"] and m[0]["found_by"] == [1, 3]
    assert m[0]["flux_from"] == "between family tori" and m[0]["values_from"] == 3
    # the same rational crossed twice (iota turned back) stays two chains
    two = chainlist.of_record(dict(nfp=NFP, crossed=[_crossed_record(), _crossed_record(
        inner=(2e-3, -0.1836), outer=(3e-3, -0.1721))]))
    assert len(chainlist.merge([(1, two)])) == 2


def test_the_largest_search_records_the_chains_its_answer_stepped_across(monkeypatch):
    """largest=True: the chains the answer's family stepped across are in the answer (found by the answering request,
    'stepped across', in the words), and a chain another request's family stalled at beyond the answer is placed
    outside the verified surface, its flux said to be extrapolated. The search's requests are given what they would be
    without the stage, which takes the time the search left, and the least certified bound is in its record."""
    surf = _circle(0.10, 4e-3, -0.21)
    fam = dict(p=-2, q=9, iota_r=-4 / 9, flux_r=-6.1e-3, shear=-3.0, half_width_est=2e-4,
               lines=[dict(residue=0.05, section=_ring(0.13, 9))], shear_at=dict(index=7, n_tori=8, flux=-5e-3))
    given = []

    def request(src, flux, b, o):
        given.append(b)
        if flux is None:
            return api.Result("range", b, 0.1, record=dict(nfp=2, chains=[fam], profile=[dict(flux=4e-3)],
                                                           range=dict(verified_lower_bound=4e-3)))
        if abs(flux - 4e-3) < 1e-12:
            r = api.Result("surface", b, 0.1, surface=surf, iota=-0.21, record=dict(nfp=2, chains=[]))
            r._family = dict(tori=[surf], axis=None, dmin=0.05, tol=3e-15)
            return r
        cert = dict(kind="edge_barrier", ok=True, rigorous=True, F_up=7e-3)
        return api.Result("no_surface", b, 0.1, record=dict(nfp=2, chains=[], certificate=cert),
                          evidence=dict(certificate=cert))

    def stage(r, src, deadline, **kw):
        # the time the search left, here all but a moment of the budget
        assert deadline - time.time() > 1000 and r.surface is surf
        return [_crossed_record()], [surf, _circle(0.05, 3.4e-4, -0.1721), _circle(0.09, 9.6e-4, -0.1836)], None
    monkeypatch.setattr(ask, "_request", request)
    monkeypatch.setattr(ask, "say", None)
    monkeypatch.setattr(crossed, "stage", stage)
    monkeypatch.setattr(api, "estimate_budget", lambda src, flux: 300.0 if flux is not None else 600.0)
    res = ask._largest(types.SimpleNamespace(id="coils", request=None, nfp=2), 1200.0, {})
    s = res.evidence["search"]
    assert res.status == "surface" and s["flux"] == 4e-3 and s["no_surface_above_Wb"] == 7e-3
    by = {(c["p"], c["q"]): c for c in res.evidence["chains"]}
    st = by[(-1, 11)]
    k = s["answered_by"]["request"]
    assert st["found_in"] == ["stepped across"] and st["found_by"] == [k]
    assert st["between_verified_Wb"] == [3.4e-4, 9.6e-4] and s["requests"][k - 1]["crossed"]["chains"] == 1
    out = by[(-2, 9)]
    assert out["flux_from"] == "extrapolated" and out["between_verified_Wb"] == [4e-3, None]
    assert "reaches_inside_verified" not in out and "half_width_cap_Wb" not in out
    words = " ".join(s["words"])
    assert "stepped across: iota = -2/11 at 0.000866 Wb (between verified surfaces at 0.00034 and 0.00096 Wb" in words
    assert ("iota = -4/9 near 0.0061 Wb by extrapolation from the family torus at 0.005 Wb, half width about "
            "2.0e-04 Wb, its lines outside the verified surface at 0.004 Wb") in words, words
    # the requests are given the budget as without the stage: the family from the axis 0.4 of it
    assert given[0] == pytest.approx(0.4 * 1200.0, abs=1.0)


def test_the_words_say_whose_a_verified_lower_bound_is(monkeypatch):
    """A chain placed against the no-flux request's verified lower bound, which lies beyond the answer and which no
    direct request verified again, is said to lie outside that request's torus, named as such, not outside "the
    verified surface", so that the answer never prints two fluxes as verified surfaces without saying what each is
    (on CNT at 88 degrees, ratio 2.75: reached 7.543e-4 Wb, and a chain outside the lower bound at 1.967e-3 Wb)."""
    lower = _circle(0.12, 5.5e-3, -0.25)
    fam = dict(p=-2, q=9, iota_r=-4 / 9, flux_r=-6.1e-3, shear=-3.0, half_width_est=2e-4,
               lines=[dict(residue=0.05, section=_ring(0.13, 9))], shear_at=dict(index=7, n_tori=8, flux=-5e-3))

    def request(src, flux, b, o):
        if flux is None:
            return api.Result("range", b, 0.1, surface=lower, iota=-0.25, record=dict(
                nfp=2, chains=[fam], profile=[dict(flux=f, converged=True, bn_off=1e-15)
                                              for f in (1e-3, 2e-3, 3e-3, 4e-3, 5.5e-3)],
                range=dict(verified_lower_bound=5.5e-3, verified=True)))
        if flux <= 4e-3:
            # a direct request verifies up to 4e-3 Wb and not beyond, not even at the lower bound's own flux
            t = _circle(0.10 * np.sqrt(flux / 4e-3), flux, -0.21)
            r = api.Result("surface", b, 0.1, surface=t, iota=-0.21, record=dict(nfp=2, chains=[]))
            r._family = dict(tori=[t], axis=None, dmin=0.05, tol=3e-15)
            return r
        return api.Result("undetermined", b, 0.1, record=dict(nfp=2, chains=[]))

    def stage(r, src, deadline, **kw):
        return [_crossed_record()], [r.surface, _circle(0.05, 3.4e-4, -0.1721), _circle(0.09, 9.6e-4, -0.1836)], None
    monkeypatch.setattr(ask, "_request", request)
    monkeypatch.setattr(ask, "say", None)
    monkeypatch.setattr(crossed, "stage", stage)
    monkeypatch.setattr(api, "estimate_budget", lambda src, flux: 300.0 if flux is not None else 600.0)
    res = ask._largest(types.SimpleNamespace(id="coils", request=None, nfp=2), 1200.0, {})
    s = res.evidence["search"]
    assert res.status == "surface" and s["flux"] == 4e-3
    assert any(e["flux"] == 5.5e-3 and e["status"] == "undetermined" for e in s["requests"])
    out = {(c["p"], c["q"]): c for c in res.evidence["chains"]}[(-2, 9)]
    # placed as before: the lower bound is a verified torus and bounds the chain
    assert out["between_verified_Wb"] == [5.5e-3, None]
    assert out["placement"]["inner_from"] == ask.LOWER_BOUND + "the no-flux request"
    words = " ".join(s["words"])
    assert "reached: 0.004 Wb" in words
    assert ("its lines outside the no-flux request's verified torus at 0.0055 Wb, not verified again by a direct "
            "request") in words, words
    assert "outside the verified surface at 0.0055" not in words
    # the stepped-across chain, between the answer family's own verified tori, is worded as before
    assert "between verified surfaces at 0.00034 and 0.00096 Wb" in words


def _cnt(tilt_deg, ratio, d=0.313, i_pf=10469.555):
    """CNT's coils (tests/test_chains.py), with I_PF = 10,469.555 A and I_IL = ratio x I_PF."""
    import torux
    a, r_pf, h_pf = 0.405, 1.08, 0.405
    th = np.radians(tilt_deg) / 2.0
    s, c = np.sin(th), np.cos(th)
    coef = np.zeros((4, 3, 3))
    coef[0, 0], coef[0, 1], coef[0, 2] = [d, 0.0, a], [0.0, a * s, 0.0], [0.0, -a * c, 0.0]
    coef[1, 0], coef[1, 1], coef[1, 2] = [-d, 0.0, -a], [0.0, a * s, 0.0], [0.0, a * c, 0.0]
    coef[2, 0], coef[2, 1], coef[2, 2] = [0.0, 0.0, r_pf], [0.0, r_pf, 0.0], [h_pf, 0.0, 0.0]
    coef[3, 0], coef[3, 1], coef[3, 2] = [0.0, 0.0, r_pf], [0.0, -r_pf, 0.0], [-h_pf, 0.0, 0.0]
    i_il = ratio * i_pf
    return torux.Source(id=f"cnt_tilt{tilt_deg:g}_ratio{ratio:g}", parts=[torux.FourierCurves(coef, np.array(
        [i_il, -i_il, -i_pf, i_pf]))], nfp=2, stellsym=True)


@pytest.mark.slow
def test_cnt_64_degrees_ratio_4_5_steps_across_2_11_and_1_5():
    """CNT at 64 degrees and I_IL / I_PF = 4.5, asked for 2.5e-3 Wb: the family from the axis steps across iota = -2/11
    and -1/5 without stalling. Both are recorded with their O and X lines, every crossing between the two family tori
    either side, those tori verified, and the resonance's flux between them."""
    from torux.analysis import analyse
    from torux import chainlist as CL
    rec = analyse(_cnt(64, 4.5), flux=2.5e-3, budget_s=600, crossed=True)
    assert rec["status"] == "surface"
    got = {(c["p"], c["q"]): c for c in CL.of_record(dict(nfp=2, crossed=rec["crossed"]))}
    for pq in ((-1, 11), (-1, 10)):
        c = got[pq]
        assert c["found_in"] == "stepped across" and len(c["O"]) >= 1 and len(c["X"]) >= 1
        lo, hi = c["between_verified_Wb"]
        assert lo < c["flux_Wb"] < hi and 0 < c["O_residue"] < 1 and all(x["residue"] < 0 for x in c["X"])
        assert c["half_width_Wb"] < hi - lo
    assert 1e-3 < got[(-1, 10)]["flux_Wb"] < 2.5e-3 and got[(-1, 11)]["flux_Wb"] < got[(-1, 10)]["flux_Wb"]


def test_the_stage_after_a_search_that_used_its_budget_takes_at_most_its_own_time(monkeypatch):
    """A search that ends at its deadline: the stage is given CROSSED_S (at most CROSSED_SHARE of the budget) from
    then, and the budget's words say how long it ran past the budget."""
    S = ask._Search(types.SimpleNamespace(id="coils", request=None, nfp=2), "largest", None, 100.0, {})
    S.deadline = time.time() - 1.0
    surf = _circle(0.10, 4e-3, -0.21)
    r = api.Result("surface", 100.0, 0.1, surface=surf, iota=-0.21, record=dict(nfp=2))
    r._ask_entry = dict(flux=4e-3, chains=[], wall_s=1.0)
    S.log.append(r._ask_entry)
    got = {}

    def stage(r_, src, deadline, **kw):
        got["left"] = deadline - time.time()
        time.sleep(1.0)
        return [_crossed_record()], [surf], None
    monkeypatch.setattr(crossed, "stage", stage)
    monkeypatch.setattr(ask, "say", None)
    ask._crossed_stage(S, r)
    assert 0 < got["left"] <= min(ask.CROSSED_S, ask.CROSSED_SHARE * 100.0) + 0.01
    assert r._ask_entry["crossed"]["after_budget_s"] >= 1.0
    res = S.finish(r, [])
    assert any(w.startswith("budget: 100 s, as given, and ") and "after it for the chains" in w
               for w in res.evidence["search"]["words"])


@pytest.mark.slow
def test_cnt_90_degrees_ratio_4_7_the_4_13_chain_has_no_axis_for_its_o_line():
    """CNT at 90 degrees with the IL centres at +-0.38475 m and I_IL / I_PF = 4.70 (Sengupta's case), asked beyond its
    last surface (3.4057e-3 Wb): the family stalls at iota = -4/13. In 0.2.1 the chain's "O line" there was the magnetic
    axis gone round 13 times (13 crossings at R = 0.2917505 m, residue 0.0493447 = sin^2(13 pi iota_axis / 2)) and its
    half width came from it. The chain's lines are both hyperbolic, and no line of the chain is the axis."""
    import torux
    a = torux.solve(_cnt(90, 4.70, d=0.38475), flux=0.0035991634013031416, budget_s=300)
    ax = (a.record or {}).get("axes") or []
    R_axis = float(np.sum(ax[0]["rc"]))
    c = [c for c in a.evidence["chains"] if (c["p"], c["q"]) == (-2, 13) and c["found_in"] == "family"]
    assert c, a.evidence["chains"]
    c = c[0]
    lines = c["O"] + c["X"]
    assert lines and "half_width_Wb" not in c and c["O"] == []
    for ln in lines:
        P = np.array(ln["section"])
        assert len(P) == 13 and len({tuple(np.round(p, 8)) for p in P}) == 13
        assert np.min(np.hypot(P[:, 0] - R_axis, P[:, 1])) > 1e-3
        assert not 0 < ln["residue"] < 1
        assert min(abs(ln["residue"] + 2.3117672), abs(ln["residue"] - 6.4181372)) < 1e-5, ln["residue"]
