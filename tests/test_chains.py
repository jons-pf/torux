"""The island chains a request identifies, in its answer whatever the outcome (torux.chainlist): read from the
records the stages write, kept per request and for a whole search, through save and load. No solve: the records
are as torux.analysis writes them."""
import types

import numpy as np
import pytest

from torux import api, ask, chainlist, chains
from torux.analysis import chain_record
from torux.answer import Answer
from torux.orbits import ClosedLine

NFP = 2


def _line(residue, R=0.3, a=0.01):
    """A symmetric closed line of period 6 (q = 6 field periods) round (R, 0) at radius a."""
    return ClosedLine(q=6, nfp=NFP, phi0=0.0, a=np.array([R, a]), b=np.array([0.0, a]), res=1e-13, converged=True,
                      residue=residue)


def _chain():
    ch = chains.Chain(p=-1, q=6, iota_r=-1 / 3, lines=[_line(2.5e-5), _line(-2.4e-5, a=0.012)], flux_r=-0.00145,
                      shear=-3.3, half_width_est=4.9e-4)
    ch.O, ch.X = ch.lines[:1], ch.lines[1:]
    ch.info["shear_at"] = dict(index=6, n_tori=10, flux=-0.00129, iota=-0.33253)
    return ch


def test_the_record_of_a_chain_keeps_every_crossing_of_its_lines():
    """A chain the family met is recorded with each closed line's q crossings of the plane phi = 0, not only the
    first, and with the family torus its shear (and so its half width) was taken at."""
    ch = _chain()
    rec = chain_record(ch)
    for d, ln in zip(rec["lines"], ch.lines):
        assert d["section"] == ln.section_points(0.0).tolist() and len(d["section"]) == 6
        assert d["R0"] == d["section"][0] and d["residue"] == ln.residue
    assert rec["shear_at"] == ch.info["shear_at"]
    assert (rec["p"], rec["q"], rec["half_width_est"], rec["flux_r"]) == (-1, 6, 4.9e-4, -0.00145)


def test_a_chain_says_which_family_torus_its_shear_is_from(monkeypatch):
    """torux.chains.analyse_chain: the shear of the pendulum half width is the family's at one torus, and the record
    says which (its index, flux and iota)."""
    monkeypatch.setattr(chains, "chain_lines", lambda *a, **k: [])
    tori = [types.SimpleNamespace(flux=f, iota=i, nfp=NFP) for f, i in
            ((0.00065, -0.33033), (0.00099, -0.33148), (0.00129, -0.33253), (0.00183, -0.33423))]
    ch = chains.analyse_chain(None, tori, -1, 6)
    k = int(np.argmin([abs(t.iota + 1 / 3) for t in tori]))
    assert ch.info["shear_at"] == dict(index=k, n_tori=4, flux=tori[k].flux, iota=tori[k].iota)
    fl, io = np.array([t.flux for t in tori]), np.array([t.iota for t in tori])
    assert ch.shear == float(np.gradient(io, fl)[k])


def test_a_chain_left_without_its_elliptic_line_has_it_traced(monkeypatch):
    """torux.chains.analyse_chain: where Newton from the family's torus finds no elliptic line of the chain (it ended on
    the magnetic axis), the line found by tracing (traced_chain_lines) is the chain's, its half width rests on it and
    the record and torux.chainlist say how many lines were traced, and the jump's ladder (width_scale) is left as it was,
    on Newton's elliptic lines only. With an elliptic line from Newton, or without the axis, nothing is traced."""
    X, O = _line(-0.16, a=0.012), _line(0.83)
    tori = [types.SimpleNamespace(flux=f, iota=i, nfp=NFP) for f, i in
            ((0.00065, -0.33033), (0.00099, -0.33148), (0.00129, -0.33253))]
    calls = []

    def traced(engine, torus, axis, p, q, known=(), deadline=None, info=None, verbose=False):
        calls.append((p, q, list(known)))
        return [O]
    monkeypatch.setattr(chains, "traced_chain_lines", traced)
    monkeypatch.setattr(chains, "chain_lines", lambda *a, **k: [X])
    ch = chains.analyse_chain(None, tori, -1, 6, axis=object())
    assert calls == [(-1, 6, [X])] and ch.O == [O] and ch.X == [X] and ch.info["traced_lines"] == 1
    assert ch.half_width_est == chains.pendulum_half_width(0.83, 6, NFP, ch.shear) and "width_scale" not in ch.info
    rec = chain_record(ch)
    assert rec["traced_lines"] == 1 and [ln["residue"] for ln in rec["lines"]] == [-0.16, 0.83]
    got = chainlist.of_record(dict(nfp=NFP, chains=[rec]))[0]
    assert got["lines_traced"] == 1 and got["O_residue"] == 0.83 and len(got["X"]) == 1
    assert got["half_width_Wb"] == ch.half_width_est
    calls.clear()
    assert chains.analyse_chain(None, tori, -1, 6).O == [] and calls == []
    monkeypatch.setattr(chains, "chain_lines", lambda *a, **k: [X, _line(0.2, a=0.011)])
    ch = chains.analyse_chain(None, tori, -1, 6, axis=object())
    assert calls == [] and "traced_lines" not in ch.info and "width_scale" in ch.info


def _pendulum(q, p, nfp, shear_J, eps):
    """Field lines of one resonant harmonic on a sheared transform, phi as time, J = Phi / 2 pi:
    theta' = iota_r + shear_J (J - J0), J' = q eps sin(q theta - p nfp phi). The residue of its O line (theta = pi / q
    at phi = 0, J = J0) over the q field periods it takes to close, from the variational equations integrated by
    SciPy, and the separatrix half width in Phi of the pendulum it is, 2 pi 2 sqrt(eps / shear_J)."""
    from scipy.integrate import solve_ivp
    iota_r = p * nfp / q

    def f(phi, y):
        th, J = y[0], y[1]
        psi = q * th - p * nfp * phi
        M = y[2:].reshape(2, 2)
        A = np.array([[0.0, shear_J], [q * q * eps * np.cos(psi), 0.0]])
        return np.concatenate([[iota_r + shear_J * J, q * eps * np.sin(psi)], (A @ M).ravel()])
    y0 = np.concatenate([[np.pi / q, 0.0], np.eye(2).ravel()])
    sol = solve_ivp(f, (0.0, 2 * np.pi * q / nfp), y0, rtol=1e-12, atol=1e-15, method="DOP853")
    M = sol.y[2:, -1].reshape(2, 2)
    assert abs(sol.y[0, -1] - (np.pi / q + 2 * np.pi * p)) < 1e-9 and abs(sol.y[1, -1]) < 1e-12   # it closed
    return (2.0 - np.trace(M)) / 4.0, 2 * np.pi * 2 * np.sqrt(eps / shear_J)


def test_the_pendulum_half_width_is_the_pendulums(monkeypatch):
    """torux.chains' half width of a chain, from its O line's residue and the family's shear, against the separatrix
    of the pendulum whose lines have that residue: equal to the small-residue order, for chains of 1 to 23 islands.
    The family's tori sample iota = iota_r + shear (Phi - Phi_r)."""
    nfp = 2
    for p, q, shear_Phi, eps in ((-1, 6, -3.3, -2e-9), (1, 1, 5.0, 3e-10), (-4, 23, -6.2, -1e-11), (2, 5, 0.8, 1e-9)):
        shear_J = 2 * np.pi * shear_Phi
        R, width = _pendulum(q, p, nfp, shear_J, eps)
        assert 0 < R < 1e-3
        Fr = 0.0015
        tori = [types.SimpleNamespace(flux=Fr + d, iota=p * nfp / q + shear_Phi * d, nfp=nfp)
                for d in (-6e-4, -3e-4, 2e-4, 5e-4)]
        monkeypatch.setattr(chains, "chain_lines", lambda *a, R=R, **k: [types.SimpleNamespace(residue=R)])
        ch = chains.analyse_chain(None, tori, p, q)
        assert abs(ch.shear - shear_Phi) < 1e-9 * abs(shear_Phi) and abs(ch.flux_r - Fr) < 1e-12
        assert abs(ch.half_width_est - width) < 1e-3 * width, (p, q, ch.half_width_est, width)


def _points(q, r, R=0.3):
    """q distinct crossings round (R, 0) at radius r."""
    return [[R + r * np.cos(2 * np.pi * k / q), r * np.sin(2 * np.pi * k / q)] for k in range(q)]


def test_every_kind_of_record_gives_its_chains():
    """The family's chains, an island band's chain (torux.analysis's and torux.islandband's records) and an edge
    barrier's closed lines, each as far as its stage computed it."""
    rec = dict(nfp=NFP, chains=[chain_record(_chain())],
               certificate=dict(kind="edge_barrier", ok=False, plane_phi=0.0,
                                screened=[dict(q=5, p=-2, residue=-0.41, points=_points(5, 0.1), plane_phi=0.0,
                                               lost=[]), dict(q=7, p=-3, residue=0.2, points=_points(7, 0.11),
                                                              plane_phi=0.0, lost=[]), dict(reason="not hyperbolic")],
                                candidates=[dict(q=5, p=-2, residue=-0.41, ok=False, reason="not lost")]),
               bracket_certificate=dict(kind="island_band_manifold", ok=False, interval=[0.0013, 0.0016],
                                        chain=dict(p=-1, q=6, X_residues=[-2.4e-5], O_residues=[2.5e-5])))
    got = chainlist.of_record(rec)
    fam = [c for c in got if c["found_in"] == "family"][0]
    assert (fam["p"], fam["q"], fam["iota"], fam["flux_Wb"], fam["half_width_Wb"]) == (-1, 6, -1 / 3, 0.00145, 4.9e-4)
    assert fam["O_residue"] == 2.5e-5 and [x["residue"] for x in fam["X"]] == [-2.4e-5]
    assert len(fam["O"][0]["section"]) == 6 and fam["O"][0]["plane_phi"] == 0.0
    band = [c for c in got if c["found_in"] == "island band"][0]
    assert band["band_Wb"] == [0.0013, 0.0016] and band["O_residue"] == 2.5e-5 and not band["certified"]
    edge = {c["q"]: c for c in got if c["found_in"] == "edge barrier"}
    assert sorted(edge) == [5, 7] and not edge[5]["certified"]
    assert (edge[5]["p"], edge[5]["iota"], len(edge[5]["X"]), edge[5]["X"][0]["residue"]) == (-2, -0.8, 1, -0.41)
    assert len(edge[5]["X"][0]["section"]) == 5 and edge[7]["O"][0]["residue"] == 0.2 and edge[7]["X"] == []
    # torux.analysis.certify_request's island band, its chain a pair, and a record with no chain at all
    one = chainlist.of_record(dict(nfp=NFP, certificate=dict(kind="island_band", ok=True, chain=(-1, 6),
                                                              O_residue=2.5e-5, interval=[0.0013, 0.0016])))
    assert one[0]["certified"] and one[0]["q"] == 6 and one[0]["O"] == [dict(residue=2.5e-5)]
    assert chainlist.of_record(dict(nfp=NFP, chains=[], jumps=[])) == [] and chainlist.of_record(None) == []


def test_a_request_that_fails_keeps_its_chains(monkeypatch, tmp_path):
    """A single request by flux that ends undetermined: the chains of its record are in its evidence, and both are
    there after save and load."""
    rec = dict(nfp=NFP, status="undetermined", chains=[chain_record(_chain())], jumps=[dict(p=-1, q=6, ok=False)])
    monkeypatch.setattr(api, "_solve", lambda req, src, surfs, budget, t0, deadline: api.Result(
        "undetermined", budget, 0.0, record=rec))
    res = api.solve(types.SimpleNamespace(id="coils", request=None, parts=[]), 0.006, budget_s=10.0)
    assert res.status == "undetermined"
    assert [(c["p"], c["q"], c["found_in"]) for c in res.evidence["chains"]] == [(-1, 6, "family")]
    path = tmp_path / "a.json"
    Answer._of(res, 0.006).save(str(path))
    back = Answer.read(str(path))
    assert back.evidence["chains"] == res.evidence["chains"]
    assert back.record["chains"][0]["lines"][1]["section"] == rec["chains"][0]["lines"][1]["section"]


def _cnt(tilt_deg, ratio, i_il=170e3):
    """The Columbia Non-neutral Torus's coils: two circular interlinked coils of radius 0.405 m, centres 0.313 m from
    the machine axis, their planes tilt_deg apart, and two circular poloidal field coils of radius 1.08 m at +-0.405 m,
    with I_IL / I_PF = ratio (two field periods, stellarator symmetric)."""
    import torux
    a, d, r_pf, h_pf = 0.405, 0.313, 1.08, 0.405
    th = np.radians(tilt_deg) / 2.0
    s, c = np.sin(th), np.cos(th)
    coef = np.zeros((4, 3, 3))
    coef[0, 0], coef[0, 1], coef[0, 2] = [d, 0.0, a], [0.0, a * s, 0.0], [0.0, -a * c, 0.0]
    coef[1, 0], coef[1, 1], coef[1, 2] = [-d, 0.0, -a], [0.0, a * s, 0.0], [0.0, a * c, 0.0]
    coef[2, 0], coef[2, 1], coef[2, 2] = [0.0, 0.0, r_pf], [0.0, r_pf, 0.0], [h_pf, 0.0, 0.0]
    coef[3, 0], coef[3, 1], coef[3, 2] = [0.0, 0.0, r_pf], [0.0, -r_pf, 0.0], [-h_pf, 0.0, 0.0]
    i_pf = -i_il / ratio
    return torux.Source(id=f"cnt_tilt{tilt_deg:g}_ratio{ratio:g}", parts=[torux.FourierCurves(coef, np.array(
        [i_il, -i_il, i_pf, -i_pf]))], nfp=2, stellsym=True)


def test_a_request_past_the_last_chain_keeps_the_chain_and_names_its_edge_barrier(tmp_path):
    """CNT at 88 degrees and I_IL / I_PF = 3.632, whose surfaces end within 2e-6 Wb of the axis, asked for 3.79e-4 Wb:
    the family stalls at the iota = -2/3 chain and cannot jump it, so the only certificate tried is an edge barrier.
    The answer says so and no island band, and it keeps the chain (its p/q and lines) and the closed lines the edge
    barrier screened beyond it, with their crossings, through save and load. The elliptic line Newton from the family's
    torus converged to there is the magnetic axis gone round three times (its three crossings one point): it is left
    out, and the chain's own elliptic line is found by tracing instead (torux.chains.traced_chain_lines), through
    R = 0.2340 m on Z = 0 with residue 0.338 (an independent tracer has R = 0.23407 and 0.23401 m, residues 0.33664 and
    0.33871, at I_IL / I_PF = 3.63 and 3.6325), and the half width rests on it."""
    import torux
    a = torux.solve(_cnt(88, 3.632), flux=0.00037895125037584076, budget_s=300)
    text = " ".join(str(a).split())
    assert a.status == "undetermined" and a.record["certificate"]["kind"] == "edge_barrier"
    assert "an edge barrier, which did not hold the request" in text and "island band" not in text
    fam = [c for c in a.evidence["chains"] if c["found_in"] == "family"]
    assert [(c["p"], c["q"]) for c in fam] == [(-1, 3)] and abs(fam[0]["iota"] + 2 / 3) < 1e-15
    assert fam[0]["lines_of_lower_period"] >= 1, fam[0]
    assert all(len({tuple(np.round(p, 8)) for p in ln["section"]}) == 3 for ln in fam[0]["O"] + fam[0]["X"])
    assert fam[0]["lines_traced"] == 1 and len(fam[0]["O"]) == 1, fam[0]
    O = fam[0]["O"][0]
    assert 0.336 < O["residue"] < 0.340 and fam[0]["O_residue"] == O["residue"] and fam[0]["half_width_Wb"] > 0
    assert any(0.2338 < R < 0.2343 and abs(Z) < 1e-9 for R, Z in O["section"]), O["section"]
    edge = [c for c in a.evidence["chains"] if c["found_in"] == "edge barrier"]
    assert edge and all(ln.get("section") for c in edge for ln in c["O"] + c["X"])
    path = tmp_path / "cnt88.json"
    a.save(str(path))
    assert Answer.read(str(path)).evidence["chains"] == a.evidence["chains"]


def test_a_line_of_lower_period_is_not_a_line_of_the_chain():
    """A closed line whose q crossings are one point (the magnetic axis gone round q times, which Newton can converge
    to from a chain's starting curve) is left out of the chain, and the half width is taken again from the chain's own
    elliptic line, or there is none."""
    axis = [[0.2917505075, 0.0]] * 6
    ring = [[0.3 + 0.01 * np.cos(k * np.pi / 3), 0.01 * np.sin(k * np.pi / 3)] for k in range(6)]
    rec = dict(p=-1, q=6, iota_r=-1 / 3, flux_r=-0.00145, shear=-3.3, half_width_est=9.9e-4,
               lines=[dict(residue=0.049, section=axis), dict(residue=2.5e-5, section=ring),
                      dict(residue=-2.4e-5, section=ring[1:] + ring[:1])])
    got = chainlist.of_record(dict(nfp=NFP, chains=[rec]))[0]
    assert got["lines_of_lower_period"] == 1 and got["O_residue"] == 2.5e-5 and len(got["O"]) == 1
    assert got["half_width_Wb"] == chains.pendulum_half_width(2.5e-5, 6, NFP, -3.3)
    alone = chainlist.of_record(dict(nfp=NFP, chains=[dict(rec, lines=rec["lines"][:1] + rec["lines"][2:])]))[0]
    assert alone["O"] == [] and "half_width_Wb" not in alone and len(alone["X"]) == 1
    edge = dict(kind="edge_barrier", ok=False, plane_phi=0.0,
                screened=[dict(q=6, p=-1, residue=0.049, points=axis, plane_phi=0.0)])
    e = chainlist.of_record(dict(nfp=NFP, certificate=edge))[0]
    assert e["O"] == [] and e["lines_of_lower_period"] == 1


def test_the_line_search_knows_a_line_of_lower_period():
    """torux.chains.distinct_crossings, which chain_lines asks of every converged line: q distinct crossings for a line
    of the chain, fewer for the axis gone round q times or a line of period dividing q gone round q / d times."""
    ring = np.array(_points(6, 0.01))
    assert chains.distinct_crossings(ring)
    assert not chains.distinct_crossings(np.array([[0.2917505075, 0.0]] * 13))
    assert not chains.distinct_crossings(np.concatenate([ring[:3], ring[:3]]))


def test_a_certificate_says_what_it_rests_on():
    """torux.chainlist.provenance: each certificate that holds names its construction and gives the closed lines it
    is built on (p, q, residue, crossings of a plane), so that a tracer of one's own can check them; a lost ring
    rests on none, and a certificate that does not hold gives nothing."""
    pts = _points(5, 0.1)
    edge = dict(kind="edge_barrier", ok=True, certificate="manifold", F_up=0.02, plane_phi=0.0,
                orbit=dict(p=-2, q=5, residue=-1.4, points=pts))
    e = chainlist.provenance(edge)
    assert e["construction"] == "edge barrier" and e["bound_Wb"] == 0.02
    assert e["lines"] == [dict(p=-2, q=5, residue=-1.4, plane_phi=0.0, crossings=pts, role="X")]
    ring = chainlist.provenance(dict(kind="edge_barrier", ok=True, certificate="lost_ring", F_up=0.03))
    assert ring["construction"] == "lost ring" and ring["lines"] == []
    band = chainlist.provenance(dict(kind="island_band_manifold", ok=True, chain=dict(p=-1, q=5), phi_s=0.0,
                                     interval=[0.001, 0.0012], X_line=dict(residue=-0.2, points=pts),
                                     O_line=dict(residue=0.2, points=_points(5, 0.11))))
    assert band["construction"] == "manifold island band" and band["interval_Wb"] == [0.001, 0.0012]
    assert [ln["role"] for ln in band["lines"]] == ["X", "O"] and band["lines"][0]["q"] == 5
    old = chainlist.provenance(dict(kind="island_band", ok=True, chain=(-1, 5), interval=[0.001, 0.0012],
                                    O_line=dict(residue=0.2, points=pts, plane_phi=0.0)))
    assert old["construction"] == "island band from island tori" and old["lines"][0]["crossings"] == pts
    assert chainlist.provenance(dict(edge, ok=False)) is None


def test_a_line_says_how_well_it_closes_and_one_that_does_not_close_is_left_out():
    """Each line measured by torux.closure carries closure_m, distance_m and converged into the chain list and a
    certificate's provenance; a line neither converged nor closing to torux.closure.CLOSES_M (an edge barrier's line
    left millimetres from its closed line) is no line of its chain and is counted, and a thin chain's line that closes
    but whose place along its curve is not determined stays, not called converged."""
    from torux.closure import CLOSES_M, CONVERGED_M
    good = dict(q=5, p=-2, residue=-0.41, points=_points(5, 0.1), plane_phi=0.0, closure_m=3e-12, distance_m=4e-12,
                converged=True)
    far = dict(q=7, p=-3, residue=36.85, points=_points(7, 0.12), plane_phi=0.0, closure_m=8e-3, distance_m=8.8e-3,
               converged=False)
    moved = dict(q=7, p=-3, residue=-0.928, points=_points(7, 0.13), plane_phi=0.0, closure_m=1e-14,
                 distance_m=2e-15, converged=True, polished=dict(moved_m=8.9e-6, before=dict(residue=-0.9258)))
    edge = dict(kind="edge_barrier", ok=False, plane_phi=0.0, screened=[good, far, moved],
                candidates=[dict(q=7, p=-3, residue=36.85, ok=False, reason="not lost")])
    e = {c["q"]: c for c in chainlist.of_record(dict(nfp=NFP, certificate=edge))}
    assert [x["residue"] for x in e[5]["X"]] == [-0.41] and e[5]["X"][0]["converged"]
    assert e[5]["X"][0]["closure_m"] == 3e-12 and e[5]["X"][0]["distance_m"] == 4e-12
    assert [x["residue"] for x in e[7]["X"]] == [-0.928] and e[7]["lines_not_converged"] == 1
    assert e[7]["X"][0]["polished"]["moved_m"] == 8.9e-6
    thin = dict(residue=1.8e-9, section=_points(6, 0.01), closure_m=0.1 * CLOSES_M, distance_m=1e3 * CONVERGED_M,
                converged=False)
    off = dict(residue=0.05, section=_points(6, 0.02), closure_m=10 * CLOSES_M, distance_m=10 * CONVERGED_M,
               converged=False)
    fam = chainlist.of_record(dict(nfp=NFP, chains=[dict(p=-1, q=6, iota_r=-1 / 3, flux_r=-0.00145, shear=-3.3,
                                                         lines=[thin, off])]))[0]
    assert [x["residue"] for x in fam["O"]] == [1.8e-9] and not fam["O"][0]["converged"]
    assert fam["lines_not_converged"] == 1
    pv = chainlist.provenance(dict(kind="edge_barrier", ok=True, certificate="manifold", F_up=0.02, plane_phi=0.0,
                                   orbit=dict(good)))
    assert pv["lines"][0]["converged"] and pv["lines"][0]["distance_m"] == 4e-12


@pytest.mark.slow
def test_an_edge_barrier_line_left_millimetres_from_its_closed_line_is_brought_to_it(tmp_path):
    """CNT at 90 degrees, interlinked coils centred at +-0.34425 m, I_IL / I_PF = 2.40: torux 0.2.2's edge barrier
    stored an iota = -3/8 line (q = 16) with residue 36.854 that comes back 8 mm from its crossing after 16 periods.
    torux.closure measures it so, follows it with Newton to the closed line 5.0 mm away, residue 30.9168 (an
    independent tracer: 30.916815), and the chain list then holds that line, converged, with what it replaced."""
    import json
    import torux
    from torux.closure import CONVERGED_M, measure
    from torux.field import FieldEngine
    s, c = np.sin(np.radians(45.0)), np.cos(np.radians(45.0))
    a, d, ipf, iil = 0.405, 0.34425, 10469.555, 25126.932
    coils = [dict(xc=[d, a], xs=[0.0, 0.0], yc=[0.0, 0.0], ys=[0.0, a * s], zc=[0.0, 0.0], zs=[0.0, -a * c],
                  current_A=iil),
             dict(xc=[-d, -a], xs=[0.0, 0.0], yc=[0.0, 0.0], ys=[0.0, a * s], zc=[0.0, 0.0], zs=[0.0, a * c],
                  current_A=-iil),
             dict(xc=[0.0, 1.08], xs=[0.0, 0.0], yc=[0.0, 0.0], ys=[0.0, 1.08], zc=[0.405, 0.0], zs=[0.0, 0.0],
                  current_A=-ipf),
             dict(xc=[0.0, 1.08], xs=[0.0, 0.0], yc=[0.0, 0.0], ys=[0.0, -1.08], zc=[-0.405, 0.0], zs=[0.0, 0.0],
                  current_A=ipf)]
    path = tmp_path / "cnt90.json"
    path.write_text(json.dumps(dict(nfp=2, stellarator_symmetric=True, coils=coils)))
    src = torux.load(str(path))
    stored = [
          [0.13671087260630477, 0.024407669785918566],
          [0.08628096387271104, 0.1971778162654607],
          [0.23795940372874386, 0.20800461641574625],
          [0.22056399931450385, -0.2395998784555492],
          [0.0754329395189476, -0.1834428602903634],
          [0.13221910294492017, -0.0011205112281504276],
          [0.0762004884890356, 0.1806403308128552],
          [0.21582580329349446, 0.24519150936643946],
          [0.24363746440627612, -0.19479090043535732],
          [0.09278077182064481, -0.20193012608115515],
          [0.13873420469912057, -0.033967864676247905],
          [0.11175210809677075, 0.12779168171233582],
          [0.12921966010475622, 0.29671657015706143],
          [0.29504912540877437, 0.01721841528543905],
          [0.1380605034567579, -0.31470025422344194],
          [0.09589761033373585, -0.14098984157157435]]
    ln = dict(q=16, p=-3, residue=36.85391858406757, points=stored, plane_phi=0.0)
    measure(FieldEngine(src, dmin=0.05), src, [(ln, 0.0, "screened")], axis_point=[0.22988850487658802, 0.0], polish_s=120)
    assert ln["converged"] and ln["distance_m"] <= CONVERGED_M and abs(ln["residue"] - 30.916815) < 1e-4, ln
    before = ln["polished"]["before"]
    assert before["residue"] == 36.85391858406757 and before["closure_m"] > 7e-3 and not before["distance_m"] < 1e-3
    assert abs(ln["polished"]["moved_m"] - 5.03e-3) < 5e-5
    edge = dict(kind="edge_barrier", ok=False, plane_phi=0.0, screened=[ln])
    got = chainlist.of_record(dict(nfp=NFP, certificate=edge))[0]
    assert [round(x["residue"], 4) for x in got["X"]] == [30.9168] and got["X"][0]["converged"]
    assert "lines_not_converged" not in got


def test_the_family_stalls_at_the_first_rational_it_meets():
    """CNT at 90 degrees, I_IL / I_PF = 4.70: the family stalls at iota -0.31689. The lowest-order rational ahead is
    -2/13 (iota -4/13), which the jump tries, but the family meets -3/19 (iota -6/19) first, and that is the chain it
    stalled at. Where no rational of q <= 30 lies short of the lowest-order one, that one is the first."""
    assert chains.nearest_rational_ahead(-0.31689, 1e-4, NFP) == (-2, 13)
    assert chains.first_rational_ahead(-0.31689, (-2, 13), NFP) == (-3, 19)
    assert chains.first_rational_ahead(-0.3075, (-2, 13), NFP) is None
    # the rows: the chain met first says the family stalled at it, the one of lowest order past it says not
    first, low = chain_record(_chain()), dict(chain_record(_chain()), p=-2, q=13, iota_r=-4 / 13)
    rows = chainlist.of_record(dict(nfp=NFP, chains=[dict(first, stalled_at=True), dict(low, stalled_at=False)]))
    fam = {(c["p"], c["q"]): c for c in rows if c["found_in"] == "family"}
    assert fam[(-1, 6)]["stalled_at"] and not fam[(-2, 13)]["stalled_at"]
    assert fam[(-1, 6)]["family_iota"] == [-0.33253] and fam[(-1, 6)]["flux_from"] == "extrapolated"
    # a record of 0.2.2 names one chain at a stall: that one
    assert chainlist.of_record(dict(nfp=NFP, chains=[first]))[0]["stalled_at"]
    m = chainlist.merge([(1, [dict(fam[(-2, 13)], request=1)]), (2, [dict(fam[(-2, 13)], stalled_at=True, request=2)])])
    assert m[0]["stalled_at"]
    m = chainlist.merge([(1, [dict(fam[(-2, 13)], request=1)])])
    assert not m[0]["stalled_at"]

