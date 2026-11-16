"""The island chains a request identifies, in its answer whatever the outcome (torux.chainlist): read from the
records the stages write, kept per request and for a whole search, through save and load. No solve: the records
are as torux.analysis writes them."""
import types

import numpy as np

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
    barrier screened beyond it, with their crossings, through save and load. The elliptic line the chain's search
    converged to there is the magnetic axis gone round three times (its three crossings one point): it is left out,
    and no half width rests on it."""
    import torux
    a = torux.solve(_cnt(88, 3.632), flux=0.00037895125037584076, budget_s=300)
    text = " ".join(str(a).split())
    assert a.status == "undetermined" and a.record["certificate"]["kind"] == "edge_barrier"
    assert "an edge barrier, which did not hold the request" in text and "island band" not in text
    fam = [c for c in a.evidence["chains"] if c["found_in"] == "family"]
    assert [(c["p"], c["q"]) for c in fam] == [(-1, 3)] and abs(fam[0]["iota"] + 2 / 3) < 1e-15
    assert fam[0]["lines_of_lower_period"] >= 1, fam[0]
    assert all(len({tuple(np.round(p, 8)) for p in ln["section"]}) == 3 for ln in fam[0]["O"] + fam[0]["X"])
    if not fam[0]["O"]:
        assert "half_width_Wb" not in fam[0] and "O_residue" not in fam[0]
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
