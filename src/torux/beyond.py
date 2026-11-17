"""A torus of the axis's family beyond an island chain that the fixed-iota jump could not cross.

The family from the magnetic axis stalls at a chain and torux.chains.jump_over tries fixed-iota Newton beyond it from
the family extrapolated across. Across a wide island (78 degrees, ratio 5.2 of the CNT scan: the 4/9 chain takes
2.05e-3 to 2.96e-3 Wb of a family that stalls at 2.04e-3) the extrapolation is no start, and every flux beyond came
back undetermined although surfaces run on to 5.26e-3 Wb. Here the start comes from the field lines instead:

    traced_jump(engine, source, cont, tori, chain, axis, direction, deadline=None) -> (TorusState or None, log)

Lines are launched on the outboard symmetry line from the family's last torus outward (torux.seeds.SectionTracer)
and each is classified by torux.seeds._orbit_record: lost, chaotic, island, undecided, or regular (an invariant
circle about the axis fitted to its section to 1e-5 of R). The regular ones whose rotation lies beyond the resonance,
nearest the chain first, then the undecided ones whose circle closed, seed a torus (torux.seeds.torus_from_circle) on
which Newton at the orbit's own transform is run. A torus is returned only when that Newton converged with the
off-grid B.n below the continuation's acceptance, its sections are simple closed curves around the axis
(torux.analysis.torus_topology) and it encloses more flux of the same sign than the family's last torus. It is a
start for the family, not a surface: the family continues from it and whatever is answered from there passes the
solver's refinement, guard and independent verification as any surface does.

The search for the largest (torux.ask) runs it as a stage of its own. Its requests are solved with a dict on the
source (torux.analysis.analyse(traced=...)): a request whose family stalls at a chain with no torus from the jump
says where (stall); the stage traces beyond the chain the families stalled at nearest the largest verified surface
(pending, cross); every later request's family solves the torus found again when it reaches that chain (reuse) and
goes on from it.
"""
from __future__ import annotations

import math
import time

import numpy as np

from . import budget as _budget

# the lines launched: NPROBE on the outboard symmetry line from the last torus outward over SPAN times its distance
# from the axis (the surfaces beyond the 4/9 chain at 78 degrees, 5.2 run to 1.6 times the stalled torus's)
NPROBE = 16
SPAN = 1.0
# the orbits tried for a torus, nearest the chain first
MAX_ORBITS = 4
# periods a line is followed: at least this, and three sweeps of its circle next to the chain's rational
NPER_MIN, NPER_MAX = 300, 1200
# lines beyond the resonance not yet regular are followed this many times longer (at most NPER_LONG periods)
RETRACE, NPER_LONG = 4, 4800
# a rotation this close to the chain's is locked to it (an island's line), not beyond it
LOCKED = 1e-6


def traced_jump(engine, source, cont, tori, chain, axis, direction, deadline=None, verbose=False, keep=None):
    """The torus beyond ``chain`` (torux.chains.Chain) from traced lines, or None, with the log of what was tried.
    ``tori``: the family up to the chain (its last torus the one beyond which lines are launched), ``cont`` its
    torux.family.Continuation, ``direction`` the sign of the family's step in iota (the side beyond the resonance).
    ``keep``: a dict in which the torus found is kept under (p, q) for the later requests of the same question
    (reuse)."""
    from .seeds import (SectionTracer, bphi_sign, outboard_point, trace_engine, _orbit_record, fit_circle,
                        refine_rotation, torus_from_circle)
    from .analysis import torus_topology
    t_start = time.time()
    deadline = _budget.earliest(deadline)
    last = tori[-1]
    nfp = int(last.nfp)
    log = dict(p=int(chain.p), q=int(chain.q), outcome="none", tries=[])

    def left():
        return np.inf if deadline is None else deadline - time.time()

    def finish(sol=None, why=None):
        log["wall_s"] = round(time.time() - t_start, 1)
        if sol is not None:
            log["outcome"] = "torus"
            log["torus"] = dict(flux=float(sol.flux), iota=float(sol.iota), M=int(sol.M), N=int(sol.N),
                                bn_off=float(sol.bn_off))
        elif why:
            log["outcome"] = why
        if verbose:
            print(f"   traced jump {chain.p}/{chain.q}: {log['outcome']} in {log['wall_s']:.0f} s", flush=True)
        return sol, log

    if not getattr(source, "stellsym", True) or not np.isfinite(last.flux) or last.flux == 0:
        return finish(why="not applicable")
    Ra, _ = axis.at(np.array([0.0]))
    Rax = float(Ra[0])
    Rb, _ = outboard_point(last)
    d = Rb - Rax
    if not np.isfinite(d) or d <= 0:
        return finish(why="no outboard distance")
    sgn = bphi_sign(engine, axis)
    try:
        teng = trace_engine(engine, source)
        tr = SectionTracer(teng, nfp, axis, sign=sgn)
        # (calibrated where the surfaces beyond the chain are sought: points nearer the coils ask many times the
        # steps of every line, 318 s of tracing beyond the 3/14 chain at 78 degrees, 5.2)
        tr.calibrate(np.stack([Rax + d * np.array([0.5, 1.0, 1.0 + 0.25 * SPAN]), np.zeros(3)], -1))
    except Exception as e:  # noqa: BLE001  (no tracer: no traced jump)
        return finish(why=f"tracer: {e!r}")
    if left() < 10:
        return finish(why="budget")
    rot_r = chain.iota_r / nfp
    rot_last = last.iota / nfp
    # three sweeps of the circle next to the chain's rational, as torux.seeds.locate sets them
    sweep = 1.0 / max(chain.q * abs(rot_last - chain.p / chain.q), 1e-6)
    nper = int(np.clip(max(150.0 / max(abs(rot_last), 1e-3), 3.0 * sweep), NPER_MIN, NPER_MAX))
    R0s = Rb + d * SPAN * (np.arange(1, NPROBE + 1) / NPROBE)
    t0 = time.time()
    o = tr.orbits(np.stack([R0s, 0 * R0s], -1), nper, deadline=deadline)
    probes = [_orbit_record(R0, o["sec"][i], o["dang"][i], o["lost"][i], engine) for i, R0 in enumerate(R0s)]
    secs = {i: o["sec"][i] for i in range(len(R0s))}
    log["nper"], log["periods"], log["trace_s"] = nper, int(o["n"]), round(time.time() - t0, 1)
    if o["n"] < nper:
        return finish(why="budget")         # (lines cut short are not classified)
    F_last = abs(float(last.flux))

    def past(r):
        """Rotation beyond the resonance, on the far side from the family, and not locked to it."""
        x = r.get("rot")
        return x is not None and np.isfinite(x) and (x - rot_r) * direction > LOCKED

    # lines beyond the resonance that are not regular yet (next to the chain the averages converge slowly):
    # the nearest followed RETRACE times longer, once
    again = sorted([i for i, r in enumerate(probes) if r["kind"] in ("undecided", "chaotic") and past(r)],
                   key=lambda i: -(probes[i].get("digits") or 0))[:MAX_ORBITS]
    if again and left() > 10:
        t0 = time.time()
        o2 = tr.orbits(np.stack([R0s[again], 0 * R0s[again]], -1), min(RETRACE * nper, NPER_LONG), deadline=deadline)
        for j, i in enumerate(again):
            rec = _orbit_record(R0s[i], o2["sec"][j], o2["dang"][j], o2["lost"][j], engine)
            if rec["n"] > probes[i]["n"]:
                probes[i] = rec
                secs[i] = o2["sec"][j]
        log["retrace"] = dict(lines=len(again), periods=int(o2["n"]), wall_s=round(time.time() - t0, 1))
    log["probes"] = [dict(R0=r["R0"], kind=r["kind"], rot=_f(r.get("rot")), digits=_f(r.get("digits")),
                          flux=_f(r.get("flux"))) for r in probes]
    # regular orbits first, then undecided ones whose circle closed (Newton and the acceptance decide), nearest the
    # chain first in each
    ok = [i for i, r in enumerate(probes) if r["kind"] in ("regular", "undecided") and past(r)
          and np.isfinite(r.get("flux", np.nan)) and r["flux"] > F_last and (r["kind"] == "regular" or
                                                                             r.get("digits", 0) >= 5)]
    beyond = sorted(ok, key=lambda i: (probes[i]["kind"] != "regular", probes[i]["flux"]))
    log["regular_beyond"] = sum(1 for i in ok if probes[i]["kind"] == "regular")
    log["undecided_beyond"] = sum(1 for i in ok if probes[i]["kind"] != "regular")
    if not beyond:
        if isinstance(keep, dict) and left() > 10:
            keep.setdefault((int(chain.p), int(chain.q)), None)    # (none: not traced again in this question)
        return finish(why="no regular orbit beyond the chain")
    for i in beyond[:MAX_ORBITS]:
        if left() < 10:
            return finish(why="budget")
        r = probes[i]
        sec = secs[i]
        ent = dict(R0=r["R0"], rot=float(r["rot"]), flux=float(r["flux"]))
        log["tries"].append(ent)
        t1 = time.time()
        try:
            rot = float(r["rot"])
            C = fit_circle(sec, rot)
            if C is not None and len(sec) > 1500 and r["digits"] < 13:
                rot, _ = refine_rotation(sec, rot, max(10 * 10 ** (-r["digits"]), 1e-12), nm=min(C.nm, 64))
                C = fit_circle(sec, rot)
            if C is None or C.nm < 8:
                ent["fail"] = "circle"
                continue
            C.rot = rot
            M = int(min(cont.Mmax, max(last.M, C.nm, 24)))
            N = int(min(cont.Nmax, max(last.N, 24)))
            st = torus_from_circle(tr, C, rot, M, N)
            if st is None:
                ent["fail"] = "line lost over a period"
                continue
            prob = cont.problem(M, N)
            st.flux = float(prob.flux_of(st.rc, st.zs))
            ent.update(M=M, N=N, seed_flux=st.flux)
            iota = rot * nfp
            with _budget.scope(deadline):
                sol = cont.solve(st, st.flux, M, N, maxit=15, fixed_iota=iota, patience=3)
                ent.update(conv=bool(sol.converged), bn=_f(sol.bn_off))
                if not sol.converged or not np.isfinite(sol.bn_off):
                    ent["fail"] = "Newton"
                    continue
                if sol.bn_off > cont.bn_cont:
                    sol = cont.raise_until(sol, sol.flux, cont.bn_cont, fixed_iota=iota)
                    ent.update(raised=[int(sol.M), int(sol.N)], bn=_f(sol.bn_off))
            if not (sol.bn_off < cont.bn_accept and abs(sol.flux) > F_last and sol.flux * last.flux > 0):
                ent["fail"] = "not a torus beyond the family's last"
                continue
            topo = torus_topology(sol, axis)
            if not topo["ok"]:
                ent["fail"] = "sections not simple curves around the axis"
                continue
            ent["wall_s"] = round(time.time() - t1, 1)
            if isinstance(keep, dict):
                keep[(int(chain.p), int(chain.q))] = sol.resized(sol.M, sol.N)
            return finish(sol)
        except Exception as e:  # noqa: BLE001  (one orbit's failure costs that orbit)
            ent["fail"] = repr(e)
        finally:
            ent.setdefault("wall_s", round(time.time() - t1, 1))
    if isinstance(keep, dict) and left() > 10:
        keep.setdefault((int(chain.p), int(chain.q)), None)
    return finish(why="no orbit gave a torus")


def reuse(keep, cont, tori, chain, axis):
    """The torus kept (traced_jump's ``keep``) for this chain by an earlier request of the same question, solved again
    at its flux on this request's continuation, when it lies beyond the family's last torus: (TorusState or None, log
    or None). The same checks as traced_jump's hold."""
    if not isinstance(keep, dict):
        return None, None
    T = keep.get((int(chain.p), int(chain.q)))
    last = tori[-1]
    if T is None or not (abs(T.flux) > abs(last.flux) and T.flux * last.flux > 0):
        return None, None
    from .analysis import torus_topology
    t0 = time.time()
    log = dict(p=int(chain.p), q=int(chain.q), outcome="none", reused=dict(flux=float(T.flux), iota=float(T.iota)))
    try:
        sol = cont.solve(T, T.flux, T.M, T.N, maxit=15, patience=3)
        if sol.converged and np.isfinite(sol.bn_off) and sol.bn_off > cont.bn_cont:
            sol = cont.raise_until(sol, sol.flux, cont.bn_cont)
        ok = (sol.converged and np.isfinite(sol.bn_off) and sol.bn_off < cont.bn_accept
              and abs(sol.flux) > abs(last.flux) and sol.flux * last.flux > 0 and torus_topology(sol, axis)["ok"])
    except Exception as e:  # noqa: BLE001
        log["fail"] = repr(e)
        ok = False
    log["wall_s"] = round(time.time() - t0, 1)
    if not ok:
        log["outcome"] = "reused torus not solved again"
        return None, log
    log["outcome"] = "torus"
    log["torus"] = dict(flux=float(sol.flux), iota=float(sol.iota), M=int(sol.M), N=int(sol.N),
                        bn_off=float(sol.bn_off))
    return sol, log


def stall(keep, tori, chain, axis, direction, dmin):
    """Where a request's family stalled at ``chain`` with no torus beyond it from the jump: kept in ``keep['stalls']``
    (the outermost such torus for each chain the question has not traced yet), for cross."""
    if not isinstance(keep, dict) or (int(chain.p), int(chain.q)) in keep:
        return
    k = (int(chain.p), int(chain.q))
    st = keep.setdefault("stalls", {})
    last = tori[-1]
    if k in st and abs(st[k]["tori"][-1].flux) >= abs(last.flux):
        return
    st[k] = dict(tori=[t.resized(t.M, t.N) for t in tori[-2:]], chain=chain, axis=axis, direction=float(direction),
                 dmin=float(dmin), flux=abs(float(last.flux)))


def pending(keep, floor=0.0):
    """The chains a request's family stalled at that the question has not traced yet, where the family's last torus
    encloses at least ``floor`` (Wb; the largest verified surface: a chain inside it stops nothing), innermost first
    (the one that stops the families nearest the largest verified surface)."""
    if not isinstance(keep, dict):
        return []
    st = keep.get("stalls") or {}
    return sorted([k for k in st if k not in keep and st[k]["flux"] >= floor * (1 - 1e-9)], key=lambda k: st[k]["flux"])


def cross(source, keep, deadline=None, verbose=False, floor=0.0):
    """The question's own stage for the first chain pending(keep, floor) gives: traced_jump from the family's last
    torus there, on an engine and continuation of its own, the torus found (or that none was) kept for the later
    requests. Returns traced_jump's log with the chain's p, q, or None when nothing is pending."""
    ks = pending(keep, floor)
    if not ks:
        return None
    k = ks[0]
    st = keep["stalls"][k]
    from .field import FieldEngine
    from .family import Continuation
    t0 = time.time()
    ax = st["axis"]
    try:
        eng = FieldEngine(source, dmin=st["dmin"])
        cont = Continuation(eng, source.nfp, M0=6, N0=max(16, ax.N), Mmax=128, Nmax=max(128, ax.N),
                            lean_source=source, lean_dmin=st["dmin"])
        cont.fmeas_off = True
        with _budget.scope(deadline):
            _, log = traced_jump(eng, source, cont, st["tori"], st["chain"], ax, st["direction"], deadline=deadline,
                                 verbose=verbose, keep=keep)
    except Exception as e:  # noqa: BLE001  (the stage costs its time, never the question)
        log = dict(p=k[0], q=k[1], outcome=f"error: {e!r}")
    if k not in keep:
        keep[k] = None              # (nothing found, or the stage's time ran out: not traced again in this question)
    log["from_flux_Wb"] = st["flux"]
    log["stage_s"] = round(time.time() - t0, 1)
    return log


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None
