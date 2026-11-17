"""How well each closed field line an answer stores closes.

Every closed line a request's record keeps (a chain's lines, the lines an edge barrier screened, found and was built
on, a certificate's lines) and every line of the chains the answer's family stepped across carries

    closure_m    |P^q(x) - x|: how far from its stored crossing x the line comes back after its q field periods
    distance_m   |(M - I)^-1 (P^q(x) - x)|: Newton's step from x, the distance from x to the closed line of the map
                 there (M the q-period map's Jacobian at x)
    converged    distance_m <= CONVERGED_M

with P the period map of torux.barrier.FieldSection (the edge barrier's accurate tracer: Gragg-Bulirsch-Stoer at a
calibrated step, lines with a large error estimate traced again with more steps) from the plane of x.  A line whose
crossings come from that map (the edge barrier's) has them from the map's own Newton; any other line (a chain's,
solved as a closed curve by torux.orbits) is followed from x for its q periods, with its own monodromy where the
record keeps it and by central differences otherwise.

Where a residue is small (|residue| below about 1e-6: a thin chain) M - I is nearly singular, so distance_m is large
however well the line closes: its crossings lie on its chain's curve, closing to closure_m, but its place along that
curve is not determined, and it is not called converged.

The lines the edge barrier's orbit search screened are its detection map's orbits polished by a few Newton steps:
some are not converged (in torux 0.2.2, 29 of 157 came back more than 1e-4 m away and one residue read 36.85 for a
line of 30.92).  measure_record follows each such line on with Newton, its steps at most POLISH_STEP_M, for at most
POLISH_S seconds a record, and puts the closed line it reaches in its place (with polished: the move, and the
residue and measure before) when that line is converged, has q distinct crossings and goes round the magnetic axis
as often (p) as the line it replaces.  A chain's line (solved as a closed curve) that does not even close to
CLOSES_M is followed on the same way where its residue leaves Newton a well posed step (|residue| >= 1e-6), and
replaced only by a closed line within CHAIN_MOVE_M of it.  A line that is still not converged keeps its measure, and
torux.chainlist lists it with its chain only where it closes (closure_m <= CLOSES_M); otherwise it counts it in
lines_not_converged.

No certificate rests on these measures and none is changed by them: a certificate's own lines are measured, never
moved.
"""
from __future__ import annotations

import math
import time

import numpy as np

CONVERGED_M = 1e-6         # a line is converged when Newton's step from its stored crossing is at most this (m)
CLOSES_M = 1e-7            # a line that is not converged is still listed with its chain when it closes to this (m)
POLISH_STEP_M = 1e-3       # the largest Newton step taken when a screened line is brought to convergence (m)
POLISH_IT = 30
POLISH_S = 60.0            # seconds a record may spend bringing its lines to convergence
CHAIN_MOVE_M = 1e-4        # a chain's line is replaced only by a closed line this near it (m)
SAME_POINT_M = 1e-8        # (torux.chainlist): crossings nearer each other than this are one point


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _pts(ln):
    P = ln.get("points", ln.get("section", ln.get("crossings")))
    try:
        P = np.asarray(P, float).reshape(-1, 2)
    except (TypeError, ValueError):
        return None
    return P if len(P) and np.all(np.isfinite(P)) else None


def _cert_lines(c, out, seen):
    """The closed lines a certificate record keeps (and the records it nests): (line dict, plane, from the edge
    barrier's orbit search)."""
    if not isinstance(c, dict) or id(c) in seen:
        return
    seen.add(id(c))
    ph = _num(c.get("plane_phi", c.get("phi_s", 0.0))) or 0.0
    for k in ("orbit", "O_line", "X_line"):
        ln = c.get(k)
        if isinstance(ln, dict) and _pts(ln) is not None:
            out.append((ln, _num(ln.get("plane_phi", ph)) or 0.0, None))
    for k in ("screened", "orbits_found", "X_orbits"):
        for ln in c.get(k) or []:
            if isinstance(ln, dict) and _pts(ln) is not None:
                out.append((ln, _num(ln.get("plane_phi", ph)) or 0.0, "screened" if k == "screened" else None))
    for k in ("manifold", "edge_barrier", "survivor", "certificate"):
        _cert_lines(c.get(k), out, seen)


def record_lines(rec):
    """Every closed line a request's record keeps: [(line dict, plane phi, kind)], each dict once; kind 'screened'
    (an edge barrier's screened line), 'chain' (a chain's line) or None (a certificate's line, or an orbit the edge
    barrier's search found: measured, never moved)."""
    rec = rec if isinstance(rec, dict) else {}
    out, seen = [], set()
    fams = list(rec.get("chains") or []) + list((rec.get("range_family") or {}).get("chains") or [])
    for c in fams + list(rec.get("crossed") or []):
        if isinstance(c, dict):
            for ln in c.get("lines") or []:
                if isinstance(ln, dict) and _pts(ln) is not None:
                    out.append((ln, _num(ln.get("plane_phi", 0.0)) or 0.0, "chain"))
    certs = [rec.get(k) for k in ("certificate", "bracket_certificate", "edge_barrier", "bracket_edge_barrier")]
    certs.append((rec.get("range") or {}).get("upper_bound_evidence"))
    certs += [a for a in (rec.get("proxy_band_attempts") or []) if isinstance(a, dict)]
    for c in certs:
        _cert_lines(c, out, seen)
    uniq, ids = [], set()
    for t in out:
        if id(t[0]) not in ids:
            ids.add(id(t[0]))
            uniq.append(t)
    return uniq


def _section(engine, source, phi, x, sp=None):
    from . import barrier as BR
    nfp = int(source.nfp)
    R, Z = float(x[0]), float(x[1])
    B = np.asarray(engine.evaluate(np.array([[R * math.cos(phi), R * math.sin(phi), Z]]), grad=False, A=False)["B"])
    Bphi = -B[0, 0] * math.sin(phi) + B[0, 1] * math.cos(phi)
    sign = 1.0 if Bphi > 0 else -1.0
    if sp is None:
        pts = np.array([[R, Z], [R * 0.98, Z], [R * 1.02, Z]])
        sp, _ = BR.calibrate_steps(engine, nfp, sign, pts)
    sec = BR.FieldSection(engine, source, sign=sign, scale=0.5 * R, steps_per_period=sp, phi0=phi)
    sec.stop_on_err = True
    return sec, sp


def _put(ln, closure, step):
    ln["closure_m"] = None if closure is None or not math.isfinite(closure) else float(closure)
    ln["distance_m"] = None if step is None or not math.isfinite(step) else float(step)
    ln["converged"] = bool(ln["distance_m"] is not None and ln["distance_m"] <= CONVERGED_M)


def _follow(sec, items):
    """closure and Newton step of lines [(dict, P (q, 2), M or None)] on the section's map: one trace a line where
    M is known, central differences (torux.barrier.orbits_many) where it is not."""
    from . import barrier as BR
    known = [(ln, P, M) for ln, P, M in items if M is not None]
    other = [(ln, P) for ln, P, M in items if M is None]
    if known:
        X = np.array([P[0] for _, P, _ in known])
        Y = X.copy()
        qs = np.array([len(P) for _, P, _ in known])
        lost = np.zeros(len(Y), bool)
        for k in range(int(qs.max())):
            act = np.flatnonzero((qs > k) & ~lost)
            if not len(act):
                break
            y, lo, _ = sec.step(Y[act], np.ones(len(act)))
            Y[act] = y
            lost[act] |= lo
        for i, (ln, P, M) in enumerate(known):
            if lost[i]:
                _put(ln, None, None)
                continue
            F = Y[i] - X[i]
            try:
                s = float(np.linalg.norm(np.linalg.solve(np.asarray(M, float) - np.eye(2), F)))
            except np.linalg.LinAlgError:
                s = math.inf
            _put(ln, float(np.linalg.norm(F)), s)
    if other:
        orbs = [BR.Orbit(len(P), P, np.eye(2)) for _, P in other]
        got = BR.orbits_many(sec, orbs, polish=0)
        for (ln, P), o in zip(other, got):
            if o is None:
                _put(ln, None, None)
            else:
                _put(ln, o.info["closure"], o.info.get("step"))


def _newton(sec, x, q, until):
    """Newton on the section's map from x for a closed line of q periods, steps capped at POLISH_STEP_M: the orbit of
    least closure it met (torux.barrier.Orbit), None where the line was lost or the time ran out.  Kept on the source,
    so that the same stored line met again (another request's edge barrier) costs nothing."""
    from . import barrier as BR
    cache = getattr(sec.source, "_closed_line_newton", None)
    if cache is None:
        cache = {}
        try:
            sec.source._closed_line_newton = cache
        except (AttributeError, TypeError):
            pass
    key = (round(float(sec.phi0), 12), int(q), round(float(x[0]), 14), round(float(x[1]), 14))
    if key in cache:
        return cache[key]
    x = np.array(x, float)
    best = None
    for _ in range(POLISH_IT):
        if time.time() > until:
            return None
        o = BR._orbit_once(sec, x, q)
        if o is None:
            break
        if best is None or o.info["closure"] < best.info["closure"]:
            best = o
        if o.info["step"] <= 1e-3 * CONVERGED_M:
            break
        s = np.linalg.solve(o.M - np.eye(2), o.info["F"])
        n = float(np.linalg.norm(s))
        if not math.isfinite(n):
            break
        x = x - (s * (POLISH_STEP_M / n) if n > POLISH_STEP_M else s)
    cache[key] = best
    return best


def _polish(sec, ln, P, axis_point, until, max_move=None):
    """Newton on the section's map from a line's first crossing (_newton): the closed line it reaches replaces the
    stored one when it is converged, has q distinct crossings, the same p (where the line has one) and lies at most
    max_move (m) from it (None: anywhere)."""
    from . import barrier as BR
    q = len(P)
    best = _newton(sec, P[0], q, until)
    o = best
    if o is None or not (o.info["step"] <= CONVERGED_M):
        return False
    d = np.linalg.norm(o.pts[:, None, :] - o.pts[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    if q > 1 and not np.min(d) > SAME_POINT_M:
        return False
    if max_move is not None and not float(np.linalg.norm(o.pts[0] - P[0])) <= max_move:
        return False
    if ln.get("p") is not None:
        if axis_point is None or abs(sec.phi0) > 1e-12:
            return False
        try:
            p = BR.rotation_number(sec, o.pts[0], np.asarray(axis_point, float), q)
        except Exception:  # noqa: BLE001
            p = None
        if p is None or int(p) != int(ln["p"]):
            return False
    before = dict(residue=ln.get("residue"), closure_m=ln.get("closure_m"), distance_m=ln.get("distance_m"),
                  crossing=[float(v) for v in P[0]])
    ln["section" if "section" in ln and "points" not in ln else "points"] = o.pts.tolist()
    ln["residue"] = float(o.residue)
    _put(ln, o.info["closure"], o.info["step"])
    ln["polished"] = dict(moved_m=float(np.linalg.norm(o.pts[0] - P[0])), before=before)
    return True


def measure(engine, source, items, axis_point=None, polish_s=POLISH_S, sp=None):
    """Measure lines [(dict, plane phi, polishable)] in place (module docstring); a line that has its measure
    already (from the map that made it) is only given converged.  The polishable ones that are not converged are
    brought to convergence where they can be, for at most polish_s seconds."""
    if engine is None or not items:
        return
    for ln, _, _ in items:
        if ln.get("closure_m") is not None or ln.get("distance_m") is not None:
            ln["converged"] = bool(_num(ln.get("distance_m")) is not None and ln["distance_m"] <= CONVERGED_M)
    todo = [t for t in items if "converged" not in t[0]]
    planes = sorted({round(ph, 12) for _, ph, _ in items})
    secs = {}
    for ph in planes:
        grp = [t for t in todo if round(t[1], 12) == ph]
        need_sec = grp or any(_to_polish(ln, k) for ln, p_, k in items if round(p_, 12) == ph)
        if not need_sec:
            continue
        x0 = _pts((grp or [t for t in items if round(t[1], 12) == ph])[0][0])[0]
        sec, sp = _section(engine, source, ph, x0, sp)
        secs[ph] = sec
        if grp:
            fol = []
            for ln, _, _ in grp:
                M = ln.get("monodromy", ln.get("M"))
                try:
                    M = None if M is None else np.asarray(M, float).reshape(2, 2)
                except (TypeError, ValueError):
                    M = None
                fol.append((ln, _pts(ln), M))
            _follow(sec, fol)
    until = time.time() + polish_s
    for ln, ph, kind in sorted(items, key=lambda t: len(_pts(t[0]))):
        if _to_polish(ln, kind) and time.time() < until:
            sec = secs.get(round(ph, 12))
            if sec is not None:
                try:
                    _polish(sec, ln, _pts(ln), axis_point, until, max_move=None if kind == "screened" else CHAIN_MOVE_M)
                except Exception:  # noqa: BLE001  (a line that cannot be polished keeps its measure)
                    pass


def _to_polish(ln, kind):
    """An edge barrier's screened line that is not converged; a chain's line that does not even close to CLOSES_M,
    where its residue (|residue| >= 1e-6) leaves Newton a well posed step."""
    if ln.get("converged") is not False:
        return False
    if kind == "screened":
        return True
    c, r = _num(ln.get("closure_m")), _num(ln.get("residue"))
    return kind == "chain" and (c is None or c > CLOSES_M) and r is not None and abs(r) >= 1e-6


def measure_record(engine, source, rec, polish_s=POLISH_S):
    """Every closed line of a request's record measured, the edge barrier's screened ones brought to convergence
    where they can be (module docstring).  Never raises: a line it could not measure has no measure."""
    t0 = time.time()
    try:
        items = record_lines(rec)
        if not items:
            return
        ax = None
        for k in ("edge_barrier", "bracket_edge_barrier", "certificate", "bracket_certificate"):
            c = (rec or {}).get(k)
            if isinstance(c, dict) and c.get("axis_point") is not None:
                ax = c["axis_point"]
                break
        sp = None
        for k in ("edge_barrier", "bracket_edge_barrier", "certificate", "bracket_certificate"):
            c = (rec or {}).get(k)
            if isinstance(c, dict) and isinstance(c.get("tracer_steps"), dict) and c["tracer_steps"].get("steps"):
                sp = int(c["tracer_steps"]["steps"])
                break
        measure(engine, source, items, axis_point=ax, polish_s=polish_s, sp=sp)
        # what was done, in the record: lines measured, brought to convergence, still not converged, seconds
        rec["closed_lines"] = dict(measured=sum(1 for ln, _, _ in items if "converged" in ln),
                                   polished=sum(1 for ln, _, _ in items if ln.get("polished")),
                                   not_converged=sum(1 for ln, _, _ in items if ln.get("converged") is False),
                                   wall_s=round(time.time() - t0, 2))
    except Exception:  # noqa: BLE001  (measuring costs the measures, never the answer)
        pass
