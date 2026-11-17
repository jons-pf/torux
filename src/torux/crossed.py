"""The island chains a family of surfaces steps across, and where every chain lies against verified surfaces.

The family from the magnetic axis (torux.family) steps outward in flux, and between two of its tori iota passes
every rational p/q in between.  Where the chain of such a rational is thin enough, the family's Newton converges on
both sides and nothing stops it: the chain is stepped across.  crossed_chains lists every one of them up to a
period QMAX, and for each:

    the rational p/q (q field periods, p poloidal turns: iota = p nfp / q, torux.chains)
    its closed lines (O elliptic, X hyperbolic), found from the family torus nearest the rational in iota, each with
        its residue and its crossings (R, Z) of the plane phi = 0, and kept only when every crossing lies between the
        two tori either side (a closed line is inside or outside an invariant torus as a whole)
    the two tori either side: refined to the rounding floor and accepted as every verified surface is, by the solver's
        own checks (off-grid B.n, the resonance guard, the quadrature check, sections simple and around the axis) and
        the independent one (torux.verify), the fluxes of the nearest two that pass being the certified interval
        that holds the chain
    the flux of its resonance, linear in iota between the two family tori either side
    the pendulum half width (torux.chains.pendulum_half_width) from the O line's residue and the shear between those
        two tori, which lie on both sides of the chain

place(chain, verified) says where any chain with crossings lies against a set of verified tori: the largest that
holds all its crossings outside and the least that holds them inside, so that a chain the family stalled at, which
lies beyond its last torus, is given the verified surface below it as a certified bound, and its flux extrapolated
from the family is said to be extrapolated.
"""
from __future__ import annotations

import time

import numpy as np

QMAX = 30          # the largest q (field periods of the chain's lines) the family's crossings are listed for


def crossings(tori, nfp, qmax=QMAX):
    """(p, q, i) for every reduced p/q with q <= qmax strictly between the rotations per period of tori[i] and
    tori[i + 1] (the family in order of flux)."""
    from .chains import rationals_between
    out = []
    for i in range(len(tori) - 1):
        a, b = tori[i].iota / nfp, tori[i + 1].iota / nfp
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        for p, q in rationals_between(a, b, qmax):
            out.append((int(p), int(q), i))
    return out


def section(torus, phi=0.0, n=None):
    """The torus's cross-section at the cylindrical angle phi: (n, 2) points (R, Z), counter-clockwise or not."""
    n = n or max(1024, 8 * int(torus.M))
    th = 2 * np.pi * np.arange(n) / n
    R, Z = torus.geometry_grid(th, np.array([phi]))
    return np.stack([R[:, 0], Z[:, 0]], -1)


def inside(curve, pts):
    """For each point (R, Z): inside the closed polygon `curve` (even-odd rule), and its distance to the polygon."""
    P = np.asarray(curve)
    Q = np.roll(P, -1, axis=0)
    pts = np.atleast_2d(np.asarray(pts, float))
    ins, dist = [], []
    for x, y in pts:
        c = (P[:, 1] > y) != (Q[:, 1] > y)
        dy = np.where(Q[:, 1] != P[:, 1], Q[:, 1] - P[:, 1], 1e-300)
        xin = P[:, 0] + (y - P[:, 1]) * (Q[:, 0] - P[:, 0]) / dy
        ins.append(bool(np.sum(c & (x < xin)) % 2 == 1))
        d = Q - P
        t = np.clip(((x - P[:, 0]) * d[:, 0] + (y - P[:, 1]) * d[:, 1]) / np.maximum(np.sum(d * d, 1), 1e-300), 0, 1)
        dist.append(float(np.min(np.hypot(P[:, 0] + t * d[:, 0] - x, P[:, 1] + t * d[:, 1] - y))))
    return np.array(ins), np.array(dist)


def line_points(line, phi=0.0):
    """A ClosedLine's crossings (R, Z) of the plane phi (all q of them), or a record's, (points, its plane)."""
    if hasattr(line, "section_points"):
        return np.asarray(line.section_points(phi)), phi
    return (np.asarray(line.get("section") or [], float).reshape(-1, 2),
            float(line.get("plane_phi", 0.0) or 0.0))


def between(lines, inner, outer, phi=0.0, margin=1e-9):
    """Every crossing of every line outside `inner`'s section and inside `outer`'s (either None: no condition),
    each farther than margin (m) from it, in the plane of the line's crossings. False for a line with none."""
    for ln in lines:
        P, ph = line_points(ln, phi)
        if not len(P):
            return False
        if inner is not None:
            i, d = inside(section(inner, ph), P)
            if np.any(i) or np.any(d < margin):
                return False
        if outer is not None:
            i, d = inside(section(outer, ph), P)
            if not np.all(i) or np.any(d < margin):
                return False
    return True


def place(lines, verified):
    """Where closed lines lie against verified tori: (inner, outer), the torus of largest flux with every crossing of
    every line outside it and the one of least flux with every crossing inside it (None where there is none, and
    both None without a crossing). A closed line lies inside or outside an invariant torus as a whole."""
    lines = [ln for ln in lines if len(line_points(ln)[0])]
    inner = outer = None
    if not lines:
        return None, None
    for T in sorted(verified, key=lambda t: abs(float(t.flux))):
        if between(lines, T, None):
            inner = T
        elif outer is None and between(lines, None, T):
            outer = T
    return inner, outer


def _line_record(ln, phi=0.0):
    from .analysis import _monodromy
    return dict(residue=float(ln.residue), res=float(ln.res), section=line_points(ln, phi)[0].tolist(), plane_phi=phi,
                **(_monodromy(ln) if phi == 0.0 else {}))


def crossed_chains(engine, cont, source, tori, axis, dmin, tol, nfp, qmax=QMAX, deadline=None, exclude=(),
                   verified=None, verbose=False, independent=True):
    """The chains the family `tori` steps across (module docstring), as records (dicts).  exclude: (p, q) left out
    (the chains the family stalled at, analysed already).  verified: a dict index -> (ok, refined torus) shared
    between calls (filled here).  Stops at the deadline: the chains not reached then are listed with their
    rational and the family tori either side only (cut=True)."""
    from .chains import chain_lines, pendulum_half_width
    from .analysis import range_candidate, torus_topology
    verified = {} if verified is None else verified
    t0 = time.time()

    def check(i):
        """The family torus i refined and accepted (cached): (ok, refined torus or None)."""
        if i in verified:
            return verified[i]
        if deadline is not None and time.time() > deadline:
            return (None, None)
        try:
            r = range_candidate(cont, source, dict(f=None, kind="family", torus=tori[i]), tori, dmin, tol,
                                deadline=deadline, axis=axis, label="crossed")
        except Exception as e:  # noqa: BLE001  (a failed check costs the bracket, never the answer)
            verified[i] = (False, None)
            if verbose:
                print(f"   crossed: torus {i} check failed: {e!r}")
            return verified[i]
        ok, best = bool(r.get("ok")), r.get("best")
        if ok and best is not None:
            topo = torus_topology(best, axis)
            ok = bool(topo["ok"])
        if ok and independent:
            # and the independent check every verified surface has (torux.verify: a second field code, B.n at the
            # reference's limit, the flux on random sections), its lines left out
            from .verify import verify_torus, acceptance
            try:
                v = verify_torus(source, best, nlines=0, deadline=deadline)
                ok = bool(acceptance(v))
            except Exception:  # noqa: BLE001
                ok = False
        verified[i] = (ok, best if ok else None)
        return verified[i]

    out = []
    # the lowest q first, so that a deadline leaves out the finest chains
    for p, q, i in sorted(crossings(tori, nfp, qmax), key=lambda c: (c[1], c[2])):
        if (p, q) in exclude or (-p, q) in exclude:
            continue
        a, b = tori[i], tori[i + 1]
        iota_r = p * nfp / q
        rec = dict(p=p, q=q, iota_r=iota_r, kind="crossed",
                   family_Wb=[abs(float(a.flux)), abs(float(b.flux))],
                   family_iota=[float(a.iota), float(b.iota)])
        shear = (b.iota - a.iota) / (b.flux - a.flux) if b.flux != a.flux else np.nan
        rec["shear"] = float(shear)
        rec["flux_r"] = float(a.flux + (iota_r - a.iota) / shear) if np.isfinite(shear) and shear else None
        rec["flux_from"] = "interpolated"
        if deadline is not None and time.time() > deadline:
            rec["cut"] = True
            out.append(rec)
            continue
        # the nearest verified tori either side: the family's own first, then further out, at most three each way
        lo = hi = None
        for j in range(i, max(-1, i - 3), -1):
            ok, best = check(j)
            if ok:
                lo = (j, best)
                break
        for j in range(i + 1, min(len(tori), i + 4)):
            ok, best = check(j)
            if ok:
                hi = (j, best)
                break
        if lo is not None:
            rec["inner_Wb"] = abs(float(lo[1].flux))
            rec["inner_iota"] = float(lo[1].iota)
        if hi is not None:
            rec["outer_Wb"] = abs(float(hi[1].flux))
            rec["outer_iota"] = float(hi[1].iota)
        start = a if abs(a.iota - iota_r) <= abs(b.iota - iota_r) else b
        info = {}
        try:
            lines = chain_lines(engine, start, p, q, deadline=deadline, info=info)
        except Exception as e:  # noqa: BLE001
            lines = []
            rec["lines_error"] = repr(e)
        if info.get("cut"):
            rec["cut"] = True
        if info.get("lower_period"):
            rec["lower_period"] = int(info["lower_period"])
        # only lines between the two family tori either side: another chain of the same rational elsewhere (a
        # profile that turns back) is not this one
        kept, away = [], 0
        for ln in lines:
            if between([ln], a, b):
                kept.append(ln)
            else:
                away += 1
        if away:
            rec["lines_elsewhere"] = away
        rec["lines"] = [_line_record(ln) for ln in kept]
        O = [ln for ln in kept if 0 < ln.residue < 1]
        if O and np.isfinite(shear) and shear:
            RO = max(ln.residue for ln in O)
            rec["half_width_est"] = pendulum_half_width(RO, q, nfp, shear)
        if kept and lo is not None and hi is not None:
            # where the lines lie against the verified tori either side, checked: what makes the interval certified
            rec["between_verified"] = bool(between(kept, lo[1], hi[1]))
        out.append(rec)
        if verbose:
            print(f"   crossed {p}/{q} iota {iota_r:.6f}: family {rec['family_Wb']}, verified "
                  f"{rec.get('inner_Wb')} .. {rec.get('outer_Wb')}, lines {len(kept)} (away {away}), "
                  f"half width {rec.get('half_width_est')}, {time.time() - t0:.1f} s", flush=True)
    return out


def stage(r, source, deadline, qmax=QMAX, verbose=False):
    """The chains the family behind the verified surface of the Result r stepped across (crossed_chains), with the
    time up to deadline: (records, verified tori: r's surface and every family torus the stage verified, axis).
    (None, [], None) when r keeps no family."""
    fam = getattr(r, "_family", None)
    if not fam or r.surface is None or getattr(r, "_engine", None) is None:
        return None, [], None
    from .family import Continuation
    from . import budget as _budget
    eng, ax, tori = r._engine, fam["axis"], fam["tori"]
    cont = Continuation(eng, source.nfp, M0=6, N0=max(16, ax.N), Mmax=128, Nmax=max(128, ax.N), lean_source=source,
                        lean_dmin=fam["dmin"])
    cont.fmeas_off = True
    verified = {}
    with _budget.scope(deadline):
        recs = crossed_chains(eng, cont, source, tori, ax, fam["dmin"], fam["tol"], source.nfp, qmax=qmax,
                              deadline=deadline, verified=verified, verbose=verbose)
    vt = [r.surface] + [t for ok, t in verified.values() if ok and t is not None]
    return recs, vt, ax


def recheck(recs, verified):
    """After torux.closure moved a chain's line to its closed line (polished), whether its lines still lie between
    the two verified tori either side (between_verified), checked again with the tori the stage verified."""
    by = {abs(float(T.flux)): T for T in verified}
    for c in recs or []:
        lines = [ln for ln in c.get("lines") or [] if isinstance(ln, dict)]
        if c.get("between_verified") is None or not any(ln.get("polished") for ln in lines):
            continue
        lo, hi = by.get(c.get("inner_Wb")), by.get(c.get("outer_Wb"))
        c["between_verified"] = bool(lo is not None and hi is not None and between(lines, lo, hi))
    return recs


def annotate(chains, verified, *_):
    """Each chain (torux.chainlist entries) with between_verified_Wb, the fluxes [inner, outer] of the verified tori
    with every crossing of its lines outside and inside them (outer None: none holds them), unless it has one (a
    stepped-across chain has the stage's own, checked against its two verified tori either side)."""
    for c in chains:
        if c.get("between_verified_Wb") is not None:
            continue
        lines = [ln for ln in (c.get("O") or []) + (c.get("X") or []) if ln.get("section")]
        inner, outer = place(lines, verified)
        if inner is not None or outer is not None:
            c["between_verified_Wb"] = [None if inner is None else abs(float(inner.flux)),
                                        None if outer is None else abs(float(outer.flux))]
            f, hw = c.get("flux_Wb"), c.get("half_width_Wb")
            if inner is not None and outer is not None:
                lo, hi = abs(float(inner.flux)), abs(float(outer.flux))
                c["half_width_cap_Wb"] = 0.5 * (hi - lo)
                if f is not None and hw is not None and hw > max(0.0, min(f - lo, hi - f)):
                    c["half_width_pendulum_Wb"] = hw
                    c["half_width_Wb"] = max(0.0, min(f - lo, hi - f))
            elif inner is not None and f is not None and hw is not None and f - hw < abs(float(inner.flux)):
                c["reaches_inside_verified"] = True
    return chains
