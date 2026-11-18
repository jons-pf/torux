"""Island chains next to a torus family: their closed lines, residues and estimated bands; jumps.

Conventions: iota per toroidal turn, rotation per field period rho = iota / nfp; the chain
p/q (reduced, p signed like iota) has its lines close after q field periods with p poloidal
turns, resonant harmonic (m, n) = (q, p) of the torus basis cos(m theta - n nfp phi).

Pendulum estimate of the chain's half width in enclosed flux Phi (Wb), from the O line's
residue R_O and the shear diota/dPhi at the resonance.  The resonant harmonic cos(q theta - p nfp phi)
makes a pendulum in psi = q theta - p nfp phi with J = Phi / 2 pi: psi' = q iota_J dJ,
dJ' = q eps sin psi (iota_J = d iota / dJ, ' = d / d phi).  Its separatrix half width is
dJ = 2 sqrt(eps / iota_J), and over the line's q field periods (phi = 2 pi q / nfp) the O line turns
by q sqrt(iota_J eps) 2 pi q / nfp = 2 sqrt(R_O) (R_O = sin^2 of half that angle, small).  So
    dPhi_half = 2 nfp sqrt(R_O) / (pi q^2 |d iota / d Phi|)
(the standard map of the q-period map in the angle q theta, twist 2 pi q^2 drho/dJ).  It assumes the
shear constant across the chain and R_O small: an estimate, not a bound.  The shear analyse_chain gives it is
the slope of iota over the family's last step before the chain (np.gradient at the end of the family): iota steepens
as the tori near the chain, and where the family stops depends on its steps, which scale with the square root of
the requested flux, so one chain's estimate differs between requests while its residue does not.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from . import budget as _budget
from . import orbits as OR
from .torus import TorusState


@dataclass
class Chain:
    p: int
    q: int
    iota_r: float
    lines: list = field(default_factory=list)       # ClosedLine (converged, with residues)
    O: list = field(default_factory=list)
    X: list = field(default_factory=list)
    flux_r: float = np.nan                           # flux of the resonant surface (interpolated)
    shear: float = np.nan                            # d iota / d Phi at the resonance
    half_width_est: float = np.nan                   # pendulum estimate (Wb)
    info: dict = field(default_factory=dict)


def rationals_between(rho_a, rho_b, qmax):
    """Reduced p/q (q <= qmax) strictly between rho_a and rho_b (either order)."""
    lo, hi = min(rho_a, rho_b), max(rho_a, rho_b)
    out = set()
    for q in range(1, qmax + 1):
        for p in range(math.ceil(lo * q), math.floor(hi * q) + 1):
            if lo < p / q < hi and math.gcd(p, q) == 1:
                out.add((p, q))
    return sorted(out, key=lambda t: t[1])


def nearest_rational_ahead(iota, diota, nfp, qmax=40, window=0.05):
    """The lowest-order rational rho = p/q ahead of iota in the direction diota, within window*|iota|."""
    rho = iota / nfp
    step = math.copysign(window * max(abs(iota), 1e-3) / nfp, diota if diota != 0 else 1.0)
    cands = rationals_between(rho, rho + step, qmax)
    if not cands:
        return None
    # lowest q first; among equal q the nearest
    cands.sort(key=lambda t: (t[1], abs(t[0] / t[1] - rho)))
    return cands[0]


def first_rational_ahead(iota, limit, nfp, qmax=30):
    """The rational rho = p/q (q <= qmax) a family at iota meets first on its way to the rational limit (p, q): the
    nearest to iota / nfp strictly between the two, or None where there is none.  nearest_rational_ahead names the
    lowest-order rational ahead, which may lie past others: at 90 degrees, I_IL / I_PF = 4.70 the family of CNT stalls
    at iota -0.31689, where it names -2/13 (iota -4/13) with -3/19 (iota -6/19) in between."""
    rho, lim = iota / nfp, limit[0] / limit[1]
    cands = rationals_between(rho, lim, qmax)
    if not cands:
        return None
    return min(cands, key=lambda t: (abs(t[0] / t[1] - rho), t[1]))


SAME_POINT_M = 1e-8     # crossings of one closed line nearer each other than this (m) are one point


def distinct_crossings(pts, tol=SAME_POINT_M):
    """Whether the q crossings (q, 2) of a closed line of period q are q distinct points. A line of lower period d
    (dividing q) closed after q periods repeats its d crossings q / d times: the magnetic axis, gone round q times, is
    such a line, and Newton started on a chain's curve can converge to it."""
    P = np.asarray(pts, float).reshape(-1, 2)
    if len(P) < 2:
        return True
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return bool(np.min(d) > tol)


def chain_lines(engine, torus: TorusState, p, q, K=None, verbose=False, deadline=None, info=None, refine=True):
    """Closed lines of the chain p/q started on the torus's curves theta = theta0 + iota_r phi for the
    symmetric starting angles; returns converged, distinct lines with residues, each with q distinct crossings
    (a line of lower period the Newton converged to is left out, counted in info["lower_period"]).
    deadline (absolute, None: none):
    no start after it and each Newton cut at it (info["cut"] = True then; the lines found so far are returned).
    refine=False: the lines as first solved, their residues not yet refined (torux.orbits.line_monodromy)."""
    nfp = torus.nfp
    iota_r = p / q * nfp
    starts = []
    for th in (0.0, math.pi, -math.pi * p / q, math.pi - math.pi * p / q):
        starts.append(th)
    K = K or max(24, 10 * q + 2 * torus.N)
    lines = []
    seen = []           # (crossings as first solved, their fate): a start that reaches them again is not refined
    for th0 in starts:
        for phi0 in (0.0, math.pi / nfp):
            # the line through the symmetry plane phi0 at theta0 + iota_r phi0 in {0, pi}
            gR = lambda ph, t0=th0: torus.geometry_on(t0 + iota_r * ph, ph)[0]
            gZ = lambda ph, t0=th0: torus.geometry_on(t0 + iota_r * ph, ph)[1]
            thc = th0 + iota_r * phi0
            if not (abs(math.sin(thc)) < 1e-9):
                continue
            if deadline is not None and time.time() > deadline:
                if info is not None:
                    info["cut"] = True
                return lines
            ln = OR.solve_closed_line(engine, nfp, q, gR, gZ, phi0=phi0, K=K, symmetric=True, deadline=deadline)
            if ln.info.get("cut") and info is not None:
                info["cut"] = True
            if not ln.converged:
                if verbose:
                    print(f"    chain {p}/{q} start theta0 {th0:.3f} phi0 {phi0:.3f}: not converged ({ln.res:.2e})")
                continue
            pts0 = ln.section_points(0.0)
            was = next((w for P, w in seen if np.min(np.linalg.norm(pts0[:, None, :] - P[None], axis=-1)) < 1e-8), None)
            if was == "lower period" and info is not None:
                info["lower_period"] = info.get("lower_period", 0) + 1
            if was is not None:
                continue
            ln = OR.line_monodromy(engine, ln, deadline=deadline, refine=refine)
            pts = ln.section_points(0.0)
            seen.append((pts0, "lower period" if not distinct_crossings(pts) else "line"))
            if not distinct_crossings(pts):
                # a line of lower period gone round q / d times (the magnetic axis is one): not a line of p/q
                if verbose:
                    print(f"    chain {p}/{q} start theta0 {th0:.3f} phi0 {phi0:.3f}: a line of lower period "
                          f"(residue {ln.residue:.4e}), left out")
                if info is not None:
                    info["lower_period"] = info.get("lower_period", 0) + 1
                continue
            dup = False
            for other in lines:
                op = other.section_points(0.0)
                dmin = np.min(np.linalg.norm(pts[:, None, :] - op[None, :, :], axis=-1))
                if dmin < 1e-8:
                    dup = True
                    break
            if verbose:
                print(f"    chain {p}/{q} start theta0 {th0:.3f} phi0 {phi0:.3f}: residue {ln.residue:.4e} dup {dup}")
            if not dup:
                lines.append(ln)
    return lines


TRACED_REACH = (0.5, 0.9)   # the traced search's half-lines: from half the torus's own distance from the axis on each to
                            # 0.9 of the axis's distance from the nearest part of the source
TRACED_NSCAN = 240          # starts on each half-line, spaced geometrically
TRACED_QMAX = 30            # chains of higher period are not traced
TRACED_WALL_S = 90.0        # the traced search's own limit (s), within the caller's deadline


def traced_chain_lines(engine, torus, axis, p, q, known=(), deadline=None, info=None, verbose=False):
    """The chain p/q's stellarator-symmetric elliptic line by tracing, for a chain beyond the torus `torus` (the family's
    last before it).  Newton from the torus's curves (chain_lines) can end on the magnetic axis there: the chain lies
    beyond the torus, its elliptic line often many times farther from the axis on one half-line than the torus's
    curves reach (CNT at 88 degrees, I_IL / I_PF = 3.5: the torus 2.6e-3 m from the axis, the line 3.2e-2 m), and the
    axis is the nearer fixed point.  On each of the four symmetric half-lines, from TRACED_REACH[0] times the torus's
    crossing to TRACED_REACH[1] times the axis's distance from the source, the closed lines of period q and winding p
    nearest the axis are solved outward until an elliptic one (torux.chainfind.symmetric_lines_outward), and of those the
    one nearest the torus (the mean distance of its crossings of the plane phi = 0 from the torus's section) is the
    chain's: a farther elliptic line of p/q belongs to another chain of the same p/q.  It has q distinct crossings
    and is none of `known` (ClosedLine list).  Returns [line] or [], and info["traced"] (when given) says what was tried."""
    from .chainfind import symmetric_lines_outward
    # one search a chain and source: the requests of one search that stall at the same chain share it (a search cut by
    # its deadline is not kept)
    cache = getattr(engine.source, "_traced_chain_lines", None)
    if cache is None:
        cache = {}
        try:
            engine.source._traced_chain_lines = cache
        except AttributeError:
            pass
    key = (int(p), int(q))
    th = np.linspace(0, 2 * np.pi, 512, endpoint=False)
    TR, TZ = torus.geometry_on(th, np.zeros_like(th))
    T = np.stack([TR, TZ], -1)
    if key in cache:
        from .crossed import inside
        pick, rec = cache[key]
        # the same chain only while the torus has not passed it: a line inside this torus is another chain's of p/q,
        # and where none was found, a torus of other flux than the one searched from may stand before another chain
        f0, f1 = abs(float(rec.get("torus_flux", np.nan))), abs(float(getattr(torus, "flux", np.nan)))
        same = (all(not np.any(inside(T, ln.section_points(0.0))[0]) for ln in pick) if pick
                else bool(np.isfinite(f0) and np.isfinite(f1) and f0 / 1.25 <= f1 <= 1.25 * f0))
        if same:
            pick = [ln for ln in pick if not any(
                np.min(np.linalg.norm(ln.section_points(0.0)[:, None, :] - o.section_points(0.0)[None, :, :], axis=-1))
                < 1e-6 for o in known)]
            if info is not None:
                info["traced"] = dict(rec, reused=True)
            return list(pick)
    nfp = torus.nfp
    pts_src = engine.source.sample_points()
    win = {}
    for plane in (0, 1):
        phi_s = plane * np.pi / nfp
        Ra, Za = (float(v[0]) for v in axis.at(np.array([phi_s])))
        xa = np.array([Ra * math.cos(phi_s), Ra * math.sin(phi_s), Za])
        d_src = float(np.min(np.linalg.norm(pts_src - xa[None, :], axis=1))) if len(pts_src) else 0.5 * Ra
        R2, _ = torus.geometry_on(np.array([0.0, np.pi]), np.array([phi_s, phi_s]))
        for Rb in R2:
            side = 1 if Rb > Ra else -1
            r = abs(float(Rb) - Ra)
            r_hi = TRACED_REACH[1] * d_src
            if np.isfinite(r) and 0 < TRACED_REACH[0] * r < r_hi:
                win[(plane, side)] = (TRACED_REACH[0] * r, r_hi)
    t0 = time.time()
    d_own = t0 + TRACED_WALL_S
    d_own = d_own if deadline is None else min(deadline, d_own)
    cands, linfo = symmetric_lines_outward(engine, engine.source, axis, p, q, win, keep=lambda ln: 0 < ln.residue < 1,
                                           skip=known, deadline=d_own, nscan=TRACED_NSCAN, verbose=verbose)
    new, dist = [], []
    for ln in cands:
        pts = ln.section_points(0.0)
        if not (0 < ln.residue < 1) or not distinct_crossings(pts):
            continue
        if any(np.min(np.linalg.norm(pts[:, None, :] - o.section_points(0.0)[None, :, :], axis=-1)) < 1e-6
               for o in list(known) + new):
            continue
        ln.info["found_by"] = "traced"
        new.append(ln)
        dist.append(float(np.mean(np.min(np.linalg.norm(pts[:, None, :] - T[None, :, :], axis=-1), axis=1))))
    pick = [new[int(np.argmin(dist))]] if new else []
    res_log = None
    if pick:
        # the line resolved: its Fourier order doubled until its crossings and residue no longer move
        ln, res_log = resolved_line(engine, pick[0], deadline=d_own)
        pick = [] if ln is None or not (0 < ln.residue < 1) or not distinct_crossings(ln.section_points(0.0)) else [ln]
        if pick:
            ln.info["found_by"] = "traced"
    if info is not None:
        info["traced"] = dict(windows={f"{k[0]},{k[1]}": [float(v[0]), float(v[1])] for k, v in win.items()},
                              tried=linfo, elliptic=[dict(residue=float(l.residue), from_torus_m=d,
                                                          R0=l.section_points(0.0)[0].tolist())
                                                     for l, d in zip(new, dist)],
                              resolution=res_log, wall_s=float(time.time() - t0), cut=bool(time.time() > d_own))
    cut = time.time() > d_own or any(t.get("cut") for t in linfo) or bool(res_log and any(r.get("cut") for r in res_log))
    if not cut:
        cache[key] = (list(pick), dict(wall_s=float(time.time() - t0), found=len(pick),
                                       torus_flux=float(getattr(torus, "flux", np.nan))))
    return pick


RESOLVED_K_MAX = 1024       # the highest Fourier order resolved_line tries
RESOLVED_POS_M = 1e-6       # crossings that move less than this (m) ...
RESOLVED_RES = (1e-7, 1e-5)  # ... and a residue r that moves less than [0] + [1] |r| between two orders: resolved (the
                            # order's own error is far below what it moved from the order half its size)


def resolved_line(engine, line, deadline=None, k_max=RESOLVED_K_MAX):
    """The symmetric closed line `line` (converged) solved again at twice its Fourier order, from itself, until its
    crossings of the plane phi = 0 move less than RESOLVED_POS_M and its residue less than RESOLVED_RES between two
    orders: a line Newton converged at too low an order closes only to that order's truncation, and its residue is
    off by as much (a traced line's first order is set by its period alone).  Returns (line, log) with the last
    order's line, or (None, log) when it does not settle by k_max or a solve fails."""
    prev = line
    log = [dict(K=int(prev.K), residue=float(prev.residue))]
    while 2 * prev.K <= k_max:
        if deadline is not None and time.time() > deadline:
            log.append(dict(cut=True))
            return None, log
        ln = OR.solve_closed_line(engine, prev.nfp, prev.q, lambda ph, c=prev: c.at(ph)[0],
                                  lambda ph, c=prev: c.at(ph)[1], phi0=prev.phi0, K=2 * prev.K, symmetric=True,
                                  deadline=deadline)
        if not ln.converged:
            log.append(dict(K=int(ln.K), converged=False, res=float(ln.res)))
            return None, log
        ln = OR.line_monodromy(engine, ln)
        dpos = float(np.max(np.linalg.norm(ln.section_points(0.0) - prev.section_points(0.0), axis=-1)))
        dres = abs(ln.residue - prev.residue)
        log.append(dict(K=int(ln.K), residue=float(ln.residue), moved_m=dpos, residue_moved=float(dres)))
        if dpos < RESOLVED_POS_M and dres < RESOLVED_RES[0] + RESOLVED_RES[1] * abs(ln.residue):
            ln.info["resolved"] = True
            return ln, log
        prev = ln
    return None, log


def analyse_chain(engine, tori, p, q, verbose=False, deadline=None, axis=None):
    """Chain p/q next to the family `tori` (TorusState list): lines, residues, flux and shear at
    the resonance (interpolated in the family), pendulum half width.  deadline: chain_lines' (ch.info["cut"]).
    axis (the family's magnetic axis): where Newton from the torus's curves leaves the chain with no elliptic line, its
    symmetric lines are sought again by tracing (traced_chain_lines, ch.info["traced_lines"] the lines it added).  The
    jump's ladder (width_scale) rests on the elliptic lines from the torus's curves only, as without the traced search:
    the half width of a traced line is the record's estimate."""
    nfp = tori[-1].nfp
    iota_r = p / q * nfp
    ch = Chain(p, q, iota_r)
    # the family torus nearest in iota as the starting surface
    k = int(np.argmin([abs(t.iota - iota_r) for t in tori]))
    ch.lines = chain_lines(engine, tori[k], p, q, verbose=verbose, deadline=deadline, info=ch.info)
    O_newton = [l for l in ch.lines if 0 < l.residue < 1]
    if (axis is not None and not O_newton and q <= TRACED_QMAX
            and (deadline is None or time.time() < deadline - 10.0)):
        try:
            more = traced_chain_lines(engine, tori[k], axis, p, q, known=ch.lines, deadline=deadline, info=ch.info,
                                      verbose=verbose)
        except Exception as e:  # noqa: BLE001
            from .envfail import environment_failure
            if environment_failure(e):
                raise
            more = []
            ch.info["traced"] = dict(error=repr(e))
        if more:
            ch.lines = ch.lines + more
            ch.info["traced_lines"] = len(more)
    ch.O = [l for l in ch.lines if 0 < l.residue < 1]
    ch.X = [l for l in ch.lines if not (0 < l.residue < 1)]
    fl = np.array([t.flux for t in tori]); io = np.array([t.iota for t in tori])
    if len(tori) >= 2:
        dI = np.gradient(io, fl)
        # linear extrapolation from the last two tori in the direction of the resonance
        j = k if 0 < k < len(tori) else len(tori) - 1
        ch.shear = float(dI[j])
        ch.info["shear_at"] = dict(index=int(j), n_tori=len(tori), flux=float(fl[j]), iota=float(io[j]))
        if ch.shear != 0:
            ch.flux_r = float(fl[j] + (iota_r - io[j]) / ch.shear)
    # the half width the record reports rests only on an elliptic line whose residue is resolved above its error
    # (torux.orbits.residue_resolved): a residue within its error gives neither a type nor a width
    O_res = [l for l in ch.O if OR.residue_resolved(l)]
    if O_res and np.isfinite(ch.shear) and ch.shear != 0:
        RO = max(l.residue for l in O_res)
        ch.half_width_est = pendulum_half_width(RO, q, nfp, ch.shear)
    if O_newton and np.isfinite(ch.shear) and ch.shear != 0:
        RO = max(l.residue for l in O_newton)
        # q times the half width: the scale the jump's ladder and the island tori's width are set on, their
        # numbers as they were before the estimate was divided by q
        ch.info["width_scale"] = float(2 * nfp * math.sqrt(RO) / (math.pi * q * abs(ch.shear)))
    return ch


def pendulum_half_width(residue, q, nfp, shear):
    """The pendulum estimate (module docstring) of a chain's half width in flux (Wb) from its O line's residue, its
    period q, the field periods nfp and the shear d iota / d Phi (1/Wb)."""
    return float(2 * nfp * math.sqrt(residue) / (math.pi * q * q * abs(shear)))


def noble_offset(iota_r, nfp, q, dist):
    """An iota beyond the resonance by about dist, with a noble (golden) correction so that the
    rotation number is far from every low-order rational."""
    g = (math.sqrt(5) - 1) / 2
    return iota_r + dist * (1 + g * 1e-3)


def jump_over(cont, tori, chain: Chain, direction, max_tries=6, verbose=False, axis=None):
    """Fixed-iota Newton beyond the chain: iota = iota_r + direction * dist (1 + noble), dist from
    the chain's estimated width (or 1e-4 |iota|) doubling per try; predictor extrapolated from the
    family.  Returns the torus beyond (TorusState) or None."""
    last = tori[-1]
    nfp = last.nfp
    # candidate distances beyond the resonance (in iota): a geometric ladder from 1e-5 |iota| to
    # 3e-2 |iota|, with the pendulum estimate's 1.25 half widths (and 1.5, 2x) inserted; sorted, so
    # the nearest torus beyond the chain is tried first (the pendulum width is unreliable for wide,
    # strongly sheared chains)
    io = max(abs(last.iota), 1e-2)
    cands = list(io * np.array([1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]))
    scale = chain.info.get("width_scale", np.nan)
    if np.isfinite(scale) and np.isfinite(chain.shear) and chain.shear != 0:
        # (on q half widths in flux, the scale these three were set on: analyse_chain's width_scale)
        w = scale * abs(chain.shear)
        cands += [1.25 * w, 1.5 * w, 2.0 * w]
    cands = sorted(set(float(c) for c in cands))
    # distances well inside the chain are futile: the family itself stopped |iota_r - iota_last| short of the
    # resonance on its side, and the chain's half width on the far side is of the same order, so candidates
    # below a quarter of that gap are skipped (CoilStellaration D8r9jBW9...: the first five of eleven tries sat
    # 3e-6..3e-4 from the 2/7 resonance after the family had stopped 3.6e-3 short of it; each try two
    # high-resolution solves)
    gap = abs(chain.iota_r - last.iota)
    cands = [c for c in cands if c > 0][: max(max_tries, 8)] if gap == 0 else \
        ([c for c in cands if c >= 0.25 * gap] or cands[-3:])
    for t, dist in enumerate(cands):
        if _budget.expired():
            return None             # (each try would stop at once, after setting up its high-resolution solves)
        target_iota = noble_offset(chain.iota_r, nfp, chain.q, direction * dist)
        # predictor: extrapolate the family in iota to the target
        if len(tori) >= 2:
            a, b = tori[-2], tori[-1]
            if b.iota != a.iota:
                lam = (target_iota - b.iota) / (b.iota - a.iota)
            else:
                lam = 1.0
            lam = float(np.clip(lam, -2, 20))
            ap = a.resized(b.M, b.N)
            guess = TorusState(b.rc + lam * (b.rc - ap.rc), b.zs + lam * (b.zs - ap.zs), target_iota,
                               b.flux + lam * (b.flux - a.flux), b.M, b.N, nfp)
        else:
            guess = TorusState(last.rc.copy(), last.zs.copy(), target_iota, last.flux, last.M, last.N, nfp)
        for (M, N) in dict.fromkeys(((last.M, last.N), cont.grow(last))):
            # patience 3: a fixed-iota Newton inside the chain does not converge, and each of its iterations is a
            # high-resolution solve (CoilStellaration D8r9jBW9...: 22 such tries at M = 80-112, N = 128, 20
            # iterations each, were most of its 30 minutes)
            sol = cont.solve(guess, guess.flux, M, N, fixed_iota=target_iota, maxit=20, patience=3)
            if verbose:
                print(f"   jump try {t} iota {target_iota:.10f} M,N {M},{N} conv {sol.converged} bn {sol.bn_off:.2e} "
                      f"flux {sol.flux:.6e}")
            if (sol.converged and sol.bn_off < cont.bn_accept and abs(sol.flux) > abs(last.flux)
                    and sol.flux * last.flux > 0):
                # only a torus of the axis's family: its sections simple closed curves around the axis
                # (CoilStellaration DDMaikF39a8...: a 1/3 jump converged onto a curve through the chain's X line
                # that wound -2 times around the axis, with the flux sign flipped)
                from .analysis import torus_topology
                topo = torus_topology(sol, axis)
                if not topo["ok"]:
                    if verbose:
                        print(f"   jump try {t}: rejected, sections not simple around the axis ({topo})", flush=True)
                    continue
                sol = cont.raise_until(sol, sol.flux, cont.bn_cont, fixed_iota=target_iota)
                return sol
    return None
