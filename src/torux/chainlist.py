"""The island chains a request identified, read from its record, in one shape.

A request records each chain it meets in the record of the stage that met it: the family from the axis where it
stalled (record['chains'], torux.chains.analyse_chain), an island band's certificate (the chain it was for), an
edge barrier's (the closed lines it screened beyond the last surface and tried).  of_record(record) lists them, one dict a chain and stage:

    p, q            the chain p/q: its lines close after q field periods with p poloidal turns (torux.chains)
    iota            its rotational transform, p nfp / q
    flux_Wb         the flux of its resonant surface, interpolated in the family
    half_width_Wb   the pendulum estimate of its half width in flux (torux.chains), from O_residue and shear
    shear           d iota / d flux (1/Wb) at the family torus the estimate took it from
    O_residue       the largest residue of its elliptic lines, the one the half width is from
    O, X            its elliptic and hyperbolic closed lines: residue, and section, the line's crossings (R, Z) of
                    the plane at cylindrical angle plane_phi (radians). A line whose q crossings are fewer than q
                    distinct points (the magnetic axis gone round q times) is not one of them: lines_of_lower_period
                    counts those left out, and the half width is the chain's own elliptic line's or none.  Each
                    line measured (torux.closure) says how well it closes: closure_m, how far from its first
                    crossing it comes back after q field periods, distance_m, Newton's step from there (the distance
                    to the closed line), converged (distance_m <= torux.closure.CONVERGED_M), and polished where the
                    edge barrier's line was brought to convergence (its move, and its residue and measure before).
                    A line neither converged nor closing to torux.closure.CLOSES_M is no line of the chain: it is
                    left out and counted in lines_not_converged
    lines_traced    of its O and X lines, those found by tracing from the symmetric half-lines (torux.chains.
                    traced_chain_lines) where Newton from the family's torus left it with no elliptic line
    unresolved      its closed lines whose residue lies within the error recorded for it (residue_err, torux.orbits.
                    line_monodromy): neither O nor X, and no half width rests on them, so a chain whose lines are
                    all unresolved has no type and no half width
    band_Wb         the fluxes (in, out) an island band put round it
    found_in        the stage: 'family' (where the family stalled), 'stepped across' (torux.crossed: a rational the
                    family's iota passed between two of its tori), 'island band' or 'edge barrier'
    stalled_at      for a chain found in the family: True for the chain the family stalled at, the rational of order
                    q <= torux.crossed.QMAX it meets first beyond its last torus; False for the lowest-order rational
                    further ahead (torux.chains.nearest_rational_ahead), past that one, which the family's jump tried to
                    cross. merge: True where any request's family stalled at it, with stalled_from, the request whose
                    family met it first and that family's last torus (family_iota, family_Wb). The row's own numbers
                    (flux_Wb, family_Wb, family_iota) are those of values_from, which is that request unless another
                    computed more of the chain (its lines, a half width)
    flux_from       how flux_Wb was had: 'between family tori' (linear in iota between the two tori either side of
                    the chain, family_Wb) or 'extrapolated' (beyond the family's last torus, from the torus at
                    family_Wb with the slope of iota over the family's last step)
    family_Wb       the family tori flux_Wb is from: the two either side, or the one it was extrapolated from
    family_iota     for an extrapolated flux, the rotational transform of the torus it was extrapolated from
    between_verified_Wb  [inner, outer]: the fluxes of the verified surfaces with every crossing of the chain's lines
                    outside and inside them (outer None: no verified surface holds the chain), set by the search
                    (torux.crossed.annotate); for a stepped-across chain the two verified tori either side
    half_width_cap_Wb    (outer - inner) / 2 for a stepped-across chain: certified, its full width is at most that. Its
                    half_width_Wb is the pendulum estimate cut to what the two verified tori leave about flux_Wb
                    (half_width_pendulum_Wb the estimate, where it was cut)
    reaches_inside_verified  for a chain beyond the verified surfaces: True where flux_Wb - half_width_Wb lies inside
                    the verified surface its lines lie outside, so that the extrapolated flux is too small or the half
                    width too large

each as far as the stage computed it (a key it did not compute is absent).  merge(lists) puts the lists of several
requests together, one entry a chain, with the numbers of the requests that found it.
"""
from __future__ import annotations

import numpy as np

FAMILY, BAND, EDGE, CROSSED = "family", "island band", "edge barrier", "stepped across"


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _put(d, k, v):
    if v is not None:
        d[k] = v


def _base(p, q, nfp, found_in):
    p, q = _int(p), _int(q)
    if q is None or q <= 0:
        return None
    out = dict(p=p, q=q)
    if p is not None and nfp:
        out["iota"] = p * float(nfp) / q
    out["found_in"] = found_in
    return out


def _line(ln, phi=None):
    out = {}
    _put(out, "residue", _num(ln.get("residue")))
    _put(out, "residue_err", _num(ln.get("residue_err")))
    sec = ln.get("section", ln.get("points"))
    if sec is not None:
        out["section"] = [[float(v) for v in pt] for pt in sec]
        out["plane_phi"] = float(phi if phi is not None else ln.get("plane_phi", 0.0))
    elif ln.get("R0") is not None:
        out["section"] = [[float(v) for v in ln["R0"]]]
        out["plane_phi"] = 0.0
    _measure(out, ln)
    return out


def _measure(out, ln):
    """How well the line closes (torux.closure), where it was measured: closure_m, distance_m, converged, and
    polished when the stored line was brought to convergence."""
    if "converged" in ln:
        out["closure_m"] = _num(ln.get("closure_m"))
        out["distance_m"] = _num(ln.get("distance_m"))
        out["converged"] = bool(ln["converged"])
    if isinstance(ln.get("polished"), dict):
        out["polished"] = ln["polished"]
    return out


def _listed(ln):
    """Whether a closed line is listed with its chain: converged, or not measured, or (a thin chain's line, whose
    place along its curve is not determined) closing to torux.closure.CLOSES_M. A line that does neither is no closed
    line of the chain: the chain counts it in lines_not_converged."""
    from .closure import CLOSES_M
    if "converged" not in ln or ln["converged"]:
        return True
    c = _num(ln.get("closure_m"))
    return c is not None and c <= CLOSES_M


def _not_converged(out, lines):
    """lines without those not listed (_listed), counted in out['lines_not_converged']."""
    keep = [ln for ln in lines if _listed(ln)]
    if len(keep) < len(lines):
        out["lines_not_converged"] = out.get("lines_not_converged", 0) + len(lines) - len(keep)
    return keep


def _elliptic(res):
    return res is not None and 0 < res < 1


def _unresolved(ln):
    """Whether a closed line's residue lies within its recorded error (residue_err, torux.orbits.line_monodromy), so
    that neither its type nor a half width can be had from it. A line whose record keeps no error (one never refined,
    an edge barrier's) is taken as it is; one whose error could not be measured (None) is not resolved."""
    if "residue_err" not in ln:
        return False
    r, e = _num(ln.get("residue")), _num(ln.get("residue_err"))
    return r is None or e is None or abs(r) <= e


# crossings of one closed line nearer each other than this (m) are one: a line whose q crossings are fewer than q
# distinct points closes after fewer field periods (the magnetic axis, gone round q times, is one), so it is not a
# line of the chain p/q
SAME_POINT_M = 1e-8


def _own(ln, q):
    """Whether a closed line's crossings (section) are q distinct points, as a line of the chain p/q has (True when it
    has no crossings to tell)."""
    sec = ln.get("section", ln.get("points"))
    if not sec or q is None or len(sec) < 2:
        return True
    P = np.asarray(sec, float).reshape(-1, 2)
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return bool(np.min(d) > SAME_POINT_M)


def _family(c, nfp):
    out = _base(c.get("p"), c.get("q"), nfp, FAMILY)
    if out is None:
        return None
    _put(out, "flux_Wb", None if _num(c.get("flux_r")) is None else abs(_num(c.get("flux_r"))))
    _put(out, "half_width_Wb", _num(c.get("half_width_est")))
    _put(out, "shear", _num(c.get("shear")))
    lines = _not_converged(out, [ln for ln in (c.get("lines") or []) if isinstance(ln, dict)])
    other = [ln for ln in lines if not _own(ln, out["q"])]
    if other:
        # a line of lower period (the axis gone round q times) is not the chain's: left out, and the half width that
        # rested on it taken again from the chain's own elliptic line, or none
        lines = [ln for ln in lines if _own(ln, out["q"])]
        out["lines_of_lower_period"] = len(other)
        own_O = [_num(ln.get("residue")) for ln in lines if _elliptic(_num(ln.get("residue"))) and not _unresolved(ln)]
        out.pop("half_width_Wb", None)
        if own_O and out.get("shear") and nfp:
            from .chains import pendulum_half_width
            out["half_width_Wb"] = pendulum_half_width(max(own_O), out["q"], float(nfp), out["shear"])
    if c.get("lower_period"):
        # (torux.chains.chain_lines left them out already)
        out["lines_of_lower_period"] = out.get("lines_of_lower_period", 0) + int(c["lower_period"])
    U = [_line(ln) for ln in lines if _unresolved(ln)]
    lines = [ln for ln in lines if not _unresolved(ln)]
    O = [_line(ln) for ln in lines if _elliptic(_num(ln.get("residue")))]
    X = [_line(ln) for ln in lines if not _elliptic(_num(ln.get("residue")))]
    if O:
        out["O_residue"] = max(o["residue"] for o in O)
    out["O"], out["X"] = O, X
    if U:
        out["unresolved"] = U        # residues within their error: no type, and no half width rests on them
    if c.get("traced_lines"):
        out["lines_traced"] = int(c["traced_lines"])     # (torux.chains.traced_chain_lines) of its lines, found by tracing
    if c.get("cut"):
        out["cut"] = True            # the chain's line search was cut by the deadline
    # the rational the family met first beyond its last torus (records before 0.3 name one chain a stall: that one)
    out["stalled_at"] = bool(c.get("stalled_at", True))
    if "flux_Wb" in out:
        # torux.chains.analyse_chain: from the family torus nearest in iota with the slope of the family's step there
        out["flux_from"] = "extrapolated"
        at = c.get("shear_at") or {}
        if _num(at.get("flux")) is not None:
            out["family_Wb"] = [abs(_num(at["flux"]))]
        if _num(at.get("iota")) is not None:
            out["family_iota"] = [_num(at["iota"])]
    return out


def _crossed(c, nfp):
    """A chain the family stepped across (torux.crossed.crossed_chains)."""
    out = _base(c.get("p"), c.get("q"), nfp, CROSSED)
    if out is None:
        return None
    if _num(c.get("flux_r")) is not None:
        out["flux_Wb"] = abs(_num(c["flux_r"]))
        out["flux_from"] = "between family tori"
    fam = [abs(_num(v)) for v in (c.get("family_Wb") or []) if _num(v) is not None]
    if len(fam) == 2:
        out["family_Wb"] = fam
    _put(out, "half_width_Wb", _num(c.get("half_width_est")))
    _put(out, "shear", _num(c.get("shear")))
    lines = _not_converged(out, [ln for ln in (c.get("lines") or []) if isinstance(ln, dict)])
    own = [ln for ln in lines if _own(ln, out["q"])]
    if c.get("lower_period"):
        out["lines_of_lower_period"] = int(c["lower_period"])
    if len(own) < len(lines):
        out["lines_of_lower_period"] = out.get("lines_of_lower_period", 0) + len(lines) - len(own)
        own_O = [_num(ln.get("residue")) for ln in own if _elliptic(_num(ln.get("residue"))) and not _unresolved(ln)]
        out.pop("half_width_Wb", None)
        if own_O and out.get("shear") and nfp:
            from .chains import pendulum_half_width
            out["half_width_Wb"] = pendulum_half_width(max(own_O), out["q"], float(nfp), out["shear"])
    lines = own
    U = [_line(ln) for ln in lines if _unresolved(ln)]
    lines = [ln for ln in lines if not _unresolved(ln)]
    O = [_line(ln) for ln in lines if _elliptic(_num(ln.get("residue")))]
    X = [_line(ln) for ln in lines if not _elliptic(_num(ln.get("residue")))]
    if O:
        out["O_residue"] = max(o["residue"] for o in O)
    out["O"], out["X"] = O, X
    if U:
        out["unresolved"] = U        # residues within their error: no type, and no half width rests on them
    lo, hi = _num(c.get("inner_Wb")), _num(c.get("outer_Wb"))
    if lo is not None and hi is not None and (c.get("between_verified") is not False):
        # its lines checked between them, or (no line found) its rational between theirs: Poincare-Birkhoff puts
        # closed lines of p/q between two invariant tori whose rotations lie either side of it
        lo, hi = sorted((abs(lo), abs(hi)))
        out["between_verified_Wb"] = [lo, hi]
        # the chain lies between them, so its full width is at most hi - lo; and the half width given is never more
        # than they leave about the chain's flux, flux_Wb +- half_width_Wb within [lo, hi]
        out["half_width_cap_Wb"] = 0.5 * (hi - lo)
        f, hw = out.get("flux_Wb"), out.get("half_width_Wb")
        if hw is not None and f is not None:
            room = max(0.0, min(f - lo, hi - f))
            if hw > room:
                out["half_width_pendulum_Wb"] = hw
                out["half_width_Wb"] = room
    if c.get("between_verified") is False:
        out["lines_outside_verified"] = True
    if c.get("cut"):
        out["cut"] = True
    return out


def _band(c, nfp):
    ch = c.get("chain")
    if isinstance(ch, dict):
        p, q = ch.get("p"), ch.get("q")
    elif isinstance(ch, (list, tuple)) and len(ch) == 2:
        p, q = ch
    else:
        return None
    out = _base(p, q, nfp, BAND)
    if out is None:
        return None
    ch = ch if isinstance(ch, dict) else {}
    O = [dict(residue=r) for r in (_num(v) for v in (ch.get("O_residues") or [])) if r is not None]
    if not O and _num(c.get("O_residue")) is not None:
        O = [dict(residue=_num(c["O_residue"]))]
    X = [dict(residue=r) for r in (_num(v) for v in list(ch.get("X_residues") or [])
                                   + list(ch.get("reflection_residues") or [])) if r is not None]
    if O:
        out["O_residue"] = max(o["residue"] for o in O)
    out["O"], out["X"] = O, X
    iv = c.get("interval")
    if isinstance(iv, (list, tuple)) and len(iv) == 2 and all(_num(v) is not None for v in iv):
        out["band_Wb"] = sorted(abs(_num(v)) for v in iv)
    out["certified"] = bool(c.get("ok"))
    return out


def _edge(c, nfp):
    """One entry a chain p/q, its lines the edge barrier's: its certificate's orbit, the closed lines it screened
    beyond the last surface (with their crossings) and the candidates it tried for a barrier."""
    chains, lines = {}, set()
    orb = c.get("orbit")
    items = ([dict(orb, ok=c.get("ok"), plane_phi=c.get("plane_phi", 0.0))] if isinstance(orb, dict) else [])
    for t in items + list(c.get("screened") or []) + list(c.get("candidates") or []):
        if not isinstance(t, dict) or t.get("q") is None:
            continue
        e = _base(t.get("p"), t.get("q"), nfp, EDGE)
        if e is None:
            continue
        e = chains.setdefault((e["p"], e["q"]), dict(e, O=[], X=[], certified=False))
        e["certified"] = bool(e["certified"] or t.get("ok"))
        res = _num(t.get("residue"))
        if (e["p"], e["q"], res) in lines or t.get("points") is None:
            continue                 # the same line screened and then tried (a candidate tried keeps no crossings)
        if not _own(t, e["q"]):
            e["lines_of_lower_period"] = e.get("lines_of_lower_period", 0) + 1
            continue
        lines.add((e["p"], e["q"], res))
        if not _listed(t):
            e["lines_not_converged"] = e.get("lines_not_converged", 0) + 1
            continue
        e["O" if _elliptic(res) else "X"].append(_line(dict(t, residue=res, points=t.get("points")),
                                                       phi=t.get("plane_phi", 0.0)))
    return list(chains.values())


def _certificate(c, nfp):
    """The chains of one certificate record (an island band's or an edge barrier's, with what it nests)."""
    if not isinstance(c, dict):
        return []
    kind = str(c.get("kind") or "")
    out = []
    if kind.startswith(("island", "manifold_band")) or (not kind and "chain" in c):
        b = _band(c, nfp)
        if b is not None:
            out.append(b)
        if isinstance(c.get("manifold"), dict):
            out += _certificate(c["manifold"], nfp)
    elif kind.startswith(("edge", "exit_curve", "lost_curve")):
        out += _edge(c, nfp)
        for k in ("edge_barrier", "survivor"):
            out += _certificate(c.get(k), nfp)
    return out


def of_record(record):
    """The chains a request's record holds (module docstring), in the order its stages met them."""
    rec = record if isinstance(record, dict) else {}
    nfp = rec.get("nfp")
    out = []
    seen = {}
    for c in rec.get("crossed") or []:
        if isinstance(c, dict):
            x = _crossed(c, nfp)
            if x is not None:
                k = (abs(x["p"]) if x.get("p") is not None else None, x["q"])
                if seen.get(k):
                    x["occurrence"] = seen[k]     # the same rational again further out (iota turned back)
                seen[k] = seen.get(k, 0) + 1
                out.append(x)
    fams = list(rec.get("chains") or []) + list((rec.get("range_family") or {}).get("chains") or [])
    for c in fams:
        if isinstance(c, dict):
            f = _family(c, nfp)
            if f is not None:
                out.append(f)
    certs = [rec.get(k) for k in ("certificate", "bracket_certificate", "edge_barrier", "bracket_edge_barrier")]
    certs.append((rec.get("range") or {}).get("upper_bound_evidence"))
    certs += [a for a in (rec.get("proxy_band_attempts") or []) if isinstance(a, dict)]
    ids = set()
    for c in certs:
        if not isinstance(c, dict) or id(c) in ids:
            continue
        ids.add(id(c))
        out += _certificate(c, nfp)
    # the same record kept under two names (the design path keeps its edge barrier as its certificate too)
    uniq, keys = [], set()
    for c in out:
        key = repr(sorted(c.items()))
        if key not in keys:
            keys.add(key)
            uniq.append(c)
    return uniq


def _key(c):
    return (abs(c["p"]) if c.get("p") is not None else None, c["q"], c.get("occurrence", 0))


def _richness(c):
    # a stepped-across chain's numbers first: its flux between tori either side and its shear from both sides
    return (c["found_in"] == CROSSED and bool(c.get("O") or c.get("X") or c.get("unresolved")),
            c["found_in"] in (FAMILY, CROSSED), "half_width_Wb" in c, "flux_Wb" in c,
            sum(len(c.get(k) or []) for k in ("O", "X", "unresolved")))


def merge(lists):
    """One entry a chain (p/q, p taken without its sign) from the per-request lists [(request, chains), ...]: the
    entry of the request that computed the most of it, with found_by, the requests that found it, and found_in,
    the stages that did."""
    groups = {}
    for k, chains in lists:
        for c in chains or []:
            groups.setdefault(_key(c), []).append((k, c))
    out = []
    for key, items in groups.items():
        # of entries that computed as much, a family's that stalled at the chain first: a request that met another
        # rational first gives a family torus with that rational between it and this chain
        rank = lambda kc: (_richness(kc[1]), kc[1]["found_in"] == FAMILY and bool(kc[1].get("stalled_at", True)),
                           -kc[0])
        k_best, best = max(items, key=rank)
        fam = [(k, c) for k, c in items if c["found_in"] == FAMILY]
        stalled = [(k, c) for k, c in fam if c.get("stalled_at", True)]
        e = {kk: v for kk, v in best.items() if kk not in ("request",)}
        e["found_in"] = list(dict.fromkeys(c["found_in"] for _, c in items))
        if fam:
            # the chain some request's family stalled at, met first beyond its last torus
            e["stalled_at"] = bool(stalled)
            if stalled:
                # the family it was met first from: the request, and the transform and flux of its last torus
                k_s, c_s = (k_best, best) if best["found_in"] == FAMILY and best.get("stalled_at", True) \
                    else max(stalled, key=rank)
                e["stalled_from"] = dict(request=k_s, family_iota=(c_s.get("family_iota") or [None])[0],
                                         family_Wb=(c_s.get("family_Wb") or [None])[0])
        e["found_by"] = sorted(set(k for k, _ in items))
        e["values_from"] = k_best
        out.append(e)
    out.sort(key=lambda c: (c.get("flux_Wb") if c.get("flux_Wb") is not None else np.inf, c["q"]))
    return out


def _residue_errs(out, c):
    """Each certificate line's residue_err (a bound on its residue's error, torux.orbits.line_monodromy), where the
    record keeps one: the X line's, the O line's, the edge barrier's orbit's."""
    src = dict(X=c.get("X_line") if isinstance(c.get("X_line"), dict) else c.get("orbit"), O=c.get("O_line"))
    for ln in out.get("lines") or []:
        s = src.get(ln.get("role"))
        if isinstance(s, dict) and _num(s.get("residue_err")) is not None:
            ln["residue_err"] = _num(s["residue_err"])


def _pts(points):
    try:
        P = np.asarray(points, float).reshape(-1, 2)
    except (TypeError, ValueError):
        return None
    return [[float(v) for v in pt] for pt in P] if len(P) else None


def _cert_line(p, q, residue, points, plane_phi, ln=None):
    out = dict(p=_int(p), q=_int(q), residue=_num(residue), plane_phi=float(plane_phi or 0.0))
    out["crossings"] = _pts(points)
    return _measure(out, ln or {})


def provenance(c):
    """What a certificate record (one that holds: ok) rests on, for a reader with a tracer of their own: dict(
    construction, the theorem's name in docs/verification.md's words; lines, the closed field lines it is built on,
    each with p, q, residue, plane_phi and crossings, the line's crossings (R, Z) of that plane, and residue_err, a
    bound on the residue's error, where the record keeps one; interval_Wb or bound_Wb). None for a record that does
    not hold. A lost ring rests on no closed line (lines [])."""
    if not isinstance(c, dict) or not c.get("ok"):
        return None
    kind = str(c.get("kind") or "")
    ch = c.get("chain")
    p, q = (ch.get("p"), ch.get("q")) if isinstance(ch, dict) else (tuple(ch) if isinstance(ch, (list, tuple))
                                                                        and len(ch) == 2 else (None, None))
    out = dict(construction=kind, lines=[])
    if kind.startswith("island_band_manifold_pairs"):
        out["construction"] = "manifold island band, from pairs of hyperbolic orbits"
        out["lines"] = [_cert_line(p, q, o.get("residue"), o.get("points"), c.get("phi_s"), o)
                        for o in c.get("X_orbits") or [] if isinstance(o, dict)]
    elif kind.startswith("island_band_manifold"):
        out["construction"] = "manifold island band"
        for key in ("X_line", "O_line"):
            ln = c.get(key)
            if isinstance(ln, dict):
                out["lines"].append(dict(_cert_line(p, q, ln.get("residue"), ln.get("points"), c.get("phi_s"), ln),
                                         role=key[0]))
    elif kind == "island_band":
        out["construction"] = "island band from island tori"
        ln = c.get("O_line")
        if isinstance(ln, dict):
            out["lines"].append(dict(_cert_line(p, q, ln.get("residue"), ln.get("points"), ln.get("plane_phi"), ln),
                                     role="O"))
    elif kind.startswith(("edge", "manifold_band")):
        if c.get("certificate") == "lost_ring":
            out["construction"] = "lost ring"
        else:
            out["construction"] = "edge barrier" if kind.startswith("edge") else "manifold band"
            orb = c.get("orbit")
            if isinstance(orb, dict):
                out["lines"].append(dict(_cert_line(orb.get("p"), orb.get("q"), orb.get("residue"), orb.get("points"),
                                                    c.get("plane_phi"), orb), role="X"))
            else:
                for k in ("edge_barrier", "survivor"):
                    sub = provenance(c.get(k))
                    if sub is not None:
                        return dict(sub, **{k_: v for k_, v in out.items() if k_ not in sub})
    _residue_errs(out, c)
    iv = c.get("interval")
    if isinstance(iv, (list, tuple)) and len(iv) == 2 and all(_num(v) is not None for v in iv):
        out["interval_Wb"] = sorted(abs(_num(v)) for v in iv)
    if _num(c.get("F_up")) is not None:
        out["bound_Wb"] = abs(_num(c["F_up"]))
    return out
