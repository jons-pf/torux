"""Every chain's band in an answer held against every surface the same answer verified.

A chain's band is its flux_Wb plus or minus its half_width_Wb (the pendulum estimate about a flux had from the family
in iota, torux.chainlist) and, where widths were asked, its bracket (torux.widths).  An invariant torus is a barrier:
a chain's closed lines, its islands and its layer lie on one side of it as a whole.  So no verified surface can lie
inside a chain's band, and where one does the band is wrong on that side.  fit(chains, tori) places every surface the
answer verified against every chain by the chain's closed lines' crossings (torux.widths.side: inside or outside the
surface's section, never by iota), and then:

    placement       each chain's placement: inner_Wb and outer_Wb, the verified surfaces nearest it with every counted
                    crossing outside and inside them, and unplaced, the verified surfaces between them that no
                    crossing places (each crossing nearer the section than the line's accuracy) or that its crossings
                    put on both sides, with the nearest crossing's distance (m).  between_verified_Wb is [inner_Wb,
                    outer_Wb]
    the estimate    cut on a side only by the verified surface placed on that side:

                        flux ->     inner (verified)                outer (verified)
                    estimate:    [=====|==========*=========]          |
                    reported:          [==========*=========]          |
                                       flux_Wb +- half_width_Wb, now about the middle of what is left

                    flux_Wb and half_width_Wb are then the middle and half of what is left, flux_estimate_Wb and
                    half_width_pendulum_Wb the estimate as it was, band_cut_Wb the surfaces that cut it.  Where the
                    estimated flux itself lies beyond a surface placed by the crossings (its interpolation in iota is
                    contradicted by the chain's own lines), or a surface that cannot be placed lies inside the band,
                    or the chain has no crossings to place a surface inside its band by, no band is given: flux_Wb
                    and half_width_Wb are left out (kept as flux_estimate_Wb and half_width_pendulum_Wb) and
                    band_withdrawn says why
    the bracket     already the nearest placed surfaces either side when the closing in placed every surface of the
                    answer; a nearer one placed here moves it (bracket['moved_from_Wb']), and a surface between its
                    two that cannot be placed withdraws it: inner_Wb and outer_Wb None, the two kept as
                    bracket['withdrawn_Wb'] and bracket['withdrawn'] saying why

A chain whose band held a verified surface before the fit says so: band_held_verified_Wb, their fluxes.  A band is
so never narrowed by iota, and never on a side by a surface the crossings put on the other.
"""
from __future__ import annotations

import numpy as np

from .widths import inside, line_margin, section, SAG_SAFETY

REL_SAME = 1e-13          # two surfaces whose fluxes agree to this are one


def lines_of(c):
    """[(points (k, 2), plane phi, accuracy (m))] of every closed line of a chain entry with crossings: its O, X and
    unresolved lines (whose type is not known, but whose crossings are those of a closed line all the same)."""
    m = line_margin(c)
    out = []
    for ln in (c.get("O") or []) + (c.get("X") or []) + (c.get("unresolved") or []):
        P = np.asarray(ln.get("section") or [], float).reshape(-1, 2)
        if len(P):
            out.append((P, float(ln.get("plane_phi", 0.0) or 0.0), m))
    return out


class Surface:
    """A verified surface of the answer with its sections (by plane) kept for every chain it is placed against.
    around: whether it is a torus of the magnetic axis's family (torux.analysis.torus_topology); one that is not is
    never placed, so a band holding it is withdrawn rather than cut by it."""
    __slots__ = ("torus", "flux", "sections", "where", "around")

    def __init__(self, torus, where, around=True):
        self.torus, self.where, self.around = torus, where, bool(around)
        self.flux = abs(float(torus.flux))
        self.sections = {}


def pool(*groups):
    """One Surface a flux from groups of (torus objects, where they came from[, around the axis]), the first of two
    at one flux kept."""
    out = []
    for g in groups:
        tori, where, around = (tuple(g) + (True,))[:3]
        for t in tori or []:
            if t is None:
                continue
            try:
                s = Surface(t, where, around)
            except (TypeError, ValueError, AttributeError):
                continue
            if not np.isfinite(s.flux) or any(abs(s.flux - x.flux) <= REL_SAME * s.flux for x in out):
                continue
            out.append(s)
    return sorted(out, key=lambda s: s.flux)


def place(s, lines):
    """Where the Surface s lies against a chain's lines: ('inner' | 'outer' | 'crosses' | None, the nearest crossing's
    distance to the section (m)). A crossing counts only farther from the section than its line's accuracy and four
    times the section's chord error; 'crosses' where counted crossings lie on both sides."""
    votes, near = set(), np.inf
    if not s.around:
        return None, near
    for P, phi, m in lines:
        if phi not in s.sections:
            s.sections[phi] = section(s.torus, phi)
        poly, sag = s.sections[phi]
        ins, dist = inside(poly, P)
        near = min(near, float(np.min(dist)))
        far = dist > m + SAG_SAFETY * sag
        votes |= {"outer" if i else "inner" for i in ins[far]}
    if not votes:
        return None, near
    return (votes.pop() if len(votes) == 1 else "crosses"), near


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def held(c, surfaces):
    """The fluxes of the Surfaces strictly inside a chain's reported band as it stands: its estimate (flux_Wb +-
    half_width_Wb) and its bracket (inner_Wb to outer_Wb)."""
    out = set()
    f, h = _num(c.get("flux_Wb")), _num(c.get("half_width_Wb"))
    if f is not None and h:
        out |= {s.flux for s in surfaces if f - h < s.flux < f + h}
    br = c.get("bracket") or {}
    a, b = _num(br.get("inner_Wb")), _num(br.get("outer_Wb"))
    if a is not None and b is not None:
        out |= {s.flux for s in surfaces if a * (1 + REL_SAME) < s.flux < b * (1 - REL_SAME)}
    return sorted(out)


def fit_chain(c, surfaces):
    """One chain entry fitted (module docstring) against the Surfaces of the answer. Returns what it did: 'cut',
    'withdrawn' or None for its estimate. A chain whose band held a verified surface before is flagged
    (band_held_verified_Wb, those surfaces' fluxes)."""
    h0 = held(c, surfaces)
    if h0 and "band_held_verified_Wb" not in c:
        c["band_held_verified_Wb"] = h0
    lines = lines_of(c)
    f = _num(c.get("flux_estimate_Wb", c.get("flux_Wb")))
    h = _num(c.get("half_width_pendulum_Wb", c.get("half_width_Wb")))
    did = None
    if not lines:
        if f is not None and h is not None:
            inner = [s.flux for s in surfaces if f - h < s.flux < f + h]
            if inner:
                _withdraw(c, f, h, f"verified surfaces at {_list(inner)} Wb lie inside the estimate, and the chain "
                          "has no crossings to place them by")
                did = "withdrawn"
        return did
    sides = [(s,) + place(s, lines) for s in surfaces]
    ins = [s for s, w, _ in sides if w == "inner"]
    outs = [s for s, w, _ in sides if w == "outer"]
    lo = max(ins, key=lambda s: s.flux) if ins else None
    hi = min(outs, key=lambda s: s.flux) if outs else None
    a = lo.flux if lo is not None else -np.inf
    b = hi.flux if hi is not None else np.inf
    unplaced = [dict(flux_Wb=s.flux, nearest_m=d if np.isfinite(d) else None,
                     **({"crosses": True} if w == "crosses" else {}),
                     **({} if s.around else {"not_around_axis": True}))
                for s, w, d in sides if w not in ("inner", "outer") and a < s.flux < b]
    pl = dict(inner_Wb=None if lo is None else lo.flux, outer_Wb=None if hi is None else hi.flux,
              surfaces=len(surfaces), placed=len(ins) + len(outs))
    if unplaced:
        pl["unplaced"] = unplaced
    if a >= b:
        # a surface placed inside the chain encloses more flux than one placed outside it: the placement contradicts
        # itself (nested surfaces of one family cannot), and nothing is narrowed by it
        pl["disagreement"] = True
    c["placement"] = pl
    if lo is not None or hi is not None:
        c["between_verified_Wb"] = [pl["inner_Wb"], pl["outer_Wb"]]
        c.pop("half_width_cap_Wb", None)
        c.pop("reaches_inside_verified", None)
        if lo is not None and hi is not None and a < b:
            c["half_width_cap_Wb"] = 0.5 * (b - a)
    if f is not None and h is not None:
        if pl.get("disagreement"):
            _withdraw(c, f, h, "the verified surfaces the chain's crossings place either side disagree")
            did = "withdrawn"
        elif not a < f < b:
            s = lo if f <= a else hi
            _withdraw(c, f, h, f"its estimated flux lies {'inside' if f <= a else 'outside'} the verified surface at "
                      f"{s.flux:.10g} Wb, which the chain's crossings put {'inside' if f <= a else 'outside'} it")
            did = "withdrawn"
        else:
            lo_b, hi_b = max(f - h, a), min(f + h, b)
            inner = [u["flux_Wb"] for u in unplaced if lo_b < u["flux_Wb"] < hi_b]
            if inner:
                _withdraw(c, f, h, f"verified surfaces at {_list(inner)} Wb lie inside the estimate, and the chain's "
                          "crossings do not place them")
                did = "withdrawn"
            elif lo_b > f - h or hi_b < f + h:
                c["flux_estimate_Wb"], c["half_width_pendulum_Wb"] = f, h
                c["flux_Wb"], c["half_width_Wb"] = middle(lo_b, hi_b)
                c["band_cut_Wb"] = [lo_b if lo_b > f - h else None, hi_b if hi_b < f + h else None]
                c.pop("band_withdrawn", None)
                did = "cut"
            else:
                c["flux_Wb"], c["half_width_Wb"] = f, h
                for k in ("flux_estimate_Wb", "band_cut_Wb", "band_withdrawn"):
                    c.pop(k, None)
                if c.get("half_width_pendulum_Wb") == h:
                    c.pop("half_width_pendulum_Wb", None)
    br = c.get("bracket")
    if isinstance(br, dict) and br.get("inner_Wb") is not None and not br.get("withdrawn"):
        _bracket(br, lo, hi, unplaced, pl)
    return did


def _bracket(br, lo, hi, unplaced, pl):
    moved = []
    if lo is not None and lo.flux > br["inner_Wb"] * (1 + REL_SAME):
        moved.append(("inner", br["inner_Wb"]))
        br["inner_Wb"], br["inner_iota"] = lo.flux, float(lo.torus.iota)
        br["inner_surface"] = None
    if hi is not None and (br.get("outer_Wb") is None or hi.flux < br["outer_Wb"] * (1 - REL_SAME)):
        moved.append(("outer", br.get("outer_Wb")))
        br["outer_Wb"], br["outer_iota"] = hi.flux, float(hi.torus.iota)
        br["outer_surface"] = None
        br["one_sided"] = False
    if moved:
        br["moved_from_Wb"] = {k: v for k, v in moved}
    a, b = br["inner_Wb"], br.get("outer_Wb")
    inside_ = [u["flux_Wb"] for u in unplaced if a < u["flux_Wb"] and (b is None or u["flux_Wb"] < b)]
    if pl.get("disagreement"):
        br["disagreement"] = True
    if inside_:
        br["unplaced_Wb"] = inside_
    if b is not None and inside_:
        # a verified surface between the two that the crossings cannot place lies on one side of the chain, but not
        # one that can be told: no bracket is claimed rather than one that holds it (the two kept as withdrawn_Wb)
        br["withdrawn_Wb"] = [a, b]
        br["withdrawn"] = (f"verified surfaces at {_list(inside_)} Wb lie between its two, and the chain's crossings "
                           "do not place them")
        br["inner_Wb"] = br["outer_Wb"] = br["inner_iota"] = br["outer_iota"] = None
        br["inner_surface"] = br["outer_surface"] = None
        br["one_sided"] = False
    if b is not None and not br.get("disagreement") and not inside_:
        br["half_width_Wb"], br["flux_Wb"] = 0.5 * (b - a), 0.5 * (a + b)
    else:
        br["half_width_Wb"] = br["flux_Wb"] = None


def middle(lo, hi):
    """(middle, half) of the interval [lo, hi], the half rounded down so that middle -+ half never passes lo or hi
    (a band cut at a surface then never holds it by a rounding)."""
    m, h = 0.5 * (lo + hi), 0.5 * (hi - lo)
    while h > 0 and (m - h < lo or m + h > hi):
        h = float(np.nextafter(h, 0.0))
    return m, h


def _withdraw(c, f, h, why):
    c["flux_estimate_Wb"], c["half_width_pendulum_Wb"] = f, h
    c.pop("flux_Wb", None)
    c.pop("half_width_Wb", None)
    c.pop("band_cut_Wb", None)
    c["band_withdrawn"] = why


def _list(xs):
    return ", ".join(f"{x:.10g}" for x in xs[:4]) + (f" and {len(xs) - 4} more" if len(xs) > 4 else "")


def fit(chains, surfaces):
    """Every chain entry fitted (fit_chain) against the Surfaces (pool): a summary of what was done."""
    out = dict(surfaces=len(surfaces), chains=len(chains), held=0, cut=0, withdrawn=0, unplaced=0, brackets_moved=0,
               still_held=0)
    for c in chains:
        did = fit_chain(c, surfaces)
        out["held"] += bool(c.get("band_held_verified_Wb"))
        out["still_held"] += bool(held(c, surfaces))
        if did:
            out[did] += 1
        if (c.get("placement") or {}).get("unplaced"):
            out["unplaced"] += 1
        if (c.get("bracket") or {}).get("moved_from_Wb"):
            out["brackets_moved"] += 1
    return out
