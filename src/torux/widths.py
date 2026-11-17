"""Island chain widths measured by closing in on each chain with verified surfaces, asked for with largest=True and
widths_s (seconds of its own, off by default):

    a = torux.solve("coils.json", largest=True, budget_s=1200, widths_s=1800)
    for c in a.evidence["chains"]:
        print(c["p"], c["q"], c.get("bracket"))

What it measures.  An invariant torus is a barrier: a closed field line lies inside or outside it as a whole, and so
do the islands round a chain's elliptic (O) lines and the chaotic layer round its hyperbolic (X) lines, which are
made of field lines too.  So a verified surface with a chain's closed lines outside it encloses less flux than the
chain's islands and layer, and one with the lines inside it encloses more.  For every chain of the answer
(answer.evidence['chains']) with closed lines, the stage asks direct requests (torux.api.solve, as the search's own
requests are asked) at fluxes between the verified surfaces nearest the chain either side, and places every surface
they verify against every chain by its lines' crossings of a plane.  The bracket is the pair of verified surfaces
nearest the chain, one each side:

    flux ->
       inner (verified)          islands and layer of the chain           outer (verified)
    ..........|  slack  [====================================]  slack  |..........
              |<---------------------- bracket ---------------------->|

so the island and its layer lie inside the bracket: (outer - inner) / 2 bounds the chain's half width from above, and
(inner + outer) / 2 is its place in flux, both read from verified surfaces only.  A chain with no verified surface
beyond it is one-sided: its bracket has the inner surface only, with outer None, and none is made up.

Which side.  A surface's side is decided by the chain's crossings, never by the rotational transform (which orders
the surfaces only while iota is monotonic in flux).  Each crossing is tested against the surface's section in the
crossing's plane: inside or outside a polygon of SECTION_N points, counted only when it lies farther from the polygon
than the polygon's own chord error (four times it) and the accuracy of the line's crossings (LINE_M for lines Newton
converged from the family's tori, EDGE_LINE_M for lines an edge barrier screened).  Every counted crossing must say
the same: one that says otherwise means the surface crosses the chain (a contradiction, so the surface is not used
for that chain and the bracket says so).  A surface no crossing is counted for (too close to every line) is not used
for that chain either.  Only a surface around the magnetic axis is placed (torux.analysis.torus_topology).

Where it asks.  Each side of each chain is closed in on by its own sequence of requests: between the verified
surface on that side and the nearest flux towards the chain where a request did not verify (or the surface on the
other side), the next request goes where the chain's edge is expected, from the rotational transform of the two
surfaces and the pendulum half width (torux.chains), kept within the middle half of the interval once a request has
failed, so that the interval shrinks by a quarter at least each time.  The transform chooses where to ask and
nothing more.  A side stops when the interval left is below REL of the bracket, after MAX_SIDE requests, or when
the time ends.  Requests run each in its own process, ``jobs`` at a time (torux.batch), each with at most
REQUEST_S seconds; the requests still running when widths_s ends are stopped and count as not verified.
"""
from __future__ import annotations

import math
import os
import tempfile
import time

import numpy as np

REQUEST_S = 300.0        # the budget of one request (the solver's own for a few circular coils)
MIN_REQUEST_S = 30.0     # a request is not started with less than this left
REL = 0.01               # a side is closed when what is left to search is this share of the bracket
MAX_SIDE = 16            # requests at most for one side of one chain
SECTION_N = 1 << 14      # points of a surface's section a crossing is tested against
SAG_SAFETY = 4.0         # a crossing counts only farther than this many times the section's chord error
LINE_M = 1e-5            # m: how far the crossings of a line Newton converged (family, stepped across) may be off
EDGE_LINE_M = 1e-4       # m: the same for a line only an edge barrier screened
PENDULUM = 1.45          # the pendulum half width over the island's, as measured: only where to ask


# --------------------------------------------------------------------------- placing a surface against a chain
def section(torus, phi, n=SECTION_N):
    """A surface's section at the cylindrical angle phi: (n, 2) polygon (R, Z) and its chord error (m), the largest
    distance of the section's points halfway along each chord from the chord's midpoint."""
    th = 2 * np.pi * np.arange(2 * n) / (2 * n)
    R, Z = torus.geometry_grid(th, np.array([float(phi)]))
    P = np.stack([R[:, 0], Z[:, 0]], -1)
    V, mid = P[0::2], P[1::2]
    chord = 0.5 * (V + np.roll(V, -1, axis=0))
    return V, float(np.max(np.hypot(*(mid - chord).T)))


def inside(poly, pts):
    """For each point: inside the closed polygon (even-odd rule), and its distance to the polygon (m)."""
    P = np.asarray(poly, float)
    Q = np.roll(P, -1, axis=0)
    x = np.asarray(pts, float).reshape(-1, 2)
    px, py = x[:, 0:1], x[:, 1:2]
    c = (P[None, :, 1] > py) != (Q[None, :, 1] > py)
    dy = np.where(Q[:, 1] != P[:, 1], Q[:, 1] - P[:, 1], 1e-300)
    xin = P[None, :, 0] + (py - P[None, :, 1]) * (Q[None, :, 0] - P[None, :, 0]) / dy[None, :]
    ins = (np.sum(c & (px < xin), axis=1) % 2) == 1
    d = Q - P
    dd = np.maximum(np.sum(d * d, 1), 1e-300)
    t = np.clip(((px - P[None, :, 0]) * d[None, :, 0] + (py - P[None, :, 1]) * d[None, :, 1]) / dd[None, :], 0, 1)
    dist = np.min(np.hypot(P[None, :, 0] + t * d[None, :, 0] - px, P[None, :, 1] + t * d[None, :, 1] - py), axis=1)
    return ins, dist


def chain_lines(c):
    """[(points (k, 2), plane phi)] of a chain entry's O and X lines with crossings."""
    out = []
    for ln in (c.get("O") or []) + (c.get("X") or []):
        P = np.asarray(ln.get("section") or [], float).reshape(-1, 2)
        if len(P):
            out.append((P, float(ln.get("plane_phi", 0.0) or 0.0)))
    return out


def line_margin(c):
    """How far (m) a chain's crossings may be off: lines Newton converged from a family's tori, or only an edge
    barrier's."""
    found = c.get("found_in") or []
    found = [found] if isinstance(found, str) else found
    return LINE_M if any(f in ("family", "stepped across") for f in found) else EDGE_LINE_M


def side(torus, lines, margin, sections=None):
    """Where a verified surface lies against a chain's closed lines: 'inner' (every counted crossing outside it),
    'outer' (every counted crossing inside it), 'crosses' (counted crossings either side) or None (no crossing far
    enough from the section to count). sections: a dict phi -> (polygon, chord error) of this surface, filled here."""
    sections = {} if sections is None else sections
    votes = set()
    for P, phi in lines:
        if phi not in sections:
            sections[phi] = section(torus, phi)
        poly, sag = sections[phi]
        ins, dist = inside(poly, P)
        far = dist > margin + SAG_SAFETY * sag
        votes |= {"outer" if i else "inner" for i in ins[far]}
    if not votes:
        return None
    return votes.pop() if len(votes) == 1 else "crosses"


# --------------------------------------------------------------------------- the stage
class _Torus:
    __slots__ = ("flux", "iota", "torus", "came", "sections")

    def __init__(self, torus, came):
        self.torus, self.came = torus, came
        self.flux = abs(float(torus.flux))
        self.iota = float(torus.iota)
        self.sections = {}


class _Chain:
    def __init__(self, c, idx):
        self.entry, self.idx = c, idx
        self.lines = chain_lines(c)
        self.margin = line_margin(c)
        self.iota_r = c.get("iota")
        h = c.get("half_width_pendulum_Wb", c.get("half_width_Wb"))
        self.h = float(h) / PENDULUM if h is not None and np.isfinite(h) and h > 0 else None
        f = c.get("flux_Wb")
        self.f_est = float(f) if f is not None and np.isfinite(f) else None
        self.inner = self.outer = None           # _Torus nearest either side
        self.unusable = set()                    # fluxes of verified surfaces not placed for this chain
        self.crossing = []                       # fluxes of surfaces that cross it
        self.asked = {"inner": 0, "outer": 0}
        self.stop = {"inner": None, "outer": None}
        self.start = None                        # the bracket before closing in
        self.span0 = {}                          # side -> its interval when first asked (one-sided chains)
        self.core_s = 0.0

    def key(self):
        return (abs(self.entry["p"]) if self.entry.get("p") is not None else None, self.entry["q"],
                self.entry.get("occurrence", 0))

    def place(self, t):
        s = side(t.torus, self.lines, self.margin, t.sections)
        if s == "inner":
            if self.inner is None or t.flux > self.inner.flux:
                self.inner = t
        elif s == "outer":
            if self.outer is None or t.flux < self.outer.flux:
                self.outer = t
        else:
            self.unusable.add(t.flux)
            if s == "crosses":
                self.crossing.append(t.flux)
        return s

    def consistent(self):
        return self.inner is None or self.outer is None or self.inner.flux < self.outer.flux


def _guess(ch, s, a, u):
    """Where the chain's edge on side s is expected, between a (the verified surface on that side) and u: the
    resonance's flux linear in iota between the two surfaces either side (else the chain's own flux), less (inner)
    or more (outer) than it by the expected half width."""
    c0 = None
    if ch.inner is not None and ch.outer is not None and ch.iota_r is not None:
        i0, i1 = ch.inner.iota, ch.outer.iota
        if (i0 - ch.iota_r) * (i1 - ch.iota_r) < 0:
            c0 = ch.inner.flux + (ch.iota_r - i0) / (i1 - i0) * (ch.outer.flux - ch.inner.flux)
    if c0 is None:
        c0 = ch.f_est if ch.f_est is not None else 0.5 * (a + u)
    h = ch.h if ch.h is not None else 0.0
    if ch.inner is not None and ch.outer is not None:
        h = min(h, 0.25 * (ch.outer.flux - ch.inner.flux))
    return c0 - 1.1 * h if s == "inner" else c0 + 1.1 * h


class Stage:
    """The closing in on the chains of one answer: chains (entries of answer.evidence['chains']), the verified
    surfaces known (torus objects), the fluxes known not to verify (failed) or not to hold a surface beyond them
    (bound), the magnetic axis (for the topology check of new surfaces) and a runner of requests."""

    def __init__(self, chains, tori, axis=None, failed=(), bound=None, verbose=False):
        self.axis, self.verbose = axis, verbose
        self.pool = []
        self.failed = set(float(f) for f in failed)
        self.bound = bound
        self.pending = {}                        # flux -> (chain, side)
        self.log = []
        self.chains = [_Chain(c, i) for i, c in enumerate(chains) if chain_lines(c)]
        for t in tori:
            self.add(t, "search")
        for ch in self.chains:
            ch.start = [None if ch.inner is None else ch.inner.flux, None if ch.outer is None else ch.outer.flux]

    def add(self, torus, came):
        t = _Torus(torus, came)
        if any(abs(t.flux - x.flux) <= 1e-13 * t.flux for x in self.pool):
            return []
        self.pool.append(t)
        return [(ch, ch.place(t)) for ch in self.chains]

    # ------------------------------------------------------------- one side's next flux
    def _far(self, ch):
        """The far end of a one-sided chain's inner side: the least flux beyond its inner surface known not to
        verify, or no surface's bound, or else the chain's own flux and twice its expected half width past it."""
        a = ch.inner.flux
        cand = [f for f in self.failed | ch.unusable if f > a * (1 + 1e-12)]
        if self.bound is not None and self.bound > a:
            cand.append(self.bound)
        if cand:
            return min(cand), True
        h = ch.h if ch.h is not None else 0.01 * a
        f = ch.f_est if ch.f_est is not None and ch.f_est > a else a
        return f + 2 * h, False

    def _interval(self, ch, s):
        """(a, u, tol): the verified surface on side s, the nearest flux towards the chain that stops the search
        (a failed request, an unusable surface, the other side's surface), and the slack small enough to stop at."""
        two = ch.inner is not None and ch.outer is not None
        if s == "inner":
            a = ch.inner.flux
            hi, stopped = (ch.outer.flux, False) if ch.outer is not None else self._far(ch)
            blk = [f for f in self.failed | ch.unusable if a < f < hi]
            u = min(blk) if blk else hi
            blk = blk or stopped
        else:
            a = ch.outer.flux
            lo = ch.inner.flux
            blk = [f for f in self.failed | ch.unusable if lo < f < a]
            u = max(blk) if blk else lo
        if two:
            tol = REL * (ch.outer.flux - ch.inner.flux)
        else:
            # (one-sided: of the chain's expected width, else of the interval as it first was)
            ch.span0.setdefault(s, abs(u - a))
            tol = REL * (2 * ch.h if ch.h is not None else ch.span0[s])
        tol = max(tol, 1e-9 * a)
        return a, u, bool(blk), tol

    def next_flux(self, ch, s):
        """The next flux to ask for side s of the chain, or None (and why in ch.stop[s] when the side is done)."""
        if ch.stop[s] is not None:
            return None
        if ch.inner is None:
            ch.stop[s] = "no verified surface inside the chain"
            return None
        if s == "outer" and ch.outer is None:
            return None                          # one-sided: nothing beyond it to close in from
        if not ch.consistent():
            ch.stop[s] = "the surfaces either side disagree"
            return None
        a, u, blocked, tol = self._interval(ch, s)
        sgn = 1.0 if s == "inner" else -1.0
        pend = sorted((f for f, (c2, _) in self.pending.items() if (f - a) * sgn > 0 and (u - f) * sgn > 0),
                      key=lambda f: (f - a) * sgn)
        if abs(u - a) <= tol:
            if not pend:
                ch.stop[s] = "closed"
            return None
        if ch.asked[s] >= MAX_SIDE:
            if not pend:
                ch.stop[s] = "requests"
            return None
        if not pend:
            g = _guess(ch, s, a, u)
            step = (g - a) * sgn
            span = abs(u - a)
            if blocked:
                step = min(max(step, 0.25 * span), 0.75 * span)
            else:
                step = min(max(step, 2 * tol, 0.25 * (ch.h or 0.0)), 0.9 * span)
            return a + sgn * step
        pts = [a] + pend + [u]
        gaps = [abs(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)]
        i = int(np.argmax(gaps))
        if gaps[i] <= 2 * tol:
            return None
        return 0.5 * (pts[i] + pts[i + 1])

    def _shared(self, ch, s):
        """Whether another side searches the same interval with a request running: chains beyond the last surface
        share their inner surface and its nearest failure, and one sequence of requests serves them all."""
        a, u, _, _ = self._interval(ch, s)
        for f, (c2, s2) in self.pending.items():
            if c2 is ch and s2 == s:
                return False
        for f, (c2, s2) in self.pending.items():
            if c2 is ch or c2.inner is None or (s2 == "outer" and c2.outer is None):
                continue
            a2, u2, _, _ = self._interval(c2, s2)
            if abs(a2 - a) <= 1e-13 * a and abs(u2 - u) <= 1e-13 * abs(u):
                return True
        return False

    # ------------------------------------------------------------- the loop
    def run(self, runner, deadline, request_s=REQUEST_S):
        """Close in until every side is done or the deadline: runner.start(flux, budget), runner.poll() -> [(flux,
        answer, wall_s)], runner.slots() (free processes), runner.stop() (every one still running)."""
        from .ask import verified
        t0 = time.time()
        sides = [(ch, s) for ch in self.chains for s in ("inner", "outer")]
        try:
            while True:
                for f, ans, wall in runner.poll():
                    self._take(f, ans, wall, verified)
                left = deadline - time.time()
                started = False
                while runner.slots() > 0 and left > MIN_REQUEST_S:
                    best = None
                    for ch, s in sides:
                        if ch.stop[s] is not None:
                            continue
                        n_p = sum(1 for c2, s2 in self.pending.values() if c2 is ch and s2 == s)
                        k = (n_p, ch.asked[s], -(ch.h or 0.0))
                        if best is not None and k >= best[0]:
                            continue
                        f = self.next_flux(ch, s)
                        if f is None or any(abs(f - x) <= 1e-12 * f for x in list(self.pending) + list(self.failed)):
                            continue
                        if self._shared(ch, s):
                            continue
                        best = (k, ch, s, f)
                    if best is None:
                        break
                    _, ch, s, f = best
                    ch.asked[s] += 1
                    self.pending[f] = (ch, s)
                    runner.start(f, min(request_s, left))
                    started = True
                    left = deadline - time.time()
                if not self.pending:
                    if all(ch.stop[s] is not None or self.next_flux(ch, s) is None for ch, s in sides):
                        break
                    if left <= MIN_REQUEST_S:
                        break
                if time.time() > deadline:
                    break
                if not started:
                    time.sleep(0.2)
        finally:
            for f in list(self.pending):
                ch, s = self.pending.pop(f)
                self.failed.add(f)
                self.log.append(dict(flux=f, chain=[ch.entry.get("p"), ch.entry.get("q")], side=s, status="cut",
                                     wall_s=None))
            runner.stop()
        for ch in self.chains:
            for s in ("inner", "outer"):
                if ch.stop[s] is None:
                    ch.stop[s] = "no verified surface beyond the chain" if s == "outer" and ch.outer is None \
                        else "time"
        return time.time() - t0

    def _take(self, f, ans, wall, verified):
        ch, s = self.pending.pop(f, (None, None))
        entry = dict(flux=f, chain=None if ch is None else [ch.entry.get("p"), ch.entry.get("q")], side=s,
                     status=getattr(ans, "status", None), wall_s=None if wall is None else round(wall, 1))
        if ch is not None and wall is not None:
            ch.core_s += wall
        ok = ans is not None and verified(ans)
        if not ok and ans is not None:
            from .ask import _why
            try:
                entry["why"] = _why(ans)
            except Exception:  # noqa: BLE001
                pass
        if ok and self.axis is not None:
            from .analysis import torus_topology
            try:
                ok = bool(torus_topology(ans.surface, self.axis)["ok"])
            except Exception:  # noqa: BLE001
                ok = False
            if not ok:
                entry["topology"] = False
        if ok:
            entry["flux_reached"] = abs(float(ans.surface.flux))
            entry["iota"] = float(ans.surface.iota)
            placed = self.add(ans.surface, "request")
            entry["placed"] = {f"{c2.entry.get('p')}/{c2.entry.get('q')}": w for c2, w in placed if w is not None}
        else:
            self.failed.add(f)
        self.log.append(entry)
        if self.verbose:
            print(f"   widths: {f:.9g} Wb {entry['status']} in {entry['wall_s']} s", flush=True)

    # ------------------------------------------------------------- what it found
    def results(self, surfaces=None):
        """{chain key: bracket dict} (module docstring). surfaces: a list each bracketing surface is put in once
        (surface_dict), the bracket naming it by its place there (inner_surface, outer_surface)."""
        out = {}
        where = {}

        def put(t):
            if surfaces is None or t is None:
                return None
            if id(t) not in where:
                where[id(t)] = len(surfaces)
                surfaces.append(surface_dict(t.torus))
            return where[id(t)]

        for ch in self.chains:
            a, b = ch.inner, ch.outer
            br = dict(inner_Wb=None if a is None else a.flux, outer_Wb=None if b is None else b.flux,
                      inner_iota=None if a is None else a.iota, outer_iota=None if b is None else b.iota,
                      one_sided=bool(a is not None and b is None), start_Wb=ch.start,
                      requests=ch.asked["inner"] + ch.asked["outer"], request_s=round(ch.core_s, 1),
                      stop=dict(ch.stop), line_margin_m=ch.margin)
            if surfaces is not None:
                br["inner_surface"], br["outer_surface"] = put(a), put(b)
            if a is not None and b is not None:
                br["half_width_Wb"] = 0.5 * (b.flux - a.flux)
                br["flux_Wb"] = 0.5 * (a.flux + b.flux)
            if not ch.consistent():
                br["disagreement"] = True
                br["half_width_Wb"] = br["flux_Wb"] = None
            if ch.crossing:
                br["crossing_surfaces_Wb"] = sorted(ch.crossing)
            # the nearest fluxes towards the chain where no surface was verified (no bound: the layer may lie
            # farther in)
            if a is not None:
                up = [f for f in self.failed if f > a.flux and (b is None or f < b.flux)]
                br["nearest_unverified_Wb"] = [min(up) if up else None, max(up) if up and b is not None else None]
            out[ch.key()] = br
        return out


SURFACE_M = 1e-16        # m: the bracketing surfaces kept in the answer, their coefficients to this


def surface_dict(s, eps=SURFACE_M):
    """A surface as torux.Result.to_dict writes it, for a tracer of one's own: its Fourier coefficients, each to eps
    (m), and only the modes up to the last above eps in m and |n|. dropped_m bounds what that leaves out at any point
    (the sum of the coefficients cut and of half eps for each one kept)."""
    keys = [k for k in ("rc", "rs", "zc", "zs") if hasattr(s, k)]
    C = {k: np.asarray(getattr(s, k), float) for k in keys}
    N = int(s.N)
    A = np.max([np.abs(v) for v in C.values()], axis=0)
    rows = np.nonzero(A.max(1) > eps)[0]
    cols = np.nonzero(A.max(0) > eps)[0]
    M2 = int(rows.max()) if len(rows) else 0
    N2 = int(np.max(np.abs(cols - N))) if len(cols) else 0
    kept = {k: v[:M2 + 1, N - N2:N + N2 + 1] for k, v in C.items()}
    dropped = sum(float(np.abs(v).sum() - np.abs(kept[k]).sum()) for k, v in C.items())
    dropped += 0.5 * eps * sum(v.size for v in kept.values())
    digits = int(round(-math.log10(eps)))
    return dict(nfp=int(s.nfp), M=M2, N=N2, iota=float(s.iota), flux_Wb=float(s.flux),
                phi0=float(getattr(s, "phi0", 0.0)), solved_M=int(s.M), solved_N=N, dropped_m=dropped,
                **{k: np.round(v, digits).tolist() for k, v in kept.items()})


# --------------------------------------------------------------------------- requests in their own processes
class Processes:
    """Direct requests of one source at fluxes, each in a fresh process (torux.batch), at most ``jobs`` at once.

    Where the system lets a process choose its cores (Linux), each request runs on cores of its own, TORUX_THREADS of
    them: solves left to share every core slow each other down many times over (on eight cores, four requests that
    each verify in 8 to 56 s alone came back undetermined after 215 s each, and in 8 to 56 s each when given two cores
    apiece), so then at most as many run at once as there are cores for."""

    def __init__(self, src, jobs, opts):
        from . import batch
        self.b = batch
        self.src, self.jobs = src, max(1, int(jobs))
        self.cpus = None
        try:
            mine = sorted(os.sched_getaffinity(0))
            t = batch._threads()
            self.jobs = min(self.jobs, max(1, len(mine) // t))
            if len(mine) >= 2 * t:
                self.cpus = [mine[i * t:(i + 1) * t] for i in range(len(mine) // t)]
        except (AttributeError, OSError):
            self.cpus = None
        self.args = dict(budget_s=None, tol=opts.get("tol", 3e-15), verify=opts.get("verify", True),
                         gradients=False, minor_radius=None, volume=None, largest=False)
        self.tmp = tempfile.TemporaryDirectory(prefix="torux-widths-")
        self.env = batch._child_env()
        self.running = []
        self.k = 0

    def slots(self):
        return self.jobs - len(self.running)

    def start(self, flux, budget):
        import pickle
        import subprocess
        import sys
        it = self.b._Item(self.src, "widths")
        it.flux = float(flux)
        args = dict(self.args, budget_s=float(budget))
        cores = None
        if self.cpus is not None:
            used = {tuple(c) for _, _, _, c in self.running if c is not None}
            cores = next(c for c in self.cpus if tuple(c) not in used)
        # as torux.batch._start, on its own cores
        base = os.path.join(self.tmp.name, str(self.k))
        job = dict(source=it.payload, flux=it.flux, overrun=self.b._OVERRUN, budget_file=base + ".budget",
                   choices=it.choices, **args)
        with open(base + ".job", "wb") as fh:
            pickle.dump(job, fh, protocol=pickle.HIGHEST_PROTOCOL)
        errfh = open(base + ".err", "wb")
        pin = (lambda: os.sched_setaffinity(0, cores)) if cores is not None else None
        proc = subprocess.Popen([sys.executable, "-c", self.b._WORKER, base + ".job", base + ".out"],
                                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errfh, env=self.env,
                                close_fds=True, preexec_fn=pin)
        j = self.b._Job(it, proc, base + ".out", base + ".err", errfh, time.time(), self.b._OVERRUN)
        j.choices = it.choices
        self.k += 1
        self.running.append((j, float(flux), float(budget), cores))

    def poll(self):
        out, still = [], []
        for j, f, budget, cores in self.running:
            if j.proc.poll() is None:
                if time.time() - j.t0 <= 2 * budget + 60:
                    still.append((j, f, budget, cores))
                    continue
                self.b._stop([j.proc])
            ans, _ = self.b._collect(j)
            out.append((f, ans, time.time() - j.t0))
        self.running = still
        return out

    def stop(self):
        self.b._stop([j.proc for j, _, _, _ in self.running])
        for j, _, _, _ in self.running:
            try:
                j.errfh.close()
            except OSError:
                pass
        self.running = []
        self.tmp.cleanup()


def default_jobs():
    """Processes at once: the cores this process may use divided by the threads each solve uses."""
    from .batch import _cores, _threads
    return max(1, _cores() // _threads())


def attach(chains, results):
    """Each chain entry its bracket (results from Stage.results, by chain key)."""
    for c in chains:
        k = (abs(c["p"]) if c.get("p") is not None else None, c["q"], c.get("occurrence", 0))
        if k in results:
            c["bracket"] = results[k]
    return chains


def table_cell(chains):
    """The torux.many table's chain_brackets: one entry a chain with a bracket, as JSON."""
    import json
    rows = []
    for c in chains or []:
        b = c.get("bracket")
        if not b:
            continue
        rows.append(dict(p=c.get("p"), q=c.get("q"), occurrence=c.get("occurrence", 0), inner_Wb=b.get("inner_Wb"),
                         outer_Wb=b.get("outer_Wb"), one_sided=b.get("one_sided"),
                         half_width_Wb=b.get("half_width_Wb"), flux_Wb=b.get("flux_Wb"),
                         requests=b.get("requests"), disagreement=bool(b.get("disagreement"))))
    return json.dumps(rows) if rows else None


def words(summary):
    if summary.get("error"):
        return f"chain widths: the closing in ended in an error after {summary.get('wall_s', 0):.0f} s ({summary['error']})"
    n, k = summary.get("chains", 0), summary.get("closed", 0)
    return (f"chain widths: {summary.get('two_sided', 0)} of {n} chains bracketed on both sides and "
            f"{summary.get('one_sided', 0)} on the inner side only, by {summary.get('requests', 0)} requests in "
            f"{summary.get('wall_s', 0):.0f} s, {k} side{'s' if k != 1 else ''} closed to {REL:.0%} "
            f"(each chain's bracket: answer.evidence['chains'][i]['bracket'])")


def stage(S, r, widths_s, jobs=None, verbose=False):
    """The closing in after a largest search S answered with r: (results by chain key, summary)."""
    from .chainlist import merge
    from .crossed import annotate
    t0 = time.time()
    chains = merge([(i + 1, e.get("chains")) for i, e in enumerate(S.log)])
    tori, axis = [r.surface], None
    if S.crossed is not None:
        tori, _, axis = S.crossed
        try:
            annotate(chains, *S.crossed)
        except Exception:  # noqa: BLE001
            pass
    elif getattr(r, "_family", None):
        axis = r._family.get("axis")
    bound = None
    for _, c, _ in S.certs:
        b = c.get("no_surface_above_Wb")
        if b is not None and (bound is None or b < bound):
            bound = float(b)
    st = Stage(chains, tori, axis=axis, failed=list(S.failed), bound=bound, verbose=verbose)
    jobs = default_jobs() if jobs is None else int(jobs)
    runner = Processes(S.src, jobs, S.opts)
    wall = st.run(runner, t0 + float(widths_s))
    surfaces = []
    res = st.results(surfaces)
    two = sum(1 for b in res.values() if b["inner_Wb"] is not None and b["outer_Wb"] is not None
              and not b.get("disagreement"))
    one = sum(1 for b in res.values() if b["one_sided"])
    summary = dict(budget_s=float(widths_s), wall_s=round(time.time() - t0, 1), jobs=jobs, chains=len(res),
                   two_sided=two, one_sided=one, requests=len(st.log),
                   verified=sum(1 for e in st.log if e.get("flux_reached") is not None),
                   closed=sum(1 for b in res.values() for v in b["stop"].values() if v == "closed"),
                   request_s=round(sum(e["wall_s"] or 0.0 for e in st.log), 1), loop_s=round(wall, 1),
                   log=st.log, surfaces=surfaces)
    return res, summary
