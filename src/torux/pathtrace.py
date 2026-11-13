"""The path tracer behind answer.render: torux's hero look, for one design.

Light is followed from an orthographic camera into a scene of opaque metal (the coils as round tubes, magnets as
small pills, axisymmetric conductors as the solids they sweep) standing on a white floor under one soft key light
and a sky, and the light of the plasma is gathered along each camera ray: thin fibres of light (field lines), each
a sharp core with a soft halo, and a faint glow in proportion to the length of the ray inside a closed mesh (the
volume the plasma fills).  The tracing writes linear-light buffers once.  The page's colours are applied after, by
``composite``, so that one trace gives the light page and the dark page.

    scene = Scene(caps, tris, fibres, channels, volume, glow_points, camera, frame)
    buffers = scene.render(spp)
    image = composite(buffers, scene, PALETTES["light"])      # linear light, (H, W, 3)
    write_png(path, to8(image))

Everything here is decoration-neutral geometry and light except the marks ``composite`` documents as decoration:
the worn edges, faint tarnish and hair-thin scratches on the metal, its sheen, and on the dark page a pool of
light under the design and the plasma's light on the metal and the floor round it.  None of them encodes anything.

The tracer itself is the one that drew torux's README hero (the twelve machines), made general: one design, any
number of fibre channels, numba kernels compiled once and kept in numba's cache.
"""
from __future__ import annotations

import math
import struct
import zlib

import numpy as np
from numba import njit, prange

INF = 1e30


# ============================================================================ geometry
def revolve(poly, nphi):
    """Triangles of the solid an (R, Z) polygon sweeps about z: vertices, vertex normals (smooth round z, sharp at
    the corners), and per vertex the distance along its polygon edge to that edge's two corners (m), with each
    corner's convexity: what the worn edges are drawn from."""
    q = np.asarray(poly, float)
    if 0.5 * np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1]) < 0:
        q = q[::-1]                                   # counter-clockwise in (R, Z): (dZ, -dR) points outwards
    d0 = np.roll(q, -1, axis=0) - q
    q = q[np.hypot(d0[:, 0], d0[:, 1]) > 1e-12]
    p, r = q, np.roll(q, -1, axis=0)
    d = r - p
    L = np.hypot(d[:, 0], d[:, 1])
    nR, nZ = d[:, 1] / L, -d[:, 0] / L
    dprev = p - np.roll(p, 1, axis=0)                 # the edge arriving at p
    cross_p = dprev[:, 0] * d[:, 1] - dprev[:, 1] * d[:, 0]
    convex_p = (cross_p > 0).astype(float)            # a left turn at p on a counter-clockwise loop: a convex corner
    convex_r = np.roll(convex_p, -1)
    ph = 2 * np.pi * np.arange(nphi + 1) / nphi
    c, s = np.cos(ph), np.sin(ph)

    def pt(a):                                        # (edges, nphi + 1, 3)
        return np.stack([np.outer(a[:, 0], c), np.outer(a[:, 0], s), np.repeat(a[:, 1:2], len(ph), 1)], -1)
    P, Rr = pt(p), pt(r)
    N = np.stack([np.outer(nR, c), np.outer(nR, s), np.repeat(nZ[:, None], len(ph), 1)], -1)
    a0, a1 = P[:, :-1], P[:, 1:]
    b0, b1 = Rr[:, :-1], Rr[:, 1:]
    n0, n1 = N[:, :-1], N[:, 1:]
    V = np.concatenate([np.stack([a0, b0, b1], 2), np.stack([a0, b1, a1], 2)]).reshape(-1, 3, 3)
    Nn = np.concatenate([np.stack([n0, n0, n1], 2), np.stack([n0, n1, n1], 2)]).reshape(-1, 3, 3)
    Le = np.repeat(L[:, None], nphi, 1)
    zero, full = np.zeros_like(Le), Le
    dp = np.concatenate([np.stack([zero, full, full], 2), np.stack([zero, full, zero], 2)]).reshape(-1, 3)
    dr = np.concatenate([np.stack([full, zero, zero], 2), np.stack([full, zero, full], 2)]).reshape(-1, 3)
    cvp = np.tile(np.repeat(convex_p[:, None], nphi, 1).reshape(-1), 2)
    cvr = np.tile(np.repeat(convex_r[:, None], nphi, 1).reshape(-1), 2)
    return V, Nn, dp, dr, np.stack([cvp, cvr], -1)


def section_mesh(X):
    """A closed triangle mesh through points X (nphi, ntheta, 3) on a torus: rows round the torus, columns round
    each section, both periodic."""
    i1 = np.roll(X, -1, axis=1)
    j0 = np.roll(X, -1, axis=0)
    j1 = np.roll(j0, -1, axis=1)
    A = np.stack([X, i1, j1], 2).reshape(-1, 3, 3)
    B = np.stack([X, j1, j0], 2).reshape(-1, 3, 3)
    return np.concatenate([A, B])


def tube_mesh(X):
    """A closed triangle mesh through points X (nphi, ntheta, 3) on a torus opened at its first and last sections:
    rows along it, columns round each section (periodic), each end closed by a fan from its centroid."""
    i1 = np.roll(X, -1, axis=1)
    j0, j1 = X[1:], i1[1:]
    A = np.stack([X[:-1], i1[:-1], j1], 2).reshape(-1, 3, 3)
    B = np.stack([X[:-1], j1, j0], 2).reshape(-1, 3, 3)
    caps = []
    for S in (X[0], X[-1]):
        c = np.broadcast_to(S.mean(axis=0), S.shape)
        caps.append(np.stack([c, S, np.roll(S, -1, axis=0)], 1))
    return np.concatenate([A, B] + caps)


def section_samples(R, Z, phi, shells):
    """Points filling a torus given by its sections R, Z (nphi, ntheta) at angles phi, with weights summing to its
    volume (m^3): each section's points pulled towards its centroid on ``shells`` shells.  The sources of the
    light the plasma casts on what is round it (decoration)."""
    out, w = [], []
    nph, nth = R.shape
    for j in range(nph):
        r, z = R[j], Z[j]
        area = 0.5 * abs(np.sum(r * np.roll(z, -1) - np.roll(r, -1) * z))
        rc, zc = r.mean(), z.mean()
        dvol = area * rc * 2 * np.pi / nph
        for k in range(shells):
            sk = (k + 0.5) / shells
            rr, zz = rc + sk * (r - rc), zc + sk * (z - zc)
            out.append(np.stack([rr * np.cos(phi[j]), rr * np.sin(phi[j]), zz], -1))
            w.append(np.full(nth, dvol * 2 * sk / (shells * nth)))
    X = np.concatenate(out)
    return np.concatenate([X, np.concatenate(w)[:, None]], 1)


@njit(parallel=True, cache=True)
def light_moments(X, E, reach):
    """The plasma's light at points X (n, 3): S = sum w / d^2 and the light vector L = sum w u / d^2 over the glow's
    points E (m, 4: position, weight), u the unit vector towards each.  The irradiance on a surface of normal n is
    then about max(0, (S + n.L) / 2).  Points nearer than reach count as reach away (the glow is a body)."""
    out = np.zeros((X.shape[0], 4))
    r2min = reach * reach
    for i in prange(X.shape[0]):
        s0 = 0.0
        lx = 0.0
        ly = 0.0
        lz = 0.0
        for j in range(E.shape[0]):
            dx = E[j, 0] - X[i, 0]
            dy = E[j, 1] - X[i, 1]
            dz = E[j, 2] - X[i, 2]
            d2 = dx * dx + dy * dy + dz * dz
            if d2 < r2min:
                d2 = r2min
            d = math.sqrt(d2)
            f = E[j, 3] / d2
            s0 += f
            lx += f * dx / d
            ly += f * dy / d
            lz += f * dz / d
        out[i, 0] = s0
        out[i, 1] = lx
        out[i, 2] = ly
        out[i, 3] = lz
    return out


# ============================================================================ the hierarchy of boxes
@njit(cache=True)
def build_bvh(pmin, pmax, pcen, leaf_size, nbins):
    """Binned surface-area-heuristic bounding volume hierarchy. The children of node k are left[k] and left[k] + 1."""
    n = pmin.shape[0]
    order = np.arange(n).astype(np.int32)
    maxn = 2 * n + 1
    bmin = np.empty((maxn, 3), np.float32)
    bmax = np.empty((maxn, 3), np.float32)
    left = np.full(maxn, -1, np.int32)
    start = np.zeros(maxn, np.int32)
    count = np.zeros(maxn, np.int32)
    st_node = np.empty(512, np.int64)
    st_s = np.empty(512, np.int64)
    st_e = np.empty(512, np.int64)
    st_node[0] = 0
    st_s[0] = 0
    st_e[0] = n
    sp = 1
    nn = 1
    cnt_b = np.zeros(nbins, np.int64)
    bb_min = np.empty((nbins, 3))
    bb_max = np.empty((nbins, 3))
    rc = np.zeros(nbins, np.int64)
    ra = np.zeros(nbins)
    while sp > 0:
        sp -= 1
        node = st_node[sp]
        s = st_s[sp]
        e = st_e[sp]
        lo = np.full(3, INF)
        hi = np.full(3, -INF)
        clo = np.full(3, INF)
        chi = np.full(3, -INF)
        for i in range(s, e):
            p = order[i]
            for a in range(3):
                lo[a] = min(lo[a], pmin[p, a])
                hi[a] = max(hi[a], pmax[p, a])
                clo[a] = min(clo[a], pcen[p, a])
                chi[a] = max(chi[a], pcen[p, a])
        for a in range(3):
            bmin[node, a] = lo[a]
            bmax[node, a] = hi[a]
        cnt = e - s
        axis = 0
        for a in range(1, 3):
            if chi[a] - clo[a] > chi[axis] - clo[axis]:
                axis = a
        ext = chi[axis] - clo[axis]
        cl = clo[axis]
        if cnt <= leaf_size or ext < 1e-9:
            start[node] = s
            count[node] = cnt
            continue
        for b in range(nbins):
            cnt_b[b] = 0
            for a in range(3):
                bb_min[b, a] = INF
                bb_max[b, a] = -INF
        scale = nbins / ext
        for i in range(s, e):
            p = order[i]
            b = min(nbins - 1, max(0, int((pcen[p, axis] - cl) * scale)))
            cnt_b[b] += 1
            for a in range(3):
                bb_min[b, a] = min(bb_min[b, a], pmin[p, a])
                bb_max[b, a] = max(bb_max[b, a], pmax[p, a])
        r0 = np.full(3, INF)
        r1 = np.full(3, -INF)
        rcount = 0
        for b in range(nbins - 1, 0, -1):
            rcount += cnt_b[b]
            if cnt_b[b] > 0:
                for a in range(3):
                    r0[a] = min(r0[a], bb_min[b, a])
                    r1[a] = max(r1[a], bb_max[b, a])
            rc[b] = rcount
            if rcount > 0:
                dx = r1[0] - r0[0]
                dy = r1[1] - r0[1]
                dz = r1[2] - r0[2]
                ra[b] = dx * dy + dy * dz + dz * dx
            else:
                ra[b] = 0.0
        l0 = np.full(3, INF)
        l1 = np.full(3, -INF)
        lcount = 0
        best = INF
        bestk = -1
        for b in range(0, nbins - 1):
            lcount += cnt_b[b]
            if cnt_b[b] > 0:
                for a in range(3):
                    l0[a] = min(l0[a], bb_min[b, a])
                    l1[a] = max(l1[a], bb_max[b, a])
            if lcount == 0 or rc[b + 1] == 0:
                continue
            dx = l1[0] - l0[0]
            dy = l1[1] - l0[1]
            dz = l1[2] - l0[2]
            cost = (dx * dy + dy * dz + dz * dx) * lcount + ra[b + 1] * rc[b + 1]
            if cost < best:
                best = cost
                bestk = b
        if bestk < 0:
            mid = (s + e) // 2
        else:
            i = s
            j = e - 1
            while i <= j:
                p = order[i]
                b = min(nbins - 1, int((pcen[p, axis] - cl) * scale))
                if b <= bestk:
                    i += 1
                else:
                    tmp = order[i]
                    order[i] = order[j]
                    order[j] = tmp
                    j -= 1
            mid = i
            if mid == s or mid == e:
                mid = (s + e) // 2
        lc = nn
        nn += 2
        left[node] = lc
        st_node[sp] = lc + 1
        st_s[sp] = mid
        st_e[sp] = e
        sp += 1
        st_node[sp] = lc
        st_s[sp] = s
        st_e[sp] = mid
        sp += 1
    return bmin[:nn].copy(), bmax[:nn].copy(), left[:nn].copy(), start[:nn].copy(), count[:nn].copy(), order


def bvh_of(pmin, pmax):
    if len(pmin) == 0:
        z = np.zeros((1, 3))
        return build_bvh(z + INF, z + INF, z + INF, 4, 16)
    return build_bvh(np.asarray(pmin, float), np.asarray(pmax, float), 0.5 * (pmin + pmax), 4, 16)


@njit(cache=True, fastmath=True, inline="always")
def box_entry(bmin, bmax, node, ox, oy, oz, ix, iy, iz, tmax):
    t1 = (bmin[node, 0] - ox) * ix
    t2 = (bmax[node, 0] - ox) * ix
    tn = min(t1, t2)
    tf = max(t1, t2)
    t1 = (bmin[node, 1] - oy) * iy
    t2 = (bmax[node, 1] - oy) * iy
    tn = max(tn, min(t1, t2))
    tf = min(tf, max(t1, t2))
    t1 = (bmin[node, 2] - oz) * iz
    t2 = (bmax[node, 2] - oz) * iz
    tn = max(tn, min(t1, t2))
    tf = min(tf, max(t1, t2))
    if tf >= max(tn, 0.0) and tn < tmax:
        return tn
    return INF


@njit(cache=True, fastmath=True, inline="always")
def hit_tri(V0, E1, E2, p, ox, oy, oz, dx, dy, dz):
    """Moller-Trumbore: (t, u, v), or (INF, 0, 0)."""
    e1x = E1[p, 0]
    e1y = E1[p, 1]
    e1z = E1[p, 2]
    e2x = E2[p, 0]
    e2y = E2[p, 1]
    e2z = E2[p, 2]
    px = dy * e2z - dz * e2y
    py = dz * e2x - dx * e2z
    pz = dx * e2y - dy * e2x
    det = e1x * px + e1y * py + e1z * pz
    if abs(det) < 1e-18:
        return INF, 0.0, 0.0
    inv = 1.0 / det
    tx = ox - V0[p, 0]
    ty = oy - V0[p, 1]
    tz = oz - V0[p, 2]
    u = (tx * px + ty * py + tz * pz) * inv
    if u < 0.0 or u > 1.0:
        return INF, 0.0, 0.0
    qx = ty * e1z - tz * e1y
    qy = tz * e1x - tx * e1z
    qz = tx * e1y - ty * e1x
    v = (dx * qx + dy * qy + dz * qz) * inv
    if v < 0.0 or u + v > 1.0:
        return INF, 0.0, 0.0
    t = (e2x * qx + e2y * qy + e2z * qz) * inv
    if t <= 1e-7:
        return INF, 0.0, 0.0
    return t, u, v


@njit(cache=True, fastmath=True, inline="always")
def hit_cap(CA, CB, CR, c, ox, oy, oz, dx, dy, dz):
    """A ray against a capsule (the segment AB swept by a sphere of radius r): the nearest t, or INF."""
    ax = CA[c, 0]
    ay = CA[c, 1]
    az = CA[c, 2]
    bax = CB[c, 0] - ax
    bay = CB[c, 1] - ay
    baz = CB[c, 2] - az
    oax = ox - ax
    oay = oy - ay
    oaz = oz - az
    ra = CR[c]
    baba = bax * bax + bay * bay + baz * baz
    bard = bax * dx + bay * dy + baz * dz
    baoa = bax * oax + bay * oay + baz * oaz
    rdoa = dx * oax + dy * oay + dz * oaz
    oaoa = oax * oax + oay * oay + oaz * oaz
    a = baba - bard * bard
    b = baba * rdoa - baoa * bard
    cc = baba * oaoa - baoa * baoa - ra * ra * baba
    h = b * b - a * cc
    if h >= 0.0 and a > 1e-20:
        t = (-b - math.sqrt(h)) / a
        y = baoa + t * bard
        if y > 0.0 and y < baba:
            if t > 1e-7:
                return t
            return INF
        if y <= 0.0:
            ocx = oax
            ocy = oay
            ocz = oaz
        else:
            ocx = ox - CB[c, 0]
            ocy = oy - CB[c, 1]
            ocz = oz - CB[c, 2]
        b2 = dx * ocx + dy * ocy + dz * ocz
        c2 = ocx * ocx + ocy * ocy + ocz * ocz - ra * ra
        h2 = b2 * b2 - c2
        if h2 > 0.0:
            t = -b2 - math.sqrt(h2)
            if t > 1e-7:
                return t
    return INF


@njit(cache=True, fastmath=True)
def trace(ox, oy, oz, dx, dy, dz, tmax, anyhit, bmin, bmax, left, start, count, order,
          ntri, V0, E1, E2, CA, CB, CR, stack):
    """The nearest opaque object along a ray: (t, primitive, u, v). Primitives below ntri are triangles, the rest
    capsules."""
    ix = 1.0 / dx if abs(dx) > 1e-20 else 1e20
    iy = 1.0 / dy if abs(dy) > 1e-20 else 1e20
    iz = 1.0 / dz if abs(dz) > 1e-20 else 1e20
    best = tmax
    bp = -1
    bu = 0.0
    bv = 0.0
    if box_entry(bmin, bmax, 0, ox, oy, oz, ix, iy, iz, best) >= INF:
        return best, bp, bu, bv
    stack[0] = 0
    sp = 1
    while sp > 0:
        sp -= 1
        node = stack[sp]
        if count[node] > 0:
            s = start[node]
            for k in range(s, s + count[node]):
                p = order[k]
                if p < ntri:
                    t, u, v = hit_tri(V0, E1, E2, p, ox, oy, oz, dx, dy, dz)
                    if t < best:
                        best = t
                        bp = p
                        bu = u
                        bv = v
                        if anyhit:
                            return best, bp, bu, bv
                else:
                    t = hit_cap(CA, CB, CR, p - ntri, ox, oy, oz, dx, dy, dz)
                    if t < best:
                        best = t
                        bp = p
                        if anyhit:
                            return best, bp, bu, bv
        else:
            lc = left[node]
            r = lc + 1
            tl = box_entry(bmin, bmax, lc, ox, oy, oz, ix, iy, iz, best)
            tr = box_entry(bmin, bmax, r, ox, oy, oz, ix, iy, iz, best)
            if tl < tr:
                if tr < INF:
                    stack[sp] = r
                    sp += 1
                if tl < INF:
                    stack[sp] = lc
                    sp += 1
            else:
                if tl < INF:
                    stack[sp] = lc
                    sp += 1
                if tr < INF:
                    stack[sp] = r
                    sp += 1
    return best, bp, bu, bv


@njit(cache=True, fastmath=True)
def glow_trace(ox, oy, oz, dx, dy, dz, tmax, bmin, bmax, left, start, count, order, GA, GB, GW, GC, crad, csig,
               chalo, cap, reach, gout, stack):
    """The light a ray gathers from the fibres before it meets an opaque object at tmax, added to gout per channel.
    Each fibre is a chain of straight segments with a sharp core (a round cylinder of the channel's radius: the
    chord over the diameter) and, where the channel has one, a soft halo (a Gaussian of the channel's width round
    the same line, integrated along the ray).  A glancing ray gathers more, up to cap times."""
    ix = 1.0 / dx if abs(dx) > 1e-20 else 1e20
    iy = 1.0 / dy if abs(dy) > 1e-20 else 1e20
    iz = 1.0 / dz if abs(dz) > 1e-20 else 1e20
    if box_entry(bmin, bmax, 0, ox, oy, oz, ix, iy, iz, tmax) >= INF:
        return
    stack[0] = 0
    sp = 1
    reach2 = reach * reach
    while sp > 0:
        sp -= 1
        node = stack[sp]
        if count[node] > 0:
            s = start[node]
            for k in range(s, s + count[node]):
                p = order[k]
                ax = GA[p, 0]
                ay = GA[p, 1]
                az = GA[p, 2]
                bx = GB[p, 0] - ax
                by = GB[p, 1] - ay
                bz = GB[p, 2] - az
                L = math.sqrt(bx * bx + by * by + bz * bz)
                if L < 1e-12:
                    continue
                ux = bx / L
                uy = by / L
                uz = bz / L
                qx = ox - ax
                qy = oy - ay
                qz = oz - az
                du = dx * ux + dy * uy + dz * uz
                qu = qx * ux + qy * uy + qz * uz
                pdx = dx - du * ux
                pdy = dy - du * uy
                pdz = dz - du * uz
                pqx = qx - qu * ux
                pqy = qy - qu * uy
                pqz = qz - qu * uz
                A2 = pdx * pdx + pdy * pdy + pdz * pdz
                if A2 < 1e-14:
                    continue
                B2 = pqx * pdx + pqy * pdy + pqz * pdz
                C0 = pqx * pqx + pqy * pqy + pqz * pqz
                b2 = C0 - B2 * B2 / A2
                if b2 >= reach2:
                    continue
                if abs(du) > 1e-12:
                    s1 = -qu / du
                    s2 = (L - qu) / du
                    if s1 > s2:
                        tmp = s1
                        s1 = s2
                        s2 = tmp
                elif qu < 0.0 or qu > L:
                    continue
                else:
                    s1 = -1e30
                    s2 = 1e30
                lo = max(s1, 0.0)
                hi = min(s2, tmax)
                if hi <= lo:
                    continue
                ch = GC[p]
                rr = crad[ch]
                r2 = rr * rr
                if b2 < r2:
                    sq = math.sqrt((r2 - b2) / A2)
                    tc = -B2 / A2
                    t1 = max(tc - sq, lo)
                    t2 = min(tc + sq, hi)
                    if t2 > t1:
                        gout[ch] += GW[p] * min((t2 - t1) / (2.0 * rr), cap)
                hw = chalo[ch]
                if hw > 0.0:
                    sg = csig[ch]
                    sA = math.sqrt(A2)
                    tc = -B2 / A2
                    kk = sA / (sg * 1.4142135623730951)
                    e = math.exp(-b2 / (2.0 * sg * sg))
                    gout[ch] += GW[p] * hw * min(e * 0.5 / sA * (math.erf((hi - tc) * kk) - math.erf((lo - tc) * kk)),
                                                 cap)
        else:
            lc = left[node]
            if box_entry(bmin, bmax, lc, ox, oy, oz, ix, iy, iz, tmax) < INF:
                stack[sp] = lc
                sp += 1
            if box_entry(bmin, bmax, lc + 1, ox, oy, oz, ix, iy, iz, tmax) < INF:
                stack[sp] = lc + 1
                sp += 1


@njit(cache=True, fastmath=True)
def volume_trace(ox, oy, oz, dx, dy, dz, tmax, bmin, bmax, left, start, count, order, V0, E1, E2, stack, tb):
    """The length of the ray inside the closed mesh before tmax (every crossing found, sorted, paired)."""
    ix = 1.0 / dx if abs(dx) > 1e-20 else 1e20
    iy = 1.0 / dy if abs(dy) > 1e-20 else 1e20
    iz = 1.0 / dz if abs(dz) > 1e-20 else 1e20
    if V0.shape[0] == 0 or box_entry(bmin, bmax, 0, ox, oy, oz, ix, iy, iz, INF) >= INF:
        return 0.0
    n = 0
    nmax = tb.shape[0]
    stack[0] = 0
    sp = 1
    while sp > 0:
        sp -= 1
        node = stack[sp]
        if count[node] > 0:
            s = start[node]
            for k in range(s, s + count[node]):
                p = order[k]
                t, u, v = hit_tri(V0, E1, E2, p, ox, oy, oz, dx, dy, dz)
                if t < INF and n < nmax:
                    tb[n] = t
                    n += 1
        else:
            lc = left[node]
            if box_entry(bmin, bmax, lc, ox, oy, oz, ix, iy, iz, INF) < INF:
                stack[sp] = lc
                sp += 1
            if box_entry(bmin, bmax, lc + 1, ox, oy, oz, ix, iy, iz, INF) < INF:
                stack[sp] = lc + 1
                sp += 1
    if n < 2:
        return 0.0
    for i in range(1, n):                          # an insertion sort: a ray crosses a torus a few times
        x = tb[i]
        j = i - 1
        while j >= 0 and tb[j] > x:
            tb[j + 1] = tb[j]
            j -= 1
        tb[j + 1] = x
    m = 0
    for i in range(n):                             # a crossing on an edge two triangles share counts once
        if m == 0 or tb[i] - tb[m - 1] > 1e-6:
            tb[m] = tb[i]
            m += 1
    total = 0.0
    for i in range(0, m - 1, 2):
        a = tb[i]
        b = tb[i + 1]
        if a >= tmax:
            break
        total += min(b, tmax) - a
    return total


@njit(cache=True, inline="always")
def rng_next(state):
    """xorshift64*, uniform in [0, 1)."""
    x = state[0]
    x ^= x >> np.uint64(12)
    x ^= x << np.uint64(25)
    x ^= x >> np.uint64(27)
    state[0] = x
    r = x * np.uint64(2685821657736338717)
    return (r >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True, inline="always")
def splitmix(x):
    x = x + np.uint64(0x9E3779B97F4A7C15)
    z = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@njit(cache=True, inline="always")
def onb(nx, ny, nz):
    sign = 1.0 if nz >= 0.0 else -1.0
    a = -1.0 / (sign + nz)
    b = nx * ny * a
    return 1.0 + sign * nx * nx * a, sign * b, -sign * nx, b, sign + ny * ny * a, -ny


# ============================================================================ decoration (it encodes nothing)
@njit(cache=True, inline="always")
def hash3(ix, iy, iz, salt):
    h = np.uint64(ix & 0xFFFFFFFF) * np.uint64(0x9E3779B1) ^ np.uint64(iy & 0xFFFFFFFF) * np.uint64(0x85EBCA77) \
        ^ np.uint64(iz & 0xFFFFFFFF) * np.uint64(0xC2B2AE3D) ^ np.uint64(salt) * np.uint64(0x27D4EB2F165667C5)
    h = splitmix(h)
    return (h >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True, fastmath=True)
def vnoise(x, y, z, salt):
    """Value noise in [0, 1], smooth."""
    fx = math.floor(x)
    fy = math.floor(y)
    fz = math.floor(z)
    ix = int(fx)
    iy = int(fy)
    iz = int(fz)
    tx = x - fx
    ty = y - fy
    tz = z - fz
    tx = tx * tx * (3.0 - 2.0 * tx)
    ty = ty * ty * (3.0 - 2.0 * ty)
    tz = tz * tz * (3.0 - 2.0 * tz)
    c000 = hash3(ix, iy, iz, salt)
    c100 = hash3(ix + 1, iy, iz, salt)
    c010 = hash3(ix, iy + 1, iz, salt)
    c110 = hash3(ix + 1, iy + 1, iz, salt)
    c001 = hash3(ix, iy, iz + 1, salt)
    c101 = hash3(ix + 1, iy, iz + 1, salt)
    c011 = hash3(ix, iy + 1, iz + 1, salt)
    c111 = hash3(ix + 1, iy + 1, iz + 1, salt)
    a = c000 + (c100 - c000) * tx
    b = c010 + (c110 - c010) * tx
    c = c001 + (c101 - c001) * tx
    d = c011 + (c111 - c011) * tx
    e = a + (b - a) * ty
    f = c + (d - c) * ty
    return e + (f - e) * tz


@njit(cache=True, fastmath=True)
def fbm(x, y, z, salt, octaves):
    tot = 0.0
    amp = 0.5
    norm = 0.0
    for o in range(octaves):
        tot += amp * vnoise(x, y, z, salt + o * 131)
        norm += amp
        x = x * 2.03 + 17.1
        y = y * 2.03 + 3.7
        z = z * 2.03 + 9.3
        amp *= 0.5
    return tot / norm


@njit(cache=True, fastmath=True)
def scratch_mask(x, y, z, nx, ny, nz, cell, prob, width, lmin, lmax, salt):
    """A few hair-thin scratches: in a cell of a lattice, with probability prob, a short straight cut through the
    surface (a thin slab at right angles to it), tapered at both ends.  0 to 1."""
    cx = int(math.floor(x / cell))
    cy = int(math.floor(y / cell))
    cz = int(math.floor(z / cell))
    best = 0.0
    for i in range(cx - 1, cx + 2):
        for j in range(cy - 1, cy + 2):
            for k in range(cz - 1, cz + 2):
                if hash3(i, j, k, salt) >= prob:
                    continue
                px = (i + hash3(i, j, k, salt + 1)) * cell
                py = (j + hash3(i, j, k, salt + 2)) * cell
                pz = (k + hash3(i, j, k, salt + 3)) * cell
                L = lmin + (lmax - lmin) * hash3(i, j, k, salt + 4)
                rx = x - px
                ry = y - py
                rz = z - pz
                d2 = rx * rx + ry * ry + rz * rz
                if d2 > 0.25 * L * L:
                    continue
                # a random direction made tangent to the surface: the slab's normal lies in the surface, at right
                # angles to the cut
                ax = hash3(i, j, k, salt + 5) * 2.0 - 1.0
                ay = hash3(i, j, k, salt + 6) * 2.0 - 1.0
                az = hash3(i, j, k, salt + 7) * 2.0 - 1.0
                an = ax * nx + ay * ny + az * nz
                ax -= an * nx
                ay -= an * ny
                az -= an * nz
                al = math.sqrt(ax * ax + ay * ay + az * az)
                if al < 1e-6:
                    continue
                ax /= al
                ay /= al
                az /= al
                sx = ny * az - nz * ay
                sy = nz * ax - nx * az
                sz = nx * ay - ny * ax
                dist = abs(rx * sx + ry * sy + rz * sz)
                w = width * (0.6 + 0.8 * hash3(i, j, k, salt + 8))
                if dist > 2.0 * w:
                    continue
                along = abs(rx * ax + ry * ay + rz * az) / (0.5 * L)
                taper = 1.0 - along * along
                if taper <= 0.0:
                    continue
                v = math.exp(-(dist / w) * (dist / w)) * taper * (0.5 + 0.5 * hash3(i, j, k, salt + 9))
                if v > best:
                    best = v
    return best


# ============================================================================ the path tracer
@njit(cache=True, fastmath=True, parallel=True)
def render_kernel(W, H, spp, min_spp, seed, cam, bvh, prims, mats, light, zf, maxdepth, row0, row1, col0, col1,
                  gbvh, GA, GB, GW, GC, crad, csig, chalo, gcap, greach, nch, vbvh, VV0, VE1, VE2, wear, n_glow, n_ie,
                  fx0, fy0, fh, fmap):
    """Path trace the pixels [row0, row1) x [col0, col1). Per-pixel sums (divide by the samples taken):
    acc (nmat, r, c, 12): the light at a first hit on material m, demodulated by its albedo (0:3), and the same
        light weighted by the decoration's masks: worn edge (3:6), tarnish (6:9), scratch (9:12)
    sacc (r, c, 6): the light of the sheen (untinted), and the same weighted by the worn edge
    facc (r, c, 3): the floor's light over its unshadowed value, the plasma's light on the floor, the floor's share
    gsum (r, c, nch + 1): the fibres of each channel and the length inside the plasma, seen by the first n_glow
        samples
    ie (r, c): the plasma's light on the first solid a ray meets, demodulated by its albedo
    n (r, c): the samples taken."""
    bmin, bmax, left, start, count, order = bvh
    gbmin, gbmax, gleft, gstart, gcount, gorder = gbvh
    vbmin, vbmax, vleft, vstart, vcount, vorder = vbvh
    ntri, V0, E1, E2, N0, N1, N2, TM, TDP, TDR, TCV, CA, CB, CR, CM, TLM, CLM = prims
    fny, fnx = fmap.shape
    (cx, cy, cz, fx, fy, fz, rx, ry, rz, ux, uy, uz, oscale) = cam
    (kx, ky, kz, cosa, ekr, ekg, ekb, skr, skg, skb, efr, efg, efb, l0r, l0g, l0b) = light
    (edge_w, tarn_f, tarn_lo, tarn_hi, tarn_cav, cav_r, sc_cell, sc_prob, sc_w, sc_lmin, sc_lmax, tube_wear_f,
     tube_wear_lo, tube_wear_hi) = wear
    nmat = mats.shape[0]
    nrow = row1 - row0
    ncol = col1 - col0
    acc = np.zeros((nmat, nrow, ncol, 12), np.float32)
    sacc = np.zeros((nrow, ncol, 6), np.float32)
    facc = np.zeros((nrow, ncol, 3), np.float32)
    gsum = np.zeros((nrow, ncol, nch + 1), np.float32)
    ie = np.zeros((nrow, ncol), np.float32)
    nsm = np.zeros((nrow, ncol), np.float32)
    aspect = W / H
    ktx, kty, ktz, kbx, kby, kbz = onb(kx, ky, kz)
    per8 = (nrow + 7) // 8
    for jq in prange(8 * per8):
        # rows interleaved across the threads (a transpose of 8 interleaved runs), so that the rows crowded with
        # objects are shared out, every row once
        jj = (jq % 8) * per8 + jq // 8
        if jj >= nrow:
            continue
        j = row0 + jj
        stack = np.empty(128, np.int64)
        gstack = np.empty(256, np.int64)
        vstack = np.empty(256, np.int64)
        tb = np.empty(64, np.float64)
        gout = np.zeros(nch, np.float64)
        state = np.empty(1, np.uint64)
        for ii in range(ncol):
            i = col0 + ii
            state[0] = splitmix(np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15) ^ np.uint64(j * W + i + 1))
            if state[0] == np.uint64(0):
                state[0] = np.uint64(1)
            nfl1 = 0
            ns = 0
            for s in range(spp):
                if s == min_spp and nfl1 == min_spp:
                    break       # every sample so far met the bare, unshadowed floor: the pixel is the page
                ns += 1
                px = (i + rng_next(state)) / W * 2.0 - 1.0
                py = 1.0 - (j + rng_next(state)) / H * 2.0
                ox = cx + (px * aspect * rx + py * ux) * oscale
                oy = cy + (px * aspect * ry + py * uy) * oscale
                oz = cz + (px * aspect * rz + py * uz) * oscale
                dx = fx
                dy = fy
                dz = fz
                tr = 1.0
                tg = 1.0
                tb_ = 1.0
                Lr = 0.0
                Lg = 0.0
                Lb = 0.0
                first_kind = -1
                first_mat = 0
                first_spec = False
                a0r = 1.0
                a0g = 1.0
                a0b = 1.0
                m_edge = 0.0
                m_tarn = 0.0
                m_scr = 0.0
                plasma_in = 0.0
                for depth in range(maxdepth):
                    t, p, u, v = trace(ox, oy, oz, dx, dy, dz, INF, False, bmin, bmax, left, start, count, order,
                                       ntri, V0, E1, E2, CA, CB, CR, stack)
                    tfl = INF
                    if dz < -1e-12:
                        tfl = (zf - oz) / dz
                    tend = min(t, tfl)
                    if depth == 0 and s < n_glow:
                        for c_ in range(nch):
                            gout[c_] = 0.0
                        glow_trace(ox, oy, oz, dx, dy, dz, tend, gbmin, gbmax, gleft, gstart, gcount, gorder, GA,
                                   GB, GW, GC, crad, csig, chalo, gcap, greach, gout, gstack)
                        for c_ in range(nch):
                            gsum[jj, ii, c_] += gout[c_]
                        gsum[jj, ii, nch] += volume_trace(ox, oy, oz, dx, dy, dz, tend, vbmin, vbmax, vleft, vstart,
                                                          vcount, vorder, VV0, VE1, VE2, vstack, tb)
                    if depth == 1 and first_kind == 0 and p >= 0 and t < cav_r:
                        m_tarn = min(1.0, m_tarn + tarn_cav)     # grime gathers where the metal faces metal
                    if p < 0 and tfl >= INF:
                        if dz > 0.0:
                            Lr += tr * skr
                            Lg += tg * skg
                            Lb += tb_ * skb
                        break
                    isfloor = tfl < t
                    if isfloor:
                        t = tfl
                    hx = ox + t * dx
                    hy = oy + t * dy
                    hz = oz + t * dz
                    is_tube = False
                    if isfloor:
                        nx = 0.0
                        ny = 0.0
                        nz = 1.0
                        mat = nmat - 1
                    elif p < ntri:
                        w = 1.0 - u - v
                        nx = w * N0[p, 0] + u * N1[p, 0] + v * N2[p, 0]
                        ny = w * N0[p, 1] + u * N1[p, 1] + v * N2[p, 1]
                        nz = w * N0[p, 2] + u * N1[p, 2] + v * N2[p, 2]
                        gx = E1[p, 1] * E2[p, 2] - E1[p, 2] * E2[p, 1]
                        gy = E1[p, 2] * E2[p, 0] - E1[p, 0] * E2[p, 2]
                        gz = E1[p, 0] * E2[p, 1] - E1[p, 1] * E2[p, 0]
                        if gx * dx + gy * dy + gz * dz > 0.0:
                            gx = -gx
                            gy = -gy
                            gz = -gz
                        if nx * gx + ny * gy + nz * gz < 0.0:
                            nx = -nx
                            ny = -ny
                            nz = -nz
                        nn = 1.0 / math.sqrt(nx * nx + ny * ny + nz * nz)
                        nx *= nn
                        ny *= nn
                        nz *= nn
                        mat = TM[p]
                        if depth == 0:
                            # worn edges: near a convex corner of the swept cross-section
                            dpp = w * TDP[p, 0] + u * TDP[p, 1] + v * TDP[p, 2]
                            drr = w * TDR[p, 0] + u * TDR[p, 1] + v * TDR[p, 2]
                            e1 = TCV[p, 0] * max(0.0, 1.0 - dpp / edge_w)
                            e2 = TCV[p, 1] * max(0.0, 1.0 - drr / edge_w)
                            ee = max(e1, e2)
                            m_edge = ee * ee * (3.0 - 2.0 * ee)
                    else:
                        c = p - ntri
                        ax = CA[c, 0]
                        ay = CA[c, 1]
                        az = CA[c, 2]
                        bax = CB[c, 0] - ax
                        bay = CB[c, 1] - ay
                        baz = CB[c, 2] - az
                        pax = hx - ax
                        pay = hy - ay
                        paz = hz - az
                        hh = (pax * bax + pay * bay + paz * baz) / (bax * bax + bay * bay + baz * baz + 1e-30)
                        hh = min(1.0, max(0.0, hh))
                        nx = pax - hh * bax
                        ny = pay - hh * bay
                        nz = paz - hh * baz
                        nn = 1.0 / math.sqrt(nx * nx + ny * ny + nz * nz)
                        nx *= nn
                        ny *= nn
                        nz *= nn
                        if nx * dx + ny * dy + nz * dz > 0.0:
                            nx = -nx
                            ny = -ny
                            nz = -nz
                        mat = CM[c]
                        is_tube = True
                    ar = mats[mat, 0]
                    ag = mats[mat, 1]
                    ab = mats[mat, 2]
                    if depth == 0 and s < n_ie:
                        # the light of the plasma on this point (its glow's moments: smooth, no shadows)
                        if isfloor:
                            gxf = (hx - fx0) / fh
                            gyf = (hy - fy0) / fh
                            ix0 = int(math.floor(gxf))
                            iy0 = int(math.floor(gyf))
                            if ix0 >= 0 and iy0 >= 0 and ix0 < fnx - 1 and iy0 < fny - 1:
                                tx = gxf - ix0
                                ty = gyf - iy0
                                plasma_in = (fmap[iy0, ix0] * (1 - tx) * (1 - ty) + fmap[iy0, ix0 + 1] * tx * (1 - ty)
                                             + fmap[iy0 + 1, ix0] * (1 - tx) * ty + fmap[iy0 + 1, ix0 + 1] * tx * ty)
                        elif p < ntri:
                            plasma_in = max(0.0, 0.5 * (TLM[p, 0] + nx * TLM[p, 1] + ny * TLM[p, 2] + nz * TLM[p, 3]))
                        else:
                            cc_ = p - ntri
                            plasma_in = max(0.0, 0.5 * (CLM[cc_, 0] + nx * CLM[cc_, 1] + ny * CLM[cc_, 2]
                                                        + nz * CLM[cc_, 3]))
                    if depth == 0:
                        if isfloor:
                            first_kind = 1
                        else:
                            first_kind = 0
                            first_mat = mat
                            a0r = ar
                            a0g = ag
                            a0b = ab
                            # tarnish: faint blotches. On tubes, rubbed bright patches. Scratches
                            n1 = fbm(hx * tarn_f, hy * tarn_f, hz * tarn_f, 11, 3)
                            tt = (n1 - tarn_lo) / (tarn_hi - tarn_lo)
                            tt = min(1.0, max(0.0, tt))
                            m_tarn = min(1.0, m_tarn + tt * tt * (3.0 - 2.0 * tt))
                            if is_tube:
                                n2 = fbm(hx * tube_wear_f, hy * tube_wear_f, hz * tube_wear_f, 29, 2)
                                tw = (n2 - tube_wear_lo) / (tube_wear_hi - tube_wear_lo)
                                tw = min(1.0, max(0.0, tw))
                                m_edge = max(m_edge, 0.5 * tw * tw * (3.0 - 2.0 * tw))
                            m_scr = scratch_mask(hx, hy, hz, nx, ny, nz, sc_cell, sc_prob, sc_w, sc_lmin, sc_lmax, 41)
                    eps = 2e-5
                    sx = hx + nx * eps
                    sy = hy + ny * eps
                    sz = hz + nz * eps
                    coat0 = mats[mat, 3]
                    if (not isfloor) and coat0 > 0.0:
                        cth = max(0.0, -(nx * dx + ny * dy + nz * dz))
                        om = 1.0 - cth
                        F = coat0 + (1.0 - coat0) * om * om * om * om * om
                        if rng_next(state) < F:
                            if depth == 0:
                                first_spec = True
                            rdx = dx + 2.0 * cth * nx
                            rdy = dy + 2.0 * cth * ny
                            rdz = dz + 2.0 * cth * nz
                            rough = mats[mat, 4]
                            u1 = 0.0
                            u2 = 0.0
                            u3 = 0.0
                            for _t in range(16):
                                u1 = rng_next(state) * 2.0 - 1.0
                                u2 = rng_next(state) * 2.0 - 1.0
                                u3 = rng_next(state) * 2.0 - 1.0
                                if u1 * u1 + u2 * u2 + u3 * u3 <= 1.0:
                                    break
                            qx = rdx + rough * u1
                            qy = rdy + rough * u2
                            qz = rdz + rough * u3
                            qn = 1.0 / math.sqrt(qx * qx + qy * qy + qz * qz)
                            qx *= qn
                            qy *= qn
                            qz *= qn
                            if qx * nx + qy * ny + qz * nz <= 0.0:
                                qx = rdx
                                qy = rdy
                                qz = rdz
                            dx = qx
                            dy = qy
                            dz = qz
                            ox = sx
                            oy = sy
                            oz = sz
                            continue
                    # the key light: one direction in its cone
                    u1 = rng_next(state)
                    u2 = rng_next(state)
                    ct = 1.0 - u1 * (1.0 - cosa)
                    st = math.sqrt(max(0.0, 1.0 - ct * ct))
                    ph = 6.283185307179586 * u2
                    lx = ktx * st * math.cos(ph) + kbx * st * math.sin(ph) + kx * ct
                    ly = kty * st * math.cos(ph) + kby * st * math.sin(ph) + ky * ct
                    lz = ktz * st * math.cos(ph) + kbz * st * math.sin(ph) + kz * ct
                    ndl = nx * lx + ny * ly + nz * lz
                    if ndl > 0.0:
                        ts, ps, _u, _v = trace(sx, sy, sz, lx, ly, lz, INF, True, bmin, bmax, left, start, count,
                                               order, ntri, V0, E1, E2, CA, CB, CR, stack)
                        if ps < 0:
                            if isfloor and depth == 0:
                                Lr += tr * ar * efr
                                Lg += tg * ag * efg
                                Lb += tb_ * ab * efb
                            else:
                                f = ndl / math.pi
                                Lr += tr * ar * ekr * f
                                Lg += tg * ag * ekg * f
                                Lb += tb_ * ab * ekb * f
                    u1 = rng_next(state)
                    u2 = rng_next(state)
                    rr = math.sqrt(u1)
                    ph = 6.283185307179586 * u2
                    lx0 = rr * math.cos(ph)
                    ly0 = rr * math.sin(ph)
                    lz0 = math.sqrt(max(0.0, 1.0 - u1))
                    t1x, t1y, t1z, b1x, b1y, b1z = onb(nx, ny, nz)
                    dx = t1x * lx0 + b1x * ly0 + nx * lz0
                    dy = t1y * lx0 + b1y * ly0 + ny * lz0
                    dz = t1z * lx0 + b1z * ly0 + nz * lz0
                    ox = sx
                    oy = sy
                    oz = sz
                    tr *= ar
                    tg *= ag
                    tb_ *= ab
                if first_kind == 0:
                    if first_spec:
                        sacc[jj, ii, 0] += Lr
                        sacc[jj, ii, 1] += Lg
                        sacc[jj, ii, 2] += Lb
                        sacc[jj, ii, 3] += Lr * m_edge
                        sacc[jj, ii, 4] += Lg * m_edge
                        sacc[jj, ii, 5] += Lb * m_edge
                    else:
                        qr = Lr / a0r
                        qg = Lg / a0g
                        qb = Lb / a0b
                        acc[first_mat, jj, ii, 0] += qr
                        acc[first_mat, jj, ii, 1] += qg
                        acc[first_mat, jj, ii, 2] += qb
                        acc[first_mat, jj, ii, 3] += qr * m_edge
                        acc[first_mat, jj, ii, 4] += qg * m_edge
                        acc[first_mat, jj, ii, 5] += qb * m_edge
                        acc[first_mat, jj, ii, 6] += qr * m_tarn
                        acc[first_mat, jj, ii, 7] += qg * m_tarn
                        acc[first_mat, jj, ii, 8] += qb * m_tarn
                        acc[first_mat, jj, ii, 9] += qr * m_scr
                        acc[first_mat, jj, ii, 10] += qg * m_scr
                        acc[first_mat, jj, ii, 11] += qb * m_scr
                        ie[jj, ii] += plasma_in
                elif first_kind == 1:
                    q0 = (Lr / l0r + Lg / l0g + Lb / l0b) / 3.0
                    facc[jj, ii, 0] += q0
                    facc[jj, ii, 1] += plasma_in
                    facc[jj, ii, 2] += 1.0
                    if abs(q0 - 1.0) < 1e-9:
                        nfl1 += 1
            nsm[jj, ii] = ns
    return acc, sacc, facc, gsum, ie, nsm


# ============================================================================ colour
def srgb_to_lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def hexrgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def lin(h):
    return srgb_to_lin(hexrgb(h))


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


# ============================================================================ the scene
class Camera:
    """An orthographic camera: it looks along ``forward`` from a plane through ``eye`` and shows ``half`` metres
    above and below the centre of the picture."""

    def __init__(self, forward, centre, half, distance=200.0):
        f = np.asarray(forward, float)
        self.forward = f / np.linalg.norm(f)
        r = np.cross(self.forward, [0.0, 0.0, 1.0])
        self.right = r / np.linalg.norm(r)
        self.up = np.cross(self.right, self.forward)
        self.centre = np.asarray(centre, float)
        self.eye = self.centre - distance * self.forward
        self.half = float(half)

    def tuple(self):
        e, f, r, u = self.eye, self.forward, self.right, self.up
        return (float(e[0]), float(e[1]), float(e[2]), float(f[0]), float(f[1]), float(f[2]), float(r[0]),
                float(r[1]), float(r[2]), float(u[0]), float(u[1]), float(u[2]), float(self.half))

    def floor_point(self, px, py, aspect, z):
        """Where the rays of picture positions px, py (-1 to 1) meet the plane z."""
        px, py = np.asarray(px, float), np.asarray(py, float)
        o = self.eye + (px[..., None] * aspect * self.right + py[..., None] * self.up) * self.half
        t = (z - o[..., 2]) / self.forward[2]
        return o + t[..., None] * self.forward


def make_light(L):
    """The key light as a cone of directions (its irradiance normal to it ``key``), a uniform sky, the floor's
    albedo."""
    k = np.asarray(L["dir"], float)
    k /= np.linalg.norm(k)
    cosa = math.cos(math.radians(L["cone_deg"]))
    ek = np.asarray(L["key"], float)
    sk = np.asarray(L["sky"], float)
    ef = ek * k[2] * (1 + cosa) / 2 / math.pi       # the floor: the key's irradiance over the cone, per unit albedo
    fa = np.asarray(L["floor_albedo"], float)
    l0 = fa * (ef + sk)                              # the unshadowed floor's radiance
    return (float(k[0]), float(k[1]), float(k[2]), float(cosa), float(ek[0]), float(ek[1]), float(ek[2]),
            float(sk[0]), float(sk[1]), float(sk[2]), float(ef[0]), float(ef[1]), float(ef[2]),
            float(l0[0]), float(l0[1]), float(l0[2]))


class Scene:
    """Everything the tracer needs, built once: the opaque solids and their hierarchy, the fibres and theirs, the
    closed mesh of the plasma's glow, the plasma's light on the metal and the floor, the camera, the light.

    caps: (A (n, 3), B (n, 3), radius (n,), material (n,)): round tubes and pills (material 0 metal of a coil,
        1 a magnet's steel).
    tris: (V (n, 3, 3), normals (n, 3, 3), dp (n, 3), dr (n, 3), convex (n, 2)): the swept solids (metal).
    fibres: (A (n, 3), B (n, 3), weight (n,), channel (n,)): segments of the fibres of light.
    channels: one dict per channel: radius, halo (its weight), sigma (its width) in metres.
    volume: (n, 3, 3) triangles of one closed mesh (the plasma's glow), or empty.
    glow: (m, 4) points with weights filling the plasma (its light on what is round it), or empty.
    style: the look (STYLE in torux.render): light, materials, wear, plasma, sizes."""

    def __init__(self, caps, tris, fibres, channels, volume, glow, camera, W, H, style):
        self.style = st = style
        self.camera = camera
        self.W, self.H = int(W), int(H)
        A, B, CR, CM = caps
        TV, TN, TDP, TDR, TCV = tris
        FA, FB, FW, FC = fibres
        E = np.asarray(glow, float).reshape(-1, 4)
        reach = st["plasma"]["light_reach"]
        self.capm = light_moments(0.5 * (A + B), E, reach) if len(A) and len(E) else np.zeros((len(A), 4))
        self.trim = light_moments(TV.mean(axis=1), E, reach) if len(TV) and len(E) else np.zeros((len(TV), 4))
        self.fmap = self._floor_light_map(E, reach)
        ntri = len(TV)
        tmin, tmax = (TV.min(axis=1), TV.max(axis=1)) if ntri else (np.zeros((0, 3)), np.zeros((0, 3)))
        cmin = np.minimum(A, B) - CR[:, None]
        cmax = np.maximum(A, B) + CR[:, None]
        self.bvh = bvh_of(np.concatenate([tmin, cmin]), np.concatenate([tmax, cmax]))
        f32 = np.float32
        self.prims = (ntri, TV[:, 0].astype(f32), (TV[:, 1] - TV[:, 0]).astype(f32),
                      (TV[:, 2] - TV[:, 0]).astype(f32), TN[:, 0].astype(f32), TN[:, 1].astype(f32),
                      TN[:, 2].astype(f32), np.zeros(ntri, np.int32), TDP.astype(f32), TDR.astype(f32),
                      TCV.astype(f32), A.astype(f32), B.astype(f32), CR.astype(f32), CM.astype(np.int32),
                      self.trim.astype(f32), self.capm.astype(f32))
        self.nch = len(channels)
        self.crad = np.array([c["radius"] for c in channels], float)
        self.chalo = np.array([c.get("halo", 0.0) for c in channels], float)
        self.csig = np.array([max(c.get("sigma", 0.0), 1e-12) for c in channels], float)
        self.greach = float(max(np.max(self.crad), np.max(np.where(self.chalo > 0, 3.0 * self.csig, 0.0))))
        if len(FA) == 0:
            FA = FB = np.zeros((1, 3)) + 1e9
            FW = np.zeros(1)
            FC = np.zeros(1, np.int32)
        gmin = np.minimum(FA, FB) - self.greach
        gmax = np.maximum(FA, FB) + self.greach
        self.gbvh = bvh_of(gmin, gmax)
        self.glow = (FA.astype(f32), FB.astype(f32), FW.astype(f32), np.asarray(FC, np.int32))
        VT = np.asarray(volume, float).reshape(-1, 3, 3)
        if len(VT):
            vmin, vmax = VT.min(axis=1), VT.max(axis=1)
            self.vbvh = bvh_of(vmin - 1e-6, vmax + 1e-6)
            self.vprims = (VT[:, 0].astype(f32), (VT[:, 1] - VT[:, 0]).astype(f32), (VT[:, 2] - VT[:, 0]).astype(f32))
        else:
            self.vbvh = bvh_of(np.zeros((0, 3)), np.zeros((0, 3)))
            self.vprims = (np.zeros((0, 3), f32), np.zeros((0, 3), f32), np.zeros((0, 3), f32))
        M = st["materials"]
        rows = []
        for k in ("coil", "magnet"):
            a = lin(M[k]["albedo_ref"])
            rows.append([a[0], a[1], a[2], M[k]["sheen"][0], M[k]["sheen"][1]])
        fa = st["light"]["floor_albedo"]
        rows.append([fa[0], fa[1], fa[2], 0.0, 0.0])
        self.mats = np.array(rows, float)
        self.light = make_light(st["light"])
        w = st["wear"]
        self.wear = (float(w["edge_width"]), float(w["tarnish_freq"]), float(w["tarnish_range"][0]),
                     float(w["tarnish_range"][1]), float(w["tarnish_cavity"]), float(w["cavity_reach"]),
                     float(w["scratch_cell"]), float(w["scratch_prob"]), float(w["scratch_width"]),
                     float(w["scratch_length"][0]), float(w["scratch_length"][1]), float(w["tube_wear_freq"]),
                     float(w["tube_wear_range"][0]), float(w["tube_wear_range"][1]))
        self.stats = dict(triangles=ntri, capsules=len(A), fibre_segments=int(len(self.glow[0])),
                          glow_triangles=len(VT), glow_points=len(E))

    def _floor_light_map(self, E, reach):
        """The plasma's light on the floor, (S + L_z) / 2 on a grid of the floor's rectangle in view."""
        st = self.style
        z = st["floor_z"]
        aspect = self.W / self.H
        c = self.camera.floor_point(np.array([-1.0, 1.0, -1.0, 1.0]), np.array([-1.0, -1.0, 1.0, 1.0]), aspect, z)
        x0, x1 = c[:, 0].min(), c[:, 0].max()
        y0, y1 = c[:, 1].min(), c[:, 1].max()
        h = st["plasma"]["floor_map_step"] * st["unit"]
        nx = min(int(math.ceil((x1 - x0) / h)) + 2, 2048)
        ny = min(int(math.ceil((y1 - y0) / h)) + 2, 2048)
        h = max(h, (x1 - x0) / (nx - 2), (y1 - y0) / (ny - 2))
        gx, gy = np.meshgrid(x0 + h * np.arange(nx), y0 + h * np.arange(ny))
        X = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)], -1)
        if len(E):
            m = light_moments(X, E, reach)
            val = np.maximum(0.0, 0.5 * (m[:, 0] + m[:, 3])).reshape(ny, nx)
        else:
            val = np.zeros((ny, nx))
        return (float(x0), float(y0), float(h), val.astype(np.float32))

    def render(self, rows, cols, spp):
        """The buffers of the pixels rows x cols at spp samples per pixel (fewer on the bare floor)."""
        st = self.style
        n_glow = min(st["plasma"]["glow_samples"], st["min_spp"], spp)
        n_ie = min(st["plasma"]["indirect_samples"], spp)
        acc, sacc, facc, gsum, ie, ns = render_kernel(
            self.W, self.H, int(spp), int(st["min_spp"]), int(st["seed"]), self.camera.tuple(), self.bvh,
            self.prims, self.mats, self.light, float(st["floor_z"]), int(st["max_depth"]), int(rows[0]),
            int(rows[1]), int(cols[0]), int(cols[1]), self.gbvh, self.glow[0], self.glow[1], self.glow[2],
            self.glow[3], self.crad, self.csig, self.chalo, float(st["plasma"]["glancing_cap"]), self.greach,
            int(self.nch), self.vbvh, self.vprims[0], self.vprims[1], self.vprims[2], self.wear, int(n_glow),
            int(n_ie), self.fmap[0], self.fmap[1], self.fmap[2], self.fmap[3])
        n = np.maximum(ns, 1.0)
        nie = np.maximum(np.minimum(ns, n_ie), 1.0)
        return dict(A=acc / n[None, ..., None], S=sacc / n[..., None], F=facc[..., 0] / n, FC=facc[..., 2] / n,
                    FI=facc[..., 1] / nie, G=gsum[..., :self.nch] / n_glow, V=gsum[..., self.nch] / n_glow,
                    IE=ie / nie, NS=ns)


# ============================================================================ the page's colours
def floor_fade(frame, rows, cols, fade):
    """1 inside, falling smoothly to 0 within ``fade`` pixels of the frame's border."""
    y = np.arange(rows[0], rows[1])
    x = np.arange(cols[0], cols[1])
    dy = np.minimum(y - frame[0], frame[1] - 1 - y)[:, None]
    dx = np.minimum(x - frame[2], frame[3] - 1 - x)[None, :]
    s = np.clip(np.minimum(dx, dy) / max(fade, 1e-9), 0.0, 1.0)
    return s * s * (3 - 2 * s)


def pool_mask(scene, rows, cols, centre, radius, power):
    """The pool of light on the dark page's floor: one soft disc on the floor round the design (decoration)."""
    W, H = scene.W, scene.H
    j = np.arange(rows[0], rows[1]) + 0.5
    i = np.arange(cols[0], cols[1]) + 0.5
    px = np.broadcast_to((i / W * 2.0 - 1.0)[None, :], (len(j), len(i)))
    py = np.broadcast_to((1.0 - j / H * 2.0)[:, None], (len(j), len(i)))
    X = scene.camera.floor_point(px, py, W / H, scene.style["floor_z"])
    q = ((X[..., 0] - centre[0]) ** 2 + (X[..., 1] - centre[1]) ** 2) / radius ** 2
    return np.exp(-q ** power)


def composite(buf, scene, pal, rows, cols, pool=None):
    """The linear-light picture of the pixels rows x cols for one page: the metal in its colours with the
    decoration's masks, its sheen, the floor (the page's colour times the shadow, and on the dark page a pool of
    light), then the light of the plasma and the fibres of each channel.

    Decoration, encoding nothing: the worn edges, tarnish and scratches (their masks come from the tracer), the
    sheen, the pool, and the plasma's light on the metal and the floor round it."""
    st = scene.style
    K = np.array(scene.light[10:13]) + np.array(scene.light[7:10])   # the unshadowed floor's radiance per albedo
    bg = lin(pal["bg"])
    fade = floor_fade((0, scene.H, 0, scene.W), rows, cols, st["edge_fade"] * scene.W / 1676.0)[..., None]
    F = buf["F"][..., None].astype(float)          # the floor's light over its unshadowed value, times its share
    C = buf["FC"][..., None].astype(float)         # the share of the pixel that is floor
    if pal.get("floor_pool") and pool is not None:
        fp = pal["floor_pool"]
        m = pool_mask(scene, rows, cols, pool[0], pool[1], st["pool_power"])[..., None] * fade
        img = (bg + (lin(fp["colour"]) - bg) * m) * (C - (C - F) * fade * fp["shadow_depth"])
    else:
        img = bg * (C - (C - F) * fade)            # the shadows fade out towards the frame's border
    for mi, key in enumerate(("coil", "magnet")):
        mp = pal[key]
        base, worn, tarn, scr = lin(mp["base"]), lin(mp["worn"]), lin(mp["tarnish"]), lin(mp["scratch"])
        A = buf["A"][mi].astype(float)
        col = A[..., 0:3] * base + A[..., 3:6] * (worn - base) * mp["worn_amount"] \
            + A[..., 6:9] * (tarn - base) * mp["tarnish_amount"] + A[..., 9:12] * (scr - base) * mp["scratch_amount"]
        img += col / K
    S = buf["S"].astype(float)
    img += (S[..., 0:3] * pal["sheen"] + S[..., 3:6] * pal["sheen_worn"]) * lin(pal["sheen_tint"]) / K
    pl = pal["plasma"]
    if pl.get("indirect", 0.0) > 0.0:
        cI = lin(pl["indirect_colour"])
        img += pl["indirect"] * buf["IE"][..., None] * cI * lin(pal["coil"]["base"])
        t0 = pl.get("indirect_floor_from", 0.0)          # only the glow near the plasma, not the far field
        fi = np.clip(buf["FI"][..., None] - t0, 0.0, None) / max(1.0 - t0, 1e-6)
        img += pl["indirect_floor"] * fi * cI * fade
    G = buf["G"].astype(float)
    V = buf["V"].astype(float)[..., None]
    order = pal["channel_order"]
    for k in order:                                    # channels drawn under the glow first, then the glow
        if k == "glow":
            cV = lin(pl["body"])
            a = pl["body_gain"] * (1.0 - np.exp(-pl["body_strength"] * V))
            img = img + cV * a if pl["mode"] == "emit" else img * (1.0 - a) + cV * a
            continue
        ch = pal["channels"][k]
        g = G[..., ch["index"]][..., None]
        c = lin(ch["colour"])
        a = ch["gain"] * (1.0 - np.exp(-ch["strength"] * g))
        if ch["mode"] == "emit":
            img = img + c * a
            if ch.get("hot_gain", 0.0) > 0.0:
                img = img + lin(ch["hot"]) * ch["hot_gain"] * np.clip(g * ch["strength"] - ch["hot_from"], 0.0, None)
        else:
            img = img * (1.0 - a) + c * a
    return img


def to8(lin_img):
    return (lin_to_srgb(lin_img) * 255 + 0.5).astype(np.uint8)


def write_png(path, img):
    """An 8-bit RGB picture as a PNG file, by the standard library alone (zlib): no imaging package needed."""
    img = np.ascontiguousarray(img, np.uint8)
    H, W = img.shape[:2]
    raw = np.empty((H, 1 + 3 * W), np.uint8)
    raw[:, 0] = 2                                    # every row filtered by its difference from the row above
    rows = img.reshape(H, 3 * W).astype(np.int16)
    raw[:, 1:] = (rows - np.vstack([np.zeros((1, 3 * W), np.int16), rows[:-1]])).astype(np.uint8)

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")
        fh.write(chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)))
        fh.write(chunk(b"IDAT", zlib.compress(raw.tobytes(), 6)))
        fh.write(chunk(b"IEND", b""))


def read_png(path):
    """The 8-bit RGB pictures write_png writes (and any PNG of 8-bit RGB or RGBA without interlacing), as an array."""
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path}: not a PNG file")
    pos, idat, W = 8, [], None
    while pos < len(data):
        n = struct.unpack(">I", data[pos:pos + 4])[0]
        kind = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + n]
        if kind == b"IHDR":
            W, H, depth, colour = struct.unpack(">IIBB", body[:10])
            if depth != 8 or colour not in (2, 6) or body[12] != 0:
                raise ValueError(f"{path}: only 8-bit RGB or RGBA without interlacing is read here")
            ch = 3 if colour == 2 else 4
        elif kind == b"IDAT":
            idat.append(body)
        pos += 12 + n
    raw = np.frombuffer(zlib.decompress(b"".join(idat)), np.uint8).reshape(H, 1 + ch * W)
    out = np.zeros((H, ch * W), np.int32)
    prev = np.zeros(ch * W, np.int32)
    for r in range(H):
        f, line = raw[r, 0], raw[r, 1:].astype(np.int32)
        if f == 0:
            cur = line
        elif f == 2:
            cur = (line + prev) & 255
        elif f == 1:
            cur = line.copy()
            for i in range(ch, ch * W):
                cur[i] = (cur[i] + cur[i - ch]) & 255
        elif f == 3:
            cur = line.copy()
            for i in range(ch * W):
                a = cur[i - ch] if i >= ch else 0
                cur[i] = (cur[i] + (a + prev[i]) // 2) & 255
        else:
            cur = line.copy()
            for i in range(ch * W):
                a = cur[i - ch] if i >= ch else 0
                b, c = prev[i], prev[i - ch] if i >= ch else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                cur[i] = (cur[i] + pr) & 255
        out[r] = cur
        prev = cur
    return out.reshape(H, W, ch)[..., :3].astype(np.uint8)
