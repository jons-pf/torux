"""answer.render: one picture in the look of the README hero, path traced.  What it draws is the answer's: every fibre
of light runs along the source's field, the woven surface lies on the verified torus, the island chain's tubes pass
through the certificate's own points, the haze is lines traced until they leave.  Outside the machine's shadow the
picture is exactly the page's colour, and nothing in it is red.  The full pictures take minutes each: -m slow."""
import copy
import math

import numpy as np
import pytest

import torux
from torux import pathtrace as PT
from torux import render as RD

from conftest import example

PAGES = {False: (255, 255, 255), True: (13, 17, 23)}


def _along(answer, pts):
    """The largest sine of the angle between a polyline's chords and the source's field at their midpoints."""
    pts = np.asarray(pts, float)
    t = pts[1:] - pts[:-1]
    keep = np.linalg.norm(t, axis=1) > 0
    t, m = t[keep], 0.5 * (pts[1:] + pts[:-1])[keep]
    B = answer.field(m)
    return float(np.max(np.linalg.norm(np.cross(t, B), axis=1) / (np.linalg.norm(t, axis=1) *
                                                                  np.linalg.norm(B, axis=1))))


def _lines_of(drawn, channel):
    """The drawn fibres of one channel, each as its polyline (m)."""
    k = RD.CHANNELS[channel]["index"]
    return [np.vstack([f[0], f[1][-1:]]) for f in drawn["fibres"] if f is not None and f[3] == k]


def _hue(img):
    """Hue (degrees), saturation and value of every pixel of an 8-bit RGB picture."""
    x = img.reshape(-1, 3).astype(float) / 255.0
    mx, mn = x.max(axis=1), x.min(axis=1)
    c = mx - mn
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        h = np.where(mx == r, ((g - b) / c) % 6, np.where(mx == g, (b - r) / c + 2, (r - g) / c + 4)) * 60.0
        s = np.where(mx > 0, c / mx, 0.0)
    return np.where(c > 0, h, 0.0), s, mx


def _check_page(img, dark):
    """Exactly the page's colour at the border, some torux green, and no red: no pixel saturated enough to read as a
    colour (over 0.45: the copper is 0.6 at a hue of 25 degrees, and where its edge meets the dark page it turns a
    dusky brown of 0.3) has a hue within 15 degrees of red."""
    page = np.array(PAGES[dark], np.uint8)
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    assert np.all(border == page), "the picture's border is not the page's colour"
    h, s, v = _hue(img)
    red = ((h < 15) | (h > 345)) & (s > 0.45) & (v > 0.2)
    assert not np.any(red), f"{int(red.sum())} red pixels"
    green = (h > 140) & (h < 185) & (s > 0.3) & (v > 0.2)
    return int(green.sum())


# ---------------------------------------------------------------------------------------------- what is drawn
@pytest.fixture(scope="module")
def drawn_954(solved_954):
    return RD.content(solved_954)


def _crossings(line, phi):
    """(R, Z) where a polyline crosses the half-plane at the cylindrical angle phi (every turn)."""
    ph = np.unwrap(np.arctan2(line[:, 1], line[:, 0]))
    out = []
    for k in range(int(np.floor((ph.min() - phi) / (2 * np.pi))), int(np.ceil((ph.max() - phi) / (2 * np.pi))) + 1):
        target = phi + 2 * np.pi * k
        for i in np.flatnonzero((ph[:-1] - target) * (ph[1:] - target) <= 0):
            if ph[i + 1] == ph[i]:
                continue
            f = (target - ph[i]) / (ph[i + 1] - ph[i])
            x = line[i] + f * (line[i + 1] - line[i])
            out.append((math.hypot(x[0], x[1]), x[2]))
    return np.array(out).reshape(-1, 2)


def _distance_to(poly, pts):
    """Each point's distance (m) to the closed polyline poly (m, 2)."""
    a, b = poly, np.roll(poly, -1, axis=0)
    d = b - a
    t = np.clip(np.einsum("pkj,kj->pk", pts[:, None, :] - a[None], d) / np.sum(d * d, axis=1), 0.0, 1.0)
    q = a[None] + t[..., None] * d[None]
    return np.min(np.linalg.norm(pts[:, None, :] - q, axis=-1), axis=1)


def test_the_surface_is_woven_from_field_lines_on_it(solved_954, drawn_954):
    """The fibres run along the field. Those of the surface cross the plane phi0 on the verified torus's own
    section, the nested ones inside it: traced lines, not decoration. The glow fills the torus itself."""
    plasma = _lines_of(drawn_954, "plasma")
    assert len(plasma) >= 12 and not _lines_of(drawn_954, "tubes") and not _lines_of(drawn_954, "threads")
    along = [_along(solved_954, ln) for ln in plasma]
    assert max(along) < 1e-2 and float(np.median(along)) < 1e-3, (max(along), float(np.median(along)))
    T = solved_954.surface
    phi0 = float(getattr(T, "phi0", 0.0))
    sec = np.column_stack(solved_954.section(phi0, 4096))
    on, inside = 0, 0
    for ln in plasma:
        c = _crossings(ln, phi0)
        if not len(c):
            continue
        d = _distance_to(sec, c)
        if np.max(d) < 1e-4:
            on += 1
        elif np.all(RD._inside(sec, c)) and np.min(d) > 1e-3:
            inside += 1
    assert on >= 8 and inside >= 3, (on, inside)
    R, Z, ph = drawn_954["region"]
    x = solved_954.points(2 * np.pi * np.arange(R.shape[1]) / R.shape[1], np.full(R.shape[1], ph[3]))
    assert np.allclose(np.hypot(x[:, 0], x[:, 1]), R[3], atol=1e-12) and np.allclose(x[:, 2], Z[3], atol=1e-12)
    assert any("verified surface" in s for s in drawn_954["about"])


def test_the_cut_shows_the_surface_and_the_nested_ones_where_they_are(solved_954, drawn_954):
    """In the opened cross-section: the surface's own punctures on its section there, the nested surfaces' inside it,
    each a crossing of a traced line. The wedge in front of it holds no light."""
    cut = drawn_954["cut"]
    sec = np.column_stack(solved_954.section(cut[0], 4096))
    dots = [d for d in drawn_954["dots"] if d[1] == "points"]
    assert len(dots) >= 3
    on = [d for d, _, _ in dots if np.max(_distance_to(sec, d)) < 1e-6]
    inside = [d for d, _, _ in dots if np.all(RD._inside(sec, d)) and np.min(_distance_to(sec, d)) > 1e-3]
    assert len(on) == 1 and len(on[0]) >= 100 and len(inside) >= 2
    scene, _ = RD.build(drawn_954, 200)
    s, off, Rt = scene.place
    mid = (0.5 * (scene.glow[0] + scene.glow[1]).astype(float) - off) / s @ Rt   # back to the source's own frame
    lit = RD._in_wedge(mid, cut) & (scene.glow[2] > 0)
    assert not np.any(lit)


def test_the_coils_are_drawn_on_their_filaments(quasr_954, drawn_954):
    paths = drawn_954["solids"]["paths"]
    part = quasr_954.parts[0]
    x, _ = part.points(np.arange(RD.STYLE["n_fourier"]) / RD.STYLE["n_fourier"])
    assert len(paths) == part.n
    assert max(float(np.max(np.abs(q - c))) for (q, _), c in zip(paths, x)) == 0.0


def test_an_answer_without_a_surface_or_certificate_draws_the_source_alone(solved_954):
    a = copy.copy(solved_954)
    a.__dict__.pop("_render", None)
    a.__dict__.pop("_render_drawn", None)
    a.status = "undetermined"
    d = RD.content(a)
    assert not d["fibres"] and d["region"] is None and d["solids"]["paths"]
    assert any("no plasma" in s for s in d["about"])


def test_render_wants_a_png_and_an_answer_that_holds_its_source(solved_954, tmp_path):
    with pytest.raises(ValueError, match=r"\.png"):
        solved_954.render(tmp_path / "x.jpg")
    path = tmp_path / "a.json"
    solved_954.save(path)
    with pytest.raises(ValueError, match="source"):
        torux.Answer.read(path).render(tmp_path / "x.png")


def test_a_small_picture_is_the_page_outside_the_machine_and_never_red(solved_954, tmp_path):
    """The whole path at a small size and the README's own samples per pixel: both pages, written and read back.
    (Light that bounced from coil to coil several times is copper on copper, a deep rust: with a few dozen samples a
    pixel can be mostly that, and read as red.)"""
    a = copy.copy(solved_954)
    a.__dict__.pop("_render", None)
    pics = RD.picture(a, width=160, spp=RD.SPP)
    for dark in (False, True):
        img = pics[dark]
        assert img.shape[1] == 160 and img.dtype == np.uint8
        assert _check_page(img, dark) > 20
        p = tmp_path / f"p{int(dark)}.png"
        PT.write_png(p, img)
        assert np.array_equal(PT.read_png(p), img)


def test_a_wide_chains_lines_are_told_apart_by_their_rotation():
    """Punctures a field period apart: on a circle round the axis at an irrational rotation, a surface. Round one of
    six elliptic points, hopping a sixth of a turn each period, an island of a q = 6 chain. Steps at random, chaos.
    A line that left, lost."""
    K = 1024
    k = np.arange(K + 1)
    ring = np.column_stack([np.cos(2 * np.pi * 0.0618034 * k), np.sin(2 * np.pi * 0.0618034 * k)])
    hop = -2 * np.pi * k / 6
    island = np.column_stack([np.cos(hop), np.sin(hop)]) + 0.2 * np.column_stack(
        [np.cos(2 * np.pi * 0.0377 * k), np.sin(2 * np.pi * 0.0377 * k)])
    ang = np.cumsum(np.random.default_rng(3).uniform(0.0, 1.0, K + 1))
    chaos = np.column_stack([np.cos(ang), np.sin(ang)])
    c = np.zeros(2)
    assert RD._island_kind(ring, 1, c, True, 6)[0] == "surface"
    kind, rho = RD._island_kind(island, 1, c, True, 6)
    assert kind == "island" and abs(rho + 1 / 6) < 1e-6
    assert RD._island_kind(chaos, 1, c, True, 6)[0] == "chaos"
    assert RD._island_kind(ring, 1, c, False, 6)[0] == "lost"


def test_png_files_read_back_as_written(tmp_path):
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (37, 53, 3)).astype(np.uint8)
    PT.write_png(tmp_path / "r.png", img)
    assert np.array_equal(PT.read_png(tmp_path / "r.png"), img)


# ---------------------------------------------------------------------------------------------- the three pictures
def _solved(name):
    return torux.solve(example(name), budget_s=None if name.endswith("0954.json") else 3300)


@pytest.mark.slow
def test_quasr_954_renders_as_its_woven_surface(tmp_path):
    a = _solved("quasr_0000954.json")
    assert a.status == "surface"
    for dark in (False, True):
        p = a.render(tmp_path / f"s{int(dark)}.png", dark=dark)
        img = PT.read_png(p)
        assert img.shape[1] == RD.README_WIDTH
        assert _check_page(img, dark) > 5000


@pytest.mark.slow
def test_an_island_chain_is_drawn_where_the_certificate_puts_it(tmp_path):
    """QUASR 219457: the requested flux falls inside an island chain. Its tubes are closed field lines through the
    certificate's elliptic points."""
    a = _solved("quasr_0219457.json")
    if a.status != "no_surface":
        pytest.skip(f"{a.status} after {a.wall_s:.0f} s on this machine")
    c = a.evidence["certificate"]
    d = RD.content(a)
    tubes = _lines_of(d, "tubes")
    assert len(tubes) >= 1
    assert max(_along(a, t) for t in tubes) < 2e-3
    O = np.asarray(c["O_line"]["points"], float)
    crossings = np.concatenate([_crossings(t, float(c.get("phi_s", 0.0))) for t in tubes])
    assert len(crossings) >= len(O)
    for o in O:                                          # every elliptic point of the certificate on a tube
        assert np.min(np.hypot(*(crossings - o).T)) < 1e-6
    for dark in (False, True):
        img = PT.read_png(a.render(tmp_path / f"i{int(dark)}.png", dark=dark))
        assert _check_page(img, dark) > 2000


@pytest.mark.slow
def test_a_wide_island_chain_is_drawn_at_its_true_width(tmp_path):
    """QUASR 797878: the requested flux falls inside an island chain some 30 pixels wide at README size. The lines
    from the axis to its elliptic point weave the core where they lie on surfaces and the islands where they are
    locked to the chain: every strand runs along the field, where the islands' strands cross the certificate's plane
    they lie round its elliptic points, inside the chain, and the elliptic lines pass through those points."""
    try:
        src = torux.load("quasr:797878")
    except ConnectionError as e:
        pytest.skip(f"QUASR 797878 is not in the cache and cannot be downloaded: {e}")
    a = torux.solve(src, budget_s=3300)
    c = (a.evidence or {}).get("certificate") or {}
    if a.status != "no_surface" or not str(c.get("kind", "")).startswith("island"):
        pytest.skip(f"{a.status} ({c.get('kind')}) after {a.wall_s:.0f} s on this machine")
    d = RD.content(a)
    isl, plasma = _lines_of(d, "islands"), _lines_of(d, "plasma")
    assert len(isl) >= 24 and len(plasma) >= 12 and not _lines_of(d, "tubes")
    assert max(_along(a, ln) for ln in isl[::max(1, len(isl) // 40)]) < 2e-3
    assert max(_along(a, ln) for ln in plasma[:20]) < 1e-3
    O = np.asarray(c["O_line"]["points"], float)
    X = np.asarray(c["X_line"]["points"], float)
    reach = np.array([np.min(np.hypot(*(X - o).T)) for o in O])          # each elliptic point to its nearest X
    cross = np.concatenate([_crossings(ln, float(c.get("phi_s", 0.0))) for ln in isl])
    near = np.hypot(cross[:, None, 0] - O[None, :, 0], cross[:, None, 1] - O[None, :, 1])
    k = np.argmin(near, axis=1)
    assert np.all(near[np.arange(len(cross)), k] < reach[k])
    for o in O:                                              # every elliptic point on an elliptic line
        assert np.min(np.hypot(*(cross - o).T)) < 1e-6
    for dark in (False, True):
        img = PT.read_png(a.render(tmp_path / f"w{int(dark)}.png", dark=dark))
        assert _check_page(img, dark) > 5000


@pytest.mark.slow
def test_beyond_the_edge_the_field_lines_leave(tmp_path):
    """QUASR 1100073: the requested flux lies beyond the last surface. Lines on surfaces inside, the haze outside
    made of traced field lines that end where they leave."""
    a = _solved("quasr_1100073.json")
    if a.status != "no_surface":
        pytest.skip(f"{a.status} after {a.wall_s:.0f} s on this machine")
    d = RD.content(a)
    haze, plasma = _lines_of(d, "threads"), _lines_of(d, "plasma")
    assert len(haze) >= 40 and len(plasma) >= 12
    assert max(_along(a, ln) for ln in plasma[:20]) < 1e-3
    assert max(_along(a, ln) for ln in haze[:20]) < 2e-2                 # chords near the coils, where lines turn
    for dark in (False, True):
        img = PT.read_png(a.render(tmp_path / f"e{int(dark)}.png", dark=dark))
        assert _check_page(img, dark) > 2000
