"""Poincaré sections: a line started on a verified surface stays on it to near rounding and turns as the surface does,
the crossings are where the independent tracer puts them, island chains and the edge are found without a setting,
and the picture keeps the conventions of answer.plot: ink for crossings, green only for a verified surface."""
import importlib.util
import math
import signal
import subprocess
import sys
import time

import numpy as np
import pytest

import torux

from conftest import example

HAVE_MPL = importlib.util.find_spec("matplotlib") is not None
needs_mpl = pytest.mark.skipif(not HAVE_MPL, reason='drawing needs matplotlib: pip install "torux[plot]"')
if HAVE_MPL:
    import matplotlib
    matplotlib.use("Agg")


@pytest.fixture(scope="module")
def section_954(solved_954):
    """The poincare of the verified surface of QUASR 954, as torux chooses it."""
    return solved_954.poincare()


def distance_from_surface(answer, R, Z, phi):
    """How far each (R, Z) lies from the answer's surface in the plane phi, m: Newton's method on the poloidal angle
    from the nearest of 4,096 points of the section, the derivatives by central differences of answer.points."""
    th = np.linspace(0.0, 2 * np.pi, 4096, endpoint=False)
    x = answer.points(th, np.full_like(th, phi))
    Rs, Zs = np.hypot(x[:, 0], x[:, 1]), x[:, 2]
    out = []
    for r, z in zip(R, Z):
        t = th[np.argmin((Rs - r) ** 2 + (Zs - z) ** 2)]
        for _ in range(8):
            h = 1e-5
            P = answer.points(np.array([t - h, t, t + h]), np.full(3, phi))
            PR, PZ = np.hypot(P[:, 0], P[:, 1]), P[:, 2]
            dR, dZ = (PR[2] - PR[0]) / (2 * h), (PZ[2] - PZ[0]) / (2 * h)
            ddR, ddZ = (PR[2] - 2 * PR[1] + PR[0]) / h ** 2, (PZ[2] - 2 * PZ[1] + PZ[0]) / h ** 2
            g = (PR[1] - r) * dR + (PZ[1] - z) * dZ
            t -= g / (dR ** 2 + dZ ** 2 + (PR[1] - r) * ddR + (PZ[1] - z) * ddZ)
        P = answer.points(np.array([t]), np.array([phi]))[0]
        out.append(math.hypot(math.hypot(P[0], P[1]) - r, P[2] - z))
    return np.array(out)


# ------------------------------------------------------------------------------------------------ right
def test_a_line_started_on_the_verified_surface_stays_on_it(solved_954, section_954):
    """Every 4th crossing of the line started on the surface, in all three planes over all its field periods, lies
    within 1e-11 m of the surface (3.5e-12 m measured over 1,200 periods), and the line was never lost."""
    p, s = section_954, solved_954
    i = p.surface_line
    assert i is not None and not p.left[i]
    worst = 0.0
    for j, phi in enumerate(p.planes):
        R, Z = p.R[i, j, :p.followed[i]], p.Z[i, j, :p.followed[i]]
        assert np.all(np.isfinite(R)) and p.followed[i] >= p.periods
        worst = max(worst, distance_from_surface(s, R[::4], Z[::4], phi).max())
    assert worst < 1e-11, worst


def test_its_rotational_transform_is_the_answers(solved_954, section_954):
    """The rotational transform of the line on the surface, kept to the nine decimals it has settled to, is the
    answer's to 1e-8 (1.1e-10 measured)."""
    p = section_954
    assert abs(p.iota[p.surface_line] - solved_954.iota) < 1e-8, (p.iota[p.surface_line], solved_954.iota)


def test_the_crossings_are_where_the_independent_tracer_puts_them(section_954):
    """Three lines' crossings of each plane over their first ten turns against torux.reference.trace (its own field,
    Gauss-Legendre steps, no code shared): within 1e-11 m (4.3e-14 m measured)."""
    from torux.reference.trace import poincare as reference
    p = section_954
    src = p._source
    pick = [0, p.surface_line, len(p.start) // 2]
    out = reference(src, p.start[pick, 0], p.start[pick, 1], p.planes, 10, phi0=p.planes[0], doubling=False)
    # the reference's sections: every plane in order of angle after the start, a turn at a time (two periods)
    for a, i in enumerate(pick):
        for j, phi in enumerate(p.planes):
            k = np.argmin(np.abs(np.mod(out["phi"][0] - phi, 2 * np.pi)))
            ref_R = out["R"][a, :, k]
            n = p.nfp
            ours = p.R[i, j, n - 1::n][:len(ref_R)] if j == 0 else p.R[i, j, :][::n][:len(ref_R)]
            assert np.max(np.abs(ours - ref_R)) < 1e-11, (i, j)


def test_lines_on_surfaces_have_a_profile_and_the_others_none(section_954):
    """The lines inside the verified surface have a rotational transform, in order of their start: the profile.
    A line that left has crossings until it left and none after, and says where."""
    p = section_954
    prof = p.profile
    assert len(prof) >= 8 and np.all(np.diff(prof[:, 0]) > 0)
    assert np.all(np.isfinite(p.iota[:p.surface_line + 1]))
    assert np.all(np.diff(prof[:, 1]) > 0)              # 954's iota rises outward, from -0.1025 at the axis
    for i in np.flatnonzero(p.left):
        assert np.all(np.isfinite(p.left_at[i]))
        k = np.isfinite(p.R[i, 0])
        n = int(k.sum())
        assert n == p.followed[i] and np.all(k[:n]) and not np.any(k[n:])
        assert p.left_at[i, 1] <= 2 * np.pi * (n + 1) / p.nfp + 1e-9
    assert np.all(np.isnan(p.left_at[~p.left]))
    assert p.axis.shape == (3, 2) and np.all(np.isfinite(p.axis))


def test_the_lines_field_is_the_engines_to_1e_14(section_954):
    """Where the lines go (every crossing of every plane, at every field period's copy of it), the field the tracer
    sums, far coils on fewer nodes, is the engine's to 1e-14 of |B|."""
    import torux.poincare_section as P
    from torux.poincare_trace import Tracer
    p, src = section_954, section_954._source
    axis = P._axis(P._engine(src, None), src, None, p.nfp)
    eng = P._engine(src, None, P._dmin(src, axis, 0.0))
    box, _ = P._box(src)
    sign, bref = P._sign(eng, src, axis, 0.0, box)
    tr = Tracer(eng, box, eng.dmin, sign, bref=bref, region=P._region(src, axis, eng.dmin, box))
    assert np.any(np.isfinite(tr.reach[:, 1:]))                      # it does take fewer nodes somewhere
    R, Z = p.R.reshape(-1), p.Z.reshape(-1)
    phi = np.broadcast_to(p.planes[None, :, None], p.R.shape).reshape(-1)
    k = np.flatnonzero(np.isfinite(R))[::7]
    phi = phi[k] + 2 * np.pi / p.nfp * (np.arange(len(k)) % p.nfp)
    x = np.column_stack([R[k] * np.cos(phi), R[k] * np.sin(phi), Z[k]])
    B = np.asarray(eng.evaluate(x, grad=False, A=False)["B"])
    err = np.linalg.norm(tr.B_kernel(x) - B, axis=1) / np.linalg.norm(B, axis=1)
    assert err.max() < 1e-14, err.max()


def turn_of(p, i):
    """Line i's mean turn about the axis a field period (turns), from its own crossings of the first plane: the
    weighted Birkhoff average of their turns."""
    R, Z = p.R[i, 0, :p.followed[i]], p.Z[i, 0, :p.followed[i]]
    d = np.diff(np.unwrap(np.arctan2(Z - p.axis[0, 1], R - p.axis[0, 0])))
    t = (np.arange(len(d)) + 0.5) / len(d)
    w = np.exp(-1.0 / (t * (1.0 - t)))
    return float(np.sum(w * d) / np.sum(w) / (2 * np.pi))


def widest_gap_px(p, i):
    """The widest gap between line i's crossings next along its curve, over the planes, in pixels of the picture at
    838 across: on a surface a turn of rho a period puts the k-th crossing at k rho round its curve."""
    from torux.plot import poincare_pixels, poincare_windows
    px = poincare_pixels(*poincare_windows(p.R, p.Z, p.left, p.start, p.axis))
    rho, out = turn_of(p, i), 0.0
    for j in range(len(p.planes)):
        R, Z = p.R[i, j, :p.followed[i]], p.Z[i, j, :p.followed[i]]
        s = np.argsort(np.mod((np.arange(len(R)) + 1) * rho, 1.0))
        x, y = np.append(R[s], R[s][0]), np.append(Z[s], Z[s][0])
        out = max(out, float(np.hypot(np.diff(x), np.diff(y)).max()) * px)
    return out


def surface_lines(p):
    """Lines with a rotational transform whose mean turn a period is no rational of an island chain of 60 islands or
    fewer (to the line's own spread, as torux.poincare tells them)."""
    import torux.poincare_section as P
    return [i for i in np.flatnonzero(np.isfinite(p.iota) & ~p.left)
            if not P._island(turn_of(p, i), p._spread[i] / (2 * np.pi))]


def test_a_curve_means_a_surface(section_954):
    """Every line on a surface reads as a closed curve at 838 pixels across, no two of its crossings next along it
    more than 2 pixels apart, or is said to be open. QUASR 954 has lines it had to follow on for that, and one, its
    outermost, is left open (2.8 pixels after 4,800 field periods)."""
    p = section_954
    for i in surface_lines(p):
        assert widest_gap_px(p, i) <= 2.0 or p.open[i], (i, widest_gap_px(p, i))
    assert np.any(p.followed > p.periods) and p.open.sum() <= 1


def test_no_ring_of_dashes_next_to_an_island_chain():
    """QUASR 219457, whose surfaces crowd round a chain of 15 thin islands: every line on a surface closes its curve at
    838 pixels, or is drawn in grey and said to be open, so that no ring of dashes reads as the chain."""
    from torux.plot import PAGES, _poincare_picture
    p = torux.poincare(example("quasr_0219457.json"))
    for i in surface_lines(p):
        assert widest_gap_px(p, i) <= 2.0 or p.open[i], (i, widest_gap_px(p, i))
    q = _poincare_picture(p)
    assert sum(len(g) for g in q.grey) == sum(np.isfinite(p.R[p.open]).sum(axis=(0, 2)))
    assert "curves not closed" in str(p) if p.open.any() else True
    if HAVE_MPL:
        colours = {tuple(np.round(c.get_facecolor()[0], 3)) for ax in p.plot().axes for c in ax.collections
                   if len(c.get_offsets())}
        assert tuple(np.round(matplotlib.colors.to_rgba(PAGES[False]["muted"]), 3)) in colours or not p.open.any()


def test_each_transform_is_given_to_the_digits_it_has_settled_to(section_954):
    """Printed and kept to the decimals whose unit is no smaller than the difference between the averages over the
    first half and the whole (rad a period, in the transform's units), and no more: the line on the verified surface
    then agrees with the answer in every digit but the last."""
    p = section_954
    period = 2 * np.pi / p.nfp
    rows = p.lines_table().splitlines()[1:]
    for i in np.flatnonzero(np.isfinite(p.iota)):
        printed = rows[i].split()[3]
        d = len(printed.split(".")[1])
        assert float(printed) == p.iota[i] and round(p.iota[i], d) == p.iota[i]
        assert 10.0 ** -d >= p._spread[i] / period > 10.0 ** -(d + 1) or d == 13, (i, printed, p._spread[i])
    i = p.surface_line
    d = len(rows[i].split()[3].split(".")[1])
    assert abs(p.iota[i] - p._answer.iota) <= 1.5 * 10.0 ** -d


def test_a_long_call_says_how_long_first(quasr_954, monkeypatch):
    import torux.ask as A
    import torux.poincare_section as P
    said = []
    monkeypatch.setattr(P, "SAY_AFTER", 0.0)
    monkeypatch.setattr(A, "say", said.append)
    torux.poincare(quasr_954, starts=[(1.0, 0.0)], turns=20)
    assert "tracing 1 field line through 40 field periods, about" in said[0]
    assert all("following" in t for t in said[1:]) and len(said) <= 2


def test_the_time_said_is_never_one_the_lines_will_not_take(monkeypatch):
    """However slow the machine (here a thousand seconds a piece of the trace), nothing is said of lines that have all
    left by a sixteenth of their length. Lines still going then are said to take the time so far and the rest of
    the periods at the pace of the last piece: 4,000 s, and 560 periods at a thousand seconds for 30."""
    import types
    import torux.ask as A
    import torux.poincare_section as P
    said = []
    monkeypatch.setattr(A, "say", said.append)
    clock = iter(range(0, 10 ** 9, 1000))
    monkeypatch.setattr(P, "time", types.SimpleNamespace(time=lambda: next(clock)))
    progress = P._say_how_long(np.zeros((16, 2)), 600, torux.Source("gone", []))
    for done in (2, 10, 40):
        progress(done, np.array([True] * 16))
    assert said == []
    progress = P._say_how_long(np.zeros((16, 2)), 600, torux.Source("staying", []))
    for done in (2, 10, 40):
        progress(done, np.array([True] * 15 + [False]))
    assert len(said) == 1 and "staying: tracing 16 field lines through 600 field periods, about 378 minutes" in said[0]


EIGHT = [f"{1.0 + 0.01 * k:g},0" for k in range(8)]       # eight lines on surfaces of QUASR 954


def test_ctrl_c_stops_a_trace_within_seconds(quasr_954, ctrl_c, monkeypatch):
    """Ctrl-C a second into the tracing of lines that take half a minute or more: a plain KeyboardInterrupt within
    three seconds."""
    import torux.ask as A
    import torux.poincare_section as P
    sent = []
    monkeypatch.setattr(P, "SAY_AFTER", 0.0)
    monkeypatch.setattr(A, "say", lambda text: sent.append(ctrl_c(1.0)))
    with pytest.raises(KeyboardInterrupt):
        torux.poincare(quasr_954, starts=[tuple(map(float, s.split(","))) for s in EIGHT], turns=2500)
    assert sent and sent[0] and time.time() - sent[0][0] < 3.0, time.time() - sent[0][0]


def test_ctrl_c_ends_the_command_in_one_line():
    """torux poincare in a terminal: Ctrl-C while its lines are traced ends it within three seconds in one line, with
    the code a shell gives a command Ctrl-C ended, and no traceback."""
    # Python's own Ctrl-C handler, as a terminal's shell leaves it, whatever these tests were started with
    args = [sys.executable, "-c", "import signal, sys; signal.signal(signal.SIGINT, signal.default_int_handler); "
            "from torux.cli import main; sys.exit(main())", "poincare", example("quasr_0000954.json"),
            "--turns", "6000"]
    for s in EIGHT:
        args += ["--start", s]
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        said = p.stderr.readline()                  # "torux: ...: tracing 8 field lines through 12000 field periods"
        assert "tracing" in said, said
        time.sleep(1.0)
        t = time.time()
        p.send_signal(signal.SIGINT)
        p.wait(timeout=60)
        took = time.time() - t
    finally:
        if p.poll() is None:
            p.kill()
    err = p.stderr.read()
    assert p.returncode == 130 and took < 3.0, (p.returncode, took)
    assert err.strip() == "stopped: nothing written", err


# ------------------------------------------------------------------------------------------------ the knobs
def test_planes_lines_turns_and_starts_are_the_only_knobs(quasr_954):
    p = torux.poincare(quasr_954, planes=[0.3, 0.1], starts=[(1.0, 0.0), (1.05, 0.02)], turns=40)
    assert p.R.shape[:2] == (2, 2) and p.R.shape[2] >= 80 and np.allclose(p.planes, [0.3, 0.1])
    assert np.allclose(p.start, [(1.0, 0.0), (1.05, 0.02)])
    q = torux.poincare(quasr_954, lines=4, turns=30)
    assert q.R.shape[1] == 3 and q.R.shape[2] >= 60 and len(q.start) >= 4
    with pytest.raises(TypeError):
        torux.poincare(quasr_954, steps=10)


@pytest.mark.parametrize("kw, words", [
    (dict(planes=[0.0, np.pi]), "the same plane"),
    (dict(planes=[]), "one or more planes"),
    (dict(lines=0), "at least 1"),
    (dict(turns=-1), "positive"),
    (dict(starts=[(1.0, 0.0)], lines=3), "not both"),
    (dict(starts=[(-1.0, 0.0)]), "R > 0"),
    (dict(starts=[(5.0, 0.0)]), "outside the machine"),
    (dict(planes=[0, 45, 90]), "radians"),
    (dict(turns=0.2), "at least one: 0.5 turns"),
])
def test_what_cannot_be_traced_is_said(quasr_954, kw, words):
    with pytest.raises(ValueError, match=words):
        torux.poincare(quasr_954, **kw)


def test_an_answer_is_sectioned_by_its_own_poincare(solved_954):
    with pytest.raises(TypeError, match=r"answer\.poincare\(\)"):
        torux.poincare(solved_954)


def test_a_source_whose_lines_do_not_go_round_says_so_before_any_tracing(monkeypatch):
    """DIII-D's I-coils: torux.solve's proof in its own words, and no estimate of the time first. One ring, a dipole
    torux is not told is one: no axis and no line round the machine, said in a sentence."""
    import torux.ask as A
    import torux.poincare_section as P
    from torux.source import Rings, Source
    said = []
    monkeypatch.setattr(P, "SAY_AFTER", 0.0)
    monkeypatch.setattr(A, "say", said.append)
    with pytest.raises(ValueError, match="no field line goes round the machine"):
        torux.poincare(example("d3d_icoils_upper.dat"))
    ring = Source(id="ring", parts=[Rings(R=[1.0], Z=[0.0], current=[1e5])], nfp=1, stellsym=True)
    with pytest.raises(ValueError, match="no section to trace: torux found no magnetic axis"):
        torux.poincare(ring)
    assert said == []
    a = torux.solve(example("d3d_icoils_upper.dat"))
    with pytest.raises(ValueError, match="no field line goes round the machine") as e:
        a.poincare()
    assert "solve" not in str(e.value)


@needs_mpl
def test_lines_of_a_field_without_an_axis_start_across_the_machine(monkeypatch, tmp_path):
    """A tokamak's 1/R field and two rings, torux not told: lines start across the box round the conductors, not
    within a few centimetres of the rings' radius, they leave as they do, the call says no time it will not take, and
    the picture has its panels."""
    from torux.source import Rings, Source, ToroidalField
    src = Source(id="tokamak", parts=[ToroidalField(RB=2.0), Rings(R=[2.0, 2.0], Z=[0.8, -0.8], current=[5e4, 5e4])],
                 nfp=1, stellsym=True)
    import torux.ask as A
    said = []
    monkeypatch.setattr(A, "say", said.append)
    p = torux.poincare(src)
    assert said == []                           # seconds of tracing: no word of minutes first
    assert p.axis is None and np.ptp(p.start[:, 0]) > 0.5
    fig = p.plot()
    assert all(any(len(c.get_offsets()) for c in ax.collections) for ax in fig.axes)
    assert all(ax.get_position().height > 0.3 for ax in fig.axes)


# ------------------------------------------------------------------------------------------------ the picture
@needs_mpl
def test_crossings_are_ink_and_green_is_only_ever_a_verified_surface(solved_954, section_954, tmp_path):
    """The crossings are drawn in the page's ink, but those of a line on a surface whose curve has not closed, in its
    grey. The answer's surface is green where it is verified, and the same section of a torus not verified is grey
    and dashed, and no pixel is green."""
    import copy
    from PIL import Image
    import torux.plot as P
    from test_plot import _green
    fig = section_954.plot()
    ink = matplotlib.colors.to_rgba(P.PAGES[False]["ink"])
    for ax in fig.axes:
        dots = [c for c in ax.collections if len(c.get_offsets())]
        grey = matplotlib.colors.to_rgba(P.PAGES[False]["muted"])
        assert dots and np.allclose(dots[0].get_facecolor()[0], ink)
        assert all(np.allclose(c.get_facecolor()[0], grey) for c in dots[1:])
        assert len(dots) <= 1 + section_954.open.any()
        assert ax.lines[0].get_color() == P.PAGES[False]["line"] and ax.lines[0].get_linestyle() == "-"
    for dark in (False, True):
        section_954.plot(tmp_path / "v.png", dark=dark)
        with Image.open(tmp_path / "v.png") as im:
            assert _green(np.asarray(im.convert("RGB")).astype(int)) > 2_000
    not_verified = copy.copy(section_954)
    a = copy.copy(solved_954)
    a.evidence = copy.deepcopy(solved_954.evidence)
    a.evidence["independent"]["ok"] = False
    a.status = "surface_unverified"
    not_verified._answer = a
    fig = not_verified.plot()
    assert fig.axes[0].lines[0].get_linestyle() == "--"
    for dark in (False, True):
        not_verified.plot(tmp_path / "n.png", dark=dark)
        with Image.open(tmp_path / "n.png") as im:
            assert _green(np.asarray(im.convert("RGB")).astype(int)) == 0


@needs_mpl
@pytest.mark.parametrize("ext", [".png", ".pdf", ".svg"])
def test_with_a_path_the_file_is_written_and_its_path_comes_back(section_954, tmp_path, ext):
    path = tmp_path / ("section" + ext)
    assert section_954.plot(path) == path and path.stat().st_size > 10_000


@needs_mpl
def test_the_same_page_on_every_side_and_words_only_on_the_frame(section_954, tmp_path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.text import Text
    from PIL import Image
    import torux.plot as P
    from test_plot import _ink_margins
    fig = section_954.plot()
    FigureCanvasAgg(fig).draw()
    said = {t.get_text() for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()}
    frame = {"φ = 0°", "φ = 45°", "φ = 90°", "Z (m)", "R (m)"}
    assert frame <= said
    assert all(t.replace("−", "").replace(".", "").isdigit() for t in said - frame)
    for dark in (False, True):
        section_954.plot(tmp_path / "x.png", dark=dark)
        with Image.open(tmp_path / "x.png") as im:
            img = np.asarray(im.convert("RGB")).astype(int)
        assert img.shape[1] == round(P.WIDTH * P.FILE_DPI)
        m = _ink_margins(img)
        assert all(abs(v - P.MARGIN * P.FILE_DPI) <= 2 for v in m), (dark, m)


def test_without_matplotlib_one_sentence_names_the_install(monkeypatch, section_954):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    with pytest.raises(ImportError, match=r'pip install "torux\[plot\]"'):
        section_954.plot("x.png")


# ------------------------------------------------------------------------------------------------ the shell
def test_the_command_line(tmp_path):
    path = tmp_path / "q954.png"
    r = subprocess.run([sys.executable, "-m", "torux.cli", "poincare", example("quasr_0000954.json"), str(path),
                        "--lines", "3", "--turns", "30"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "rotational transform" in r.stdout and path.stat().st_size > 10_000 if HAVE_MPL else True
    r = subprocess.run([sys.executable, "-m", "torux.cli", "poincare", example("quasr_0000954.json"), "--start",
                        "1.0"], capture_output=True, text=True)
    assert r.returncode == 1 and "--start takes R,Z" in r.stderr


def _cli(*args):
    return subprocess.run([sys.executable, "-m", "torux.cli", "poincare"] + list(args), capture_output=True,
                          text=True)


def test_the_command_line_reads_as_its_usage_says(tmp_path):
    """SOURCE [SOURCE ...] [FILE] in --help. A picture torux cannot draw is refused as one, a folder that is not
    there before any tracing, and FILE after --planes' angles is FILE."""
    r = _cli("--help")
    assert "SOURCE [SOURCE ...] [FILE]" in r.stdout and "[--start R,Z]" in r.stdout
    r = _cli(example("quasr_0000954.json"), str(tmp_path / "x.jpg"), "--lines", "2", "--turns", "10")
    assert r.returncode == 1 and "name a .png, .pdf or .svg file" in r.stderr and not r.stdout
    r = _cli(example("quasr_0000954.json"), "/nonexistent-dir/x.png", "--lines", "2", "--turns", "10")
    assert r.returncode == 1 and "there is no folder '/nonexistent-dir'" in r.stderr and not r.stdout
    r = _cli(example("quasr_0000954.json"), "--turns", "0.2")
    assert r.returncode == 1 and "at least one" in r.stderr
    if HAVE_MPL:
        f = tmp_path / "two.png"
        r = _cli(example("quasr_0000954.json"), "--planes", "0", "45", str(f), "--lines", "2", "--turns", "10")
        assert r.returncode == 0 and "phi = 0°, 45°" in r.stdout and f.stat().st_size > 10_000, r.stderr


# ------------------------------------------------------------------------------------------------ every design
def chain_of(p, i, plane=0, q_max=12):
    """The islands line i goes round in a plane: q where its mean turn about the axis a field period (the weighted
    Birkhoff average of its crossings' turns) is a rational p/q, q <= q_max, to 1e-4, and every q-th crossing stays in
    one island, an arc of angle about the axis narrower than the 2 pi / q of q islands side by side. Else 0."""
    from fractions import Fraction
    R, Z = p.R[i, plane, :p.followed[i]], p.Z[i, plane, :p.followed[i]]
    if p.left[i] or not np.all(np.isfinite(R)):
        return 0
    th = np.unwrap(np.arctan2(Z - p.axis[plane, 1], R - p.axis[plane, 0]))
    d = np.diff(th)
    t = (np.arange(len(d)) + 0.5) / len(d)
    w = np.exp(-1.0 / (t * (1.0 - t)))
    rho = float(np.sum(w * d) / np.sum(w) / (2 * np.pi))
    f = Fraction(rho).limit_denominator(q_max)
    q = f.denominator
    if abs(rho - float(f)) > 1e-4 or q < 2:
        return 0
    every = th[::q] - 2 * np.pi * f.numerator * np.arange(len(th[::q]))
    return q if np.ptp(every) < 2 * np.pi / q else 0


def test_surfaces_end_well_inside_the_coils_and_beyond_them_lines_wander_and_leave():
    """QUASR 1100073: the inner half of the lines stays, most of it with a rotational transform. Lines leave only from
    the outer half, and out there a line goes round a chain of 11 islands."""
    p = torux.poincare(example("quasr_1100073.json"))
    d = np.hypot(p.start[:, 0] - p.axis[0, 0], p.start[:, 1] - p.axis[0, 1])
    inner = d < np.median(d)
    assert not p.left[inner].any() and np.isfinite(p.iota[inner]).sum() >= 0.5 * inner.sum()
    assert p.left.sum() >= 2 and not np.any(p.left & inner)
    assert any(chain_of(p, i) == 11 for i in range(len(p.start)))
    # every line that stayed has its transform: those near the edge's chains are followed on until it settles
    assert np.all(np.isfinite(p.iota[~p.left])), p.lines_table()


@pytest.mark.slow
def test_wendelstein_7x_shows_its_five_edge_islands():
    """W7-X standard: three lines turn once a toroidal turn (1/5 a field period) round its five edge islands at
    phi = 0, one of them clearly so in every plane."""
    p = torux.poincare(example("w7x.json"))
    chain = [i for i in range(len(p.start)) if chain_of(p, i) == 5]
    assert len(chain) >= 3
    assert any(all(chain_of(p, i, j) == 5 for j in range(3)) for i in chain)
    # their transform is the chain's, 1, in every digit printed, and no more digits are printed than have settled
    rows = p.lines_table().splitlines()[1:]
    for i in chain:
        d = len(rows[i].split()[3].split(".")[1])
        assert p.iota[i] == round(1.0, d) and d < 12, rows[i]


@pytest.mark.slow
def test_quasr_797878_shows_its_wide_chain():
    """QUASR 797878: lines round its wide chain of six islands, in every plane, and four or more of them with the
    chain's transform, settled (a line that wanders along the chain's separatrix stays in it too, with none)."""
    p = torux.poincare("quasr:797878")
    chain = [i for i in range(len(p.start)) if all(chain_of(p, i, j) == 6 for j in range(3))]
    settled = [i for i in chain if np.isfinite(p.iota[i])]
    assert len(settled) >= 4, p.lines_table()
    assert all(abs(p.iota[i] + 0.5) < 1e-4 for i in settled), p.lines_table()


def test_a_line_on_a_surface_past_the_conductors_has_not_left():
    """CNT at 90 degrees, I_IL / I_PF = 4.70 (IL centres 0.38475 m): the coils reach |Z| = 0.405 m, the iota = -6/19
    island |Z| = 0.501 m and torux's verified surface 0.481 m. A line has left only once it is half the conductors'
    extent past them (in 0.2.2, 5 %: |Z| <= 0.458 m, and those lines were said to have left)."""
    import torux.poincare_section as P
    from test_crossed import _cnt
    box, size = P._box(_cnt(90, 4.70, d=0.38475))
    tb = P._traced_box(box, size)
    assert box[3] < 0.501 and tb[3] > 0.501 + 0.25 and tb[2] < -0.501 - 0.25 and tb[0] >= 1e-6
    assert np.allclose(tb, [max(1e-6, box[0] - 0.5 * size), box[1] + 0.5 * size, box[2] - 0.5 * size,
                            box[3] + 0.5 * size])
