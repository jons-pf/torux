"""The residue a closed line records, and the error it records with it.

The truth: an independent tracer of the QUASR coils (Biot-Savart by the trapezoid rule with the points per coil doubled
until settled, DOP853 at rtol 1e-13 with the variational equation, Newton on the period map), which shares no code
with torux: on QUASR 1973900 (six field periods) the X line of the chain -8/15 through (0.833164, 0) in phi = 0 has
residue -9.601566e-6 to 1e-11, and its O line 9.601596e-6. torux 0.2.2 stored -1.1333e-5 and 7.879e-6 for them and
recorded a monodromy error of 2.6e-10: its Fourier curve, with the modes of the first solve, missed the line's
harmonics of the field period (the 11th has 1.6e-5 m).
"""
import math
import socket

import numpy as np
import pytest

from torux import barrier as BR
from torux import orbits as OR
from torux.chainlist import _line, provenance
from torux.chainfind import _fourier_guess

TRUE_X = -9.601566e-6          # the independent tracer's, +- 1e-11


def _online():
    try:
        socket.create_connection(("quasr.flatironinstitute.org", 443), timeout=5).close()
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _online(), reason="QUASR's server not reachable")
def test_a_thin_chains_residue_is_its_lines_and_its_error_is_honest():
    import torux
    from torux.field import FieldEngine
    src = torux.load("quasr:1973900")
    nfp, q = int(src.nfp), 15
    eng = FieldEngine(src, dmin=src.meta.get("dmin", 0.05))
    T = 2 * math.pi / nfp
    x0 = np.array([0.833164154979634, 0.0])
    B = np.asarray(eng.evaluate(np.array([[x0[0], 0.0, 0.0]]), grad=False, A=False)["B"])
    sec = BR.FieldSection(eng, src, sign=1.0 if B[0, 1] > 0 else -1.0, k=6, steps_per_period=24, reference=False,
                          scale=0.1)
    tr = sec.line_tracer(k=6, steps_per_period=96)
    o = tr.trace(x0[None], 0.0, q * T, nsteps=q * 96, record=True)
    Y = np.concatenate([[x0], np.array([r[0] for r in o["rec"]])])
    gR, gZ = _fourier_guess(Y[:-1, 0], Y[:-1, 1], q * T, 0.0)
    K0 = max(24, 10 * q + 16)              # the modes torux.chainfind gives a line of 15 periods
    coarse = OR.line_monodromy(eng, OR.solve_closed_line(eng, nfp, q, gR, gZ, K=K0), refine=False)
    assert coarse.converged and abs(coarse.residue - TRUE_X) > 0.1 * abs(TRUE_X)     # 0.2.2's measure: far off
    ln = OR.line_monodromy(eng, OR.solve_closed_line(eng, nfp, q, gR, gZ, K=K0))
    assert ln.K > K0 and ln.residue < 0
    assert abs(ln.residue - TRUE_X) < 1e-4 * abs(TRUE_X)
    # the recorded error bounds the true one (the independent tracer's own error added), and is small
    assert abs(ln.residue - TRUE_X) <= ln.info["residue_err"] + 1e-11
    assert ln.info["residue_err"] < 1e-3 * abs(TRUE_X)
    assert ln.info["monodromy_err"] >= ln.info["monodromy_err_modes"] > 0


def test_a_certificates_lines_say_how_well_their_residues_are_known():
    """torux.chainlist.provenance gives each certificate line's residue_err where the record keeps one, and the chain
    list each line's."""
    pts = [[1.0, 0.0], [0.9, 0.1], [0.9, -0.1]]
    c = dict(ok=True, kind="island_band_manifold", chain=dict(p=-1, q=3), phi_s=0.0, interval=[0.001, 0.0012],
             X_line=dict(residue=-2e-5, residue_err=3e-11, points=pts), O_line=dict(residue=2e-5, points=pts))
    e = provenance(c)
    X, O = (next(ln for ln in e["lines"] if ln["role"] == r) for r in "XO")
    assert X["residue_err"] == 3e-11 and "residue_err" not in O
    assert _line(dict(residue=2e-5, residue_err=1e-12, section=pts))["residue_err"] == 1e-12
    assert "residue_err" not in _line(dict(residue=2e-5, section=pts))


def test_a_residue_within_its_error_gives_no_type_and_no_half_width():
    """A chain's line whose residue lies within the error recorded for it (as on Hammond's second wireframe, where
    0.2.2 called the lines of 4/23 elliptic with residue 1.4e-7 and an independent tracer finds them parabolic to
    2e-13) is listed as unresolved, neither O nor X, and no half width rests on it; the search's words say so."""
    from torux.ask import _unresolved_words
    from torux.chainlist import _crossed
    from torux.orbits import ClosedLine, residue_resolved
    ring = [[1.2 + 0.01 * math.cos(2 * math.pi * k / 5), 0.01 * math.sin(2 * math.pi * k / 5)] for k in range(5)]
    rec = dict(p=2, q=5, flux_r=0.003, shear=-0.36, half_width_est=None, inner_Wb=0.0027, outer_Wb=0.0072,
               between_verified=True, lines=[dict(residue=1.4e-7, residue_err=3e-7, section=ring),
                                             dict(residue=-1.2e-7, residue_err=None, section=ring[1:] + ring[:1])])
    c = _crossed(rec, 2)
    assert c["O"] == [] and c["X"] == [] and len(c["unresolved"]) == 2
    assert "half_width_Wb" not in c and "O_residue" not in c
    assert _unresolved_words(c) == "2 lines whose residues lie within their errors, so no type and no half width"
    # resolved lines keep their type; a line never refined (no error recorded) is taken as it is
    rec["lines"] = [dict(residue=1.4e-5, residue_err=3e-12, section=ring),
                    dict(residue=-1.4e-5, section=ring[1:] + ring[:1])]
    c = _crossed(rec, 2)
    assert len(c["O"]) == 1 and len(c["X"]) == 1 and "unresolved" not in c
    ln = ClosedLine(5, 2, 0.0, np.zeros(3), np.zeros(3), residue=1e-9)
    assert residue_resolved(ln)
    ln.info["residue_err"] = 2e-9
    assert not residue_resolved(ln)
    ln.info["residue_err"] = None
    assert not residue_resolved(ln)
    ln.info["residue_err"] = 1e-12
    assert residue_resolved(ln)
