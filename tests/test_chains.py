"""Island chains as torux.chains measures them: the pendulum half width of a chain against the pendulum's own, and
the family torus its shear is taken at."""
import types

import numpy as np

from torux import chains

NFP = 2


def test_a_chain_says_which_family_torus_its_shear_is_from(monkeypatch):
    """torux.chains.analyse_chain: the shear of the pendulum half width is the family's at one torus, and the record
    says which (its index, flux and iota)."""
    monkeypatch.setattr(chains, "chain_lines", lambda *a, **k: [])
    tori = [types.SimpleNamespace(flux=f, iota=i, nfp=NFP) for f, i in
            ((0.00065, -0.33033), (0.00099, -0.33148), (0.00129, -0.33253), (0.00183, -0.33423))]
    ch = chains.analyse_chain(None, tori, -1, 6)
    k = int(np.argmin([abs(t.iota + 1 / 3) for t in tori]))
    assert ch.info["shear_at"] == dict(index=k, n_tori=4, flux=tori[k].flux, iota=tori[k].iota)
    fl, io = np.array([t.flux for t in tori]), np.array([t.iota for t in tori])
    assert ch.shear == float(np.gradient(io, fl)[k])


def _pendulum(q, p, nfp, shear_J, eps):
    """Field lines of one resonant harmonic on a sheared transform, phi as time, J = Phi / 2 pi:
    theta' = iota_r + shear_J (J - J0), J' = q eps sin(q theta - p nfp phi). The residue of its O line (theta = pi / q
    at phi = 0, J = J0) over the q field periods it takes to close, from the variational equations integrated by
    SciPy, and the separatrix half width in Phi of the pendulum it is, 2 pi 2 sqrt(eps / shear_J)."""
    from scipy.integrate import solve_ivp
    iota_r = p * nfp / q

    def f(phi, y):
        th, J = y[0], y[1]
        psi = q * th - p * nfp * phi
        M = y[2:].reshape(2, 2)
        A = np.array([[0.0, shear_J], [q * q * eps * np.cos(psi), 0.0]])
        return np.concatenate([[iota_r + shear_J * J, q * eps * np.sin(psi)], (A @ M).ravel()])
    y0 = np.concatenate([[np.pi / q, 0.0], np.eye(2).ravel()])
    sol = solve_ivp(f, (0.0, 2 * np.pi * q / nfp), y0, rtol=1e-12, atol=1e-15, method="DOP853")
    M = sol.y[2:, -1].reshape(2, 2)
    assert abs(sol.y[0, -1] - (np.pi / q + 2 * np.pi * p)) < 1e-9 and abs(sol.y[1, -1]) < 1e-12   # it closed
    return (2.0 - np.trace(M)) / 4.0, 2 * np.pi * 2 * np.sqrt(eps / shear_J)


def test_the_pendulum_half_width_is_the_pendulums(monkeypatch):
    """torux.chains' half width of a chain, from its O line's residue and the family's shear, against the separatrix
    of the pendulum whose lines have that residue: equal to the small-residue order, for chains of 1 to 23 islands.
    The family's tori sample iota = iota_r + shear (Phi - Phi_r)."""
    nfp = 2
    for p, q, shear_Phi, eps in ((-1, 6, -3.3, -2e-9), (1, 1, 5.0, 3e-10), (-4, 23, -6.2, -1e-11), (2, 5, 0.8, 1e-9)):
        shear_J = 2 * np.pi * shear_Phi
        R, width = _pendulum(q, p, nfp, shear_J, eps)
        assert 0 < R < 1e-3
        Fr = 0.0015
        tori = [types.SimpleNamespace(flux=Fr + d, iota=p * nfp / q + shear_Phi * d, nfp=nfp)
                for d in (-6e-4, -3e-4, 2e-4, 5e-4)]
        monkeypatch.setattr(chains, "chain_lines", lambda *a, R=R, **k: [types.SimpleNamespace(residue=R)])
        ch = chains.analyse_chain(None, tori, p, q)
        assert abs(ch.shear - shear_Phi) < 1e-9 * abs(shear_Phi) and abs(ch.flux_r - Fr) < 1e-12
        assert abs(ch.half_width_est - width) < 1e-3 * width, (p, q, ch.half_width_est, width)


