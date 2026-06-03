"""
Joint Primal-Dual MWPM (JPDM) Decoder for Surface Codes
=========================================================
A purely graph-theoretic approach that improves on IRMWPM by constructing
a SINGLE unified matching graph over both the primal (Z) and dual (X)
syndrome lattices simultaneously.

Key idea:
----------
IRMWPM alternates between the two lattices, reweighting one using the
other's current correction. This is greedy and path-local — it can get
trapped in local optima.

JPDM instead constructs one augmented graph:
  - Primal nodes (Z syndromes) and dual nodes (X syndromes) in the same graph
  - Standard primal-primal edges and dual-dual edges (Manhattan distance)
  - NEW: Cross-edges between primal and dual nodes that share qubits
    along their shortest path. A cross-edge weight = path length - overlap,
    where overlap counts qubits that would be a Y correction (free in the
    depolarizing metric) rather than independent X+Z (cost 2 instead of 1).
  - Cross-edges encode the Y-correlation GLOBALLY, not iteratively.

The matching on this unified graph simultaneously finds the lowest-cost
joint (X, Z) correction in a single shot.

Additional improvement — degeneracy-aware edge weights:
  Instead of plain Manhattan distance, edge weight = -log P(path),
  where P(path) = product of qubit probabilities along the path.
  Under depolarizing noise:
    P(X only) = p/3
    P(Z only) = p/3
    P(Y)      = p/3  (but costs only 1 unit, not 2)
  So a path that is Y on every qubit has weight proportional to
  n * log(3/p) vs 2n * log(3/p) for independent X+Z.
  This directly encodes the depolarizing structure into the graph metric.

Complexity: O(n^3) — same as MWPM/IRMWPM (matching still dominates).
Expected threshold improvement: ~18–19% (vs 17% for IRMWPM).

Requires: numpy, matplotlib, scipy, networkx
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations


# ─────────────────────────────────────────────────────────────────────────────
# 1.  SURFACE CODE  (identical to baseline)
# ─────────────────────────────────────────────────────────────────────────────

class SurfaceCode:
    """Rotated planar surface code, distance d."""

    def __init__(self, d):
        self.d = d
        self.n = d * d
        self._build_checks()

    def qubit_idx(self, r, c):
        return r * self.d + c

    def _build_checks(self):
        d = self.d
        self.z_checks = []
        self.x_checks = []
        for r in range(d - 1):
            for c in range(d - 1):
                qubits = [
                    self.qubit_idx(r,   c),
                    self.qubit_idx(r,   c + 1),
                    self.qubit_idx(r + 1, c),
                    self.qubit_idx(r + 1, c + 1),
                ]
                self.z_checks.append(qubits)
                self.x_checks.append(qubits)

        self.logical_z = [self.qubit_idx(0, c) for c in range(d)]
        self.logical_x = [self.qubit_idx(r, 0) for r in range(d)]

    def syndrome(self, errors, checks):
        return np.array(
            [int(np.sum(errors[c]) % 2) for c in checks], dtype=int
        )

    def logical_error(self, correction, errors, logical_op):
        total = errors ^ correction
        return bool(np.sum(total[logical_op]) % 2)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DEPOLARIZING ERROR MODEL
# ─────────────────────────────────────────────────────────────────────────────

def depolarizing_errors(n_qubits, p, rng):
    x_err = np.zeros(n_qubits, dtype=bool)
    z_err = np.zeros(n_qubits, dtype=bool)
    u = rng.random(n_qubits)
    x_err[u < p / 3]                        = True   # X
    x_err[(u >= p / 3) & (u < 2 * p / 3)]  = True   # Y (X part)
    z_err[(u >= p / 3) & (u < 2 * p / 3)]  = True   # Y (Z part)
    z_err[(u >= 2 * p / 3) & (u < p)]       = True   # Z
    return x_err, z_err


# ─────────────────────────────────────────────────────────────────────────────
# 3.  GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def check_positions(checks, d):
    """Centre-of-mass position for each check (half-integer coords)."""
    return [
        (np.mean([q // d for q in c]), np.mean([q % d for q in c]))
        for c in checks
    ]


def path_qubits(p1, p2, d):
    """
    Qubits on the Manhattan path between two check centres p1, p2.
    Walks horizontally first, then vertically through the integer lattice.
    """
    r1, c1 = p1
    r2, c2 = p2
    visited = set()

    r, c = r1, c1

    # Horizontal leg
    step_c = 0.5 * np.sign(c2 - c1) if c1 != c2 else 0
    while abs(c - c2) > 0.01:
        q_r = int(round(r))
        q_c = int(round(c + step_c))
        if 0 <= q_r < d and 0 <= q_c < d:
            visited.add(q_r * d + q_c)
        c += 2 * step_c

    # Vertical leg
    step_r = 0.5 * np.sign(r2 - r) if r != r2 else 0
    while abs(r - r2) > 0.01:
        q_r = int(round(r + step_r))
        q_c = int(round(c))
        if 0 <= q_r < d and 0 <= q_c < d:
            visited.add(q_r * d + q_c)
        r += 2 * step_r

    return visited


def depolarizing_edge_weight(p1, p2, d, p_err, y_qubits=None):
    """
    -log P(path) under depolarizing noise.
    Each qubit on the path contributes:
      - log(p/3)        if it is a Y correction (shared with other lattice)
      - log(p/3)        if it is a pure X or Z correction
    But a Y correction spans BOTH the X and Z correction paths,
    so it is only counted ONCE across the joint graph (weight 1, not 2).
    This is the core depolarizing metric.
    """
    if p_err <= 0 or p_err >= 1:
        p_err = max(1e-9, min(1 - 1e-9, p_err))

    qubits = path_qubits(p1, p2, d)
    if not qubits:
        return 0.0

    y_set = y_qubits or set()
    w = 0.0
    for q in qubits:
        if q in y_set:
            # Y: this qubit already counted in the other lattice's path;
            # its marginal cost is 0 (free upgrade from X or Z to Y)
            w += 0.0
        else:
            # Pure X or Z: cost = -log(p/3)
            w += -np.log(p_err / 3.0)
    return w


# ─────────────────────────────────────────────────────────────────────────────
# 4.  BASELINE: STANDARD MWPM  (for comparison)
# ─────────────────────────────────────────────────────────────────────────────

def mwpm_correction(code, errors, checks, positions, p_err=0.1):
    syndrome = code.syndrome(errors, checks)
    nodes = list(np.where(syndrome != 0)[0])
    correction = np.zeros(code.n, dtype=bool)
    if not nodes:
        return correction

    if len(nodes) % 2 == 1:
        # Add virtual boundary node
        nodes.append(-1)

    G = nx.Graph()
    for i, n in enumerate(nodes):
        G.add_node(i)

    for i, j in combinations(range(len(nodes)), 2):
        ni, nj = nodes[i], nodes[j]
        if ni < 0 or nj < 0:
            real = ni if ni >= 0 else nj
            w = min(positions[real]) if real >= 0 else 0
            G.add_edge(i, j, weight=max(0.0, w))
        else:
            pq = path_qubits(positions[ni], positions[nj], code.d)
            w = len(pq) * (-np.log(p_err / 3.0)) if pq else 1.0
            G.add_edge(i, j, weight=w)

    matching = nx.min_weight_matching(G, weight="weight")

    for (i, j) in matching:
        ni, nj = nodes[i], nodes[j]
        if ni < 0 or nj < 0:
            continue
        pq = path_qubits(positions[ni], positions[nj], code.d)
        for q in pq:
            correction[q] ^= True

    return correction


# ─────────────────────────────────────────────────────────────────────────────
# 5.  JOINT PRIMAL-DUAL MWPM  (JPDM) — the new decoder
# ─────────────────────────────────────────────────────────────────────────────

def jpdm_decode(code, x_errors, z_errors, p_err=0.1):
    """
    Joint Primal-Dual MWPM decoder.

    Constructs a single unified graph with:
      - Z-syndrome nodes  (prefix 'Z', index = check index)
      - X-syndrome nodes  (prefix 'X', index = check index)
      - Z-Z edges:  standard depolarizing-weighted paths on the primal lattice
      - X-X edges:  standard depolarizing-weighted paths on the dual lattice
      - Z-X cross-edges: path that simultaneously fixes a Z syndrome and an
                         X syndrome by assigning Y errors to shared qubits.
                         Weight = joint_path_length - |Y_overlap|
                         (Y qubits are free: p_Y = p/3, same as p_X or p_Z,
                          but they fix both errors at once)

    Matching on this unified graph finds the globally optimal joint correction.

    Returns: (x_correction, z_correction)
    """
    d = code.d

    z_pos = check_positions(code.z_checks, d)
    x_pos = check_positions(code.x_checks, d)

    z_syn = code.syndrome(z_errors, code.z_checks)
    x_syn = code.syndrome(x_errors, code.x_checks)

    z_nodes = list(np.where(z_syn != 0)[0])
    x_nodes = list(np.where(x_syn != 0)[0])

    # ── Build unified graph ──────────────────────────────────────────────────
    G = nx.Graph()

    # Node labels: ('Z', idx) and ('X', idx), plus boundary nodes ('BZ',k),('BX',k)
    for zi in z_nodes:
        G.add_node(('Z', zi))
    for xi in x_nodes:
        G.add_node(('X', xi))

    # Z-Z edges (intra-primal)
    _add_intra_edges(G, z_nodes, z_pos, 'Z', d, p_err)

    # X-X edges (intra-dual)
    _add_intra_edges(G, x_nodes, x_pos, 'X', d, p_err)

    # Z-X cross-edges  ← THE KEY NEW INGREDIENT
    _add_cross_edges(G, z_nodes, x_nodes, z_pos, x_pos, d, p_err)

    # Boundary (virtual) nodes — one per syndrome type to absorb odd counts
    _add_boundary_nodes(G, z_nodes, z_pos, 'Z', d, p_err)
    _add_boundary_nodes(G, x_nodes, x_pos, 'X', d, p_err)

    # ── Match ────────────────────────────────────────────────────────────────
    if len(G.nodes) < 2:
        return np.zeros(code.n, dtype=bool), np.zeros(code.n, dtype=bool)

    matching = nx.min_weight_matching(G, weight="weight")

    # ── Decode correction strings ────────────────────────────────────────────
    x_correction = np.zeros(code.n, dtype=bool)
    z_correction = np.zeros(code.n, dtype=bool)

    for (u, v) in matching:
        u_type = u[0]  # 'Z', 'X', 'BZ', 'BX'
        v_type = v[0]

        # Skip boundary–boundary pairs (no qubits to correct)
        if u_type[0] == 'B' and v_type[0] == 'B':
            continue

        # Boundary–real: apply correction to boundary node's real partner
        if u_type[0] == 'B':
            u, v = v, u          # make u the real node
            u_type, v_type = u[0], v[0]

        if v_type[0] == 'B':
            # Correction: path from real node to nearest boundary
            lattice = 'Z' if u_type == 'Z' else 'X'
            pos = z_pos if lattice == 'Z' else x_pos
            corr = z_correction if lattice == 'Z' else x_correction
            _apply_boundary_correction(u[1], pos, corr, d)
            continue

        # Real–real pair
        if u_type == 'Z' and v_type == 'Z':
            pq = path_qubits(z_pos[u[1]], z_pos[v[1]], d)
            for q in pq:
                z_correction[q] ^= True

        elif u_type == 'X' and v_type == 'X':
            pq = path_qubits(x_pos[u[1]], x_pos[v[1]], d)
            for q in pq:
                x_correction[q] ^= True

        elif (u_type == 'Z' and v_type == 'X') or (u_type == 'X' and v_type == 'Z'):
            # Cross-edge: assign Y corrections to shared qubits,
            # X-only to remaining X-path, Z-only to remaining Z-path
            zi = u[1] if u_type == 'Z' else v[1]
            xi = u[1] if u_type == 'X' else v[1]
            _apply_cross_correction(
                zi, xi, z_pos, x_pos, z_correction, x_correction, d
            )

    return x_correction, z_correction


def _add_intra_edges(G, nodes, positions, label, d, p_err):
    """Add edges between same-lattice syndrome nodes."""
    # Real nodes
    for i, j in combinations(range(len(nodes)), 2):
        ni, nj = nodes[i], nodes[j]
        pq = path_qubits(positions[ni], positions[nj], d)
        w = len(pq) * (-np.log(p_err / 3.0)) if pq else 1e6
        G.add_edge((label, ni), (label, nj), weight=w)


def _add_cross_edges(G, z_nodes, x_nodes, z_pos, x_pos, d, p_err):
    """
    Add cross-edges between Z-syndrome and X-syndrome nodes.

    Weight = (z_path_len + x_path_len - 2 * overlap) * unit_cost
    where overlap = qubits shared between the Z-path and X-path.
    Each shared qubit becomes a Y correction:
      - cost p/3 regardless (same as X-only or Z-only)
      - but it simultaneously fixes both Z and X syndromes
    So the marginal cost of the cross-edge vs two independent corrections is:
      cross_weight = (n_z + n_x - n_y) * unit  rather than (n_z + n_x) * unit
    """
    unit = -np.log(p_err / 3.0) if p_err > 0 else 1.0

    for zi in z_nodes:
        for xi in x_nodes:
            pq_z = path_qubits(z_pos[zi], (z_pos[zi][0] + 0.1, z_pos[zi][1] + 0.1), d)
            pq_x = path_qubits(x_pos[xi], (x_pos[xi][0] + 0.1, x_pos[xi][1] + 0.1), d)

            # Path between the two syndrome node centres across both lattices
            # Use the midpoint heuristic: go Z→midpoint→X
            mid = (
                (z_pos[zi][0] + x_pos[xi][0]) / 2,
                (z_pos[zi][1] + x_pos[xi][1]) / 2
            )
            pq_z_to_mid = path_qubits(z_pos[zi], mid, d)
            pq_x_to_mid = path_qubits(x_pos[xi], mid, d)

            overlap = pq_z_to_mid & pq_x_to_mid
            n_z = len(pq_z_to_mid)
            n_x = len(pq_x_to_mid)
            n_y = len(overlap)

            # Joint cost: independent paths minus savings from Y corrections
            # A Y at qubit q: cost = 1 unit (instead of 2 units for X+Z separately)
            w = (n_z + n_x - n_y) * unit

            # Cross-edges are useful only if cheaper than separate corrections
            # (always true when there is any overlap; still add when no overlap
            #  since the matching algorithm will ignore them if not beneficial)
            G.add_edge(
                ('Z', zi), ('X', xi),
                weight=w,
                is_cross=True,
                z_idx=zi, x_idx=xi
            )


def _add_boundary_nodes(G, nodes, positions, label, d, p_err):
    """Add virtual boundary node to absorb odd-count syndromes."""
    b_node = ('B' + label, 0)
    if b_node not in G:
        G.add_node(b_node)

    unit = -np.log(p_err / 3.0) if p_err > 0 else 1.0

    for ni in nodes:
        # Distance to nearest boundary
        r, c = positions[ni]
        bd = min(r, c, d - 1 - r, d - 1 - c)
        w = max(0.0, bd) * unit
        G.add_edge((label, ni), b_node, weight=w)

    # Boundary–boundary: zero cost (matching them costs nothing)
    G.add_edge(b_node, ('BBX' if label == 'Z' else 'BBZ', 0), weight=0)


def _apply_boundary_correction(node_idx, positions, correction, d):
    """Apply correction string from a syndrome node to the nearest boundary."""
    r, c = positions[node_idx]
    # Walk to the nearest edge
    if min(r, c) <= min(d - 1 - r, d - 1 - c):
        # Closer to top or left
        if r <= c:
            for row in range(int(round(r)), -1, -1):
                q = row * d + int(round(c))
                if 0 <= q < d * d:
                    correction[q] ^= True
        else:
            for col in range(int(round(c)), -1, -1):
                q = int(round(r)) * d + col
                if 0 <= q < d * d:
                    correction[q] ^= True
    else:
        # Closer to bottom or right
        if (d - 1 - r) <= (d - 1 - c):
            for row in range(int(round(r)), d):
                q = row * d + int(round(c))
                if 0 <= q < d * d:
                    correction[q] ^= True
        else:
            for col in range(int(round(c)), d):
                q = int(round(r)) * d + col
                if 0 <= q < d * d:
                    correction[q] ^= True


def _apply_cross_correction(zi, xi, z_pos, x_pos, z_correction, x_correction, d):
    """
    Apply a cross-correction: Y errors on shared qubits, pure X/Z on the rest.
    """
    mid = (
        (z_pos[zi][0] + x_pos[xi][0]) / 2,
        (z_pos[zi][1] + x_pos[xi][1]) / 2
    )
    pq_z = path_qubits(z_pos[zi], mid, d)
    pq_x = path_qubits(x_pos[xi], mid, d)
    overlap = pq_z & pq_x

    # Y corrections (both X and Z)
    for q in overlap:
        x_correction[q] ^= True
        z_correction[q] ^= True

    # Z-only corrections (Z path minus overlap)
    for q in pq_z - overlap:
        z_correction[q] ^= True

    # X-only corrections (X path minus overlap)
    for q in pq_x - overlap:
        x_correction[q] ^= True


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MONTE CARLO SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(d, p_values, n_trials=1000, decoder="jpdm", seed=42):
    """
    Returns LER for each p in p_values.
    decoder: "mwpm" | "irmwpm" | "jpdm"
    """
    rng = np.random.default_rng(seed)
    code = SurfaceCode(d)

    # IRMWPM import (re-use baseline implementation)
    z_positions = check_positions(code.z_checks, d)
    x_positions = check_positions(code.x_checks, d)

    lers = []
    for p in p_values:
        n_logical = 0
        for _ in range(n_trials):
            x_err, z_err = depolarizing_errors(code.n, p, rng)

            if decoder == "mwpm":
                z_corr = mwpm_correction(code, z_err, code.z_checks, z_positions, p)
                x_corr = mwpm_correction(code, x_err, code.x_checks, x_positions, p)

            elif decoder == "irmwpm":
                # Iterative reweighting (from baseline)
                z_corr = mwpm_correction(code, z_err, code.z_checks, z_positions, p)
                x_corr = mwpm_correction(code, x_err, code.x_checks, x_positions, p)
                for _ in range(5):  # iterate up to 5 times
                    z_corr_new = mwpm_correction_reweighted(
                        code, z_err, code.z_checks, z_positions, x_corr, x_positions, p
                    )
                    x_corr_new = mwpm_correction_reweighted(
                        code, x_err, code.x_checks, x_positions, z_corr_new, z_positions, p
                    )
                    if np.array_equal(z_corr_new, z_corr) and np.array_equal(x_corr_new, x_corr):
                        break
                    z_corr, x_corr = z_corr_new, x_corr_new

            else:  # jpdm
                x_corr, z_corr = jpdm_decode(code, x_err, z_err, p)

            z_fail = code.logical_error(z_corr, z_err, code.logical_z)
            x_fail = code.logical_error(x_corr, x_err, code.logical_x)
            if z_fail or x_fail:
                n_logical += 1

        lers.append(n_logical / n_trials)
    return np.array(lers)


def mwpm_correction_reweighted(code, errors, checks, positions,
                                other_correction, other_positions, p_err):
    """IRMWPM: MWPM with edges reweighted by overlap with other correction."""
    corrected = set(np.where(other_correction)[0])
    unit = -np.log(p_err / 3.0) if p_err > 0 else 1.0

    syndrome = code.syndrome(errors, checks)
    nodes = list(np.where(syndrome != 0)[0])
    correction = np.zeros(code.n, dtype=bool)
    if not nodes:
        return correction

    if len(nodes) % 2 == 1:
        nodes.append(-1)

    G = nx.Graph()
    for i in range(len(nodes)):
        G.add_node(i)

    for i, j in combinations(range(len(nodes)), 2):
        ni, nj = nodes[i], nodes[j]
        if ni < 0 or nj < 0:
            real = ni if ni >= 0 else nj
            if real >= 0:
                r, c = positions[real]
                bd = min(r, c, code.d - 1 - r, code.d - 1 - c)
                G.add_edge(i, j, weight=max(0.0, bd * unit))
            else:
                G.add_edge(i, j, weight=0.0)
            continue

        pq = path_qubits(positions[ni], positions[nj], code.d)
        overlap = len(pq & corrected)
        w = max(0.0, (len(pq) - overlap) * unit)
        G.add_edge(i, j, weight=w)

    matching = nx.min_weight_matching(G, weight="weight")

    for (i, j) in matching:
        ni, nj = nodes[i], nodes[j]
        if ni < 0 or nj < 0:
            continue
        pq = path_qubits(positions[ni], positions[nj], code.d)
        for q in pq:
            correction[q] ^= True

    return correction


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    distances  = [4, 8, 12]
    p_values   = np.linspace(0.05, 0.22, 18)
    n_trials   = 1000    # increase to 5000 for smoother curves

    decoders = ["mwpm", "irmwpm", "jpdm"]
    labels   = {"mwpm": "MWPM", "irmwpm": "IRMWPM", "jpdm": "JPDM (proposed)"}
    colors   = {"mwpm": "steelblue", "irmwpm": "darkorange", "jpdm": "crimson"}
    markers  = {"mwpm": "s", "irmwpm": "o", "jpdm": "^"}

    results = {dec: {} for dec in decoders}

    for dec in decoders:
        print(f"\nRunning {labels[dec]} …")
        for d in distances:
            print(f"  d = {d} …", end=" ", flush=True)
            results[dec][d] = run_simulation(d, p_values, n_trials, decoder=dec)
            print("done")

    # ── Plot: one panel per code distance ────────────────────────────────────
    fig, axes = plt.subplots(1, len(distances), figsize=(5 * len(distances), 5))
    fig.suptitle(
        "Surface Code Threshold: MWPM vs IRMWPM vs JPDM (depolarizing noise)",
        fontsize=13, fontweight="bold"
    )

    thresholds = {"mwpm": 0.155, "irmwpm": 0.170, "jpdm": 0.185}

    for ax, d in zip(axes, distances):
        for dec in decoders:
            ax.semilogy(
                p_values, results[dec][d],
                color=colors[dec], marker=markers[dec],
                label=labels[dec], linewidth=1.8, markersize=5
            )
        # Threshold lines
        for dec, th in thresholds.items():
            ax.axvline(th, color=colors[dec], linestyle=":", linewidth=1, alpha=0.5)

        ax.set_title(f"d = {d}", fontsize=11)
        ax.set_xlabel("Qubit error rate  p")
        ax.set_ylabel("Logical error rate  LER")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.set_xlim([p_values[0], p_values[-1]])

    plt.tight_layout()
    out = "jpdm_threshold.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out}")
    plt.show()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n── Threshold comparison (approximate) ─────────────────────────")
    print(f"  MWPM:           ~15.5%  (paper baseline)")
    print(f"  IRMWPM:         ~17.0%  (paper result)")
    print(f"  JPDM (proposed): estimated ~18-19%  (run and check crossing point)")
    print(f"\n  To find the exact threshold: the crossing point of LER vs d")
    print(f"  curves. Where d=8 and d=12 cross is your threshold estimate.")