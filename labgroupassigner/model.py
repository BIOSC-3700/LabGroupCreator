import numpy as np
from enum import Enum
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

from labgroupassigner.errors import SolverError


class SolverStatus(Enum):
    OPTIMAL = "optimal"
    TIME_LIMIT = "time_limit"
    INFEASIBLE = "infeasible"


def group_sizes(n):
    """Sizes for n students: only 3s and 4s,
    maximizing 4s.

    Returns a list of sizes (4s first), or None if
    n is too small (< 6).
    """
    if n < 6:
        return None
    for a in range(n // 4, -1, -1):
        rem = n - 4 * a
        if rem % 3 == 0:
            return [4] * a + [3] * (rem // 3)
    return None


def build_and_solve(
    data,
    *,
    balance_weight=1.0,
    diversity_weight=1.0,
    pronoun_weight=1.0,
    one_she_penalty=10.0,
    timeout_minutes=1.0,
    status_callback=None,
    progress_callback=None,
):
    """Formulate and solve the group assignment MIP.

    Returns a dict with 'assignments' (0-based group
    per student), 'objective', 'success', and 'status'.
    """
    log = status_callback or print
    n = data["n_students"]
    scores = data["total_scores"]
    cat_scores = data["cat_scores"]
    is_she = data["is_she"]
    pairs = data["same_name_pairs"]
    use_pronoun = data["use_pronoun_constraint"]
    n_cat = len(data["categories"])

    # Compute group sizes
    sizes = data.get("group_sizes")
    if sizes is None:
        sizes = group_sizes(n)
    if sizes is None:
        raise SolverError(
            f"Cannot form groups from {n} students "
            f"(need at least 6)"
        )

    g = len(sizes)

    # -- Variable indices (0-based) --
    n_x = n * g

    def idx_x(i, j):
        return j * n + i

    idx_min_total = n_x
    idx_max_total = n_x + 1
    n_max_cat = g * n_cat

    def idx_max_cat(j, c):
        return n_x + 2 + c * g + j

    n_base = n_x + 2 + n_max_cat

    if use_pronoun:
        n_vars = n_base + 4 * g

        def idx_has_she(j):
            return n_base + j

        def idx_has_two_she(j):
            return n_base + g + j

        def idx_d_plus(j):
            return n_base + 2 * g + j

        def idx_d_minus(j):
            return n_base + 3 * g + j
    else:
        n_vars = n_base

    # -- Objective --
    obj = np.zeros(n_vars)
    obj[idx_min_total] = -balance_weight
    obj[idx_max_total] = balance_weight
    for j in range(g):
        for c in range(n_cat):
            obj[idx_max_cat(j, c)] = -diversity_weight
    if use_pronoun:
        for j in range(g):
            obj[idx_d_plus(j)] = pronoun_weight
            obj[idx_d_minus(j)] = pronoun_weight
            obj[idx_has_she(j)] = one_she_penalty
            obj[idx_has_two_she(j)] = -one_she_penalty

    # -- Count constraint rows --
    n_rows = (
        n
        + g
        + g
        + g
        + n * g * n_cat
        + len(pairs) * g
    )
    if use_pronoun:
        n_rows += 5 * g

    A = lil_matrix((n_rows, n_vars))
    lb = np.full(n_rows, -np.inf)
    ub = np.full(n_rows, np.inf)
    row = 0

    if progress_callback:
        progress_callback("Building constraint matrix")

    # Block 1: each student in exactly one group
    for i in range(n):
        for j in range(g):
            A[row, idx_x(i, j)] = 1.0
        lb[row] = 1.0
        ub[row] = 1.0
        row += 1

    # Block 2: group size (per-group right-hand side)
    for j in range(g):
        for i in range(n):
            A[row, idx_x(i, j)] = 1.0
        lb[row] = float(sizes[j])
        ub[row] = float(sizes[j])
        row += 1

    # Block 3: min_total <= group_total
    for j in range(g):
        A[row, idx_min_total] = 1.0
        for i in range(n):
            A[row, idx_x(i, j)] = -scores[i]
        lb[row] = -np.inf
        ub[row] = 0.0
        row += 1

    # Block 4: max_total >= group_total
    for j in range(g):
        A[row, idx_max_total] = 1.0
        for i in range(n):
            A[row, idx_x(i, j)] = -scores[i]
        lb[row] = 0.0
        ub[row] = np.inf
        row += 1

    # Block 5: max_cat[j,c] >= cat_scores[i,c]*x[i,j]
    for j in range(g):
        for c in range(n_cat):
            for i in range(n):
                A[row, idx_max_cat(j, c)] = 1.0
                A[row, idx_x(i, j)] = (
                    -cat_scores[i, c]
                )
                lb[row] = 0.0
                ub[row] = np.inf
                row += 1

    # Block 6: same-name exclusion
    for i1, i2 in pairs:
        for j in range(g):
            A[row, idx_x(i1, j)] = 1.0
            A[row, idx_x(i2, j)] = 1.0
            lb[row] = -np.inf
            ub[row] = 1.0
            row += 1

    if use_pronoun:
        # Block 7a: she_count >= has_she
        for j in range(g):
            for i in range(n):
                A[row, idx_x(i, j)] = is_she[i]
            A[row, idx_has_she(j)] = -1.0
            lb[row] = 0.0
            ub[row] = np.inf
            row += 1

        # Block 7b: she_count <= sizes[j] * has_she
        for j in range(g):
            for i in range(n):
                A[row, idx_x(i, j)] = is_she[i]
            A[row, idx_has_she(j)] = -float(sizes[j])
            lb[row] = -np.inf
            ub[row] = 0.0
            row += 1

        # Block 8a: she_count >= 2 * has_two_she
        for j in range(g):
            for i in range(n):
                A[row, idx_x(i, j)] = is_she[i]
            A[row, idx_has_two_she(j)] = -2.0
            lb[row] = 0.0
            ub[row] = np.inf
            row += 1

        # Block 8b: she_count <= 1 + (s-1)*has_two_she
        for j in range(g):
            for i in range(n):
                A[row, idx_x(i, j)] = is_she[i]
            A[row, idx_has_two_she(j)] = -float(
                sizes[j] - 1
            )
            lb[row] = -np.inf
            ub[row] = 1.0
            row += 1

        # Block 9: she_count - d+ + d- == 2
        for j in range(g):
            for i in range(n):
                A[row, idx_x(i, j)] = is_she[i]
            A[row, idx_d_plus(j)] = -1.0
            A[row, idx_d_minus(j)] = 1.0
            lb[row] = 2.0
            ub[row] = 2.0
            row += 1

    # -- Integrality --
    integrality = np.zeros(n_vars)
    integrality[:n_x] = 1
    if use_pronoun:
        integrality[n_base:n_base + g] = 1
        integrality[
            n_base + g:n_base + 2 * g
        ] = 1

    # -- Variable bounds --
    var_lb = np.zeros(n_vars)
    var_ub = np.empty(n_vars)
    var_ub[:n_x] = 1.0
    var_ub[idx_min_total] = np.inf
    var_ub[idx_max_total] = np.inf
    mc_start = n_x + 2
    var_ub[mc_start:mc_start + n_max_cat] = 5.0
    if use_pronoun:
        var_ub[n_base:n_base + g] = 1.0
        var_ub[
            n_base + g:n_base + 2 * g
        ] = 1.0
        var_ub[
            n_base + 2 * g:n_base + 4 * g
        ] = np.inf

    # -- Solve --
    if progress_callback:
        progress_callback("Solving MILP")

    result = milp(
        c=obj,
        constraints=LinearConstraint(
            A.tocsc(), lb, ub
        ),
        integrality=integrality,
        bounds=Bounds(lb=var_lb, ub=var_ub),
        options={
            "time_limit": timeout_minutes * 60,
        },
    )

    if result.x is None:
        msg = (
            "No feasible solution found. "
            f"Solver status: {result.message}"
        )
        if use_pronoun:
            n_she_total = int(is_she.sum())
            msg += (
                f"\nPronoun distribution: "
                f"{n_she_total} she/unknown, "
                f"{n - n_she_total} he"
            )
        raise SolverError(msg)

    if result.success:
        status = SolverStatus.OPTIMAL
        log(
            "Optimal solution found. "
            f"Objective: {result.fun:.2f}"
        )
    else:
        status = SolverStatus.TIME_LIMIT
        log(
            "Time limit reached. "
            "Using best solution. "
            f"Objective: {result.fun:.2f}"
        )

    # -- Extract assignments --
    x_vals = result.x[:n_x]
    assignments = np.empty(n, dtype=int)
    for i in range(n):
        for j in range(g):
            if x_vals[idx_x(i, j)] > 0.5:
                assignments[i] = j
                break

    return {
        "assignments": assignments,
        "objective": result.fun,
        "success": result.success,
        "status": status,
        "group_sizes": sizes,
    }
