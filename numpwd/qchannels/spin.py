"""Module to compute spin operator Partial Wave Decompositions
"""

from typing import Dict
from typing import List
from typing import Tuple

from itertools import product

from sympy import S
from sympy import Matrix
from sympy.physics.quantum.cg import CG


def op_pwd(matrix: Dict[Tuple[S, S, S, S], complex], j_max_x2: int = 2) -> Dict:
    """Computes the operator Partial Wave Decomposition of the operator.

        op_pwd[j_out, j_in, xi, mxi] = (2*xi + 1)/(2*j_out + 1)
            * sum(CG(j_in, m1; xi, mxi; j_out, m2)
            * matrix[(j_out, m2, j_in, m1)], {m1, m2})

    The matrix must map 'matrix(j_out, m2, j_in, m1) -> complex'.
    This routine returns a dictionary which contains all non-zero channels with
    'j_in, j_out in [0, j_max_x2/2]' in 1/2 steps.
    """
    out = []

    for j_in_x2, j_out_x2 in product(range(j_max_x2 + 1), range(j_max_x2 + 1)):
        j_in = S(f"{j_in_x2}/2")
        j_out = S(f"{j_out_x2}/2")
        for xi_x2 in range(abs(j_in_x2 - j_out_x2), j_in_x2 + j_out_x2 + 1):
            xi = S(f"{xi_x2}/2")
            for mxi_x2 in range(-xi_x2, xi_x2 + 1, 2):
                mxi = S(f"{mxi_x2}/2")
                val = 0
                for m1_x2 in range(-j_in_x2, j_in_x2 + 1, 2):
                    m1 = S(f"{m1_x2}/2")
                    m2 = mxi + m1
                    if abs(m2) > j_out:
                        continue

                    val += (
                        CG(j_in, m1, xi, mxi, j_out, m2).doit()
                        * matrix[(j_out, m2, j_in, m1)]
                        if (j_out, m2, j_in, m1) in matrix
                        else 0
                    )

                if val != 0:
                    val *= (2 * xi + 1) / (2 * j_out + 1)
                    out.append(
                        {
                            "j_out": j_out,
                            "j_in": j_in,
                            "xi": xi,
                            "mxi": mxi,
                            "val": val.simplify(),
                        }
                    )

    return out


def pauli_contract_subsystem(
    matrix: Dict[Tuple[int, int, int, int], complex]
) -> Dict[Tuple[int, int, int, int], complex]:
    """Computes spin contraction of two nucleon pauli spin half operators:

        op[(j_out, mj_out, j_in, mj_in)] = sum(
            CG(1/2, ms1_out, 1/2, ms2_out, j_out, mj_out)
            * CG(1/2, ms1_in, 1/2, ms2_in, j_out, mj_in)
            * matrix[(ms2_out, ms1_out, ms1_in, ms2_in)]
        )

        Channels which correspond to zero are excluded from the output.
    """
    s_half = S("1/2")
    ms_range = [S("-1/2"), S("1/2")]
    spin_range = [0, 1]

    op_dict = {}

    for j_out, j_in in product(*[spin_range] * 2):
        for mj_out, mj_in in product(range(-j_out, j_out + 1), range(-j_in, j_in + 1)):
            res = 0
            for ms2_out, ms1_out, ms1_in, ms2_in in product(*[ms_range] * 4):
                if ms1_out + ms2_out != mj_out or ms1_in + ms2_in != mj_in:
                    continue

                tres = (
                    CG(s_half, ms1_out, s_half, ms2_out, j_out, mj_out)
                    * CG(s_half, ms1_in, s_half, ms2_in, j_in, mj_in)
                ).doit()

                tres *= (
                    matrix[(ms2_out, ms1_out, ms1_in, ms2_in)]
                    if (ms2_out, ms1_out, ms1_in, ms2_in) in matrix
                    else 0
                )

                res += tres

            if res != 0:
                op_dict[(j_out, mj_out, j_in, mj_in)] = res

    return op_dict


def pauli_substitution(
    n_nucleon: int, ms_out: S, ms_in: S, pauli_symbol: str = "sigma"
) -> Dict[str, complex]:
    """Substitudes pauli matrix indices and spin quantum numbers into sympy expression

        < 1/2 ms_out | sigma_{n_partilce, a} | 1/2 ms_in >
            -> vec(ms_out) @ sigma_a @ vec(ms_in)

        The input 'n_nucleon' specifies which pauli matrix will be substituted.
    """
    spin_vec = {S("1/2"): Matrix([1, 0]), S("-1/2"): Matrix([0, 1])}
    pauli_mat = {
        0: Matrix([[1, "+0"], ["+0", "+1"]]),
        1: Matrix([[0, "+1"], ["+1", "+0"]]),
        2: Matrix([[0, "-I"], ["+I", "+0"]]),
        3: Matrix([[1, "+0"], ["+0", "-1"]]),
    }

    return {
        f"{pauli_symbol}{n_nucleon}{a}": spin_vec[ms_out].dot(sigma @ spin_vec[ms_in])
        for a, sigma in pauli_mat.items()
    }


def expression_to_matrix(
    op_expression: S, pauli_symbol: str = "sigma"
) -> Dict[Tuple[S, S, S, S], complex]:
    """Converts pauli matrix expression to callable matrix for spins.

    'op_expression' is a sympy expression containing pauli matrices like 'tau11 tau12'
    where the first index corresponds to the pauli matrix and the second to the nucleon
    subsystem.
    """

    matrix = {}

    ms_range = [S("-1/2"), S("1/2")]

    for ms2_out, ms1_out, ms1_in, ms2_in in product(*[ms_range] * 4):

        substitutions = pauli_substitution(
            1, ms1_out, ms1_in, pauli_symbol=pauli_symbol
        )
        substitutions.update(
            pauli_substitution(2, ms2_out, ms2_in, pauli_symbol=pauli_symbol)
        )

        val = op_expression.subs(substitutions).simplify()

        if val != 0:
            matrix[(ms2_out, ms1_out, ms1_in, ms2_in)] = val

    return matrix


def dict_to_data(
    matrix: Dict[Tuple[S, S, S, S], S], columns: List[str]
) -> List[Dict[str, S]]:
    """Converts operator maps to list of entries
    """

    data = []

    for keys, val in matrix.items():
        data.append(dict(zip(columns, keys)))
        data[-1]["val"] = val

    return data
