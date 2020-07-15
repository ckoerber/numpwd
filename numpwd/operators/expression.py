def decompose_operator(
    spin_momentum_expression, isospin_expression, momentum_mesh, substitutions,
):
    """

    Arguments:
        isospin_expression: List of isospin matrix elements (dicts) which must have
            the keys ["t_o", "mt_o", "t_i", "mt_i", "expr"].
    """

    #   Check isospin
    ie = isospin_expressions.copy()
    iso_mat = {}
    required_isospin_keys = set(["t_o", "mt_o", "t_i", "mt_i", "expr"])
    for el in ie:
        if required_isospin_keys != set(el.keys()):
            raise KeyError(
                f"Spin element {el} does not provide all required spin keys."
            )
        iso_mat[(["t_o"], ["mt_o"], ["t_i"], ["mt_i"])] = float(el["expr"])

    spin_decomposition = get_spin_matrix_element_ex(
        expr, pauli_symbol="sigma", ex_label="_ex"
    )

    for decomposition in spin_decomposition:
        expr = decomposition["val"]
        for subs in substitutions:
            expr = expr.subs(subs)

        expr = expr.rewrite("exp").expand()
