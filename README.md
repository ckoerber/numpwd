# numpwd

Python module to convert two-nucleon operators from math-expressions (sympy) to partial wave decomposed arrays in momentum space.

## Description

This module exports two-nucleon operators (like external currents) specified by latex-like expressions in momentum space to arrays in a partial wave basis:

$$
	\hat O
	=
	f(\hat p_o, \hat p_i, \hat q)
	\vec sigma_1 \cdot (\vec p_o - \vec p_i)
	\vec sigma_1 \cdot (\vec p_o - \vec p_i)
	\to
	O_{(l_o s_o)j_o m_{j_o} (l_i s_i)j_i m_{j_i}}(p_o, p_i, \vec{q})
$$
where ` o, i`  indicate incoming or outgoing quantities and nuclear channels are coupled in an `(ls)j` scheme.

It utilizes sympy to convert expressions to arrays on CPU (numpy) or GPU  (cupy) and integrates out angular dependence to project on partial wave channels.

### Details

For general spin-dependent operators, array dimensions can be quite large: `(c_o, c_i, p_o, p_i, q, x_o, x_i, phi_o, phi_i)`, where
`c`  specifies the number of `(ls)j`  channels ~ 60
`p` specifies the momentum grid  ~ 60
`q` the external momentum values ~ 10
`x` the polar angle values ~ 20
`phi` the azimuthal angle values ~ 50

For this reason, this module utilizes the Wigner–Eckart theorem to reduce the number of allowed channels. For a large class of operators of interest, this also allows to run one integration analytically (also done under the hood).

To further speed up computations, intermediate results like Clebsch-Gordan coefficients are repeated integrals are cached.

The notebook folder specifies more details and provides examples.

## Install
Install via pip
```bash
pip install [-e] [--user] .
```

## Authors
* [@ckoerber](https://www.ckoerber.com).
