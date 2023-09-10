# numpwd

**Note: This repository is not beeing maintained as of 2022.**

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

## Details

This module semi-numerically computes the partial wave decomposition of operators

\begin{multline}
    O_{(l_o s_o)j_o m_{j_o} (l_i s_i)j_i m_{j_i}}(p_o, p_i, \vec{q})
    =
    \sum\limits_{m_{s_o} m_{s_i}}
    \sum\limits_{m_{l_o} m_{l_i}}
    \left\langle
        l_o m_{l_o}, s_o m_{s_o} \big\vert j_o m_{j_o}
    \right\ranlge
    \left\langle
        l_i m_{l_i}, s_i m_{s_i} \big\vert j_i m_{j_i}
    \right\ranlge
    \\ \times
    \int d x_o d x_i d \phi_o d \phi_i
    Y_{l_o m_{l_o}}^*(x_o, \phi_o)
    Y_{l_i m_{l_i}}(x_i, \phi_i)
    \\ \times
    \left\langle
        \vec p_o; s_o m_{s_o}
        \big\vert
        \hat O(\vec p_o, \vec p_i, \vec q)
        \big\vert
        \vec p_i; s_i m_{s_i}
    \right\rangle
\end{multline}

For general spin-dependent operators, array dimensions can exceed available resources by far. For example, arrays (complex numbers) in intermediate computations are of the shape `(c_s, p_o, p_i, q, x_o, x_i, phi_o, phi_i)`, where

* `c_s` specifies the number of in and outgoing spin channels ~ 10
* `p` specifies the momentum grid  ~ 60
* `q` the external momentum values ~ 10
* `x` the polar angle values ~ 20
* `phi` the azimuthal angle values ~ 50

which corresponds to `10 * 60**2 * 10 * 20**2 * 50**2 * 16B ~ 5TB` if one would try to entirely allocate this array at once.

To allow computational optimizations like vectorization, this module utilizes the Wigner–Eckart theorem to reduce the number of allowed channels.
For a large class of operators of interest, this furthermore allows to run one angular integration analytically (done under the hood) such that intermediate arrays for large operators are at the size of `5GB`.

To further speed up computations, intermediate results like Clebsch-Gordan coefficients and repeated integrals are cached.

The docs folder contains a tex file specifying the mathematical aspect and the notebook folder specifies more details and provides examples.

## Install
Install via pip
```bash
pip install {-e} {--user} .{[gpu]}
```

Curly brackets are optional,

* `--user` installs the module for the current user only
* `-e` symlinks this directory into your python import path, if you update the files in this repo, your imports will update too (useful for development)
* `[gpu]` add GPU support. This requires an CUDA (NVIDIA) capable GPU.

## Authors
* [@ckoerber](https://www.ckoerber.com).
