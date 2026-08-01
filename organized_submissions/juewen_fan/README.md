# Gradient-Plateau SBA

Entrant: Juewen Fan

`gradient_plateau_sba.py` implements a two-stage black-box attack. The first
stage estimates gradients in a low-frequency DCT subspace, uses Adam-style
updates, and includes rescue, plateau-breakthrough, and zero-gradient restart
mechanisms. The second stage applies an SBA coordinate-search fallback.

The module expects the competition framework to provide
`attack_problem.one_image_problem.SingleImageProblem`.
