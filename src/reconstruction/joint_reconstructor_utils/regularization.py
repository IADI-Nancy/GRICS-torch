"""Cache regularization scaling across solves at a resolution level."""


def _assign_cached_reg_scale(params, Data_res, cache_key, solver, reference_vec):
    if not params.cg_use_reg_scale_proxy:
        solver.reg_scale = 1.0
        return

    cache = Data_res.setdefault("_reg_scale_cache", {})
    if cache_key not in cache:
        cache[cache_key] = solver._update_regularization_scale(reference_vec)
    solver.reg_scale = cache[cache_key]
