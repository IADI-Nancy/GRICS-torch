"""Strict, file-owned configuration. TOML values precede explicit call overrides."""
from pathlib import Path
from types import SimpleNamespace
import math
import tomllib

ISMRMRD_READER_DATA_TYPES = {
    f'{source}-{physiology}'
    for source in ('ismrmrd', 'siemens')
    for physiology in ('saec', 'polaris', 'physio_text', 'physio_array')
}
REAL_DATA_TYPES = {'preprocessed-real'} | ISMRMRD_READER_DATA_TYPES
SYNTHETIC_DATA_TYPES = {'shepp-logan', 'from_image'}

_RIGID_MOTION_KEYS = {
    "num_motion_events",
    "rigid_motion_amplitude_scale",
    "max_tx",
    "max_ty",
    "max_phi",
    "max_center_x",
    "max_center_y",
    "max_tx_3d",
    "max_ty_3d",
    "max_tz_3d",
    "max_rx_3d",
    "max_ry_3d",
    "max_rz_3d",
    "max_center_x_3d",
    "max_center_y_3d",
    "max_center_z_3d",
    "motion_tau",
}

_NONRIGID_MOTION_KEYS = {
    "nonrigid_motion_amplitude",
    "nonrigid_discrete_s_scale",
    "nonrigid_resp_cycles_min",
    "nonrigid_resp_cycles_max",
    "nonrigid_diaphragm_level",
    "nonrigid_diaphragm_sharpness",
    "nonrigid_lateral_sigma_lr",
    "nonrigid_lateral_sigma_ap",
    "nonrigid_ap_fraction",
    "nonrigid_lr_fraction",
    "nonrigid_anterior_bias",
    "nonrigid_inferior_gain",
    "nonrigid_top_decay",
}

_PATH_KEYS = {'output_root', 'workflow_label', 'cache_root'}
_RUNTIME_KEYS = {
    'save_debug_plots', 'use_deterministic_algorithms', 'runtime_device', 'verbose', 'print_to_console',
    'remove_temporary_data_after_run', 'cache_preprocessed_data', 'jupyter_notebook_flag', 'flip_for_display',
    'seed', 'seed_enabled',
}
_NORMALIZATION_KEYS = {'normalize_kspace', 'kspace_norm_mode', 'kspace_norm_eps'}
_REAL_DATA_KEYS = {'rawdata_sensor_type'}
_ISMRMRD_READER_KEYS = {'print_raw_calibration_lines'}
_PHYSIO_CLOCK_KEYS = {'physio_clock_drift_seconds'}
_POLARIS_KEYS = {'polaris_channel_mode'}
_CSM_ESPIRIT_KEYS = {'coil_sensitivity_method', 'espirit_calibration_width', 'espirit_kernel_width', 'espirit_max_iter'}
_CSM_ODILLE_SPLINE_KEYS = {'coil_sensitivity_method', 'spline_magnitude_smoothing', 'spline_phase_smoothing', 'coil_sensitivity_eps'}
_CSM_KEYS = _CSM_ESPIRIT_KEYS | _CSM_ODILLE_SPLINE_KEYS
_RECONSTRUCTION_KEYS = {
    'reconstruction_dimension', 'reconstruction_motion_type', 'N_motion_states',
    'N_motion_states_per_level', 'motion_binning_mode', 'motion_quantization_bins',
    'ResolutionLevels', 'GN_iterations_per_level',
    'update_motion_on_final_iteration', 'gn_early_stopping', 'save_reconstruction_outputs',
    'cg_stop_on_stagnation', 'cg_true_residual_interval', 'cg_stagnation_consecutive_steps', 'cg_stagnation_countdown_steps',
    'cg_use_reg_scale_proxy', 'cg_reg_scale_num_probes', 'lambda_r', 'lambda_m',
    'max_iter_recon', 'max_iter_motion', 'tol_recon', 'tol_motion',
}
_SAMPLING_KEYS = {'kspace_sampling_type', 'NshotsPerNex', 'Nex', 'acceleration_factor', 'calibration_lines'}
_SHEPP_KEYS = {'data_dimension', 'N_SheppLogan', 'Ncoils_SheppLogan', 'Nz_SheppLogan',
               'SheppLoganFillFraction'}
_IMAGE_KEYS = {'data_dimension', 'Ncoils_input', 'image_resize_factor'}
_RIGID_GEOMETRY_KEYS = {'FoVxy_mm', 'FoVz_mm'}
_MOTION_KEYS = {'motion_simulation_config_dimension', 'simulated_motion_type', 'check_simulated_motion_consistency'} | _RIGID_GEOMETRY_KEYS | _RIGID_MOTION_KEYS | _NONRIGID_MOTION_KEYS
_FILE_SCHEMAS = {
    'general': {'paths': _PATH_KEYS, 'runtime': _RUNTIME_KEYS,
                'kspace_normalization': _NORMALIZATION_KEYS},
    'real_data': {'real_data': _REAL_DATA_KEYS},
    'ismrmrd_reader': {'ismrmrd_reader': _ISMRMRD_READER_KEYS},
    'polaris': {'polaris': _POLARIS_KEYS | _PHYSIO_CLOCK_KEYS},
    'physio': {'physio': _PHYSIO_CLOCK_KEYS},
    'reconstruction': {'reconstruction': _RECONSTRUCTION_KEYS},
    'sampling': {'sampling': _SAMPLING_KEYS},
    'shepp-logan': {'shepp_logan': _SHEPP_KEYS},
    'from_image': {'from_image': _IMAGE_KEYS},
    'motion': {'motion': _MOTION_KEYS},
    'postprocessing': {'postprocessing': {'normalize_image_by_grics_reference'}},
    'coil_sensitivity': {'coil_sensitivity': _CSM_KEYS},
}
_OVERRIDE_KEYS = (_PATH_KEYS | _RUNTIME_KEYS | _NORMALIZATION_KEYS | (_CSM_KEYS - {'coil_sensitivity_method'}) |
                  _RECONSTRUCTION_KEYS | _SAMPLING_KEYS | _SHEPP_KEYS | _IMAGE_KEYS |
                  _MOTION_KEYS | _REAL_DATA_KEYS | _ISMRMRD_READER_KEYS | _POLARIS_KEYS | _PHYSIO_CLOCK_KEYS)
_BOOL_KEYS = {
    'save_debug_plots', 'check_simulated_motion_consistency', 'use_deterministic_algorithms',
    'print_raw_calibration_lines', 'verbose', 'print_to_console', 'remove_temporary_data_after_run', 'cache_preprocessed_data',
    'jupyter_notebook_flag', 'flip_for_display', 'seed_enabled', 'normalize_kspace',
    'update_motion_on_final_iteration', 'gn_early_stopping',
    'save_reconstruction_outputs', 'cg_stop_on_stagnation', 'cg_use_reg_scale_proxy',
}


def _require(cfg, keys, context):
    missing = sorted(set(keys) - cfg.keys())
    if missing:
        raise ValueError(f'Missing {context} settings: {missing}. Set them in the owning TOML or overrides.')


def _load_toml_flat(path, kind, _include_chain=()):
    """Load one file-owned TOML configuration, expanding common motion files.

    Only ``[motion].include`` is supported. It names one relative TOML fragment;
    included settings are read first, and the selecting file may not redefine them.
    This keeps every effective setting unambiguous.
    """
    path = Path(path).resolve()
    if path in _include_chain:
        chain = ' -> '.join(str(item) for item in (*_include_chain, path))
        raise ValueError(f'Motion configuration include cycle: {chain}.')
    schema = _FILE_SCHEMAS[kind]
    with path.open('rb') as handle:
        data = tomllib.load(handle)
    unknown_sections = data.keys() - schema.keys()
    if unknown_sections:
        raise ValueError(f'{path}: invalid {kind} sections: {sorted(unknown_sections)}.')
    result = {}
    for section, entries in data.items():
        if not isinstance(entries, dict):
            raise ValueError(f'{path}: expected a [{section}] table.')
        include = entries.get('include')
        if include is not None:
            if kind != 'motion' or section != 'motion':
                raise ValueError(f'{path}: include is allowed only in a [motion] table.')
            if not isinstance(include, str) or not include.strip():
                raise ValueError(f'{path}: motion.include must be one nonempty relative path.')
            include_path = Path(include)
            if include_path.is_absolute() or '..' in include_path.parts:
                raise ValueError(f'{path}: motion.include must be a relative path below its directory.')
            included = _load_toml_flat(path.parent / include_path, 'motion', (*_include_chain, path))
            duplicate = result.keys() & included.keys()
            if duplicate:
                raise ValueError(f'{path}: duplicate settings from included common motion files: {sorted(duplicate)}.')
            result.update(included)
        for key, value in entries.items():
            if key == 'include':
                continue
            if key not in schema[section]:
                raise ValueError(f'{path}: setting {section}.{key} is not allowed in this file/section.')
            if key in result:
                raise ValueError(f'{path}: setting {key} duplicates an included motion setting.')
            result[key] = value
    return result


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}; got {value!r}.')
    return value


def _number(value, name, *, minimum=None, positive=False, maximum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number; got {value!r}.')
    if positive and value <= 0:
        raise ValueError(f'{name} must be > 0.')
    if minimum is not None and value < minimum:
        raise ValueError(f'{name} must be >= {minimum}.')
    if maximum is not None and value > maximum:
        raise ValueError(f'{name} must be <= {maximum}.')
    return value


def _choice(value, name, choices):
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f'{name} must be one of {sorted(choices)}; got {value!r}.')


def _apply_overrides(cfg, overrides, allowed):
    for key, value in (overrides or {}).items():
        if key not in allowed:
            raise ValueError(f'Unknown or misplaced configuration override: {key}.')
        cfg[key] = value


def _validate_general(cfg):
    _require(cfg, _PATH_KEYS | _RUNTIME_KEYS | _NORMALIZATION_KEYS, 'general')
    for key in _BOOL_KEYS & cfg.keys():
        if type(cfg[key]) is not bool:
            raise ValueError(f'{key} must be a boolean.')
    for key in _PATH_KEYS:
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise ValueError(f'{key} must be a nonempty path string.')
    _integer(cfg['seed'], 'seed', 0)
    _choice(cfg['runtime_device'], 'runtime_device', {'cpu', 'gpu'})
    _choice(cfg['kspace_norm_mode'], 'kspace_norm_mode', {'rms', 'max'})
    _number(cfg['kspace_norm_eps'], 'kspace_norm_eps', positive=True)


def _validate_real_data(cfg):
    _require(cfg, _REAL_DATA_KEYS, 'real-data')
    if not isinstance(cfg['rawdata_sensor_type'], str) or not cfg['rawdata_sensor_type']:
        raise ValueError('rawdata_sensor_type must be a nonempty string.')


def _validate_csm(cfg):
    _require(cfg, {'coil_sensitivity_method'}, 'coil sensitivity')
    method = cfg['coil_sensitivity_method']
    _choice(method, 'coil_sensitivity_method', {'espirit', 'odille-spline'})
    expected = _CSM_ESPIRIT_KEYS if method == 'espirit' else _CSM_ODILLE_SPLINE_KEYS
    supplied = cfg.keys() & _CSM_KEYS
    if supplied != expected:
        missing = sorted(expected - supplied)
        irrelevant = sorted(supplied - expected)
        raise ValueError(f'coil sensitivity config for {method!r} must contain exactly {sorted(expected)}; missing={missing}, irrelevant={irrelevant}.')
    if method == 'espirit':
        for key in ('espirit_calibration_width', 'espirit_kernel_width', 'espirit_max_iter'):
            _integer(cfg[key], key)
        if cfg['espirit_kernel_width'] > cfg['espirit_calibration_width']:
            raise ValueError('espirit_kernel_width cannot exceed espirit_calibration_width.')
    else:
        for key in ('spline_magnitude_smoothing', 'spline_phase_smoothing'):
            _number(cfg[key], key, minimum=0)
        _number(cfg['coil_sensitivity_eps'], 'coil_sensitivity_eps', positive=True)


def _validate_source(cfg):
    _choice(cfg['reconstruction_dimension'], 'reconstruction_dimension', {'2D', '3D'})
    # Real input dimensionality is taken from the explicitly selected reconstruction tag.
    if 'data_dimension' not in cfg and cfg['data_type'] in REAL_DATA_TYPES:
        cfg['data_dimension'] = cfg['reconstruction_dimension']
    _require(cfg, {'data_dimension'}, 'data source')
    _choice(cfg['data_dimension'], 'data_dimension', {'2D', '3D'})
    if cfg['data_dimension'] != cfg['reconstruction_dimension']:
        raise ValueError('data_dimension must match reconstruction_dimension.')
    if cfg['data_type'] == 'shepp-logan':
        _require(cfg, _SHEPP_KEYS, 'Shepp-Logan source')
        for key in ('N_SheppLogan', 'Nz_SheppLogan', 'Ncoils_SheppLogan'):
            _integer(cfg[key], key)
        if cfg['Ncoils_SheppLogan'] % 4:
            raise ValueError('Ncoils_SheppLogan must be divisible by 4.')
        dim = '3D' if cfg['Nz_SheppLogan'] > 1 else '2D'
        if dim != cfg['data_dimension']:
            raise ValueError('Nz_SheppLogan conflicts with data_dimension.')
        _number(cfg['SheppLoganFillFraction'], 'SheppLoganFillFraction', positive=True, maximum=1)
    if cfg['data_type'] == 'from_image':
        _require(cfg, _IMAGE_KEYS, 'image source')
        if cfg['data_dimension'] != '2D':
            raise ValueError('from_image supports only 2D data.')
        _integer(cfg['Ncoils_input'], 'Ncoils_input')
        if cfg['Ncoils_input'] % 4:
            raise ValueError('Ncoils_input must be divisible by 4.')
        _number(cfg['image_resize_factor'], 'image_resize_factor', positive=True)


def _validate_sampling(cfg):
    _choice(cfg['kspace_sampling_type'], 'kspace_sampling_type', {'linear', 'interleaved', 'random', 'from-data'})
    if cfg['kspace_sampling_type'] == 'from-data':
        if cfg['data_type'] not in REAL_DATA_TYPES:
            raise ValueError('from-data sampling requires real input data.')
        extra = (cfg.keys() & _SAMPLING_KEYS) - {'kspace_sampling_type'}
        if extra:
            raise ValueError(f'from-data sampling reads acquisition counts from data; remove {sorted(extra)}.')
        return
    _require(cfg, _SAMPLING_KEYS, 'simulated sampling')
    for key in ('NshotsPerNex', 'Nex', 'acceleration_factor'):
        _integer(cfg[key], key)
    _integer(cfg['calibration_lines'], 'calibration_lines', 0)
    if cfg['acceleration_factor'] > 1 and cfg['calibration_lines'] == 0:
        raise ValueError('Accelerated sampling requires positive calibration_lines.')
    cfg['Nshots'] = cfg['NshotsPerNex'] * cfg['Nex']
    if cfg['data_type'] == 'shepp-logan':
        validate_sampling_size(SimpleNamespace(**cfg), cfg['N_SheppLogan'], cfg['Nz_SheppLogan'])


def validate_sampling_size(params, ny, nz):
    """Validate data-dependent sampling bounds once the image dimensions are known."""
    if params.kspace_sampling_type == 'from-data':
        return
    if params.calibration_lines > ny:
        raise ValueError('calibration_lines cannot exceed the ky matrix size.')
    if params.acceleration_factor > ny:
        raise ValueError('acceleration_factor cannot exceed the ky matrix size.')
    acquired = set(range(0, ny, params.acceleration_factor))
    if params.acceleration_factor > 1:
        start = (ny - params.calibration_lines) // 2
        acquired.update(range(start, start + params.calibration_lines))
    if params.NshotsPerNex > len(acquired) * nz:
        raise ValueError('NshotsPerNex cannot exceed the number of acquired readouts per repetition.')
    if nz > 1 and params.kspace_sampling_type in {'linear', 'interleaved'}:
        # These 3D orders assign ky groups (with all kz partitions) to each shot.
        for shot in range(params.NshotsPerNex):
            if params.kspace_sampling_type == 'linear':
                candidates = range(shot * ny // params.NshotsPerNex, (shot + 1) * ny // params.NshotsPerNex)
            else:
                candidates = range(shot, ny, params.NshotsPerNex)
            if not acquired.intersection(candidates):
                raise ValueError('Sampling configuration creates an empty 3D shot.')
    return len(acquired) * nz * params.Nex


def _validate_motion(cfg):
    _require(cfg, {'simulated_motion_type'}, 'motion')
    mode = cfg['simulated_motion_type']
    _choice(mode, 'simulated_motion_type', {'as-it-is', 'rigid-realistic', 'rigid-per-shot', 'non-rigid-realistic', 'non-rigid-per-shot'})
    supplied = cfg.keys() & (_RIGID_MOTION_KEYS | _NONRIGID_MOTION_KEYS)
    if mode == 'as-it-is':
        if cfg['data_type'] not in REAL_DATA_TYPES or cfg['kspace_sampling_type'] != 'from-data':
            raise ValueError('as-it-is motion requires real data with from-data sampling; reordered data requires simulated motion.')
        if supplied or 'motion_simulation_config_dimension' in cfg or 'check_simulated_motion_consistency' in cfg:
            raise ValueError('as-it-is motion cannot have synthetic motion parameters or simulation diagnostics.')
        return
    _require(cfg, {'motion_simulation_config_dimension'}, 'motion')
    _choice(cfg['motion_simulation_config_dimension'], 'motion_simulation_config_dimension', {'2D', '3D'})
    if cfg['motion_simulation_config_dimension'] != cfg['data_dimension']:
        raise ValueError('motion_simulation_config_dimension must match data_dimension.')
    rigid = mode.startswith('rigid-')
    if rigid:
        required = {'rigid_motion_amplitude_scale'}
        if cfg['data_dimension'] == '2D':
            required |= {'max_tx', 'max_ty', 'max_phi', 'max_center_x', 'max_center_y'}
        else:
            required |= {'max_tx_3d', 'max_ty_3d', 'max_tz_3d', 'max_rx_3d', 'max_ry_3d', 'max_rz_3d', 'max_center_x_3d', 'max_center_y_3d', 'max_center_z_3d'}
        if mode.endswith('-realistic'):
            required |= {'num_motion_events', 'motion_tau'}
    else:
        common = _NONRIGID_MOTION_KEYS - {'nonrigid_discrete_s_scale', 'nonrigid_motion_amplitude', 'nonrigid_resp_cycles_min', 'nonrigid_resp_cycles_max'}
        required = common | ({'nonrigid_discrete_s_scale'} if mode.endswith('-per-shot') else {'nonrigid_motion_amplitude', 'nonrigid_resp_cycles_min', 'nonrigid_resp_cycles_max'})
        required |= {'check_simulated_motion_consistency'}
    irrelevant = supplied - required
    if rigid and 'check_simulated_motion_consistency' in cfg:
        raise ValueError('check_simulated_motion_consistency is only valid for non-rigid simulation.')
    if not rigid and cfg.keys() & _RIGID_GEOMETRY_KEYS:
        raise ValueError(f'Rigid-motion geometry is incompatible with {mode}: {sorted(cfg.keys() & _RIGID_GEOMETRY_KEYS)}.')
    if irrelevant:
        raise ValueError(f'Parameters incompatible with {mode}: {sorted(irrelevant)}.')
    _require(cfg, required, f'{mode} motion')
    if rigid:
        _require(cfg, {'FoVxy_mm'} | ({'FoVz_mm'} if cfg['data_dimension'] == '3D' else set()), 'motion field of view')
        for key in required:
            if key in {'num_motion_events', 'motion_tau'}:
                _integer(cfg[key], key)
            else:
                _number(cfg[key], key, minimum=None if key.startswith('max_center_') else 0)
    else:
        if type(cfg['check_simulated_motion_consistency']) is not bool:
            raise ValueError('check_simulated_motion_consistency must be a boolean.')
        for key in required - {'check_simulated_motion_consistency'}:
            _number(cfg[key], key, minimum=-1 if key == 'nonrigid_diaphragm_level' else 0)
        for key in ('nonrigid_lateral_sigma_lr', 'nonrigid_lateral_sigma_ap', 'nonrigid_diaphragm_sharpness'):
            _number(cfg[key], key, positive=True)
        _number(cfg['nonrigid_diaphragm_level'], 'nonrigid_diaphragm_level', minimum=-1, maximum=1)
        _number(cfg['nonrigid_anterior_bias'], 'nonrigid_anterior_bias', minimum=0, maximum=1)
        if mode.endswith('-realistic'):
            if cfg['nonrigid_resp_cycles_min'] > cfg['nonrigid_resp_cycles_max']:
                raise ValueError('nonrigid_resp_cycles_min cannot exceed nonrigid_resp_cycles_max.')



def _validate_reconstruction(cfg):
    required = _RECONSTRUCTION_KEYS - {'motion_quantization_bins', 'cg_reg_scale_num_probes'}
    _require(cfg, required, 'reconstruction')
    if cfg['motion_binning_mode'] == 'kspace_energy':
        _require(cfg, {'motion_quantization_bins'}, 'kspace-energy motion binning')
    elif 'motion_quantization_bins' in cfg:
        raise ValueError('motion_quantization_bins is only valid when motion_binning_mode="kspace_energy".')
    if cfg['cg_use_reg_scale_proxy']:
        _require(cfg, {'cg_reg_scale_num_probes'}, 'CG regularization-scale proxy')
    elif 'cg_reg_scale_num_probes' in cfg:
        raise ValueError('cg_reg_scale_num_probes is only valid when cg_use_reg_scale_proxy=true.')
    _choice(cfg['reconstruction_motion_type'], 'reconstruction_motion_type', {'rigid', 'non-rigid'})
    levels = cfg['ResolutionLevels']
    if not isinstance(levels, list) or not levels:
        raise ValueError('ResolutionLevels must be a nonempty list.')
    for value in levels:
        _number(value, 'ResolutionLevels entry', positive=True, maximum=1)
    if any(a >= b for a, b in zip(levels, levels[1:])) or levels[-1] != 1:
        raise ValueError('ResolutionLevels must increase strictly and end at 1.0.')
    iterations = cfg['GN_iterations_per_level']
    if not isinstance(iterations, list) or len(iterations) != len(levels):
        raise ValueError('GN_iterations_per_level must contain one positive integer per resolution level.')
    for value in iterations:
        _integer(value, 'GN_iterations_per_level entry')
    for key in ('max_iter_recon', 'max_iter_motion', 'cg_true_residual_interval'):
        _integer(cfg[key], key)
    if cfg['cg_use_reg_scale_proxy']:
        _integer(cfg['cg_reg_scale_num_probes'], 'cg_reg_scale_num_probes')
    for key in ('cg_stagnation_consecutive_steps', 'cg_stagnation_countdown_steps'):
        _integer(cfg[key], key, 0)
    for key in ('tol_recon', 'tol_motion'):
        _number(cfg[key], key, positive=True)
    for key in ('lambda_r', 'lambda_m'):
        values = cfg[key]
        if key == 'lambda_r' and isinstance(values, list):
            if len(values) != len(levels):
                raise ValueError('lambda_r must have one value per resolution level.')
        else:
            values = [values]
        for value in values:
            _number(value, key, minimum=0)
    _choice(cfg['motion_binning_mode'], 'motion_binning_mode', {'kmeans', 'kspace_energy'})
    if cfg['motion_binning_mode'] == 'kspace_energy':
        _integer(cfg['motion_quantization_bins'], 'motion_quantization_bins', 2)
    states = _integer(cfg['N_motion_states'], 'N_motion_states')
    per_shot = cfg['simulated_motion_type'].endswith('-per-shot')
    if per_shot and 'Nshots' in cfg:
        cfg['N_motion_states'] = cfg['Nshots']
        if states != cfg['N_motion_states']:
            print(f"[config] Per-shot simulation: N_motion_states changed from {states} to {cfg['N_motion_states']} (shot count).", flush=True)
    schedule = cfg['N_motion_states_per_level']
    if schedule == 'full':
        return
    if not isinstance(schedule, list) or len(schedule) != len(levels):
        raise ValueError('N_motion_states_per_level must be "full" or one integer per resolution level.')
    for value in schedule:
        _integer(value, 'N_motion_states_per_level entry')
        if per_shot and cfg['kspace_sampling_type'] == 'from-data':
            raise ValueError('Use N_motion_states_per_level="full" when shot counts come from data.')
        if value > cfg['N_motion_states']:
            raise ValueError('N_motion_states_per_level cannot exceed N_motion_states.')
        if cfg['reconstruction_motion_type'] == 'rigid' and value != cfg['N_motion_states']:
            raise ValueError('Per-level motion-state reduction requires non-rigid reconstruction.')


def _apply_notebook_logging(cfg, overrides):
    """Quiet notebook logs unless the caller explicitly overrides that setting."""
    if not cfg['jupyter_notebook_flag']:
        return
    for key in ('verbose', 'print_to_console'):
        if key not in (overrides or {}) and cfg[key]:
            cfg[key] = False
            print(f"[config] Notebook mode: {key} changed from True to False. Use overrides to keep it enabled.", flush=True)


def load_postprocessing_config(path, *, overrides=None):
    cfg = _load_toml_flat(path, 'postprocessing')
    allowed = _FILE_SCHEMAS['postprocessing']['postprocessing']
    _apply_overrides(cfg, overrides, allowed)
    _require(cfg, allowed, 'postprocessing')
    if type(cfg['normalize_image_by_grics_reference']) is not bool:
        raise ValueError('normalize_image_by_grics_reference must be a boolean.')
    return SimpleNamespace(**cfg)


def load_config(*, data_type, reconstruction_config, coil_sensitivity_config,
                shepp_logan_config=None, from_image_config=None, real_data_config=None,
                ismrmrd_reader_config=None, polaris_config=None, physio_config=None, sampling_config=None,
                motion_simulation_config=None, overrides=None):
    _choice(data_type, 'data_type', REAL_DATA_TYPES | SYNTHETIC_DATA_TYPES)
    root = Path(__file__).resolve().parents[2] / 'config'
    cfg = _load_toml_flat(root / 'general.toml', 'general')
    cfg['data_type'] = data_type
    cfg.update(_load_toml_flat(reconstruction_config, 'reconstruction'))
    cfg.update(_load_toml_flat(coil_sensitivity_config, 'coil_sensitivity'))
    if shepp_logan_config is not None and data_type != 'shepp-logan':
        raise ValueError('shepp_logan_config is only valid for shepp-logan data.')
    if from_image_config is not None and data_type != 'from_image':
        raise ValueError('from_image_config is only valid for from_image data.')
    saec_data = data_type in {'ismrmrd-saec', 'siemens-saec'}
    ismrmrd_reader_data = data_type in ISMRMRD_READER_DATA_TYPES
    if real_data_config is not None and not saec_data:
        raise ValueError('real_data_config is only valid for ISMRMRD or Siemens SAEC data.')
    if saec_data:
        if real_data_config is None:
            raise ValueError(f'{data_type} requires a real_data_config.')
        cfg.update(_load_toml_flat(real_data_config, 'real_data'))
    if ismrmrd_reader_config is not None and not ismrmrd_reader_data:
        raise ValueError('ismrmrd_reader_config is only valid for ISMRMRD or Siemens raw data.')
    if ismrmrd_reader_data:
        if ismrmrd_reader_config is None:
            raise ValueError(f'{data_type} requires an ismrmrd_reader_config.')
        cfg.update(_load_toml_flat(ismrmrd_reader_config, 'ismrmrd_reader'))
    polaris_data = data_type in {'ismrmrd-polaris', 'siemens-polaris'}
    if polaris_config is not None and not polaris_data:
        raise ValueError('polaris_config is only valid for ISMRMRD or Siemens Polaris data.')
    if polaris_data:
        if polaris_config is None:
            raise ValueError(f'{data_type} requires a polaris_config.')
        cfg.update(_load_toml_flat(polaris_config, 'polaris'))
    generic_physio_data = data_type in {
        f'{source}-{kind}' for source in ('ismrmrd', 'siemens')
        for kind in ('physio_text', 'physio_array')
    }
    if physio_config is not None and not generic_physio_data:
        raise ValueError('physio_config is only valid for physiological text or array inputs.')
    if generic_physio_data:
        cfg.update(_load_toml_flat(
            physio_config if physio_config is not None else root / 'real_data/physio.toml', 'physio'))
    if data_type in SYNTHETIC_DATA_TYPES:
        path = shepp_logan_config if data_type == 'shepp-logan' else from_image_config
        if path is None:
            raise ValueError(f'{data_type} requires its source configuration file.')
        cfg.update(_load_toml_flat(path, data_type))
    if sampling_config is not None:
        cfg.update(_load_toml_flat(sampling_config, 'sampling'))
    elif data_type in REAL_DATA_TYPES and (overrides or {}).get('kspace_sampling_type', 'from-data') == 'from-data':
        cfg.update(_load_toml_flat(root / 'sampling_simulation/from_data.toml', 'sampling'))
    if motion_simulation_config is not None:
        cfg.update(_load_toml_flat(motion_simulation_config, 'motion'))
    elif data_type in REAL_DATA_TYPES:
        cfg.update(_load_toml_flat(root / 'motion_simulation/as_is.toml', 'motion'))
    _apply_overrides(cfg, overrides, _OVERRIDE_KEYS)
    _require(cfg, {'kspace_sampling_type'}, 'sampling')
    source_only = (_SHEPP_KEYS | _IMAGE_KEYS) - {'data_dimension'}
    permitted_source = _SHEPP_KEYS if data_type == 'shepp-logan' else _IMAGE_KEYS if data_type == 'from_image' else set()
    misplaced = (cfg.keys() & source_only) - permitted_source
    if misplaced:
        raise ValueError(f'Source settings incompatible with {data_type}: {sorted(misplaced)}.')
    _validate_general(cfg)
    if saec_data:
        _validate_real_data(cfg)
    elif cfg.keys() & _REAL_DATA_KEYS:
        raise ValueError(f'Real-data settings incompatible with {data_type}: {sorted(cfg.keys() & _REAL_DATA_KEYS)}.')
    if ismrmrd_reader_data:
        _require(cfg, _ISMRMRD_READER_KEYS, 'ISMRMRD-reader')
        if type(cfg['print_raw_calibration_lines']) is not bool:
            raise ValueError('print_raw_calibration_lines must be a boolean.')
    elif cfg.keys() & _ISMRMRD_READER_KEYS:
        raise ValueError(f'ISMRMRD-reader settings incompatible with {data_type}: {sorted(cfg.keys() & _ISMRMRD_READER_KEYS)}.')
    if polaris_data:
        _require(cfg, _POLARIS_KEYS, 'Polaris')
        _choice(cfg['polaris_channel_mode'], 'polaris_channel_mode', {'all', 'largest-amplitude'})
    elif cfg.keys() & _POLARIS_KEYS:
        raise ValueError(f'Polaris settings incompatible with {data_type}: {sorted(cfg.keys() & _POLARIS_KEYS)}.')
    if polaris_data or generic_physio_data:
        _require(cfg, _PHYSIO_CLOCK_KEYS, 'physiological sensor')
        cfg['physio_clock_drift_seconds'] = float(_number(
            cfg['physio_clock_drift_seconds'], 'physio_clock_drift_seconds'))
    elif saec_data:
        # SAEC uses its Siemens stop trigger; retain zero for the shared preparer.
        shift = _number(cfg.get('physio_clock_drift_seconds', 0.0), 'physio_clock_drift_seconds')
        if shift != 0:
            raise ValueError('physio_clock_drift_seconds is supported only for Polaris, text and array physiology.')
        cfg['physio_clock_drift_seconds'] = 0.0
    elif cfg.keys() & _PHYSIO_CLOCK_KEYS:
        raise ValueError(f'Physiological clock settings incompatible with {data_type}.')
    _validate_csm(cfg)
    _validate_source(cfg)
    _validate_sampling(cfg)
    _validate_motion(cfg)
    _validate_reconstruction(cfg)
    if data_type == 'shepp-logan':
        spatial = [cfg['N_SheppLogan'], cfg['N_SheppLogan']]
        if cfg['data_dimension'] == '3D':
            spatial.append(cfg['Nz_SheppLogan'])
        params = SimpleNamespace(**cfg)
        validate_reconstruction_size(params, spatial)
        validate_calibration_size(params, spatial, has_reference=False)
        readouts = validate_sampling_size(params, cfg['N_SheppLogan'], cfg['Nz_SheppLogan'])
        validate_motion_readout_count(params, readouts)
    _apply_notebook_logging(cfg, overrides)
    return SimpleNamespace(**cfg)


def validate_reconstruction_size(params, spatial_shape):
    """Reject resolution levels that collapse a known image axis to zero."""
    # downsample_data keeps a genuinely 3D volume at depth >= 2 even when
    # its requested coarse depth rounds to zero or one. Only the other axes
    # can therefore collapse; retain the existing checks for 2D inputs.
    checked_shape = spatial_shape[:2] if len(spatial_shape) == 3 and spatial_shape[2] > 1 else spatial_shape
    for level in params.ResolutionLevels:
        if any(int(size * level) < 1 for size in checked_shape):
            raise ValueError("ResolutionLevels would produce an empty spatial dimension.")


def validate_calibration_size(params, spatial_shape, *, has_reference):
    """Validate requested calibration support against actual data dimensions."""
    if params.coil_sensitivity_method == 'espirit':
        if params.espirit_calibration_width > min(spatial_shape):
            raise ValueError('espirit_calibration_width exceeds the smallest encoded spatial dimension.')


def validate_motion_readout_count(params, readouts):
    """Validate motion-state/event counts when acquisition size is known."""
    if type(params.N_motion_states) is int and params.N_motion_states > readouts:
        raise ValueError('N_motion_states cannot exceed the number of acquired readouts.')
    if params.simulated_motion_type == 'rigid-realistic' and params.num_motion_events > readouts:
        raise ValueError('num_motion_events cannot exceed the number of acquired readouts.')
