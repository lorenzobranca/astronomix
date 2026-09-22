"""
Build the chemistry configuration from a carbox reaction network file.

Turns a carbox network CSV into the pair of containers the chemistry module
consumes: a static ``ChemistryConfig`` (species count and ordering, solver, and
the network's structural half) and a dynamic ``ChemistryParams`` (the network's
array half, unit-conversion factors, physical parameters and rate modifiers).

The reaction network (a carbox ``JNetwork``) is an Equinox module, i.e. a JAX
pytree of arrays plus structure. Because it holds a Python list of reactions it
is not hashable, so it is carried in the dynamic ``ChemistryParams`` rather than
the static ``ChemistryConfig``; it stays differentiable in its rate constants.

WARNING: This helper imports carbox. It is an optional dependency; the import is
kept local so that ``import astronomix`` does not require carbox unless this
helper is actually called.
"""

# typing
from typing import Tuple

# numerics
import numpy as np

# jax
import jax.numpy as jnp

# astronomix constants
from astronomix._modules._chemistry.chemistry_options import EMULATOR, KVAERNO5

# astronomix containers
from astronomix._modules._chemistry.chemistry_options import (
    ChemistryConfig,
    ChemistryParams,
)

# Default location of KROME's Neufeld & Kaufman CO cooling table.
DEFAULT_CO_COOLING_TABLE_PATH = "/export/scratch/lbranca/krome/data/coolCO.dat"


def _load_co_cooling_table(table_path):
    """Load KROME's ``coolCO.dat`` into a dense 3D table and its grid bounds.

    The file lists, per row, the integer grid indices (i, j, k) followed by
    ``log10(T)``, ``log10(n_H + n_H2)``, ``log10(N_CO)`` and ``log10`` of the
    cooling coefficient. The grid is uniform in each log axis.

    Args:
        table_path: Path to ``coolCO.dat``.

    Returns:
        A tuple ``(table, bounds)`` with the table shaped
        ``(n_logT, n_logn, n_logNCO)`` and the six grid limits
        ``[logT_min, logT_max, logn_min, logn_max, logNCO_min, logNCO_max]``.
    """
    rows = np.loadtxt(table_path, comments="#")
    index_temperature = rows[:, 0].astype(int) - 1
    index_density = rows[:, 1].astype(int) - 1
    index_column = rows[:, 2].astype(int) - 1
    log_temperature = rows[:, 3]
    log_density = rows[:, 4]
    log_column = rows[:, 5]
    log_cooling = rows[:, 6]

    table = np.zeros(
        (
            index_temperature.max() + 1,
            index_density.max() + 1,
            index_column.max() + 1,
        )
    )
    table[index_temperature, index_density, index_column] = log_cooling
    bounds = np.array(
        [
            log_temperature.min(),
            log_temperature.max(),
            log_density.min(),
            log_density.max(),
            log_column.min(),
            log_column.max(),
        ]
    )
    return jnp.asarray(table), jnp.asarray(bounds)


def build_chemistry_from_network_file(
    network_csv_path: str,
    network_format: str,
    number_density_unit_cgs: float,
    temperature_unit_kelvin: float,
    time_unit_seconds: float,
    cosmic_ray_rate: float = 1e-17,
    fuv_field: float = 1.0,
    visual_extinction: float = 2.0,
    absolute_tolerance: float = 1e-18,
    relative_tolerance: float = 1e-10,
    solver: int = KVAERNO5,
    max_steps: int = 4096,
    reaction_chunk_size: int = 0,
    thermochemistry: bool = False,
    dust_to_gas_ratio: float = 1e-2,
    floor_temperature: float = 1e1,
    hydrogen_molecule_formation_rate_coefficient: float = 3e-17,
    cooling_courant: float = 0.1,
    co_cooling: bool = False,
    co_cooling_table_path: str = DEFAULT_CO_COOLING_TABLE_PATH,
) -> Tuple[ChemistryConfig, ChemistryParams, Tuple[str, ...]]:
    """Assemble the chemistry containers from a carbox network file.

    Args:
        network_csv_path: Path to the carbox reaction-network CSV.
        network_format: carbox parser format, e.g. ``"latent_tgas"``.
        number_density_unit_cgs: Number density [cm^-3] per code number-density
            unit (converts species densities into CGS for the network).
        temperature_unit_kelvin: Kelvin per code (rescaled) temperature unit.
        time_unit_seconds: Seconds per code time unit.
        cosmic_ray_rate: Cosmic-ray ionization rate [s^-1].
        fuv_field: FUV radiation field strength (Draine units).
        visual_extinction: Visual extinction Av [mag].
        absolute_tolerance: Absolute tolerance of the stiff solver.
        relative_tolerance: Relative tolerance of the stiff solver.
        solver: Stiff-solver tag (see ``chemistry_options``).
        max_steps: Maximum internal Diffrax steps per cell per hydro step.
        reaction_chunk_size: React the grid in sequential chunks of this many
            cells (bounds the stiff solver's peak memory) instead of a single
            vmap over the whole grid. Zero (default) reacts the whole grid at
            once. See ``ChemistryConfig.reaction_chunk_size``.
        thermochemistry: When True, evolve the temperature with the abundances
            (heating/cooling) and write the result back into the pressure field.
        dust_to_gas_ratio: Dust-to-gas mass ratio (grain photoelectric heating).
        floor_temperature: Lower bound [K] on the post-reaction temperature.
        hydrogen_molecule_formation_rate_coefficient: H + H -> H2 grain formation
            rate coefficient [cm^3 s^-1] (H2 formation heating).
        co_cooling: When True, add tabulated CO rotational cooling (loads the
            Neufeld & Kaufman table from ``co_cooling_table_path``).
        co_cooling_table_path: Path to KROME's ``coolCO.dat``.

    Returns:
        A tuple ``(chemistry_config, chemistry_params, species_names)``. The
        species ordering is returned so the caller can place initial abundances
        into the matching state slots.
    """

    # carbox is an optional dependency; import locally so the package imports
    # without it when chemistry is unused.
    from carbox.network import Network
    from carbox.parsers import parse_chemical_network

    # A dense stoichiometry matrix keeps the per-cell right-hand side a plain
    # matmul, which vmaps cleanly across the grid (the default sparse BCOO does
    # not).
    parsed_network = parse_chemical_network(network_csv_path, network_format)
    dense_network = Network(
        parsed_network.species,
        parsed_network.reactions,
        use_sparse=False,
    )
    reaction_network = dense_network.get_ode()

    species_names = tuple(species.name for species in dense_network.species)
    number_of_species = len(species_names)
    number_of_reactions = reaction_network.reactions_number

    # Resolve the species the thermochemistry terms reference. A species absent
    # from the network maps to -1, and its heating/cooling contribution drops out.
    def species_index(name):
        return species_names.index(name) if name in species_names else -1

    # Load the CO cooling table only when requested.
    co_cooling_table = jnp.array([])
    co_cooling_bounds = jnp.array([])
    if co_cooling:
        co_cooling_table, co_cooling_bounds = _load_co_cooling_table(
            co_cooling_table_path
        )

    chemistry_config = ChemistryConfig(
        chemistry=True,
        number_of_chemical_species=number_of_species,
        species_names=species_names,
        number_of_reactions=number_of_reactions,
        solver=solver,
        max_steps=max_steps,
        reaction_chunk_size=reaction_chunk_size,
        thermochemistry=thermochemistry,
        hydrogen_index=species_index("H"),
        molecular_hydrogen_index=species_index("H2"),
        electron_index=species_index("E"),
        atomic_oxygen_index=species_index("O"),
        ionized_hydrogen_index=species_index("H+"),
        helium_index=species_index("He"),
        ionized_carbon_index=species_index("C+"),
        co_cooling=co_cooling,
        carbon_monoxide_index=species_index("CO"),
    )

    chemistry_params = ChemistryParams(
        network=reaction_network,
        number_density_unit_cgs=number_density_unit_cgs,
        temperature_unit_kelvin=temperature_unit_kelvin,
        time_unit_seconds=time_unit_seconds,
        cosmic_ray_rate=cosmic_ray_rate,
        fuv_field=fuv_field,
        visual_extinction=visual_extinction,
        dust_to_gas_ratio=dust_to_gas_ratio,
        floor_temperature=floor_temperature,
        cooling_courant=cooling_courant,
        hydrogen_molecule_formation_rate_coefficient=(
            hydrogen_molecule_formation_rate_coefficient
        ),
        atol=absolute_tolerance,
        rtol=relative_tolerance,
        rate_modifier_a=jnp.ones(number_of_reactions),
        rate_modifier_b=jnp.zeros(number_of_reactions),
        co_cooling_table=co_cooling_table,
        co_cooling_bounds=co_cooling_bounds,
    )

    return chemistry_config, chemistry_params, species_names


def _species_elements_and_charge(name):
    """Element counts and charge of a species from its name (e.g. ``"H3O+"``)."""
    import re

    if name.upper() == "E":
        return {}, -1
    charge = name.count("+") - name.count("-")
    core = name.replace("+", "").replace("-", "")
    counts = {}
    for element, number in re.findall(r"([A-Z][a-z]?)(\d*)", core):
        if element:
            counts[element] = counts.get(element, 0) + (int(number) if number else 1)
    return counts, charge


def _species_hydrogen_and_charge(name):
    counts, charge = _species_elements_and_charge(name)
    return counts.get("H", 0), charge


def attach_emulator(
    chemistry_config: ChemistryConfig,
    chemistry_params: ChemistryParams,
    emulator_npz_path: str,
    project_conservation: bool = True,
    dense_emulator_npz_path: str = None,
    dense_threshold_cgs: float = 0.0,
) -> Tuple[ChemistryConfig, ChemistryParams]:
    """Replace the stiff solve by a neural emulator exported to npz.

    The npz (see ``emulator_dataset/export_fcnn_to_npz.py`` and
    ``export_multionet_to_npz.py`` in the cloud-collision project) holds the
    standardisation of the 17 quantities and of log10 nH, the time grid, the
    quantity names, the activation and residual flag, and the weights:

    * ``architecture`` absent or ``"fcnn"``: dense layers ``W<i>``/``b<i>``
      (``n_layers`` of them) on ``[state, tau, log10 nH]``;
    * ``architecture == "multionet"``: branch layers ``branch_W<i>``/``branch_b<i>``
      (``n_branch_layers``) on ``[state, log10 nH]`` and trunk layers
      ``trunk_W<i>``/``trunk_b<i>`` (``n_trunk_layers``) on ``[tau]``.

    Optional ``domain_log_nh`` / ``domain_log_t`` ([min, max] of the training set)
    enable the training-domain guard (``emulator_domain_margin_dex``): cells outside
    are left unchanged rather than extrapolated. The species order must match the
    registered network's, and thermochemistry must be on (the emulator returns the
    temperature).

    Args:
        chemistry_config: Configuration built by ``build_chemistry_from_network_file``.
        chemistry_params: Its parameters.
        emulator_npz_path: The exported model.
        project_conservation: Rescale H-bearing species to the hydrogen budget and
            reset electrons to neutrality after every emulator step.
        dense_emulator_npz_path: Optional second exported model for the cells at or
            above ``dense_threshold_cgs`` hydrogen nuclei per cm^3 (same species,
            activation and residual flag; own standardisation, time grid and
            domain). See ``ChemistryConfig.emulator_dense_threshold_cgs``.
        dense_threshold_cgs: The density split [cm^-3]; required (> 0) with a dense
            model.

    Returns:
        The updated (config, params) with ``solver == EMULATOR``.
    """
    if not chemistry_config.thermochemistry:
        raise ValueError("the emulator predicts the temperature: enable thermochemistry")

    def parse(path):
        """Load one exported model; returns (data, architecture, weights, biases, trunk_w, trunk_b)."""
        data = np.load(path, allow_pickle=True)
        names = tuple(str(name) for name in data["quantity_names"][:-1])
        if names != tuple(chemistry_config.species_names):
            raise ValueError(
                f"emulator species {names} do not match the network's {chemistry_config.species_names}"
            )
        architecture = str(data["architecture"]) if "architecture" in data else "fcnn"

        def layers(prefix, count_key):
            count = int(data[count_key])
            weights = tuple(jnp.asarray(data[f"{prefix}W{i}"], dtype=jnp.float64) for i in range(count))
            biases = tuple(jnp.asarray(data[f"{prefix}b{i}"], dtype=jnp.float64) for i in range(count))
            return weights, biases

        n_quantities = len(names) + 1
        if architecture == "fcnn":
            weights, biases = layers("", "n_layers")
            trunk_weights, trunk_biases = (), ()
            if weights[0].shape[1] != n_quantities + 2 or weights[-1].shape[0] != n_quantities:
                raise ValueError(f"fcnn layer shapes {[w.shape for w in weights]} do not fit {n_quantities} quantities")
        elif architecture == "multionet":
            weights, biases = layers("branch_", "n_branch_layers")
            trunk_weights, trunk_biases = layers("trunk_", "n_trunk_layers")
            if weights[0].shape[1] != n_quantities + 1 or trunk_weights[0].shape[1] != 1:
                raise ValueError("multionet expects the branch net on [state, log10 nH] and the trunk net on [tau]")
            if weights[-1].shape[0] != trunk_weights[-1].shape[0] or weights[-1].shape[0] < n_quantities:
                raise ValueError("branch and trunk nets must emit the same number of outputs, at least one per quantity")
        else:
            raise ValueError(f"unknown emulator architecture {architecture!r}")
        return data, architecture, weights, biases, trunk_weights, trunk_biases

    def domain(data, key):
        return jnp.asarray(data[key], dtype=jnp.float64) if key in data else jnp.array([])

    data, architecture, weights, biases, trunk_weights, trunk_biases = parse(emulator_npz_path)
    names = tuple(str(name) for name in data["quantity_names"][:-1])
    hydrogen, charge = zip(*(_species_hydrogen_and_charge(name) for name in names))
    parsed = [_species_elements_and_charge(name)[0] for name in names]
    elements = sorted({element for counts in parsed for element in counts})
    element_matrix = np.array(
        [[counts.get(element, 0) for element in elements] for counts in parsed], dtype=np.float64
    )

    config = chemistry_config._replace(
        solver=EMULATOR,
        emulator_architecture=architecture,
        emulator_activation=str(data["activation"]).lower(),
        emulator_residual=bool(data["residual"]),
        emulator_project_conservation=project_conservation,
    )
    params = chemistry_params._replace(
        emulator_weights=weights,
        emulator_biases=biases,
        emulator_trunk_weights=trunk_weights,
        emulator_trunk_biases=trunk_biases,
        emulator_input_mean=jnp.asarray(data["input_mean"], dtype=jnp.float64),
        emulator_input_std=jnp.asarray(data["input_std"], dtype=jnp.float64),
        emulator_param_mean=float(data["param_mean"]),
        emulator_param_std=float(data["param_std"]),
        emulator_time_grid=jnp.asarray(data["time_grid_seconds"], dtype=jnp.float64),
        emulator_hydrogen_atoms=jnp.asarray(hydrogen, dtype=jnp.float64),
        emulator_charges=jnp.asarray(charge, dtype=jnp.float64),
        emulator_element_matrix=jnp.asarray(element_matrix),
        emulator_domain_log_nh=domain(data, "domain_log_nh"),
        emulator_domain_log_t=domain(data, "domain_log_t"),
    )
    if dense_emulator_npz_path is None:
        return config, params
    if not dense_threshold_cgs > 0:
        raise ValueError("a dense-gas emulator needs dense_threshold_cgs > 0")
    dense, dense_architecture, dense_weights, dense_biases, dense_trunk_w, dense_trunk_b = parse(
        dense_emulator_npz_path
    )
    if str(dense["activation"]).lower() != config.emulator_activation or bool(dense["residual"]) != config.emulator_residual:
        raise ValueError("the dense-gas emulator must share the main model's activation and residual flag")
    config = config._replace(
        emulator_dense_threshold_cgs=float(dense_threshold_cgs),
        emulator_dense_architecture=dense_architecture,
    )
    params = params._replace(
        emulator_dense_weights=dense_weights,
        emulator_dense_biases=dense_biases,
        emulator_dense_trunk_weights=dense_trunk_w,
        emulator_dense_trunk_biases=dense_trunk_b,
        emulator_dense_input_mean=jnp.asarray(dense["input_mean"], dtype=jnp.float64),
        emulator_dense_input_std=jnp.asarray(dense["input_std"], dtype=jnp.float64),
        emulator_dense_param_mean=float(dense["param_mean"]),
        emulator_dense_param_std=float(dense["param_std"]),
        emulator_dense_time_grid=jnp.asarray(dense["time_grid_seconds"], dtype=jnp.float64),
        emulator_dense_domain_log_nh=domain(dense, "domain_log_nh"),
        emulator_dense_domain_log_t=domain(dense, "domain_log_t"),
    )
    return config, params
