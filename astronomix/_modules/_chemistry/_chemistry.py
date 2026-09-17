"""
Per-cell astrochemistry as an operator-split source term.

Once per hydro step this module reacts the chemical species carried in the state
array. The species advect with the flow as passive volumetric densities (handled
by the finite-volume solver); here we freeze the fluid and, in every cell,
integrate a carbox reaction network over the hydro time step with a stiff Diffrax
solver. Concretely the driver:

  1. reads the species block and the local temperature from the primitive state,
  2. crosses the unit boundary from astronomix code units into the CGS units the
     network expects (number densities in cm^-3, temperature in Kelvin, time in
     seconds),
  3. integrates the network per cell (vmapped over the grid), and
  4. writes the updated species back into the state array.

The carbox network is reassembled from the static structure held in the
configuration and the dynamic array leaves held in the parameters, mirroring how
the neural-network cooling curve is embedded (see ``_modules/_cooling``).

When ``chemistry_config.thermochemistry`` is enabled the temperature is evolved
*together* with the abundances (an extra ODE variable per cell), and the evolved
temperature is converted back into the pressure field — so chemistry and hydro
exchange energy. When it is disabled the species react at a fixed temperature and
the energy field is left untouched. The finite-difference source-term path is a
deliberate follow-up.

WARNING: The network integration uses Diffrax, which is an optional dependency.
It is imported lazily inside the driver so that ``import astronomix`` does not
require Diffrax (or carbox) when the chemistry module is inactive.
"""

# general
from functools import partial

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix._modules._chemistry.chemistry_options import EMULATOR, KVAERNO5
from astronomix.option_classes.simulation_config import STATE_TYPE

# astronomix containers
from astronomix._modules._chemistry.chemistry_options import ChemistryConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._modules._chemistry._thermochemistry import temperature_derivative
from astronomix._modules._cooling._cooling import (
    get_pressure_from_temperature,
    get_temperature_from_pressure,
)


def _advance_single_cell(
    cell_abundances,
    cell_temperature_kelvin,
    time_step_seconds,
    reaction_network,
    rate_modifier_a,
    rate_modifier_b,
    cosmic_ray_rate,
    fuv_field,
    visual_extinction,
    absolute_tolerance,
    relative_tolerance,
    solver,
    max_steps,
    thermochemistry,
    adiabatic_index,
    dust_to_gas_ratio,
    hydrogen_molecule_formation_rate_coefficient,
    hydrogen_index,
    molecular_hydrogen_index,
    electron_index,
    atomic_oxygen_index,
    ionized_hydrogen_index,
    helium_index,
    ionized_carbon_index,
    co_cooling,
    carbon_monoxide_index,
    co_cooling_table,
    co_cooling_bounds,
):
    """Advance one cell's chemistry over ``time_step_seconds`` with a stiff solve.

    This is the 0-D box integration carbox performs, reduced to a single short
    step so it can serve as an operator-split source term (no snapshot grid, just
    integrate from zero to the hydro time step and read the end state). It is
    written for a single cell and vmapped over the grid by the driver.

    With ``thermochemistry`` enabled the integrated state is the abundances with
    the temperature (Kelvin) appended as a final element, so the temperature
    evolves under net heating/cooling while the reaction rates see the changing
    temperature. Otherwise only the abundances are integrated, at fixed
    temperature.

    Args:
        cell_abundances: Absolute species number densities [cm^-3] for this cell.
        cell_temperature_kelvin: Gas temperature in Kelvin for this cell.
        time_step_seconds: Chemistry sub-step length in seconds (the hydro step).
        reaction_network: The reassembled carbox ``JNetwork`` right-hand side.
        rate_modifier_a: Per-reaction multiplicative rate modifier.
        rate_modifier_b: Per-reaction additive rate modifier.
        cosmic_ray_rate: Cosmic-ray ionization rate [s^-1].
        fuv_field: FUV radiation field strength (Draine units).
        visual_extinction: Visual extinction Av [mag].
        absolute_tolerance: Absolute tolerance for the stiff solver.
        relative_tolerance: Relative tolerance for the stiff solver.
        solver: The Diffrax solver instance to integrate with.
        max_steps: Maximum internal Diffrax steps.
        thermochemistry: When True, evolve the temperature alongside the species.
        adiabatic_index: The adiabatic index gamma (for dT/dt).
        dust_to_gas_ratio: The dust-to-gas mass ratio (for photoelectric heating).
        hydrogen_molecule_formation_rate_coefficient: H + H -> H2 grain formation
            rate coefficient [cm^3 s^-1] (for H2 formation heating).
        hydrogen_index: Index of atomic H in the abundances (or -1).
        molecular_hydrogen_index: Index of H2 (or -1).
        electron_index: Index of the electron (or -1).
        atomic_oxygen_index: Index of atomic O (or -1).
        ionized_hydrogen_index: Index of H+ (or -1).
        helium_index: Index of He (or -1).
        ionized_carbon_index: Index of C+ (or -1).
        co_cooling: Whether to add tabulated CO rotational cooling.
        carbon_monoxide_index: Index of CO (or -1).
        co_cooling_table: The 3D CO cooling table.
        co_cooling_bounds: The CO table grid limits.

    Returns:
        The end-of-step cell state: the updated abundances [cm^-3] alone, or the
        abundances with the updated temperature [K] appended when
        ``thermochemistry`` is enabled.
    """

    # Diffrax is an optional dependency; keep the import local so the package
    # imports without it when chemistry is inactive.
    import diffrax as dx

    # Diffrax's implicit solve goes through Lineax, whose structural check is
    # sharding-strict on jax 0.10 and would otherwise block the solve under any
    # mesh; make it sharding-agnostic so the chemistry can run multi-GPU.
    from ._sharding_compat import ensure_lineax_sharding_compatibility

    ensure_lineax_sharding_compatibility()

    # The chemistry right-hand side; the physical parameters shared across cells
    # are closed over rather than threaded through Diffrax ``args``.
    def chemistry_derivative(time, abundances, temperature_kelvin):
        # The stiff solver probes intermediate states that can transiently
        # overshoot to negative abundances or a non-physical temperature; clamp
        # both before the network evaluates its rate products (T enters the
        # Arrhenius/temperature-dependent rates), so a pathological cell cannot
        # emit NaNs/infs that would poison the whole vmapped solve. Physical
        # states are untouched.
        abundances = jnp.maximum(abundances, 0.0)
        temperature_kelvin = jnp.clip(temperature_kelvin, 3.0, 1.0e9)
        return reaction_network(
            time,
            abundances,
            temperature_kelvin,
            cosmic_ray_rate,
            fuv_field,
            visual_extinction,
            rate_modifier_a,
            rate_modifier_b,
        )

    if thermochemistry:
        # State = [abundances, temperature]; the temperature evolves under net
        # heating/cooling and feeds back into the reaction rates.
        def vector_field(time, state, args):
            abundances = state[:-1]
            temperature_kelvin = state[-1]
            abundance_derivative = chemistry_derivative(
                time,
                abundances,
                temperature_kelvin,
            )
            temperature_rate = temperature_derivative(
                abundances,
                temperature_kelvin,
                adiabatic_index,
                cosmic_ray_rate,
                fuv_field,
                dust_to_gas_ratio,
                hydrogen_molecule_formation_rate_coefficient,
                hydrogen_index,
                molecular_hydrogen_index,
                electron_index,
                atomic_oxygen_index,
                ionized_hydrogen_index,
                helium_index,
                ionized_carbon_index,
                co_cooling,
                carbon_monoxide_index,
                co_cooling_table,
                co_cooling_bounds,
            )
            return jnp.concatenate(
                [abundance_derivative, jnp.reshape(temperature_rate, (1,))]
            )

        initial_state = jnp.concatenate(
            [cell_abundances, jnp.reshape(cell_temperature_kelvin, (1,))]
        )
    else:
        # Temperature is a fixed parameter of the reaction rates for this step.
        def vector_field(time, abundances, args):
            return chemistry_derivative(time, abundances, cell_temperature_kelvin)

        initial_state = cell_abundances

    ode_term = dx.ODETerm(vector_field)

    solution = dx.diffeqsolve(
        ode_term,
        solver,
        t0=0.0,
        t1=time_step_seconds,
        dt0=jnp.minimum(1e-6, time_step_seconds * 1e-3),
        y0=initial_state,
        args=None,
        stepsize_controller=dx.PIDController(
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        ),
        saveat=dx.SaveAt(t1=True),
        max_steps=max_steps,
    )

    # Only the end state matters for an operator-split sub-step. Guard against a
    # cell whose stiff solve did not converge (e.g. hit ``max_steps`` on a
    # pathological, strongly-compressed cell): Diffrax then returns a non-success
    # result and its final value can be non-finite. Rather than let one such cell
    # poison the whole vmapped grid with a NaN, keep that cell's pre-reaction
    # state for this operator-split sub-step (it reacts again next step). The
    # element-wise finite check is a final backstop.
    final_state = solution.ys[-1]
    solve_succeeded = solution.result == dx.RESULTS.successful
    final_state = jnp.where(solve_succeeded, final_state, initial_state)
    final_state = jnp.where(jnp.isfinite(final_state), final_state, initial_state)
    return final_state


def _emulator_activation(name):
    """The hidden-layer activation by its (lower-case) PyTorch name.

    ``gelu`` is the exact erf form, as ``torch.nn.GELU()`` (JAX's default is the
    tanh approximation, which differs at the 1e-3 level).
    """
    return {
        "softplus": jax.nn.softplus,
        "gelu": lambda x: jax.nn.gelu(x, approximate=False),
        "silu": jax.nn.silu,
        "relu": jax.nn.relu,
        "tanh": jnp.tanh,
    }[name]


def _dense_stack(features, weights, biases, activation):
    """``act(W h + b)`` through all but the last layer, which is linear."""
    hidden = features
    for weight, bias in zip(weights[:-1], biases[:-1]):
        hidden = activation(weight @ hidden + bias)
    return weights[-1] @ hidden + biases[-1]


def _emulator_network(state, tau, parameter, chemistry_config, chemistry_params):
    """Evaluate the emulator on one cell's standardised inputs.

    Args:
        state: Standardised ``[log10 x_i, log10 T]`` (``n_quantities``).
        tau: Index fraction of the elapsed time on the model's time grid (scalar).
        parameter: Standardised ``log10 n_H`` (scalar).

    Returns:
        The network output in standardised units (``n_quantities``), before the
        residual skip.

    ``fcnn``: one dense stack on ``concat(state, tau, parameter)`` (the CODES
    ``FullyConnected`` input layout).

    ``multionet``: the branch net on ``concat(state, parameter)`` and the trunk net
    on ``tau`` each emit ``output_factor * n_quantities`` values, which are split
    into ``n_quantities`` contiguous chunks (any remainder goes one-per-chunk to the
    first chunks, as CODES does) and dotted chunk by chunk. The fixed parameter is
    fed to the branch net (CODES ``params_branch=True``, the only layout exported).
    """
    p = chemistry_params
    activation = _emulator_activation(chemistry_config.emulator_activation)
    tau = jnp.reshape(tau, (1,))
    parameter = jnp.reshape(parameter, (1,))
    if chemistry_config.emulator_architecture == "fcnn":
        features = jnp.concatenate([state, tau, parameter])
        return _dense_stack(features, p.emulator_weights, p.emulator_biases, activation)
    if chemistry_config.emulator_architecture == "multionet":
        branch = _dense_stack(
            jnp.concatenate([state, parameter]), p.emulator_weights, p.emulator_biases, activation
        )
        trunk = _dense_stack(tau, p.emulator_trunk_weights, p.emulator_trunk_biases, activation)
        n_quantities = state.shape[0]
        total = branch.shape[0]
        base, remainder = divmod(total, n_quantities)  # static: shapes are known at trace time
        sizes = [base + 1 if i < remainder else base for i in range(n_quantities)]
        bounds = [0]
        for size in sizes:
            bounds.append(bounds[-1] + size)
        return jnp.stack(
            [jnp.dot(branch[lo:hi], trunk[lo:hi]) for lo, hi in zip(bounds[:-1], bounds[1:])]
        )
    raise ValueError(f"unknown emulator architecture {chemistry_config.emulator_architecture!r}")


def _emulate_single_cell(
    cell_abundances,
    cell_temperature_kelvin,
    time_step_seconds,
    chemistry_config,
    chemistry_params,
):
    """Advance one cell with the neural emulator instead of the stiff solve.

    Mirrors the operator the emulator was trained on (a CODES ``FullyConnected`` /
    ``FullyConnectedResidual`` or ``MultiONet`` / ``MultiONetResidual`` surrogate):
    the input is the standardised ``[log10 x_i, log10 T]`` with
    ``x_i = n_i / n_H,nuclei``, the index fraction of the elapsed time on the
    model's time grid, and the standardised ``log10 n_H``; the output is the
    standardised ``[log10 x_i(t), log10 T(t)]`` (added to the input for the
    residual variants). See ``_emulator_network`` for the two architectures.

    Afterwards, optionally, the H-bearing species are rescaled to the cell's
    hydrogen-nuclei budget (which the hydro conserves and the emulator does not
    know about) and the electrons are reset to charge neutrality.

    Args:
        cell_abundances: Species number densities [cm^-3].
        cell_temperature_kelvin: Gas temperature [K].
        time_step_seconds: Elapsed time [s] (the hydro step).
        chemistry_config: Static configuration (activation, residual flag, roles).
        chemistry_params: The emulator leaves (weights, standardisation, grid).

    Returns:
        ``concat(abundances [cm^-3], T [K])`` as ``_advance_single_cell`` with
        thermochemistry does.
    """
    p = chemistry_params
    hydrogen_atoms = p.emulator_hydrogen_atoms
    hydrogen_nuclei = jnp.maximum(jnp.dot(hydrogen_atoms, cell_abundances), 1e-30)

    log_fractions = jnp.log10(jnp.clip(cell_abundances / hydrogen_nuclei, 1e-30, None))
    log_temperature = jnp.log10(jnp.clip(cell_temperature_kelvin, 1.0, 1e9))
    state = jnp.concatenate([log_fractions, jnp.reshape(log_temperature, (1,))])
    standardised = (state - p.emulator_input_mean) / p.emulator_input_std
    # Keep the network inside the range it was trained on; the residual below is
    # still added to the TRUE (unclipped) state, so this only bounds the correction.
    clip = chemistry_config.emulator_input_clip_sigma
    network_input = jnp.clip(standardised, -clip, clip) if clip > 0 else standardised

    grid = p.emulator_time_grid
    tau = jnp.interp(
        jnp.clip(time_step_seconds, grid[0], grid[-1]),
        grid,
        jnp.linspace(0.0, 1.0, grid.shape[0]),
    )
    parameter = (jnp.log10(hydrogen_nuclei) - p.emulator_param_mean) / p.emulator_param_std
    parameter = jnp.clip(parameter, -clip, clip) if clip > 0 else parameter

    output = _emulator_network(network_input, tau, parameter, chemistry_config, p)
    if chemistry_config.emulator_residual:
        output = output + standardised

    new_state = output * p.emulator_input_std + p.emulator_input_mean
    # Cap the per-step change (dex). The stiff solve never moves a species by more
    # than ~0.2 dex or T by more than ~0.5 dex in one hydro step on this network,
    # so anything beyond is extrapolation error, not chemistry.
    change = new_state - state
    cap_x = chemistry_config.emulator_max_log_change
    cap_t = chemistry_config.emulator_max_log_temperature_change
    if cap_x > 0:
        change = change.at[:-1].set(jnp.clip(change[:-1], -cap_x, cap_x))
    if cap_t > 0:
        change = change.at[-1].set(jnp.clip(change[-1], -cap_t, cap_t))
    new_state = state + change
    new_abundances = 10.0 ** new_state[:-1] * hydrogen_nuclei
    new_temperature = 10.0 ** new_state[-1]

    if chemistry_config.emulator_project_conservation:
        # Elements: rescale each element's species so the element's atoms sum to
        # the budget the cell entered with (the hydro conserves them, the emulator
        # does not know about them). Species sharing elements (CO, HCO+, ...) make
        # the passes interact, so sweep the elements twice; the residual
        # non-conservation after that is far below the emulator's own error.
        element_matrix = p.emulator_element_matrix  # (species, elements)
        budgets = jnp.maximum(cell_abundances @ element_matrix, 1e-30)
        for _ in range(2):
            for element in range(element_matrix.shape[1]):
                atoms = element_matrix[:, element]
                predicted = jnp.maximum(jnp.dot(atoms, new_abundances), 1e-30)
                scale = budgets[element] / predicted
                new_abundances = jnp.where(atoms > 0, new_abundances * scale, new_abundances)
        # Charge: electrons balance the ions.
        electron = chemistry_config.electron_index
        if electron >= 0:
            charges = p.emulator_charges.at[electron].set(0.0)
            electrons = jnp.maximum(jnp.dot(charges, new_abundances), 1e-30)
            new_abundances = new_abundances.at[electron].set(electrons)

    new_abundances = jnp.where(jnp.isfinite(new_abundances), new_abundances, cell_abundances)
    new_temperature = jnp.where(jnp.isfinite(new_temperature), new_temperature, cell_temperature_kelvin)
    return jnp.concatenate([new_abundances, jnp.reshape(new_temperature, (1,))])


@partial(jax.jit, static_argnames=("chemistry_config", "registered_variables"))
def update_chemistry(
    primitive_state: STATE_TYPE,
    registered_variables: RegisteredVariables,
    chemistry_config: ChemistryConfig,
    simulation_params: SimulationParams,
    time_step: float,
) -> STATE_TYPE:
    """React the chemical species of the primitive state for one time step.

    Reads the species block and the local temperature, integrates the reaction
    network in every cell over ``time_step`` and writes the updated species
    densities back. When ``chemistry_config.thermochemistry`` is enabled the
    temperature is evolved alongside the species and the resulting pressure is
    written back into the energy field; otherwise the energy field is untouched.

    Args:
        primitive_state: The primitive state array.
        registered_variables: The registered variables (locate the species block).
        chemistry_config: The static chemistry configuration (network structure,
            species count, solver, thermochemistry flag, species-role indices).
        simulation_params: The simulation parameters (network leaves, unit
            factors, physical parameters, tolerances, gamma).
        time_step: The hydro time step, in code units.

    Returns:
        The primitive state with the species block advanced by one reaction step
        (and, with thermochemistry, the pressure updated by heating/cooling).
    """

    chemistry_params = simulation_params.chemistry_params

    # -------------------------------------------------------------
    # =============== ↓ Pick the reaction network and solver ↓ ====
    # -------------------------------------------------------------

    # The network rides in the (dynamic) parameters rather than the static
    # configuration, because a carbox ``JNetwork`` holds a Python list of
    # reactions and so is not hashable.
    reaction_network = chemistry_params.network

    if chemistry_config.solver == EMULATOR:
        solver = None  # no ODE solve: the neural emulator replaces it
    elif chemistry_config.solver == KVAERNO5:
        import diffrax as dx

        from ._sharding_compat import ensure_lineax_sharding_compatibility

        ensure_lineax_sharding_compatibility()
        solver = dx.Kvaerno5()
    else:
        raise ValueError(
            f"Unsupported chemistry solver tag: {chemistry_config.solver}"
        )

    # -------------------------------------------------------------
    # =============== ↑ Pick the reaction network and solver ↑ ====
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # =============== ↓ Cross into CGS units ↓ ====================
    # -------------------------------------------------------------

    density = primitive_state[registered_variables.density_index]
    pressure = primitive_state[registered_variables.pressure_index]

    # The pressure-to-temperature relation returns the rescaled code temperature;
    # scale it into Kelvin for the rate expressions.
    code_temperature = get_temperature_from_pressure(
        density,
        pressure,
        chemistry_params.hydrogen_mass_fraction,
        chemistry_params.metal_mass_fraction,
    )
    temperature_kelvin = code_temperature * chemistry_params.temperature_unit_kelvin

    # Clamp the temperature that seeds the stiff solve to a physical range. In a
    # run with extreme density contrast a floored/shocked cell can yield a
    # non-physical P/rho (hence T); starting the solver from a sane value keeps
    # the reaction network well-posed (the thermochemistry rate expressions apply
    # their own internal clamp too).
    temperature_kelvin = jnp.clip(
        temperature_kelvin, chemistry_params.floor_temperature, 1.0e9
    )

    time_step_seconds = time_step * chemistry_params.time_unit_seconds

    # The species block holds code-unit number densities; the network works in
    # absolute number densities [cm^-3].
    species_start = registered_variables.chemistry_species_index
    number_of_species = registered_variables.num_chemical_species
    species_code_units = primitive_state[
        species_start : species_start + number_of_species
    ]
    species_number_densities = (
        species_code_units * chemistry_params.number_density_unit_cgs
    )

    # -------------------------------------------------------------
    # =============== ↑ Cross into CGS units ↑ ====================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # =============== ↓ React every cell (vmapped) ↓ ==============
    # -------------------------------------------------------------

    # Flatten the spatial grid so each cell is one row: the species array goes
    # from ``(num_species, *grid)`` to ``(num_cells, num_species)`` and the
    # temperature from ``(*grid,)`` to ``(num_cells,)``.
    grid_shape = temperature_kelvin.shape
    number_of_cells = temperature_kelvin.size

    species_per_cell = species_number_densities.reshape(
        number_of_species, number_of_cells
    ).T
    temperature_per_cell = temperature_kelvin.reshape(number_of_cells)

    # Bind everything that is shared across cells; vmap only over the per-cell
    # abundances and temperature.
    if chemistry_config.solver == EMULATOR:
        react_one_cell = partial(
            _emulate_single_cell,
            time_step_seconds=time_step_seconds,
            chemistry_config=chemistry_config,
            chemistry_params=chemistry_params,
        )
    else:
        react_one_cell = partial(
            _advance_single_cell,
            time_step_seconds=time_step_seconds,
            reaction_network=reaction_network,
            rate_modifier_a=chemistry_params.rate_modifier_a,
            rate_modifier_b=chemistry_params.rate_modifier_b,
            cosmic_ray_rate=chemistry_params.cosmic_ray_rate,
            fuv_field=chemistry_params.fuv_field,
            visual_extinction=chemistry_params.visual_extinction,
            absolute_tolerance=chemistry_params.atol,
            relative_tolerance=chemistry_params.rtol,
            solver=solver,
            max_steps=chemistry_config.max_steps,
            thermochemistry=chemistry_config.thermochemistry,
            adiabatic_index=simulation_params.gamma,
            dust_to_gas_ratio=chemistry_params.dust_to_gas_ratio,
            hydrogen_molecule_formation_rate_coefficient=(
                chemistry_params.hydrogen_molecule_formation_rate_coefficient
            ),
            hydrogen_index=chemistry_config.hydrogen_index,
            molecular_hydrogen_index=chemistry_config.molecular_hydrogen_index,
            electron_index=chemistry_config.electron_index,
            atomic_oxygen_index=chemistry_config.atomic_oxygen_index,
            ionized_hydrogen_index=chemistry_config.ionized_hydrogen_index,
            helium_index=chemistry_config.helium_index,
            ionized_carbon_index=chemistry_config.ionized_carbon_index,
            co_cooling=chemistry_config.co_cooling,
            carbon_monoxide_index=chemistry_config.carbon_monoxide_index,
            co_cooling_table=chemistry_params.co_cooling_table,
            co_cooling_bounds=chemistry_params.co_cooling_bounds,
        )
    # At full grid resolution a single ``vmap`` over every cell materialises the
    # stiff solver's per-cell working set for the whole grid at once, which is
    # the memory bottleneck (the state itself is small). When a chunk size is
    # configured and smaller than the grid, react the grid in sequential chunks
    # via ``jax.lax.map``: it vmaps ``react_one_cell`` over ``batch_size`` cells
    # at a time (so a saturated chunk keeps full throughput) and scans over the
    # chunks (so only one chunk's working set is live). The per-cell solves are
    # independent, so this is numerically identical to the unchunked vmap and
    # only bounds peak memory.
    chunk_size = chemistry_config.reaction_chunk_size
    if 0 < chunk_size < number_of_cells:
        reacted_per_cell = jax.lax.map(
            lambda cell: react_one_cell(cell[0], cell[1]),
            (species_per_cell, temperature_per_cell),
            batch_size=chunk_size,
        )
    else:
        reacted_per_cell = jax.vmap(react_one_cell)(
            species_per_cell,
            temperature_per_cell,
        )

    # -------------------------------------------------------------
    # =============== ↑ React every cell (vmapped) ↑ ==============
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # =============== ↓ Cross back and write the state ↓ ==========
    # -------------------------------------------------------------

    # The first ``number_of_species`` columns are the reacted abundances; with
    # thermochemistry a final column carries the evolved temperature (Kelvin).
    reacted_species_per_cell = reacted_per_cell[:, :number_of_species]

    # Undo the flatten and the CGS scaling, then write the species block back.
    reacted_species_number_densities = reacted_species_per_cell.T.reshape(
        number_of_species, *grid_shape
    )
    reacted_species_code_units = (
        reacted_species_number_densities / chemistry_params.number_density_unit_cgs
    )
    primitive_state = primitive_state.at[
        species_start : species_start + number_of_species
    ].set(reacted_species_code_units)

    # With thermochemistry, convert the evolved temperature back into a pressure
    # and write the energy field, mirroring the cooling module's round-trip.
    if chemistry_config.thermochemistry:
        reacted_temperature_kelvin = reacted_per_cell[:, number_of_species].reshape(
            grid_shape
        )

        # Cooling limiter: cap the fractional temperature change per operator-
        # split step to ``cooling_courant``. Applying the full stiff-cooling
        # temperature drop in one step makes the fixed-grid hydro (which then sees
        # the new pressure) go unstable when the cooling time is far below the CFL
        # step; limiting |dT|/T per step bounds the pressure change the hydro sees
        # and keeps the coupling stable, while the timestep stays at the (fast)
        # hydro CFL. A cell that wants to cool further simply does so over the next
        # few steps (it converges to the same equilibrium). This is the standard
        # remedy for stiff radiative cooling coupled to a grid scheme.
        maximum_fractional_change = chemistry_params.cooling_courant
        reacted_temperature_kelvin = jnp.clip(
            reacted_temperature_kelvin,
            temperature_kelvin * (1.0 - maximum_fractional_change),
            temperature_kelvin * (1.0 + maximum_fractional_change),
        )

        # Never let heating/cooling push the temperature below the floor.
        reacted_temperature_kelvin = jnp.maximum(
            reacted_temperature_kelvin,
            chemistry_params.floor_temperature,
        )

        # Kelvin -> rescaled code temperature (inverse of the read above), then
        # code temperature -> pressure via the shared equation of state.
        reacted_code_temperature = (
            reacted_temperature_kelvin / chemistry_params.temperature_unit_kelvin
        )
        new_pressure = get_pressure_from_temperature(
            density,
            reacted_code_temperature,
            chemistry_params.hydrogen_mass_fraction,
            chemistry_params.metal_mass_fraction,
        )
        primitive_state = primitive_state.at[
            registered_variables.pressure_index
        ].set(new_pressure)

    # -------------------------------------------------------------
    # =============== ↑ Cross back and write the state ↑ ==========
    # -------------------------------------------------------------

    return primitive_state

