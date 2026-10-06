"""
Convenience functions for operations that combine multiple elements
of an array based on some stencil, e.g. b_i <- a_{i + 1} + a_{i - 1}.
Allows for code "closer to the math".
"""

# general
from functools import partial

# typing
from typing import Tuple, Union
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped

# jax
import jax
import jax.numpy as jnp

# astronomix (sharding-aware roll)
from astronomix._finite_volume._sharding import distributed_roll


def custom_roll(input_array: jnp.ndarray, shift: int, axis: int) -> jnp.ndarray:
    """Periodic roll of ``input_array`` by ``shift`` along ``axis``.

    Routes through :func:`distributed_roll`, which is a plain ``jnp.roll`` on a
    single device (or an unsharded axis) and a ``ppermute``-based global roll when
    the finite-volume step runs inside a ``shard_map`` and ``axis`` is the
    spatially-sharded one. An earlier version used a slice+concatenate for fusion,
    but that lowers to an all-gather on a sharded axis; the roll keeps the
    single-device numerics identical while letting the stencils shard.

    This helper is deliberately not ``jax.jit``-wrapped: it must be traced inline
    so the active sharding context (a thread-local read at trace time) is honoured
    rather than being fixed by a cached compilation.

    Args:
        input_array: The array to roll.
        shift: The (signed) number of positions to roll by.
        axis: The axis along which to roll.

    Returns:
        The rolled array.
    """
    return distributed_roll(input_array, shift, axis)


def _shift(input_array: jnp.ndarray, shift: int, axis: int) -> jnp.ndarray:
    """Shift ``input_array`` by ``shift`` along ``axis``.

    A thin indirection over :func:`custom_roll`: the shift is currently periodic,
    but routing every stencil through this single entry point leaves room to
    support other boundary conditions later without touching call sites.
    """
    return custom_roll(input_array, shift, axis)

# Not ``jax.jit``-wrapped: it calls ``custom_roll``, which must trace inline so
# the active sharding context is honoured (see ``custom_roll``).
def _stencil_add(
    input_array: jnp.ndarray,
    indices: Tuple[int, ...],
    factors: Tuple[Union[float, Float[Array, ""]], ...],
    axis: int,
) -> jnp.ndarray:
    """
    Combines elements of an array additively
        output_i <- sum_j factors_j * input_array_{i + indices_j}

    Args:
        input_array: The array to operate on.
        indices: output_i <- sum_j factors_j * input_array_{i + indices_j}
        factors: output_i <- sum_j factors_j * input_array_{i + indices_j}
        axis: The axis along which to operate.

    Returns:
        output_i <- sum_j factors_j * input_array_{i + indices_j}
    """

    output = sum(
        factor * custom_roll(input_array, -index, axis=axis)
        for factor, index in zip(factors, indices)
    )

    return output
