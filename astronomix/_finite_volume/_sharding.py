"""Multi-GPU sharding support for the native-JAX finite-volume solver.

The finite-volume step is a stencil computation: it reads neighbouring cells via
periodic rolls (``custom_roll``). Under JAX's GSPMD auto-partitioning a
``jnp.roll`` on a spatially-sharded axis cannot be lowered (it becomes a
width-``N-1`` slice that is not divisible by the mesh), so the solver cannot be
sharded that way. Instead the step is run inside a ``shard_map`` where each
device holds a contiguous slab of the domain as a *local* array, and the two
collectives the step needs are provided explicitly:

* **Distributed roll** (:func:`distributed_roll`): a global periodic roll of the
  sharded axis, built from a local roll plus a ``ppermute`` of the wrapped
  boundary slab from the neighbouring device. ``custom_roll`` routes through this
  when a :func:`shard_axis_context` is active, so every stencil becomes correct
  across devices with no halo-width bookkeeping.
* **All-gather / scatter** (:func:`all_gather_sharded_axis` /
  :func:`scatter_sharded_axis`): the self-gravity Poisson solve is a global FFT
  and cannot be done per-slab, so its density is gathered to a full replicated
  array, solved, and the local slab of the result taken back.

Because ``custom_roll`` is called on arrays of different ranks (the 4-D state
``(var, x, y, z)``, a 3-D field component ``(x, y, z)``, a ``(3, x, y, z)``
field, ...), the sharded axis is identified by its **position from the end**:
the spatial axes are always trailing, and the sharded spatial dimension sits at
a fixed offset from the end (e.g. X in a 3-D run is the third-from-last axis,
``-3``). This is invariant across the array ranks the stencils use.

The context is a thread-local set by the ``shard_map`` wrapper during tracing;
``custom_roll`` and the gravity solver read it at trace time.
"""

# general
import threading
from contextlib import contextmanager

# jax
import jax
import jax.numpy as jnp

try:  # jax >= 0.8 promoted shard_map out of experimental
    from jax.shard_map import shard_map as _shard_map
except ImportError:  # pragma: no cover
    from jax.experimental.shard_map import shard_map as _shard_map


_shard_state = threading.local()


@contextmanager
def fv_sharding_context(sharding):
    """Expose the finite-volume ``NamedSharding`` to the jitted step at trace time.

    Mirrors the Pallas path's ``pallas_mesh_context``: the outer
    ``time_integration`` wraps the (traced) jit call in this context so the step,
    which does not receive ``sharding`` as an argument, can pick it up via
    :func:`get_active_fv_sharding` and wrap the finite-volume evolve in a
    ``shard_map``. ``None`` (single device) leaves the step unwrapped.
    """
    previous = getattr(_shard_state, "sharding", None)
    _shard_state.sharding = sharding
    try:
        yield
    finally:
        _shard_state.sharding = previous


def get_active_fv_sharding():
    """Return the ``NamedSharding`` set by :func:`fv_sharding_context`, or ``None``."""
    return getattr(_shard_state, "sharding", None)


class ShardAxisContext:
    """The one spatially-sharded axis, described independently of array rank.

    Attributes:
        axis_name: The mesh axis name used by ``ppermute`` / ``all_gather``.
        axis_from_end: Position of the sharded axis counted from the end of the
            array (e.g. ``-3`` for X in a 3-D run). Rank-invariant across the
            state, field and field-component arrays the stencils operate on.
        num_devices: Number of devices the axis is split across.
    """

    def __init__(self, axis_name, axis_from_end, num_devices):
        self.axis_name = axis_name
        self.axis_from_end = axis_from_end
        self.num_devices = num_devices


def get_shard_axis_context():
    """Return the active :class:`ShardAxisContext`, or ``None``."""
    return getattr(_shard_state, "context", None)


@contextmanager
def shard_axis_context(axis_name, axis_from_end, num_devices):
    """Activate a sharding context for the enclosed trace."""
    previous = getattr(_shard_state, "context", None)
    _shard_state.context = ShardAxisContext(axis_name, axis_from_end, num_devices)
    try:
        yield
    finally:
        _shard_state.context = previous


def _is_sharded_axis(array, axis, context):
    """Whether ``axis`` of ``array`` is the sharded spatial axis."""
    axis_from_end = (axis % array.ndim) - array.ndim
    return axis_from_end == context.axis_from_end


def distributed_roll(input_array, shift, axis):
    """Global periodic roll of the sharded ``axis`` inside a ``shard_map``.

    A plain local ``jnp.roll`` wraps the shift within each device's slab, which is
    wrong at the internal slab boundaries. This corrects it: the ``|shift|`` cells
    that wrapped locally are overwritten with the neighbouring device's boundary
    slab, exchanged via ``jax.lax.ppermute``. The result equals a global roll of
    the full (unsharded) array. Only small shifts (the stencil reach, ``|shift|``
    well below the slab size) are used.

    When no sharding context is active, or ``axis`` is not the sharded axis, this
    is a plain ``jnp.roll`` (so single-device numerics are unchanged).
    """
    context = get_shard_axis_context()
    if context is None or not _is_sharded_axis(input_array, axis, context):
        return jnp.roll(input_array, shift, axis=axis)

    axis = axis % input_array.ndim
    axis_name = context.axis_name
    num_devices = context.num_devices
    axis_length = input_array.shape[axis]
    normalized_shift = int(shift) % axis_length
    if normalized_shift == 0:
        return input_array

    rolled = jnp.roll(input_array, shift, axis=axis)

    if int(shift) > 0:
        # ``rolled[0:width]`` wrapped from this slab's tail but should hold the
        # previous device's last ``width`` cells; each device sends its tail
        # forward (d -> d+1).
        width = normalized_shift
        send_slab = jax.lax.slice_in_dim(
            input_array, axis_length - width, axis_length, axis=axis
        )
        received = jax.lax.ppermute(
            send_slab,
            axis_name,
            perm=[(d, (d + 1) % num_devices) for d in range(num_devices)],
        )
        rolled = jax.lax.dynamic_update_slice_in_dim(rolled, received, 0, axis)
    else:
        # ``rolled[-width:]`` should hold the next device's first ``width`` cells;
        # each device sends its head backward (d -> d-1).
        width = (-int(shift)) % axis_length
        send_slab = jax.lax.slice_in_dim(input_array, 0, width, axis=axis)
        received = jax.lax.ppermute(
            send_slab,
            axis_name,
            perm=[(d, (d - 1) % num_devices) for d in range(num_devices)],
        )
        rolled = jax.lax.dynamic_update_slice_in_dim(
            rolled, received, axis_length - width, axis
        )
    return rolled


def _resolve_sharded_axis(sharding, ndim):
    """From a ``NamedSharding``, find the single sharded array axis.

    Returns ``(array_axis, axis_name, num_devices, axis_from_end)`` for the one
    partition-spec entry whose mesh axis has more than one device. Assumes a
    single spatial axis is sharded (mesh like ``(1, num_gpus, 1, 1)``).
    """
    mesh = sharding.mesh
    spec = sharding.spec
    for array_axis, axis_name in enumerate(spec):
        if axis_name is not None and mesh.shape[axis_name] > 1:
            return (
                array_axis,
                axis_name,
                mesh.shape[axis_name],
                array_axis - ndim,
            )
    return None


def run_in_shard_map(function, primitive_state, sharding, replicated_args=()):
    """Run ``function(local_primitive_state, *replicated_args)`` in a ``shard_map``.

    ``primitive_state`` is the one sharded input/output (partitioned by
    ``sharding.spec``); ``replicated_args`` are dynamic array arguments (e.g. the
    timestep and the parameters) passed with a fully-replicated spec. They must be
    passed as arguments rather than closed over, because under JAX's explicit-mesh
    mode ``shard_map`` cannot capture inputs that carry a sharding. Purely static
    arguments (the config, registered variables, ``None``) may still be closed
    over by ``function``. The sharding context is activated inside the body so
    that ``custom_roll`` and the gravity Poisson solve emit their collectives.
    Returns the reassembled global state.
    """
    resolved = _resolve_sharded_axis(sharding, primitive_state.ndim)
    if resolved is None:
        return function(primitive_state, *replicated_args)
    _array_axis, axis_name, num_devices, axis_from_end = resolved

    def body(local_state, *args):
        with shard_axis_context(axis_name, axis_from_end, num_devices):
            return function(local_state, *args)

    replicated_spec = jax.sharding.PartitionSpec()
    in_specs = (sharding.spec,) + tuple(replicated_spec for _ in replicated_args)

    return _shard_map(
        body,
        mesh=sharding.mesh,
        in_specs=in_specs,
        out_specs=sharding.spec,
        check_rep=False,
    )(primitive_state, *replicated_args)


def distributed_max(scalar):
    """Global maximum of a per-device scalar across the sharded axis.

    Inside a ``shard_map`` a plain ``jnp.max`` reduces only over the LOCAL slab, so
    every device gets a different answer. Any control-flow decision taken on such a
    value diverges between devices -- and if the divergent branch contains a
    collective (``ppermute`` from :func:`distributed_roll`, say), the devices
    deadlock: one leaves the loop while the other waits for a partner that will
    never arrive. Reducing with ``pmax`` first makes the decision unanimous.

    Outside a sharding context this is the identity, so single-device numerics are
    bit-for-bit unchanged.
    """
    context = get_shard_axis_context()
    if context is None:
        return scalar
    return jax.lax.pmax(scalar, context.axis_name)


def all_gather_sharded_axis(local_array, axis):
    """Gather the (sharded) ``axis`` into a full replicated array.

    Used by the self-gravity Poisson FFT, which needs the whole density. Returns
    the array unchanged when no sharding context is active. ``axis`` is the
    array's own sharded-axis position (e.g. 0 for a 3-D density field).
    """
    context = get_shard_axis_context()
    if context is None:
        return local_array
    return jax.lax.all_gather(local_array, context.axis_name, axis=axis, tiled=True)


def scatter_sharded_axis(full_array, axis):
    """Take this device's slab out of a full replicated array along ``axis``.

    The inverse of :func:`all_gather_sharded_axis`. Returns the array unchanged
    when no sharding context is active.
    """
    context = get_shard_axis_context()
    if context is None:
        return full_array
    num_devices = context.num_devices
    full_length = full_array.shape[axis]
    slab = full_length // num_devices
    device_index = jax.lax.axis_index(context.axis_name)
    return jax.lax.dynamic_slice_in_dim(full_array, device_index * slab, slab, axis=axis)
