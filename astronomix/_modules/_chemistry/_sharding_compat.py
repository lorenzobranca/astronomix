"""jax-0.10 sharding compatibility shim for the Diffrax stiff solve.

The chemistry advance is a per-cell stiff ODE solved with Diffrax's implicit
``Kvaerno5``. Each Newton iteration of that solve calls into Lineax (Diffrax's
linear-solver dependency), which guards its ravel/unravel with a structural
check, :func:`lineax._misc.structure_equal`. That check compares the *pytree
structure* of the solve vector against the operator's declared output structure
by round-tripping both through ``jax.eval_shape`` and ``equinox.tree_equal``.

As of jax 0.10, ``jax.eval_shape`` evaluated inside a mesh context stamps a
concrete ``.sharding`` onto every returned ``ShapeDtypeStruct``. The solve
vector and the declared output structure then carry *different* shardings even
when their shape and dtype agree, so ``tree_equal`` returns ``False`` and Lineax
raises ``"pytree does not match out_structure"``. The net effect is that the
whole Diffrax solve fails to trace under **any** sharding (``shard_map``, GSPMD,
pmap) — which is exactly what blocked the chemistry from running multi-GPU,
while single-device runs (no sharding to compare) were unaffected.

The structural check was only ever meant to compare shape and dtype, so the fix
is to make :func:`structure_equal` sharding-agnostic: strip the sharding off the
``ShapeDtypeStruct`` leaves before comparing. This restores the pre-0.10
behaviour. Applying it is idempotent and a no-op on jax versions whose
``eval_shape`` does not attach a sharding.
"""

# general
import sys

# jax
import jax


def ensure_lineax_sharding_compatibility():
    """Make Lineax's ``structure_equal`` ignore array sharding (jax 0.10+).

    Called right after Diffrax is imported for a chemistry solve. Rebinds the
    sharding-strict ``structure_equal`` — in its defining module and in every
    module that imported it by name — to a variant that compares only shape and
    dtype. Safe to call repeatedly; returns immediately once patched.
    """
    try:
        import lineax._misc as lineax_misc
    except ImportError:  # Lineax absent -> nothing to patch.
        return

    if getattr(lineax_misc, "_astronomix_sharding_patched", False):
        return

    strip_weak_dtype = lineax_misc.strip_weak_dtype
    import equinox

    def structure_equal(first, second):
        # Reduce both operands to their shape/dtype skeleton, discarding the
        # ``.sharding`` that jax 0.10's ``eval_shape`` attaches under a mesh, then
        # compare structurally exactly as Lineax intended.
        def shape_dtype_only(tree):
            evaluated = strip_weak_dtype(jax.eval_shape(lambda: tree))
            return jax.tree.map(
                lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype),
                evaluated,
            )

        return equinox.tree_equal(
            shape_dtype_only(first), shape_dtype_only(second)
        ) is True

    # The original was imported ``from .._misc import structure_equal`` into the
    # solver modules, so each holds its own binding; rebind all of them.
    original = lineax_misc.structure_equal
    lineax_misc.structure_equal = structure_equal
    for module in list(sys.modules.values()):
        if getattr(module, "structure_equal", None) is original:
            module.structure_equal = structure_equal

    lineax_misc._astronomix_sharding_patched = True
