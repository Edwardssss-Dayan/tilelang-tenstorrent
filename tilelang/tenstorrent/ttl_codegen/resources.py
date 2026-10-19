"""Resolve current Device DFB families to finite TTL binding groups."""

from __future__ import annotations

from dataclasses import dataclass

from tvm import IRModule, arith, tirx


class ResourceResolutionError(NotImplementedError):
    """A Device resource cannot be resolved to one static logical DFB group."""


_DFB_OPERAND_POSITIONS = {
    "tl.tt.dfb_reserve": (0,),
    "tl.tt.dfb_wait": (0,),
    "tl.tt.dfb_copy_wait": (0,),
    "tl.tt.dfb_release": (0,),
    "tl.tt.dfb_load": (0,),
    "tl.tt.dfb_pipe_send": (1,),
    "tl.tt.dfb_pipe_recv": (1,),
    "tl.tt.dfb_pipe_wait": (1,),
    "tl.tt.tensor_to_dfb_nd": (1,),
    "tl.tt.dfb_to_tensor_nd": (0,),
    "tl.tt.gemm_update": (0, 1),
    "tl.tt.accumulator_materialize": (1,),
    "tl.tt.compute_value_store": (1,),
    "tl.tt.compute_value_gemm": (1, 2),
    "tl.tt.dfb_local_copy": (0, 1),
    "tl.tt.dfb_gather_copy": (0, 1),
    "tl.tt.dfb_collective_reduce": (0, 1, 2),
}


def _find_column_row(column, target: int) -> int | None:
    count = int(column.count)
    values = [int(value) for value in column.values]
    period = int(column.period)
    step = int(column.step)
    if period:
        for residue, base in enumerate(values):
            if residue >= count:
                break
            if step == 0:
                if target == base:
                    return residue
                continue
            delta = target - base
            if delta % step:
                continue
            quotient = delta // step
            row = residue + quotient * period
            if quotient >= 0 and row < count:
                return row
        return None
    if values:
        try:
            return values.index(target)
        except ValueError:
            return None
    start = int(column.start)
    if step == 0:
        return 0 if count and target == start else None
    delta = target - start
    if delta % step:
        return None
    row = delta // step
    return row if 0 <= row < count else None


def _column_value(column, row: int) -> int:
    count = int(column.count)
    if row < 0 or row >= count:
        raise ResourceResolutionError(f"column row {row} is outside [0, {count})")
    values = [int(value) for value in column.values]
    period = int(column.period)
    if period:
        return values[row % period] + int(column.step) * (row // period)
    if values:
        return values[row]
    return int(column.start) + int(column.step) * row


@dataclass(frozen=True)
class ResolvedDFB:
    family_id: int
    family_row: int
    storage_group_id: int


class CurrentDFBResolver:
    """Resolve static DFB IDs without materializing current descriptor tables."""

    def __init__(self, mod: IRModule):
        attrs = mod.attrs
        self._families = list(attrs["tt.dfb_families"])
        self._relations = list(attrs["tt.dfb_storage_group_relations"])
        self._groups = {
            int(group.storage_group_id): group
            for group in attrs["tt.dfb_storage_groups"]
        }
        if len(self._families) != len(self._relations):
            raise ResourceResolutionError(
                "DFB families and storage-group relations have different lengths"
            )
        self._cache: dict[int, ResolvedDFB] = {}
        for family_id, relation in enumerate(self._relations):
            if int(relation.dfb_family_id) != family_id:
                raise ResourceResolutionError(
                    f"storage relation {int(relation.dfb_family_id)} does not "
                    f"match family {family_id}"
                )

    def resolve(self, dfb_id: int) -> ResolvedDFB:
        if dfb_id in self._cache:
            return self._cache[dfb_id]
        result = None
        for family_id, family in enumerate(self._families):
            row = _find_column_row(family.fields["dfb_id"], dfb_id)
            if row is None:
                continue
            group_id = _column_value(
                self._relations[family_id].storage_group_ids, row
            )
            if group_id not in self._groups:
                raise ResourceResolutionError(
                    f"DFB {dfb_id} resolves to missing storage group {group_id}"
                )
            candidate = ResolvedDFB(family_id, row, group_id)
            if result is not None:
                raise ResourceResolutionError(
                    f"DFB {dfb_id} appears in more than one descriptor family"
                )
            result = candidate
        if result is None:
            raise ResourceResolutionError(
                f"DFB {dfb_id} is missing from current descriptor families"
            )
        self._cache[dfb_id] = result
        return result

    def groups_for_slot(self, slot: str) -> tuple[int, ...]:
        """Return groups whose typed producer/consumer contract uses ``slot``."""

        return tuple(
            sorted(
                group_id
                for group_id, group in self._groups.items()
                if str(group.producer_slot) == slot or str(group.consumer_slot) == slot
            )
        )


def resolve_current_function_groups(mod: IRModule) -> dict[str, tuple[int, ...]]:
    """Return the finite storage-group bindings required by each kernel slot.

    Generation IDs may remain affine in compact loops or Core coordinates.
    Bindings come from the finite typed producer/consumer group contract rather
    than statement expansion.
    """

    resolver = CurrentDFBResolver(mod)
    analyzer = arith.Analyzer()
    result: dict[str, tuple[int, ...]] = {}
    for global_var, function in mod.functions.items():
        slot = str(function.attrs["tt.kernel_slot"])
        groups = set(resolver.groups_for_slot(slot))
        failure: list[ResourceResolutionError] = []

        def visit(node):
            if failure:
                return
            if not isinstance(node, tirx.Call) or not hasattr(node.op, "name"):
                return
            name = str(node.op.name)
            positions = _DFB_OPERAND_POSITIONS.get(name, ())
            if name == "tl.tt.dfb_compute":
                positions = range(len(node.args))
            elif name == "tl.tt.compute_value":
                positions = range(1, len(node.args))
            for position in positions:
                value = analyzer.simplify(node.args[position])
                if not isinstance(value, tirx.IntImm):
                    continue
                try:
                    resolved = resolver.resolve(int(value)).storage_group_id
                    if resolved not in groups:
                        failure.append(
                            ResourceResolutionError(
                                f"function {global_var.name_hint} operation {name} uses "
                                f"storage group {resolved} outside slot {slot} ownership"
                            )
                        )
                        return
                except ResourceResolutionError as error:
                    failure.append(error)
                    return

        tirx.stmt_functor.post_order_visit(function.body, visit)
        if failure:
            raise failure[0]
        result[str(global_var.name_hint)] = tuple(sorted(groups))
    return result


__all__ = [
    "CurrentDFBResolver",
    "ResolvedDFB",
    "ResourceResolutionError",
    "resolve_current_function_groups",
]
