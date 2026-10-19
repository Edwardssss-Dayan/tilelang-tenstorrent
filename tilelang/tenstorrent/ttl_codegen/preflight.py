"""Pure Device-IR-to-TTL mapping preflight with no TT-Lang dependency."""

from __future__ import annotations

from tvm import IRModule, tirx

from .resources import ResourceResolutionError, resolve_current_function_groups


class TTLMappingPreflightError(NotImplementedError):
    """The verified Device program has no complete current TTL mapping."""


def preflight_ttl_mapping(mod: IRModule) -> None:
    attrs = mod.attrs
    issues: set[str] = set()
    required = (
        "tt.dfb_families",
        "tt.dfb_storage_groups",
        "tt.dfb_storage_group_relations",
        "tt.communication_step_families",
        "tt.transport_families",
        "tt.kernel_order",
    )
    for name in required:
        if name not in attrs:
            issues.add(f"missing required current Device attribute {name}")

    has_control_flow = False
    operation_names: set[str] = set()
    for function in mod.functions.values():
        def visit(node):
            nonlocal has_control_flow
            has_control_flow |= isinstance(node, (tirx.For, tirx.IfThenElse))
            if isinstance(node, tirx.Call) and hasattr(node.op, "name"):
                name = str(node.op.name)
                operation_names.add(name)

        tirx.stmt_functor.post_order_visit(function.body, visit)

    if has_control_flow:
        issues.add("shared Core/SCF control-flow emission is not implemented")
    if attrs.get("tt.communication_step_families", []):
        issues.add("body-driven communication step and Pipe transport emission is not implemented")
    try:
        resolve_current_function_groups(mod)
    except (KeyError, ResourceResolutionError) as error:
        issues.add(f"DFB-family binding resolution failed: {error}")
    legacy_only = {
        "tl.tt.dfb_add",
        "tl.tt.tensor_to_dfb",
        "tl.tt.dfb_to_tensor",
    }
    unsupported = sorted(name for name in operation_names if name not in legacy_only)
    if unsupported:
        issues.add("current Device operation mappings are incomplete: " + ", ".join(unsupported))

    if not issues:
        issues.add("stage-marked shared-grid TTL module emission is not implemented")
    raise TTLMappingPreflightError(
        "TTL mapping preflight failed: " + "; ".join(sorted(issues))
    )


__all__ = ["TTLMappingPreflightError", "preflight_ttl_mapping"]
