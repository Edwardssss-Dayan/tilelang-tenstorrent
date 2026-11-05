"""Pass 12 checks actual scalar DAGs and always uses normalized formation."""

import pytest
from tilelang import tvm
from tvm import ir, tirx
from tvm.ir import Op
from tilelang.tenstorrent import transform
from tilelang.tenstorrent.pipeline import TENSTORRENT_LOWER_PASS_ORDER
from testing.python.tenstorrent.lower.test_tilelang_tenstorrent_phase2_form import add_fp32
from testing.python.tenstorrent.lower.test_tilelang_tenstorrent_phase1_pipeline import phase1_noop

TARGET = tvm.target.Target({"kind": "tenstorrent", "arch": "wormhole_b0"})


def before_formation(frontend=add_fp32):
    mod = tirx.transform.BindTarget(TARGET)(tvm.IRModule({"main": frontend}))
    for name in TENSTORRENT_LOWER_PASS_ORDER:
        if name == "FormTenstorrentDeviceProgram":
            break
        if name != "BindTarget":
            mod = getattr(transform, name)()(mod)
    return mod


def replace_add(mod, expression):
    def rewrite(node):
        if isinstance(node, tirx.Call) and isinstance(node.op, Op) and node.op.name == "tl.tt.tile_add":
            a, b, c = [arg.buffer for arg in node.args]
            attrs = dict(node.annotations)
            attrs.update({
                "tt.compute_kind": tirx.StringImm("elementwise"),
                "tt.logical_domain": c.shape,
                "tt.input_shapes": [a.shape, b.shape],
                "tt.expression": expression(tirx.BufferLoad(a, [0, 0]), tirx.BufferLoad(b, [0, 0])),
            })
            return tirx.Call("void", Op.get("tl.tt.tile_compute"), [node.args[2], node.args[0], node.args[1]], attrs, node.span)
        return None
    func = mod["main"]
    mod["main"] = func.with_body(tirx.stmt_functor.ir_transform(func.body, None, rewrite), func.span)
    return mod


@pytest.mark.parametrize("expression", [None, lambda a, b: a + b, lambda a, b: a * b, lambda a, b: a / b,
                                        lambda a, b: (a + b) * a / b])
def test_supported_operations_share_planner(expression):
    mod = before_formation()
    if expression is not None:
        mod = replace_add(mod, expression)
    formed = transform.FormTenstorrentDeviceProgram()(mod)
    assert str(formed.attrs["tt.ir_stage"]) == "device"
    assert "tt.dfb_families" in formed.attrs
    formed = transform.InferTenstorrentComputeRequirements()(formed)
    transform.VerifyTenstorrentDeviceIR()(formed)


@pytest.mark.parametrize("expression,reason", [
    (lambda a, b: a + tirx.call_intrin("float32", "tirx.erf", b), "tirx.erf"),
    (lambda a, b: a * tirx.FloorMod(a, b), "FloorMod"),
])
def test_nested_unsupported_is_reported_before_formation(expression, reason):
    mod = replace_add(before_formation(), expression)
    before = ir.save_json(mod)
    with pytest.raises(NotImplementedError, match=reason):
        transform.FormTenstorrentDeviceProgram()(mod)
    assert ir.save_json(mod) == before


@pytest.mark.parametrize("frontend", [add_fp32, phase1_noop])
def test_all_inputs_require_normalized_topology(frontend):
    mod = before_formation(frontend)
    mod["main"] = mod["main"].without_attr("tt.topology_normalized")
    with pytest.raises(ValueError, match="NormalizeTenstorrentTopology"):
        transform.FormTenstorrentDeviceProgram()(mod)


@pytest.mark.parametrize("write_output", [False, True])
def test_dfb_dead_chain_and_tensor_read_bookkeeping(write_output):
    mod = before_formation()
    if not write_output:
        copies = []
        tirx.stmt_functor.post_order_visit(
            mod["main"].body,
            lambda node: copies.append(node)
            if isinstance(node, tirx.Call)
            and isinstance(node.op, Op)
            and node.op.name == "tl.tileop.copy"
            else None,
        )
        output_copy = copies[-1]
        body = tirx.stmt_functor.ir_transform(
            mod["main"].body,
            None,
            lambda node: tirx.Evaluate(0)
            if isinstance(node, tirx.Evaluate) and node.value.same_as(output_copy)
            else None,
            ["tirx.Evaluate"],
        )
        mod["main"] = mod["main"].with_body(body)
    formed = transform.FormTenstorrentDeviceProgram()(mod)
    operations = []
    for function in formed.functions.values():
        tirx.stmt_functor.post_order_visit(
            function.body,
            lambda node: operations.append(node.op.name)
            if isinstance(node, tirx.Call) and isinstance(node.op, Op)
            else None,
        )
    assert bool(operations.count("tl.tt.dfb_compute")) == write_output
    assert bool(operations.count("tl.tt.tensor_to_dfb_nd")) == write_output
    assert operations.count("tl.tt.dfb_to_tensor_nd") == int(write_output)
    if not write_output:
        assert not formed.attrs["tt.dfb_families"]
        assert all(not function.attrs["tt.tensor_arg_indices"] for function in formed.functions.values())
    transform.VerifyTenstorrentDeviceIR()(transform.InferTenstorrentComputeRequirements()(formed))


def test_empty_body_uses_common_planner():
    mod = transform.FormTenstorrentDeviceProgram()(before_formation(phase1_noop))
    assert str(mod.attrs["tt.ir_stage"]) == "device"
    assert not mod.attrs["tt.dfb_families"]
    transform.VerifyTenstorrentDeviceIR()(transform.InferTenstorrentComputeRequirements()(mod))


def test_unsupported_actual_operation_has_no_noop_fallback():
    mod = before_formation(phase1_noop)

    def rewrite(node):
        if isinstance(node, tirx.Evaluate):
            return tirx.Evaluate(tirx.call_extern("int32", "unsupported_formation_operation"))
        return None

    func = mod["main"]
    mod["main"] = func.with_body(tirx.stmt_functor.ir_transform(func.body, None, rewrite), func.span)
    with pytest.raises(NotImplementedError, match="unsupported program: operation"):
        transform.FormTenstorrentDeviceProgram()(mod)


def test_unsupported_nested_dtype_reports_unsupported():
    mod = replace_add(before_formation(), lambda a, b: a + tirx.Cast("float32", tirx.Cast("float16", b)))
    with pytest.raises(NotImplementedError, match="scalar BF16/FP32"):
        transform.FormTenstorrentDeviceProgram()(mod)
