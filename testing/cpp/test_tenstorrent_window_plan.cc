// Internal plan-contract checks: compile this translation unit against the same
// headers and libraries as form_device_program.cc. No production registry hooks.
#include <tvm/ffi/base_details.h>
#undef TVM_FFI_STATIC_INIT_BLOCK
#define TVM_FFI_STATIC_INIT_BLOCK() static void UnusedTestRegistration()
#include "../../src/tenstorrent/transform/form_device_program.cc"
#include <iostream>

int main() {
  using namespace tvm;
  using namespace tvm::tirx;
  using namespace tvm::tl::tenstorrent;
  using Planner = GeneralDataflowPlanner;
  auto op = [](const Op &kind, ffi::Array<PrimExpr> args) -> Stmt {
    return Evaluate(Call(DataType::Handle(), kind, args));
  };
  Stmt issue = op(tensor_to_dfb_nd(), {Integer(0), Integer(0)});
  Stmt wait = op(dfb_copy_wait(), {Integer(0), Integer(1)});
  Stmt send = op(dfb_pipe_send(), {Integer(0), Integer(0)});
  ffi::Array<Stmt> original{issue, wait, send};
  auto verify = [&](ffi::Array<Stmt> operations) {
    Planner::VerifyWindowSchedulePlan({operations}, original, 1);
  };
  verify({issue, wait, send});
  Stmt issue1 = op(tensor_to_dfb_nd(), {Integer(0), Integer(1)});
  Stmt wait1 = op(dfb_copy_wait(), {Integer(1), Integer(1)});
  Stmt send1 = op(dfb_pipe_send(), {Integer(0), Integer(1)});
  Planner::VerifyWindowSchedulePlan(
      {{issue1, wait1, send1, issue, wait, send}},
      {issue, wait, send, issue1, wait1, send1}, 2);
  size_t rejected = 0;
  auto reject = [&](ffi::Array<Stmt> operations) {
    try {
      verify(operations);
    } catch (const std::exception &) {
      ++rejected;
      return;
    }
    throw std::runtime_error("invalid window plan accepted");
  };
  reject({issue, send, wait});
  reject({send, issue, wait});
  reject({issue, wait});
  reject({issue, wait, send, send});
  reject({issue, issue, wait, send});
  reject({wait, issue, send});
  reject({issue, wait, wait, send});
  reject({issue, send});
  reject({issue, op(dfb_copy_wait(), {Integer(0), Integer(2)}), send});
  reject({issue, op(dfb_copy_wait(), {Integer(1), Integer(1)}), send});
  reject({op(tensor_to_dfb_nd(), {Integer(0), Integer(1)}), wait, send});
  reject({Evaluate(Integer(0))});
  reject({op(dfb_copy_wait(), {})});
  reject({op(dfb_copy_wait(), {Var("id", DataType::Int(32)), Integer(1)})});
  size_t loop_rejected = 0;
  auto reject_loop = [&](const Planner::LoopSchedulePlan &plan) {
    try {
      Planner::VerifyLoopSchedulePlan(plan);
    } catch (const std::exception &) {
      ++loop_rejected;
      return;
    }
    throw std::runtime_error("invalid loop plan accepted");
  };
  For loop(Var("i", DataType::Int(32)), Integer(0), Integer(3),
           ForKind::kSerial, Evaluate(Integer(0)));
  Planner::LoopSchedulePlan plan;
  plan.analysis = {loop, LoopStrategy::kPreserve};
  reject_loop(plan);
  plan.preserved_body = loop->body;
  Planner::VerifyLoopSchedulePlan(plan);
  plan.analysis.strategy = LoopStrategy::kExpand;
  reject_loop(plan);
  plan.extent = 1025;
  reject_loop(plan);
  plan.extent = 3;
  Planner::VerifyLoopSchedulePlan(plan);
  plan.analysis.strategy = LoopStrategy::kWindow;
  plan.stages = plan.depth = 2;
  reject_loop(plan);
  plan.analysis.loop = For(loop->loop_var, loop->min, loop->extent,
                          ForKind::kSerial, loop->body, std::nullopt,
                          {{"num_stages", Integer(2)}});
  Planner::VerifyLoopSchedulePlan(plan);
  plan.depth = 1;
  reject_loop(plan);
  plan.depth = 0;
  reject_loop(plan);
  plan.stages = 33;
  plan.depth = 3;
  reject_loop(plan);
  plan.stages = plan.depth = 2;
  plan.extent = 1025;
  reject_loop(plan);
  plan.analysis.loop = For();
  reject_loop(plan);
  size_t dead_write_rejected = 0;
  for (bool value_mode : {false, true}) {
    Planner::DeadWriteAnalysis analysis;
    analysis.compute_value_mode = value_mode;
    analysis.roots = {3};
    analysis.inputs = {{3, {2}}, {2, {1}}, {1, {2}}, {5, {4}}};
    analysis.compute = {{5, false}, {3, false}, {-1, true}, {1, false}};
    analysis.transfer = {{4, false}, {2, false}, {0, false}};
    auto dead_plan = Planner::BuildDeadWritePlan(analysis);
    Planner::VerifyDeadWritePlan(analysis, dead_plan, 6);
    if (dead_plan.live != std::unordered_set<int64_t>({1, 2, 3}) ||
        dead_plan.compute != std::vector<bool>({false, true, true, true}) ||
        dead_plan.transfer != std::vector<bool>({false, true, false}) ||
        dead_plan.recompute_tensor_reads == value_mode)
      throw std::runtime_error("incorrect dead-write fixed point or masks");
    auto reject_dead_write = [&](const Planner::DeadWriteAnalysis &facts,
                                 const Planner::DeadWriteEliminationPlan &bad) {
      try {
        Planner::VerifyDeadWritePlan(facts, bad, 6);
      } catch (const std::exception &) {
        ++dead_write_rejected;
        return;
      }
      throw std::runtime_error("invalid dead-write plan accepted");
    };
    auto bad = dead_plan;
    bad.live.erase(3);
    reject_dead_write(analysis, bad);
    bad = dead_plan;
    bad.live.erase(1);
    reject_dead_write(analysis, bad);
    bad = dead_plan;
    bad.live.insert(0);
    reject_dead_write(analysis, bad);
    bad = dead_plan;
    bad.compute.pop_back();
    reject_dead_write(analysis, bad);
    bad = dead_plan;
    bad.transfer.push_back(false);
    reject_dead_write(analysis, bad);
    bad = dead_plan;
    bad.compute[0] = true;
    reject_dead_write(analysis, bad);
    bad = dead_plan;
    bad.compute[2] = false;
    reject_dead_write(analysis, bad);
    bad = dead_plan;
    bad.transfer[1] = false;
    reject_dead_write(analysis, bad);
    bad = dead_plan;
    bad.recompute_tensor_reads = value_mode;
    reject_dead_write(analysis, bad);
    auto malformed = analysis;
    malformed.roots.insert(6);
    reject_dead_write(malformed, Planner::BuildDeadWritePlan(malformed));
    malformed = analysis;
    malformed.inputs[5].push_back(-1);
    reject_dead_write(malformed, dead_plan);
    malformed = analysis;
    malformed.compute.push_back({6, false});
    reject_dead_write(malformed, Planner::BuildDeadWritePlan(malformed));
    analysis.roots.clear();
    dead_plan = Planner::BuildDeadWritePlan(analysis);
    Planner::VerifyDeadWritePlan(analysis, dead_plan, 6);
    if (!dead_plan.live.empty() ||
        dead_plan.compute != std::vector<bool>({false, false, true, false}))
      throw std::runtime_error("rootless dead writes retained");
  }
  std::cout << "dead-write plan: both modes, cyclic closure, dead components, "
            << "mandatory operations and empty roots accepted; "
            << dead_write_rejected << " malformed plans rejected\n";
  std::cout << "window plan: forwarding and independent reorder accepted, "
            << rejected << " malformed schedules and " << loop_rejected
            << " malformed loop plans rejected\n";
}
