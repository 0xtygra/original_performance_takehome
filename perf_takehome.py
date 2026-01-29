"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.vconst_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False, skip_debug=True):
        # temporarily reverting while we vectorise everything
        # instrs = []
        # for engine, slot in slots:
        #     instrs.append({engine: [slot]})
        # return instrs
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        instrs_len = 0
        curr_dests = {"alu": {},  "store": {}, "load": {}, "flow": {}}
        prev_engine = None
        # NEXT THING TO DO
        # BE ABLE TO HANDLE ALU -> LOAD -> ALU INTO 1 CYCLE
        # additionally be able to handle ALU -> load -> load -> load -> ALU if none of the ALUs depend on the loads, even if the interim loads are multiple cycles on their own
        for engine, slot in slots:
            if instrs_len == 0:
                instrs.append({engine: [slot]})
                curr_dests = {engine: {slot[1]: True}}
                instrs_len += 1
                continue
            curr_engine_len = (
                0
                if engine not in instrs[instrs_len - 1]
                else len(instrs[instrs_len - 1][engine])
            )

            if engine == "debug" and skip_debug:
                continue
            if prev_engine == engine:
                if engine == "alu" and curr_engine_len < SLOT_LIMITS[engine]:
                    # simple for now - ensure that the instr we want to pack into the same cycle doesnt depend on the write from the prev alu
                    if (
                        slot[2] not in curr_dests[engine]
                        and slot[3] not in curr_dests[engine]
                    ):
                        instrs[instrs_len - 1][engine].append(slot)
                        curr_dests[engine][slot[1]] = True
                # COMMENTED OUT AS NOT WORKING WITH MULTIPLYADD
                # elif engine == "valu" and curr_engine_len < SLOT_LIMITS[engine]:
                #     # simple for now - ensure that the instr we want to pack into the same cycle doesnt depend on the write from the prev alu
                #     if (
                #         slot[2] not in curr_dests[engine]
                #         and slot[3] not in curr_dests[engine]
                #     ):
                #         instrs[instrs_len - 1][engine].append(slot)
                #         curr_dests[engine][slot[1]] = True
                elif engine == "store" and curr_engine_len < SLOT_LIMITS[engine]:
                    instrs[instrs_len - 1][engine].append(slot)
                    # curr_dests irrelevant for store i believe, we can overwrite
                    # curr_dests[engine][slot[2]] = True
                elif engine == "load" and curr_engine_len < SLOT_LIMITS[engine]:
                    instrs[instrs_len - 1][engine].append(slot)
                else:
                    instrs.append({engine: [slot]})
                    curr_dests[engine] = {slot[1]: True}
                    instrs_len += 1
                    prev_engine = engine
                    continue

            new_engine_len = (
                0
                if engine not in instrs[instrs_len - 1]
                else len(instrs[instrs_len - 1][engine])
            )
            # we werent able to add new instr into current cycle
            if curr_engine_len == new_engine_len:
                instrs.append({engine: [slot]})
                curr_dests = {engine: {slot[1]: True}}
                instrs_len += 1
            prev_engine = engine

        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def scratch_vconst(self, val, name=None):
        if val not in self.vconst_map:
            addr = self.alloc_scratch(name, VLEN)
            self.add("load", ("const", addr, val))
            self.add("valu", ("vbroadcast", addr, addr))
            self.vconst_map[val] = addr
        return self.vconst_map[val]

    def build_hash(self, val_vec_hash_addr, tmp_vec_1, tmp_vec_2, round, i):
        for hi, (op1, val1, op2, op3, val3) in enumerate[
            tuple[str, int, str, str, int]
        ](HASH_STAGES):
            self.instrs.append({
                "valu": [
                    (op1, tmp_vec_1, val_vec_hash_addr,
                     self.scratch_vconst(val1)),
                    (op3, tmp_vec_2, val_vec_hash_addr,
                     self.scratch_vconst(val3))
                ]
            })
            self.add("valu", (op2, val_vec_hash_addr, tmp_vec_1, tmp_vec_2))
            self.add("debug", (
                "vcompare",
                val_vec_hash_addr,
                [(round, i + j, "hash_stage", hi) for j in range(VLEN)],
            ))

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        tmp1 = self.alloc_scratch("tmp1")
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        init_vars_vec = ["forest_values_p", "n_nodes"]
        for v in init_vars:
            if v not in init_vars_vec:
                self.alloc_scratch(v, 1)
            else:
                self.alloc_scratch(v, VLEN)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            if v not in init_vars_vec:
                self.add("load", ("load", self.scratch[v], tmp1))
            else:
                self.add("load", ("load", self.scratch[v], tmp1))
                self.add(
                    "valu", ("vbroadcast", self.scratch[v], self.scratch[v]))

        zero_vec_const = self.scratch_vconst(0)
        one_vec_const = self.scratch_vconst(1)
        two_vec_const = self.scratch_vconst(2)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        # Scalar scratch registers
        tmp_addr = self.alloc_scratch("tmp_addr")
        tmp_addr_2 = self.alloc_scratch("tmp_addr_2")
        tmp_vec_1 = self.alloc_scratch("tmp_vec_1", VLEN)
        tmp_vec_2 = self.alloc_scratch("tmp_vec_2", VLEN)
        tmp_vec_3 = self.alloc_scratch("tmp_vec_3", VLEN)
        tmp_vec_idx = self.alloc_scratch("tmp_vec_idx", VLEN)
        tmp_vec_val = self.alloc_scratch("tmp_vec_val", VLEN)
        tmp_vec_node_val = self.alloc_scratch("tmp_vec_node_val", VLEN)
        # vec_batch_is = self.alloc_scratch("vec_batch_is", batch_size)
        # for i in range(batch_size):
        #         self.add("load", ("const", tmp1, i))
        #         self.add("load", ("load", vec_batch_is + i, tmp1))

        for round in range(rounds):
            for i in range(0, batch_size, VLEN):
                i_const = self.scratch_const(i)
                # idx = mem[inp_indices_p + i]
                # self.add("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const))
                # self.add("alu", ("+", tmp_addr_2, self.scratch["inp_values_p"], i_const))
                self.instrs.append({"alu": [("+", tmp_addr, self.scratch["inp_indices_p"],
                                   i_const), ("+", tmp_addr_2, self.scratch["inp_values_p"], i_const)]})
                self.instrs.append(
                    {"load": [("vload", tmp_vec_idx, tmp_addr), ("vload", tmp_vec_val, tmp_addr_2)]})
                self.instrs.append({"debug": [("vcompare", tmp_vec_idx, [
                    (round, i + j, "idx") for j in range(VLEN)]),
                    ("vcompare", tmp_vec_val, [
                        (round, i + j, "val") for j in range(VLEN)])
                ]})
                # node_val = mem[forest_values_p + idx]
                self.add("valu", ("+", tmp_vec_3,
                         self.scratch["forest_values_p"], tmp_vec_idx))
                #
                for k in range(0, VLEN, 2):
                    self.instrs.append({"load": [
                        ("load", tmp_vec_node_val + k, tmp_vec_3 + k),
                        ("load", tmp_vec_node_val + k+1, tmp_vec_3 + k+1)
                    ]})

                self.add("debug", ("vcompare", tmp_vec_node_val, [
                         (round, i + j, "node_val") for j in range(VLEN)]))
                # val = myhash(val ^ node_val)
                self.add("valu", ("^", tmp_vec_val,
                         tmp_vec_val, tmp_vec_node_val))
                self.build_hash(tmp_vec_val, tmp_vec_1, tmp_vec_2, round, i)
                self.add("debug", ("vcompare", tmp_vec_val, [
                         (round, i + j, "hashed_val") for j in range(VLEN)]))
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                # self.instrs.append({
                #     "valu"
                # })
                self.add("valu", ("%", tmp_vec_1, tmp_vec_val, two_vec_const))
                self.add("valu", ("+", tmp_vec_1, tmp_vec_1, one_vec_const))
                # muladd
                self.add("valu", ("multiply_add", tmp_vec_idx,
                         tmp_vec_idx, two_vec_const, tmp_vec_1))
                self.add("debug", ("vcompare", tmp_vec_idx, [
                         (round, i + j, "next_idx") for j in range(VLEN)]))
                # idx = 0 if idx >= n_nodes else idx
                self.add("valu", ("<", tmp_vec_1, tmp_vec_idx,
                         self.scratch["n_nodes"]))
                self.add("flow", ("vselect", tmp_vec_idx,
                         tmp_vec_1, tmp_vec_idx, zero_vec_const))
                self.add("debug", ("vcompare", tmp_vec_idx, [
                         (round, i + j, "wrapped_idx") for j in range(VLEN)]))
                # mem[inp_indices_p + i] = idx
                # self.add(
                #     "alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const))
                # self.add("alu", ("+", tmp_addr_2,
                #          self.scratch["inp_values_p"], i_const))
                self.instrs.append({
                    "store": [
                        ("vstore", tmp_addr, tmp_vec_idx),
                        ("vstore", tmp_addr_2, tmp_vec_val)
                    ]
                })
                # self.instrs.append({
                #     "store": [
                #         ("vstore", tmp_addr, tmp_vec_idx),
                #         ("vstore", tmp_addr_2, tmp_vec_val)
                #     ]
                # })
                # self.add("store", ("vstore", tmp_addr, tmp_vec_idx))
                # # mem[inp_values_p + i] = val
                # self.add("store", ("vstore", tmp_addr_2, tmp_vec_val))

        # Required to match with the yield in reference_kernel2
        self.add("flow", ("pause",))


BASELINE = 147734


def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values),
                    len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p: inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p: inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p: inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p: inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p: inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p: inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5]: mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6]: mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
