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

    # function to load 2 consts into memory in one cycle
    # NOTE - a bit duplicated atm, not prio tho
    def scratch_double_const(self, val1, val2, name1=None, name2=None):
        ops = []
        if val1 not in self.const_map:
            addr = self.alloc_scratch(name1)
            ops.append(("const", addr, val1))
            self.const_map[val1] = addr
        if val2 not in self.const_map:
            addr = self.alloc_scratch(name2)
            ops.append(("const", addr, val2))
            self.const_map[val2] = addr
        if len(ops) != 0:
            self.instrs.append({"load": ops})
        return [self.const_map[val1], self.const_map[val2]]
        # return self.const_map[val]

    def scratch_vconst(self, val, name=None):
        if val not in self.vconst_map:
            addr = self.alloc_scratch(name, VLEN)
            self.add("load", ("const", addr, val))
            self.add("valu", ("vbroadcast", addr, addr))
            self.vconst_map[val] = addr
        return self.vconst_map[val]

    def build_hash(self, val_vec_hash_addr, tmp_vec_1, tmp_vec_2, round, i):
        for hash_index, (op1, val1, op2, op3, val3) in enumerate[
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
                [(round, i + j, "hash_stage", hash_index)
                 for j in range(VLEN)],
            ))

    def append_to_curr_cycle(self, engine, slot):
        num_cycles = len(self.instrs)
        if engine not in [engine for engine in self.instrs[num_cycles-1]] or len(self.instrs[num_cycles-1][engine]) >= SLOT_LIMITS[engine]:
            self.instrs.append({engine: []})
        num_cycles = len(self.instrs)
        # print(engine)
        # print(len(self.instrs[num_cycles-1][engine]))
        self.instrs[num_cycles-1][engine].append(slot)

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
        # stores the 8 disparate forest values for our current batch
        forest_values_p_vec = self.alloc_scratch(
            "forest_values_vec", batch_size)
        forest_values_vec = self.alloc_scratch("tmp_vec_node_val", batch_size)
        # vec_batch_is = self.alloc_scratch("vec_batch_is", batch_size)
        # for i in range(batch_size):
        #         self.add("load", ("const", tmp1, i))
        #         self.add("load", ("load", vec_batch_is + i, tmp1))

        input_indices = self.alloc_scratch("input_indices", batch_size)
        input_values = self.alloc_scratch("input_values", batch_size)
        tmp_vec_batch_size = self.alloc_scratch(
            "tmp_vec_batch_size", batch_size)
        # keep all these contiguous, just feels right
        for i in range(0, batch_size, VLEN*2):
            if i+VLEN < batch_size:
                self.scratch_double_const(i, i+VLEN)
            else:
                self.scratch_const(i)
        # what im thinking: we initially compute 2 addresses, store in addr1 addr2
        # then each round we: compute 2 more addresses (store them in same vars), and load 2
        # then at the end, once, we load the last 2
        zero_const = self.scratch_const(0)
        vlen_const = self.scratch_const(VLEN)
        self.instrs.append({"alu": [
            ("+", tmp_addr_2, self.scratch["inp_values_p"], zero_const),
            ("+", tmp_addr, self.scratch["inp_values_p"], vlen_const),
        ],
            "valu": [
                ("vbroadcast", input_indices, zero_const),
                ("vbroadcast", input_indices + VLEN, zero_const),
        ]})
        # now we initially have tmp_addr and tmp_addr_2 populated
        for i in range(0, batch_size-VLEN*2, VLEN*2):
            i_const = self.scratch_const(i+VLEN*2)
            i_2nd_const = self.scratch_const(i+VLEN*3)

            self.instrs.append({"load": [
                ("vload", input_values + i, tmp_addr_2),
                ("vload", input_values + i + VLEN, tmp_addr),
            ],
                "alu": [
                ("+", tmp_addr_2, self.scratch["inp_values_p"], i_const),
                ("+", tmp_addr, self.scratch["inp_values_p"], i_2nd_const),
            ],
                "valu": [
                    ("vbroadcast", input_indices+i+2*VLEN, zero_const),
                    ("vbroadcast", input_indices+i+3*VLEN, zero_const),
            ]}
            )

        final_offset = batch_size-VLEN*2
        self.instrs.append({"load": [
            ("vload", input_values + final_offset, tmp_addr_2),
            ("vload", input_values + final_offset + VLEN, tmp_addr),
        ],
            # "valu": [
            # ("vbroadcast", input_indices+final_offset, zero_const),
            # ("vbroadcast", input_indices+final_offset + VLEN, zero_const),]
        })
        # theoretically at this point we now have all of our input indices and values in 2 256 contiguous blocks of memory

        for round in range(rounds):
            for i in range(0, batch_size, VLEN):
                i_const = self.scratch_const(i)
                # idx = mem[inp_indices_p + i]
                self.instrs.append({"debug": [("vcompare", input_indices + i, [
                    (round, i + j, "idx") for j in range(VLEN)]),
                    ("vcompare", input_values + i, [
                        (round, i + j, "val") for j in range(VLEN)])
                ]})
                # node_val = mem[forest_values_p + idx]
                self.append_to_curr_cycle("valu", ("+", forest_values_p_vec + i,
                                                   self.scratch["forest_values_p"], input_indices+i))
            for i in range(0, batch_size, VLEN):
                i_const = self.scratch_const(i)
                # how can i remove this loop? this loop results in 4 cycles per loop = 4 * 16 (rounds) * 256/8 (batch size/vlen) = 2000 cycles
                # what are we doing here? loading 8 disparate forest values into memory in order to operate on the 8 long vec of them
                # when we hash
                # the forest values are readonly, they will always be disparate
                # additionally the indices cant change
                # if we need to load 8 things at once we need to vload
                # but vload loads a block of 8 contiguous values from main memory into scratch
                for k in range(0, VLEN, 2):
                    self.instrs.append({"load": [
                        ("load", forest_values_vec + i +
                         k, forest_values_p_vec + i + k),
                        ("load", forest_values_vec + i +
                         k+1, forest_values_p_vec + i + k+1)
                    ]})

                self.add("debug", ("vcompare", forest_values_vec+i, [
                         (round, i + j, "node_val") for j in range(VLEN)]))

            for i in range(0, batch_size, VLEN):
                i_const = self.scratch_const(i)
                self.append_to_curr_cycle("valu", ("^", input_values + i,
                                                   input_values + i, forest_values_vec+i))

            for i in range(0, batch_size, VLEN):
                i_const = self.scratch_const(i)
                self.build_hash(input_values + i, tmp_vec_1,
                                tmp_vec_2, round, i)
                self.add("debug", ("vcompare", input_values + i, [
                         (round, i + j, "hashed_val") for j in range(VLEN)]))

            for i in range(0, batch_size, VLEN):
                i_const = self.scratch_const(i)
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                self.append_to_curr_cycle("valu", ("%", tmp_vec_batch_size + i,
                                                   input_values + i, two_vec_const))

            for i in range(0, batch_size, VLEN):
                i_const = self.scratch_const(i)
                self.append_to_curr_cycle(
                    "valu", ("+", tmp_vec_batch_size + i, tmp_vec_batch_size + i, one_vec_const))

            # this does almost nothing lmao saves us like 50 cycles
            if round == forest_height:
                for i in range(0, batch_size, VLEN):
                    self.append_to_curr_cycle(
                        "valu", ("vbroadcast", input_indices+i, zero_vec_const))
            else:
                for i in range(0, batch_size, VLEN):
                    i_const = self.scratch_const(i)
                    # idx = 2*idx + (1 if val % 2 == 0 else 2)
                    self.append_to_curr_cycle("valu", ("multiply_add", input_indices+i,
                                                       input_indices+i, two_vec_const, tmp_vec_batch_size + i))
                    self.append_to_curr_cycle("debug", ("vcompare", input_indices+i, [
                        (round, i + j, "next_idx") for j in range(VLEN)]))
                    # idx = 0 if idx >= n_nodes else idx
                    # WHAT IS THE IDX VALUE CUTOFF FOR WRAPPING?
                    # 2*idx + 2 >= n_nodes -> idx >= 1/2 * n_nodes - 1 IF VALUE IS EVEN
                    # if n_nodes is 1000
                    # idx 499 -> 999 = no wrap
                    # idx 498 -> 996 + 2 = no wrap
                    # idx 500 -> wrap

                for i in range(0, batch_size, VLEN):
                    self.append_to_curr_cycle(
                        "valu", ("%", input_indices+i, input_indices+i, self.scratch["n_nodes"]))

                for i in range(0, batch_size, VLEN):
                    # i_const = self.scratch_const(i)
                    # self.append_to_curr_cycle("flow", ("vselect", input_indices+i,
                    #                                    tmp_vec_batch_size + i, input_indices+i, zero_vec_const))
                    self.append_to_curr_cycle("debug", ("vcompare", input_indices+i, [
                        (round, i + j, "wrapped_idx") for j in range(VLEN)]))

                # mem[inp_indices_p + i] = idx

        # Required to match with the yield in reference_kernel2
        i_const = self.scratch_const(0)
        self.instrs.append({"alu": [("+", tmp_addr, self.scratch["inp_indices_p"],
                                     i_const), ("+", tmp_addr_2, self.scratch["inp_values_p"], i_const)]})

        # WE NEED TO VSTORE AT THE input_indices_p and input_values_p
        # now we initially have tmp_addr and tmp_addr_2 populated
        for i in range(0, batch_size-VLEN, VLEN):
            i_const = self.scratch_const(i+VLEN)

            self.instrs.append({"store": [
                ("vstore", tmp_addr, input_indices + i),
                ("vstore", tmp_addr_2, input_values + i)
            ],
                "alu": [
                ("+", tmp_addr, self.scratch["inp_indices_p"],
                 i_const), ("+", tmp_addr_2, self.scratch["inp_values_p"], i_const)]}
            )

        final_offset = batch_size-VLEN
        self.instrs.append({"store": [
            ("vstore", tmp_addr, input_indices + final_offset),
            ("vstore", tmp_addr_2, input_values + final_offset)
        ]})
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
