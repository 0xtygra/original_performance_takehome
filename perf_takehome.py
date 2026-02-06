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

    # TODO: CAN ADD THESE TO OTHER INITIAL ALLOCATIONS TOO
    def load_hash_values(self):
        loads = []
        vbroadcasts = []
        for hash_index, (op1, val1, op2, op3, val3) in enumerate[
            tuple[str, int, str, str, int]
        ](HASH_STAGES):
            addr1 = self.alloc_scratch(None, VLEN)
            addr3 = self.alloc_scratch(None, VLEN)
            loads.append(("const", addr1, val1))
            loads.append(("const", addr3, val3))
            vbroadcasts.append(("vbroadcast", addr1, addr1))
            vbroadcasts.append(("vbroadcast", addr3, addr3))
            self.vconst_map[val1] = addr1
            self.vconst_map[val3] = addr3

        self.instrs.append({"load": loads[0:2]})
        for i in range(2, len(loads), 2):
            self.instrs.append({"load": loads[i:i+2],
                                "valu": vbroadcasts[i-2:i]})
        self.instrs.append({"valu": vbroadcasts[-2:]})

    # all 3 vecs must be batch_size long
    # this only takes 80 cycles per call
    def build_hash(self, input_values_vec, tmp_vec_1, tmp_vec_2, round, i, batch_size):
        valu_stage_1 = []
        valu_stage_2 = []
        for hash_index, (op1, val1, op2, op3, val3) in enumerate[
            tuple[str, int, str, str, int]
        ](HASH_STAGES):
            # 0 2 4 are all mul add
            # 1 3 5 unchanges
            # before we had 2 stage 1s for every stage 2, now we have 1 2 1 2 1 2 stage 1s for 6 stage 2s = 9 for 6
            # these stages: add input and val1, left shift (multiply) input by val3, then add the results of both
            # instead we can multiply input and add the result of the first add in the same op
            for j in range(0, batch_size, VLEN):
                if hash_index % 2 == 0:
                    valu_stage_1.append((op1, tmp_vec_1 + j, input_values_vec + j,
                                        self.scratch_vconst(val1)))
                    valu_stage_2.append(("multiply_add", input_values_vec + j, input_values_vec + j, self.scratch_vconst(val3),
                                        tmp_vec_1 + j))
                else:
                    valu_stage_1.append((op1, tmp_vec_1 + j, input_values_vec + j,
                                        self.scratch_vconst(val1)))
                    valu_stage_1.append((op3, tmp_vec_2 + j, input_values_vec + j,
                                        self.scratch_vconst(val3)))
                    valu_stage_2.append(
                        (op2, input_values_vec + j, tmp_vec_1 + j, tmp_vec_2 + j))

        i = 0
        while len(valu_stage_2) + len(valu_stage_1) != 0:
            if i > 2:
                self.instrs.append(
                    {"valu": valu_stage_2[0:SLOT_LIMITS["valu"]]})
                valu_stage_2 = valu_stage_2[SLOT_LIMITS["valu"]:]
            else:
                self.instrs.append(
                    {"valu": valu_stage_1[0:SLOT_LIMITS["valu"]]})
                valu_stage_1 = valu_stage_1[SLOT_LIMITS["valu"]:]
            i = (i+1) % 5

        # for j in range(0, len(valu_stage_1) + len(valu_stage_2), SLOT_LIMITS["valu"]):
        #     self.instrs.append(
        #         {"valu": valu_stage_1[j:j+SLOT_LIMITS["valu"]]})
        # for j in range(0, len(valu_stage_2), SLOT_LIMITS["valu"]):
        #     self.instrs.append(
        #         {"valu": valu_stage_2[j:j+SLOT_LIMITS["valu"]]})

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
        self.load_hash_values()

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
        forest_level_0_val = self.alloc_scratch("forest_level_0_val")

        self.instrs.append(
            {"load": [("load", forest_level_0_val, self.scratch["forest_values_p"])]})
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
        one_const = self.scratch_const(1)

        three_const = self.scratch_const(3)
        four_const = self.scratch_const(4)
        five_const = self.scratch_const(5)
        six_const = self.scratch_const(6)

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
                self.append_to_curr_cycle("valu", ("+", forest_values_p_vec + i,
                                                   self.scratch["forest_values_p"], input_indices+i))
            # saves us 500 cycles
            if round % (forest_height+1) == 0:
                for i in range(0, batch_size, VLEN*2):
                    self.instrs.append(
                        {"valu": [
                            ("vbroadcast", forest_values_vec + i, forest_level_0_val),
                            ("vbroadcast", forest_values_vec +
                             i + VLEN, forest_level_0_val)
                        ]})
                for i in range(0, batch_size, VLEN):
                    i_const = self.scratch_const(i)
                    self.append_to_curr_cycle("valu", ("^", input_values + i,
                                                       input_values + i, forest_values_vec+i))
            elif round % (forest_height + 1) == 1:
                # our input_indices are [1,1,2,1,2,2,1,2,...]
                forest_val_1 = forest_values_vec
                forest_val_2 = forest_values_vec + 1
                forest_val_1_vec = forest_values_vec+VLEN
                forest_val_2_vec = forest_values_vec+2*VLEN
                offset = 128
                index_mask = forest_values_vec+offset
                index_forest_xor_1 = forest_values_vec + offset + VLEN
                index_forest_xor_2 = forest_values_vec + offset + 2*VLEN
                self.instrs.append(
                    {"alu": [("+", tmp_addr, self.scratch["forest_values_p"], one_const)],
                     "flow": [("add_imm", tmp_addr_2, self.scratch["forest_values_p"], 2)]})
                self.instrs.append(
                    {"load": [("load", forest_val_1, tmp_addr),
                              ("load", forest_val_2, tmp_addr_2)]})
                self.instrs.append({
                    "valu": [
                        ("vbroadcast", forest_val_1_vec, forest_val_1),
                        ("vbroadcast", forest_val_2_vec, forest_val_2),
                    ]
                })
                # at this point our forest_values_vec has the 1st level of our forest loaded in at indexes 1 and 2
                # if our input_indices has a lsb of 0, we need forest_values_vec[0], otherwise 1
                # let's use forest_values_vec + 128 for this
                # creating our mask as well as a and b options
                self.instrs.append({
                    "valu": [("&", index_mask, one_vec_const, input_indices),
                             ("^", index_forest_xor_1,
                              input_values, forest_val_1_vec),
                             ("^", index_forest_xor_2,
                              input_values, forest_val_2_vec),
                             ]
                })
                # forest_values_vec now has a mask
                for i in range(VLEN, batch_size, VLEN):
                    self.instrs.append({
                        "valu": [("&", index_mask, one_vec_const, input_indices + i),
                                 ("^", index_forest_xor_1,
                                  input_values+i, forest_val_1_vec),
                                 ("^", index_forest_xor_2,
                                  input_values+i, forest_val_2_vec),
                                 ],
                        "flow": [("vselect", input_values+i - VLEN, index_mask, index_forest_xor_1, index_forest_xor_2)]
                    })
                self.instrs.append({
                    "flow": [("vselect", input_values+batch_size-VLEN, index_mask, index_forest_xor_1, index_forest_xor_2)]
                })

            elif round % (forest_height+1) == 2:
                # our input_indices are [1,1,2,1,2,2,1,2,...]
                forest_val_1 = forest_values_vec
                forest_val_2 = forest_values_vec + 1
                forest_val_3 = forest_values_vec + 2
                forest_val_4 = forest_values_vec + 3

                addr1 = forest_values_vec + 4
                addr2 = forest_values_vec + 5
                addr3 = forest_values_vec + 6
                addr4 = forest_values_vec + 7

                forest_val_1_vec = forest_values_vec+VLEN
                forest_val_2_vec = forest_values_vec+2*VLEN
                forest_val_3_vec = forest_values_vec+3*VLEN
                forest_val_4_vec = forest_values_vec+4*VLEN

                three_vconst = forest_values_vec + 5*VLEN
                four_vconst = forest_values_vec + 6*VLEN
                five_vconst = forest_values_vec + 7*VLEN
                six_vconst = forest_values_vec + 8*VLEN

                offset = 128
                index_mask1 = forest_values_vec+offset
                index_mask2 = forest_values_vec+offset+VLEN
                index_mask3 = forest_values_vec+offset+VLEN*2
                index_mask4 = forest_values_vec+offset+VLEN*3
                index_forest_xor_1 = forest_values_vec + offset + VLEN*4
                index_forest_xor_2 = forest_values_vec + offset + VLEN*5
                index_forest_xor_3 = forest_values_vec + offset + VLEN*6
                index_forest_xor_4 = forest_values_vec + offset + VLEN*7

                self.instrs.append(
                    {"alu": [("+", addr1, self.scratch["forest_values_p"], three_const),
                             ("+", addr2,
                              self.scratch["forest_values_p"], four_const),
                             ("+", addr3,
                              self.scratch["forest_values_p"], five_const),
                             ("+", addr4,
                              self.scratch["forest_values_p"], six_const)
                             ]})
                self.instrs.append(
                    {"load": [("load", forest_val_1, addr1),
                              ("load", forest_val_2, addr2)],
                     "valu": [
                         ("vbroadcast", three_vconst, three_const),
                         ("vbroadcast", four_vconst, four_const),
                         ("vbroadcast", five_vconst, five_const),
                         ("vbroadcast", six_vconst, six_const),
                    ]
                    })
                self.instrs.append({
                    "load": [("load", forest_val_3, addr3),
                             ("load", forest_val_4, addr4)],
                    "valu": [
                        ("vbroadcast", forest_val_1_vec, forest_val_1),
                        ("vbroadcast", forest_val_2_vec, forest_val_2),
                    ]
                })
                self.instrs.append({
                    "valu": [("vbroadcast", forest_val_3_vec, forest_val_3),
                             ("vbroadcast", forest_val_4_vec, forest_val_4),]
                })

                for i in range(0, batch_size, VLEN):
                    self.instrs.append({
                        "valu": [
                            ("==", index_mask1, input_indices+i, three_vconst),
                            ("==", index_mask2, input_indices+i, four_vconst),
                            ("==", index_mask3, input_indices+i, five_vconst),
                            ("==", index_mask4, input_indices+i, six_vconst),
                        ]
                    })

                    self.instrs.append({
                        "valu": [
                            ("*", index_forest_xor_1,
                             forest_val_1_vec, index_mask1),
                            ("*", index_forest_xor_2,
                             forest_val_2_vec, index_mask2),
                            ("*", index_forest_xor_3,
                             forest_val_3_vec, index_mask3),
                            ("*", index_forest_xor_4,
                             forest_val_4_vec, index_mask4),
                        ]
                    })

                    self.instrs.append({
                        "valu": [
                            ("^", input_values+i, input_values+i, index_forest_xor_1)
                        ]
                    })
                    self.instrs.append({
                        "valu": [
                            ("^", input_values+i, input_values+i, index_forest_xor_2)
                        ]
                    })
                    self.instrs.append({
                        "valu": [
                            ("^", input_values+i, input_values+i, index_forest_xor_3)
                        ]
                    })
                    self.instrs.append({
                        "valu": [
                            ("^", input_values+i, input_values+i, index_forest_xor_4)
                        ]
                    })

            else:
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

            print(len(self.instrs))
            # this is 150 cycles each round
            # 256 elements in the vector, 3 parts of each hash stage, 6 hash stages
            # valu 6 slot limit + doing 8 at a time
            # -> 256 * 3 * 6 / (6*8) = 96
            self.build_hash(input_values, forest_values_p_vec,
                            forest_values_vec, round, i, batch_size)
            for i in range(0, batch_size, VLEN):
                self.add("debug", ("vcompare", input_values + i, [
                         (round, i + j, "hashed_val") for j in range(VLEN)]))

            print(len(self.instrs))
            for i in range(0, batch_size, VLEN):
                i_const = self.scratch_const(i)
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                self.append_to_curr_cycle("valu", ("%", forest_values_p_vec + i,
                                                   input_values + i, two_vec_const))

            for i in range(0, batch_size, VLEN):
                i_const = self.scratch_const(i)
                self.append_to_curr_cycle(
                    "valu", ("+", forest_values_p_vec + i, forest_values_p_vec + i, one_vec_const))

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
                                                       input_indices+i, two_vec_const, forest_values_p_vec + i))
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
