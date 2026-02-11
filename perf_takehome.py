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
import copy

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


def calc_cycle_inputs_outputs(inputs, outputs, engine, engine_instrs):
    if engine == "alu":
        for instr in engine_instrs:
            inputs.add(instr[2])
            inputs.add(instr[3])
            outputs.add(instr[1])
    # same as alu but vector
    elif engine == "valu":
        for instr in engine_instrs:
            if instr[0] == "vbroadcast":
                inputs.add(instr[2])
                for j in range(VLEN):
                    outputs.add(instr[1]+j)
            elif instr[0] == "multiply_add":
                for j in range(VLEN):
                    outputs.add(instr[1]+j)
                    inputs.add(instr[2]+j)
                    inputs.add(instr[3]+j)
                    inputs.add(instr[4]+j)
            else:
                for j in range(VLEN):
                    inputs.add(instr[2]+j)
                    inputs.add(instr[3]+j)
                    outputs.add(instr[1]+j)
    elif engine == "load":
        for instr in engine_instrs:
            outputs.add(instr[1])
            if instr[0] == "load":
                inputs.add(instr[2])
            elif instr[0] == "vload":
                for j in range(VLEN):
                    inputs.add(instr[2]+j)
            elif instr[0] == "const":
                outputs.add(instr[1])
            else:
                raise NotImplementedError(
                    f"Unknown op {engine} {instr[0]}")
            # kind of ignore load_offset for now, dont really use it
    elif engine == "store":
        # NEED TO THINK THROUGH THIS MORE, NOT A PROBLEM FOR OUR USE
        # CASE THOUGH AS WE ONLY STORE ONCE AT THE END
        for instr in engine_instrs:
            if instr[0] == "store":
                inputs.add(instr[2])
                inputs.add(instr[1])
            elif instr[0] == "vstore":
                for j in range(VLEN):
                    inputs.add(instr[2]+j)
                    inputs.add(instr[1]+j)
            else:
                raise NotImplementedError(
                    f"Unknown op {engine} {instr[0]}")
    elif engine == "flow":
        for instr in engine_instrs:
            if instr[0] == "vselect":
                for j in range(VLEN):
                    inputs.add(instr[2]+j)
                    inputs.add(instr[4]+j)
                    inputs.add(instr[3]+j)
                    outputs.add(instr[1]+j)
            elif instr[0] == "add_imm":
                inputs.add(instr[2])
                outputs.add(instr[1])
            elif instr[0] == "pause":
                # do nothing
                a = 1
            else:
                raise NotImplementedError(
                    f"Unknown op {engine} {instr[0]}")
    elif engine == "debug":
        for instr in engine_instrs:
            if instr[0] == "compare":
                inputs.add(instr[1])
            if instr[0] == "vcompare":
                for j in range(VLEN):
                    inputs.add(instr[1] + j)

    else:
        raise NotImplementedError(f"Unknown op {engine}")


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        # tuple ("engine",(op,etc))
        self.interim_instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.vconst_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def place_op_in_earliest_available_cycle(self, engine, op, earliest_valid_cycle_idx, instr_list):
        for i in range(earliest_valid_cycle_idx, len(instr_list)):
            curr_instr = instr_list[i]
            if engine not in curr_instr:
                curr_instr[engine] = []
            if len(curr_instr[engine]) < SLOT_LIMITS[engine]:
                instr_list[i][engine].append(op)
                return instr_list
        instr_list.append({engine: [op]})
        return instr_list

    # TODO: make it a bit smarter such that we can put reads and writes in the same
    # cycle where possible
    def optimise_instrs(self, instrs):
        initial_instrs = [{}]
        for ii, (engine, op) in enumerate(instrs):

            instrs_len = len(initial_instrs)
            for i in range(instrs_len):
                curr_idx = instrs_len-i-1
                curr_cycle = initial_instrs[curr_idx]
                curr_cycle_inputs = set()
                curr_cycle_outputs = set()
                for tmp_eng, instrs in curr_cycle.items():
                    calc_cycle_inputs_outputs(
                        curr_cycle_inputs, curr_cycle_outputs, tmp_eng, instrs)
                # now we know all the inputs and outputs for our current cycle
                # time to check if our current instruction has any conflicts
                op_inputs = set()
                op_outputs = set()
                calc_cycle_inputs_outputs(
                    op_inputs, op_outputs, engine, [op])
                # our current instruction has an input that is mutated in this cycle -> CANT GO IN THIS CYCLE
                if op_inputs & curr_cycle_outputs:
                    initial_instrs = self.place_op_in_earliest_available_cycle(
                        engine, op, curr_idx+1, initial_instrs)
                    break
                if op_outputs & curr_cycle_outputs:
                    initial_instrs = self.place_op_in_earliest_available_cycle(
                        engine, op, curr_idx+1, initial_instrs)
                    break
                if op_outputs & curr_cycle_inputs:
                    initial_instrs = self.place_op_in_earliest_available_cycle(
                        engine, op, curr_idx, initial_instrs)
                    break
                if engine == "flow" and op[0] == "pause":
                    initial_instrs = self.place_op_in_earliest_available_cycle(
                        engine, op, curr_idx, initial_instrs)
                    break
                # if we get here then we can go back further
                # if this is the first cycle though then we gotta put it in
                if curr_idx == 0:
                    initial_instrs = self.place_op_in_earliest_available_cycle(
                        engine, op, 0, initial_instrs)
                    break
        return initial_instrs

    def build(self, debug=False, optimise=True):
        if not optimise:
            instrs = []
            for (engine, op) in self.interim_instrs:
                instrs.append({engine: [op]})
            return instrs
        initial_instrs = self.optimise_instrs(self.interim_instrs)

        # some of our cycles are weird
        # for instr in initial_instrs:
        #     if "valu" in instr:
        #         valu_instr = instr["valu"]
        #         outputs = {}
        #         for i in range(len(valu_instr)):
        #             op = valu_instr[i]
        #             if op[1] not in outputs:
        #                 outputs[op[1]] = []
        #             outputs[op[1]].append(i)
        #         for dest, idxs in outputs.items():
        #             if len(idxs) > 1:
        #                 print(outputs)
        #                 print(valu_instr)
        #             idxs_to_pop = idxs[:-1][::-1]
        #             for idx in idxs_to_pop:
        #                 instr["valu"].pop(idx)

        # now go through instrs to see if we can replace alu with valu
        valu_converted_instrs = []
        for instr in initial_instrs:
            valu_converted_instr = copy.deepcopy(instr)
            if "valu" not in valu_converted_instr:
                valu_converted_instrs.append(valu_converted_instr)
                continue
            # find the LAST convertible valu (cant convert vbroadcast or mul add)
            # must be last as some of our cycles have internal ordering dependencies
            # does this mean we have unnecessary ops? if ordering matters then we must be
            # writing to something twice
            last_valu_op_idx = None
            for i in range(len(valu_converted_instr["valu"])):
                op = valu_converted_instr["valu"][-1-i]
                if op[0] not in ["vbroadcast", "multiply_add"]:
                    last_valu_op_idx = -1-i
                    break
            if last_valu_op_idx == None:
                valu_converted_instrs.append(valu_converted_instr)
                continue
            if len(valu_converted_instr["valu"]) == SLOT_LIMITS["valu"] \
                    and ("alu" not in valu_converted_instr):
                # take a valu and convert to 8 alus
                if "alu" not in valu_converted_instr:
                    valu_converted_instr["alu"] = []
                last_valu_op = valu_converted_instr["valu"].pop(
                    last_valu_op_idx)
                for j in range(VLEN):
                    valu_converted_instr["alu"].append(
                        (last_valu_op[0], last_valu_op[1]+j, last_valu_op[2]+j, last_valu_op[3]+j))
            valu_converted_instrs.append(valu_converted_instr)

        # re-flatten then go through again to rejig it
        ops = []
        for instr in valu_converted_instrs:
            for engine, slots in instr.items():
                for slot in slots:
                    ops.append((engine, slot))

        valu_converted_optimised_instrs = self.optimise_instrs(
            ops)

        self.instrs = valu_converted_instrs
        # self.instrs = valu_converted_optimised_instrs
        return self.instrs

        # every op has inputs and outputs
        # in general, we can move an op back in cycles as long as we want until:
        # one of our inputs is another output, or
        # our output is another output

    def add(self, engine, ops):
        if (isinstance(ops, list)):
            for op in ops:
                self.interim_instrs.append((engine, op))
        else:
            self.interim_instrs.append((engine, ops))
        # self.instrs.append({engine: [slot]})

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
            self.add("load", [("const", addr, val)])
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
            self.add("load", ops)
        return [self.const_map[val1], self.const_map[val2]]
        # return self.const_map[val]

    def scratch_vconst(self, val, name=None):
        if val not in self.vconst_map:
            addr = self.alloc_scratch(name, VLEN)
            self.add("load", [("const", addr, val)])
            self.add("valu", [("vbroadcast", addr, addr)])
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

        self.add("load", loads[0:2])
        for i in range(2, len(loads), 2):
            self.add("load", loads[i:i+2])

            self.add("valu", vbroadcasts[i-2:i])
        self.add("valu", vbroadcasts[-2:])

    # all 3 vecs must be batch_size long
    # this only takes 80 cycles per call
    def build_hash(self, input_values_vec, tmp_vec_1, tmp_vec_2, round, i, batch_size, pre_computed_hash_mul):
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

                    self.add("valu", [("multiply_add", input_values_vec+j, input_values_vec+j, pre_computed_hash_mul[hash_index], self.scratch_vconst(
                        val1))])
                else:
                    self.add("valu", [(op1, tmp_vec_1 + j, input_values_vec + j,
                                      self.scratch_vconst(val1)),
                                      (op3, tmp_vec_2 + j, input_values_vec + j,
                                      self.scratch_vconst(val3)),
                                      (op2, input_values_vec + j,
                                      tmp_vec_1 + j, tmp_vec_2 + j)
                                      ])

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int, batch_chunk_size: int = 5
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
        three_vec_const = self.scratch_vconst(3)
        seven_vec_const = self.scratch_vconst(7)
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

        # vec_batch_is = self.alloc_scratch("vec_batch_is", batch_size)
        # for i in range(batch_size):
        #         self.add("load", ("const", tmp1, i))
        #         self.add("load", ("load", vec_batch_is + i, tmp1))

        input_indices = self.alloc_scratch("input_indices", batch_size)
        input_values = self.alloc_scratch("input_values", batch_size)
        forest_level_0_val = self.alloc_scratch("forest_level_0_val")

        self.add(
            "load", [("load", forest_level_0_val, self.scratch["forest_values_p"])])
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

        # hash ops 1 3 5 can be collapsed into 1 mul_add
        pre_computed_hash_mul = []
        for hash_index, (op1, val1, op2, op3, val3) in enumerate[
            tuple[str, int, str, str, int]
        ](HASH_STAGES):
            if hash_index % 2 != 0:
                pre_computed_hash_mul.append(None)
                continue
            pre_computed_hash_mul.append(self.scratch_vconst(1 + (1 << val3)))

        vlen_const = self.scratch_const(VLEN)
        self.add("alu", [
            ("+", tmp_addr_2, self.scratch["inp_values_p"], zero_const),
            ("+", tmp_addr, self.scratch["inp_values_p"], vlen_const),
        ])
        self.add("valu", [
            ("vbroadcast", input_indices, zero_const),
            ("vbroadcast", input_indices + VLEN, zero_const),
        ])

        # now we initially have tmp_addr and tmp_addr_2 populated
        for i in range(0, batch_size-VLEN*2, VLEN*2):
            i_const = self.scratch_const(i+VLEN*2)
            i_2nd_const = self.scratch_const(i+VLEN*3)

            self.add("load", [
                ("vload", input_values + i, tmp_addr_2),
                ("vload", input_values + i + VLEN, tmp_addr),
            ])

            self.add("alu", [
                ("+", tmp_addr_2, self.scratch["inp_values_p"], i_const),
                ("+", tmp_addr, self.scratch["inp_values_p"], i_2nd_const),
            ])
            self.add("valu", [
                ("vbroadcast", input_indices+i+2*VLEN, zero_const),
                ("vbroadcast", input_indices+i+3*VLEN, zero_const),
            ]
            )

        final_offset = batch_size-VLEN*2
        self.add("load", [
            ("vload", input_values + final_offset, tmp_addr_2),
            ("vload", input_values + final_offset + VLEN, tmp_addr),
        ])
        # theoretically at this point we now have all of our input indices and values in 2 256 contiguous blocks of memory

        # we can keep the first 3 levels of the forest in memory
        forest_node_scratch_vecs = []
        num_forest_nodes_to_load = 15  # level 0,1,2,3
        for i in range(num_forest_nodes_to_load):
            node_idx = self.scratch_const(i)
            forest_node_vec = self.alloc_scratch(f"forest_node_{i}", VLEN)
            # lets us better pack things so we can get 2 loads
            addr_to_use = tmp_addr if i % 2 == 0 else tmp_addr_2
            self.add("alu", ("+", addr_to_use, node_idx,
                     self.scratch["forest_values_p"]))
            self.add("load", [
                ("load", forest_node_vec, addr_to_use)
            ])
            self.add(
                "valu", [("vbroadcast", forest_node_vec, forest_node_vec)])
            forest_node_scratch_vecs.append(forest_node_vec)

        vecs_per_batch = batch_size//VLEN
        chunk_scratch_memory = []
        for i in range(batch_chunk_size):
            chunk_scratch_memory.append({
                "node": self.alloc_scratch(f"node{i}", VLEN),
                "tmp1": self.alloc_scratch(f"tmp1{i}", VLEN),
                "tmp2": self.alloc_scratch(f"tmp2{i}", VLEN),
                "tmp3": self.alloc_scratch(f"tmp3{i}", VLEN),
            })
        for chunk_start in range(0, vecs_per_batch, batch_chunk_size):
            for round in range(rounds):
                for sub_chunk in range(batch_chunk_size):
                    chunk = sub_chunk + chunk_start
                    chunk_memory = chunk_scratch_memory[sub_chunk]
                    indices = input_indices + chunk * VLEN
                    values = input_values + chunk * VLEN
                    level = round % (forest_height+1)

                    # saves us 500 cycles
                    curr_interim_instrs = len(self.interim_instrs)
                    if level == 0:
                        forest_values_l_0 = forest_node_scratch_vecs[0]
                        self.add("valu", ("^", values,
                                          values, forest_values_l_0))
                    elif level == 1:
                        self.add(
                            "valu", ("&", chunk_memory["tmp1"], indices, values))
                        self.add(
                            "flow",
                            ("vselect", chunk_memory["node"], chunk_memory["tmp1"],
                             forest_node_scratch_vecs[1], forest_node_scratch_vecs[2]),
                        )
                        self.add("valu", ("^", values,
                                          values, chunk_memory["node"]))
                    elif level == 2:
                        self.add(
                            "valu", ("-", chunk_memory["tmp1"], indices, three_vec_const))
                        self.add(
                            "valu", ("&", chunk_memory["tmp2"], chunk_memory["tmp1"], one_vec_const))
                        self.add(
                            "valu", ("&", chunk_memory["node"], chunk_memory["tmp1"], two_vec_const))
                        self.add(
                            "flow",
                            ("vselect", chunk_memory["tmp1"], chunk_memory["tmp2"],
                             forest_node_scratch_vecs[4], forest_node_scratch_vecs[3]),
                        )
                        self.add(
                            "flow",
                            ("vselect", chunk_memory["tmp2"], chunk_memory["tmp2"],
                             forest_node_scratch_vecs[6], forest_node_scratch_vecs[5]),
                        )
                        self.add(
                            "flow",
                            ("vselect", chunk_memory["node"], chunk_memory["node"],
                             chunk_memory["tmp2"], chunk_memory["tmp1"]),
                        )
                        self.add("valu", ("^", values,
                                          values, chunk_memory["node"]))
                    else:
                        self.add("valu", ("+", chunk_memory["tmp1"],
                                          self.scratch["forest_values_p"], indices))
                        for j in range(VLEN):
                            self.add(
                                "load", ("load", chunk_memory["node"] + j, chunk_memory["tmp1"] + j))
                        self.add("valu", ("^", values,
                                          values, chunk_memory["node"]))
                    print(
                        f"round: {round}, stage: loading and xoring forest values, new instrs: {len(self.interim_instrs) - curr_interim_instrs}")

                    curr_interim_instrs = len(self.interim_instrs)
                    # this is 150 cycles each round
                    # 256 elements in the vector, 3 parts of each hash stage, 6 hash stages
                    # valu 6 slot limit + doing 8 at a time
                    # -> 256 * 3 * 6 / (6*8) = 96
                    for hash_index, (op1, val1, op2, op3, val3) in enumerate[
                        tuple[str, int, str, str, int]
                    ](HASH_STAGES):
                        # 0 2 4 are all mul add
                        # 1 3 5 unchanges
                        # before we had 2 stage 1s for every stage 2, now we have 1 2 1 2 1 2 stage 1s for 6 stage 2s = 9 for 6
                        # these stages: add input and val1, left shift (multiply) input by val3, then add the results of both
                        # instead we can multiply input and add the result of the first add in the same op
                        if hash_index % 2 == 0:
                            self.add("valu", [("multiply_add", values, values, pre_computed_hash_mul[hash_index], self.scratch_vconst(
                                val1))])
                        else:
                            self.add("valu", [(op1, chunk_memory["tmp1"], values,
                                               self.scratch_vconst(val1)),
                                              (op3, chunk_memory["tmp2"], values,
                                               self.scratch_vconst(val3)),
                                              (op2, values,
                                               chunk_memory["tmp1"], chunk_memory["tmp2"])
                                              ])
                    # self.build_hash(input_values, forest_values_p_vec,
                    #                 forest_values_vec, round, i, batch_size, pre_computed_hash_mul)
                    print(
                        f"round: {round}, stage: building hash, new instrs: {len(self.interim_instrs) - curr_interim_instrs}")

                    for i in range(0, batch_size, VLEN):
                        self.add("debug", ("vcompare", input_values + i, [
                                (round, i + j, "hashed_val") for j in range(VLEN)]))

                    curr_interim_instrs = len(self.interim_instrs)
                    # this does almost nothing lmao saves us like 50 cycles
                    if round == forest_height and False:
                        for i in range(0, batch_size, VLEN):
                            self.add(
                                "valu", ("vbroadcast", input_indices+i, zero_vec_const))
                    else:
                        self.add(
                            "valu", ("&",
                                     chunk_memory["tmp1"], values, one_vec_const)
                        )
                        self.add(
                            "valu",
                                ("+", chunk_memory["node"],
                                 chunk_memory["tmp1"], one_vec_const),
                        )
                        self.add(
                            "valu", ("multiply_add", indices, indices,
                                     two_vec_const, chunk_memory["node"])
                        )

                        # i_const = self.scratch_const(i)
                        # self.add("flow", ("vselect", input_indices+i,
                        #                                    tmp_vec_batch_size + i, input_indices+i, zero_vec_const))
                        self.add("debug", ("vcompare", indices,
                                           (round, chunk * VLEN, "wrapped_idx")))

                        # mem[inp_indices_p + i] = idx
                    print(
                        f"round: {round}, stage: calc and wrapping next index, new instrs: {len(self.interim_instrs) - curr_interim_instrs}")

        # Required to match with the yield in reference_kernel2
        i_const = self.scratch_const(0)
        self.add("alu", [("+", tmp_addr, self.scratch["inp_indices_p"],
                          i_const), ("+", tmp_addr_2, self.scratch["inp_values_p"], i_const)])

        # WE NEED TO VSTORE AT THE input_indices_p and input_values_p
        # now we initially have tmp_addr and tmp_addr_2 populated
        for i in range(0, batch_size-VLEN, VLEN):
            i_const = self.scratch_const(i+VLEN)

            self.add("store", [
                ("vstore", tmp_addr, input_indices + i),
                ("vstore", tmp_addr_2, input_values + i)
            ])
            self.add("alu", [
                ("+", tmp_addr, self.scratch["inp_indices_p"],
                 i_const), ("+", tmp_addr_2, self.scratch["inp_values_p"], i_const)]
            )

        final_offset = batch_size-VLEN
        self.add("store", [
            ("vstore", tmp_addr, input_indices + final_offset),
            ("vstore", tmp_addr_2, input_values + final_offset)
        ])
        self.add("flow", ("pause",))

        instrs = self.build(True, False)
        cycle_counts = {"valu": 0, "alu": 0, "load": 0,
                        "debug": 0, "flow": 0, "store": 0}
        engine_counts = {"valu": 0, "alu": 0, "load": 0,
                         "debug": 0, "flow": 0, "store": 0}
        for instr in instrs:
            for engine, ops in instr.items():
                cycle_counts[engine] = cycle_counts[engine]+1
                for _op in ops:
                    engine_counts[engine] = engine_counts[engine]+1
        print(cycle_counts)
        print(engine_counts)
        print(len(self.instrs))


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
