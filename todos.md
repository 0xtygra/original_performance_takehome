[] look at how to run and ignore debug
[] multie core? what is a core
[] flow? load?
[] also actually understand what store is doing and how we can better combine them
[] need to be able to pack more than just 1 type of instr into a cycle!


CURRENT IDEA: we dont need to vload the input indices, theyre all 0 at first and we can just vbroadcast the zero const into them
-> this means we can vload 2 batches of the values at once


- load a block of the forest into memory
- store the pointer to this block in memory too
- alu? 
for i in range(0,batch_size)
    alu, ^, input_value + i, in_mem_forest_vals + i
    # wrong for 2 reasons - this adds an index and a value and in_mem wont have all our values
- we can push the loads into the same cycles as the alu xors but it still wont fix our hard bottleneck of 256 (/2) cycles per batch to get the vec of our forest values into memory
- 4x speedup with vselect? how though
- we need to vload, no other way around it
- lets think of 1 vload for the 3rd round where we have 8 forest elements
- we have the whole forest in scratch and our input_indices look like [7,11,8,12,11,7,14,13]...
- lets just look at this batch for now
- forest_values_in_mem_vec is [124,1512,1521,412,9159,19394,193,13139]
- if we had the start of forest_values_in_mem_vec in mem could we: alu, +,tmp_addr, input_indices + i, forest_mem_const
    and then alu, ^, input_values + i, input_values+i,tmp_addr
    ==== this wont work as tmp_addr stores the index, but we have no way of accessing it right? it's a ptr to an index but we need it to act as just a ptr
    - compile time vs runtime issue
- only at maximum a 3x throughput increase here though hmmm
- but we could pack this with our hashing? how would it work when we have more than 256 items to get into memory
- 1536 is our max, that's 512+1024. 1024 is our max forest size
- can we do everything with just 3 vars? no we cant, need scratch constants too
- in that case then the tmp_addr could overflow. do we need a vselect 

#4477

- ONCE WE HAVE EFFICIENT BUILD, IF WE HAVE EMPTY ALUS SHOULD WE CONVERT VALU OPS INTO 8 ALUS?