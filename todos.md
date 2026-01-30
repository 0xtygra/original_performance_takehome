[] look at how to run and ignore debug
[] multie core? what is a core
[] flow? load?
[] also actually understand what store is doing and how we can better combine them
[] need to be able to pack more than just 1 type of instr into a cycle!


CURRENT IDEA: we dont need to vload the input indices, theyre all 0 at first and we can just vbroadcast the zero const into them
-> this means we can vload 2 batches of the values at once
