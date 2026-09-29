# How to add a block

1. **Declare the parameters** in `blocks/params.py`: a `ParamSchema("MyBlock",
   [ParamSpec(name, type, default, minimum=, maximum=, choices=, unit=, doc=)])`.
   Types: int, float, bool, str, dict, list. Add it to `PRIMITIVE_SCHEMAS`.
   Defaults should equal what the config files say today (there is a test for
   the existing blocks).
2. **Write the class** in `blocks/<name>.py`:

   ```python
   class MyBlock(Block):
       type_name = "MyBlock"
       schema = MY_SCHEMA
       neural = True                      # creates NEST structure
       ports = (spikes_in("ring"), spikes_out("features"), signal_out("counts"))

       def build(self, ctx, inputs):      # inputs: port -> [Connection] for build-bound spike inputs
           ring = self._source_population(self._single_input(inputs, "ring"))
           self.population = build_something(ctx.backend, self.id, ring, **self.params)
           self.outputs = {"features": self.population}      # what downstream blocks receive
           self.built = True

       def counts_sources(self):          # signal outputs that are spike-count readouts
           return {"counts": self.population.recorder_list()}
   ```

   The NEST module is `ctx.backend`; never import NEST. Build through a
   vectorised builder in `nest/populations.py` (one `Connect` per projection,
   weight matrices as `(pre, post)`, `connect_matrix` transposes for NEST) and
   raise `self.error("…")` so messages carry the block id. A block must be
   rebuildable: `build` starts from nothing (see `Encoder.build`).
   - Late-bound input (a port with `late=True`): implement `connect_late(ctx,
     connection)` on the target, or `connect_stim(ctx, ring, connection)` on a
     source that drives `Ring.stim`.
   - Signal blocks (`neural = False`): implement `reset()` and a per-tick
     method (`step`), and add a transceiver in `graph/compile.py` that reads
     its input edges (`self.read(inputs, port)`) and emits one datapack named
     by the block id.
   - Neural blocks with a per-tick input (encoders): implement `drive(angle,
     t_ms, dt_ms, centroid)`, `expire(t_ms)`, `clear()`; `GraphNestEngine`
     calls them and recalibrates the kernel after rate changes.
3. **Register** it in `blocks/__init__.py` (`REGISTRY`); the editor palette,
   `describe_types()` and the block reference follow automatically
   (`scripts/write_block_reference.py`).
4. **Test** it on the fake NEST (`test/test_blocks_fake.py`: creation calls,
   connection rows via `support_fake_nest.connection_rows`, validation
   messages) and, when it has a legacy counterpart, add a real-NEST equivalence
   test in the container with a documented tolerance and a pinned golden.
5. **Use** it in a template (`graph/templates.py`) and regenerate the examples
   (`write_examples()`; a test pins the files to the templates).
