# SKaiNET DAG DSL

Lightweight, definition-only DSL for constructing compute DAGs without executing tensor ops. It produces a `GraphProgram` that can be lowered to a `ComputeGraph` inside `skainet-compile-dag`.

## Quick start
```kotlin
import sk.ainet.lang.dag.dag
import sk.ainet.lang.graph.dsl.toComputeGraph
import sk.ainet.lang.tensor.ops.TensorSpec

val program = dag {
    val x = input("x", TensorSpec("x", listOf(1, 4), "FP32"))
    val w = parameter("w", TensorSpec("w", listOf(4, 4), "FP32"))
    val y = relu(matmul(x, w))
    output(y)
}

val graph = program.toComputeGraph() // DefaultComputeGraph with deterministic ids
```

## Notes
- The DSL is symbolic: no runtime tensors are allocated.
- Outputs must be marked via `output(...)`; if omitted, the last node’s outputs are used.
- Helper ops mirror common tensor ops; `op(operation, inputs)` stays open for custom/registry-backed operations.
- Lowering preserves node ids (`n<seq>_<op>` by default) and wires edges as `e_src_out__dst_in`, matching trace-based graphs.
- You can declare parameters/constants with an allocation-free data-style helper:
  `val w = parameter<FP32, Float>("w") { shape(4, 4) { ones() } }`
