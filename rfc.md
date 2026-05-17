# RFC: Hybrid Adaptive DSL with Optional DType Constraints

**Status:** Draft

Summary
This RFC proposes a hybrid adaptive DSL for model definition and execution.

The DSL remains architecture-first by default: it describes layer topology, tensor roles, and graph structure without requiring a fixed dtype for every tensor. Tensor dtype normally follows the loaded model file.

At the same time, the DSL may optionally express explicit dtype constraints where execution requires them. These constraints are resolved during load, compile, or lowering, before forward execution begins.

This provides two important properties:

A single DSL definition can load different GGUF quantization variants.
Strict execution targets, such as NPUs, can require specific runtime dtypes and layouts.
The key rule is:

DType annotations in the DSL describe executable requirements, not assumptions about the source file.

Motivation
GGUF models frequently use heterogeneous per-tensor quantization. A single file may contain tensors in FP16, FP32, Q8, Q4, Q4_K, or other quantized formats.

A strict DSL that hardcodes dtype into every layer has several drawbacks:

one model architecture may require multiple DSL definitions for different quant variants
mixed-precision GGUF files become awkward to represent
loading arbitrary GGUF variants becomes harder
dtype policy becomes coupled to model architecture
conversion may be forced even when the current backend could execute the source dtype directly
An adaptive DSL solves this by allowing tensor dtype to follow the file. However, pure adaptivity is not enough for constrained execution targets.

For example, an NPU may support native int8 execution but not GGUF Q8, Q4_K, or other packed quantized formats. In that case, the DSL or backend configuration must be able to require a specific executable dtype or layout.

This RFC proposes combining both approaches:

adaptive dtype behavior by default
explicit dtype constraints when needed
load/compile-time constraint resolution
backend-specific lowering before execution
Goals
Keep the DSL architecture-focused by default.
Allow one DSL definition to load multiple quantized model variants.
Support mixed-precision GGUF files.
Make dtype a first-class tensor property.
Allow explicit dtype constraints for specific ops, tensors, layers, or backends.
Resolve hard dtype requirements before forward execution.
Avoid marker-class-based dtype detection.
Avoid treating raw packed byte shape as logical tensor shape.
Separate source file dtype from executable backend dtype.
Support restricted backends such as NPUs without making the whole DSL strict.
Produce a dtype-safe prepared DAG before forward execution.
Support an optional compiled/lowered path for StableHLO, MLIR, or native optimized code.
Non-Goals
This RFC does not define a new tensor engine.
This RFC does not prescribe a specific Kotlin API.
This RFC does not require all tensors to be converted at load time.
This RFC does not require the DSL to declare every tensor dtype.
This RFC does not define exact quantization algorithms.
This RFC does not define backend-specific packed layouts.
This RFC does not require GGUF Q8 to be treated as native int8.
Definitions
Source dtype
The dtype stored in the model file.

Examples:

FP16
FP32
Q8
Q4
Q4_K
The source dtype describes what was read from disk.

Logical dtype
The dtype represented by the tensor inside the engine.

This should be explicit tensor metadata, not inferred from wrapper classes or raw storage type.

Required dtype
The dtype required by an op, layer, backend, or execution policy.

For example, an NPU backend may require int8 tensors for a given matrix multiplication.

Lowered dtype
The dtype and layout actually passed to the executable kernel.

This may differ from the source dtype if conversion or lowering occurred.

Logical shape
The shape of the tensor as seen by the graph.

For example, a quantized matrix may logically be:

[out_features, in_features]
even if it is physically stored as packed bytes.

Physical storage layout
The internal memory representation of a tensor.

For quantized tensors, this may include:

packed bytes
block structure
scales
zero points
backend-specific layout metadata
Physical storage layout is an implementation detail of the tensor representation.

Resolved DAG
A normalized internal graph produced from the DSL and loaded tensors.

The resolved DAG makes execution metadata explicit on nodes and edges, including:

tensor logical shapes
resolved dtypes
layouts
backend assignments
conversion nodes
lowering nodes
op dependencies
quantization metadata
dtype and backend constraints
The resolved DAG is a compiled intermediate representation, not necessarily the final executable artifact.

Executable plan
A scheduled and backend-aware representation derived from the resolved DAG.

The executable plan includes selected kernels, memory planning, buffer reuse, constant placement, lowered tensors, and backend-specific execution decisions.

Lowering
The process of converting high-level graph operations, tensor dtypes, layouts, or storage formats into representations required by a selected backend.

Examples include:

Q4_K weight to native int8 NPU weight
GGUF packed layout to backend-native layout
high-level projection op to backend-specific matmul op
dynamic dtype choice to fixed kernel selection
resolved DAG to StableHLO, MLIR, or native backend code
Lowering is part of graph preparation. It may happen during loading if the target backend is already known, or during an explicit compile step if backend selection happens later.

Design Overview
The DSL defines model architecture and optional dtype constraints.

The model file provides source tensors with source dtypes and logical shapes.

The loader creates engine tensors with explicit dtype metadata.

The compile or lowering phase resolves constraints against backend capabilities.

If a hard constraint can be satisfied, tensors may be converted or lowered. If it cannot be satisfied, loading or compilation fails.

Forward execution only sees resolved tensors.

flowchart TD
    A[DSL definition] --> B[Model architecture]
    A --> C[Optional dtype constraints]

    D[Model file / GGUF] --> E[Source tensors with file dtypes]

    B --> F[Graph construction]
    C --> G[Constraint resolution]
    E --> G

    G --> H{Constraints satisfied?}

    H -- Yes, as-is --> I[Use source dtype directly]
    H -- Requires conversion --> J[Lower / convert tensor]
    H -- Impossible --> K[Fail at load or compile time]

    I --> L[Resolved runtime tensors]
    J --> L

    L --> M[Kernel dispatch]
    M --> N[Execution on CPU / SIMD / NPU]
Default Adaptive Behavior
If no dtype constraint is declared, the engine should preserve the dtype provided by the model file whenever possible.

For example:

GGUF tensor: Q4_K
DSL constraint: none
Backend: CPU
Result: keep Q4_K and dispatch Q4_K-capable kernel
This allows one DSL definition to support many model variants.

flowchart LR
    A[GGUF Q4/Q8/FP16/FP32 tensor] --> B[Engine tensor with explicit dtype]
    B --> C[No hard dtype constraint]
    C --> D[Keep source dtype]
    D --> E[Dispatch by actual tensor dtype]
Explicit DType Constraints
The DSL may optionally declare that a tensor or op requires a specific dtype.

Such annotations should be interpreted as execution constraints.

They do not mean the source file must already contain that dtype.

For example:

Source tensor: Q4_K
Required dtype: int8
Backend: NPU
Resolution: lower Q4_K to backend-native int8, or fail
This allows restricted targets to express requirements without making the entire DSL strict.

flowchart TD
    A[Tensor loaded from file] --> B{Does DSL/backend require a specific dtype?}

    B -- No --> C[Keep file dtype]
    B -- Yes --> D{Does current tensor already satisfy requirement?}

    D -- Yes --> E[Use directly]
    D -- No --> F{Can it be converted/lowered?}

    F -- Yes --> G[Convert during load/compile]
    F -- No --> H[Raise load/compile error]

    C --> I[Dispatch by resolved dtype]
    E --> I
    G --> I
DType Constraints as Policies
A dtype annotation should be modeled as a policy rather than a simple claim about storage.

Useful policy categories include:

Any
No specific dtype is required.

The tensor may keep the source dtype.

Require
A hard requirement.

The executable graph is invalid unless the tensor is available in the required dtype and layout.

Prefer
A soft requirement.

The runtime should use the preferred dtype if available or cheap to produce, but may fall back to another supported dtype.

One-of
A restricted set of acceptable dtypes.

The runtime may choose any supported dtype from the allowed set.

Native Int8 vs GGUF Quantized Formats
Native int8 and GGUF quantized formats must not be treated as equivalent.

A GGUF Q8 tensor may be stored using int8-like values internally, but the tensor contract usually includes quantization metadata, block-level scales, and GGUF-specific layout semantics.

A native int8 tensor for an NPU is an executable representation expected by that backend. It may require different layout, scale handling, alignment, calibration, or memory placement.

Therefore:

GGUF Q8 != native int8
GGUF Q4_K != native int8
packed quantized storage != executable integer tensor contract
The system should represent this distinction explicitly.

flowchart LR
    A[GGUF Q8 / Q4_K] --> B[Quantized tensor format]
    B --> C[Has packing, block metadata, scales]

    D[Native int8] --> E[Backend execution format]
    E --> F[Has backend-specific layout and quant contract]

    C -. not equivalent .- F
Logical Shape vs Physical Storage
Logical shape must be part of the tensor contract.

Physical storage should not define graph-visible shape.

For example, a packed quantized tensor may occupy a one-dimensional byte segment internally, but the graph should see the tensor as its logical multidimensional shape.

flowchart TD
    A[Quantized tensor] --> B[Logical shape]
    A --> C[Physical storage]

    B --> D[Graph contract]
    C --> E[Implementation detail]

    D --> F[Shape inference]
    D --> G[Op validation]
    E --> H[Kernel-specific decoding]
The engine should avoid designs where a tensor appears as a 1D byte array at load time and is later patched into a logical 2D shape. The loader should produce properly shaped logical tensors directly.

Load and Compile Pipeline
The recommended pipeline is:

flowchart TD
    A[Read model file] --> B[Create tensors with source dtype and logical shape]
    B --> C[Build graph from DSL]
    C --> D[Attach optional dtype constraints]
    D --> E[Check backend capabilities]
    E --> F{All constraints satisfied?}

    F -- Already satisfied --> G[Use tensors as-is]
    F -- Convertible --> H[Lower tensors]
    F -- Not satisfiable --> I[Fail before execution]

    G --> J[Resolved executable graph]
    H --> J
    J --> K[Forward execution]
Constraint resolution should happen before execution. Forward execution should not need to discover that a tensor cannot run on the selected backend.

Backend Behavior
CPU / SIMD Backend
A general CPU backend should prefer adaptive execution.

It can keep GGUF source dtypes when suitable kernels exist.

flowchart TD
    A[CPU / SIMD backend] --> B[Accept multiple dtypes]
    B --> C[Keep source dtype where possible]
    C --> D[Dispatch on resolved tensor dtype]
Restricted NPU Backend
A restricted backend should declare supported executable dtypes and layouts.

If required tensors are not already in that form, they must be lowered before execution.

flowchart TD
    A[NPU backend] --> B[Supports limited executable formats]
    B --> C[Require native dtype/layout]
    C --> D[Lower tensors before execution]
    D --> E[Dispatch to NPU kernel]
Compiled Execution Path
The adaptive tensor-engine path should remain the default execution model for flexible GGUF loading and heterogeneous quantization.

However, the system may also expose a separate compiled execution path. In this context, the DSL itself is not the final artifact. The DSL is converted into a resolved DAG, and the resolved DAG may then be converted into an executable plan or lowered further into StableHLO, MLIR, or native optimized code.

The recommended model is:

DSL source
→ resolved DAG
→ executable plan
→ optional backend lowering
flowchart LR
    A[DSL definition] --> B[Graph builder]
    B --> C[Resolved DAG]
    C --> D[Execution planner]
    D --> E[Executable plan]
    E --> F[Runtime / backend execution]
The resolved DAG should contain:

op nodes
tensor edges
logical shapes
source dtypes
resolved execution dtypes
layouts
backend assignments
quantization metadata
explicit conversion or lowering nodes
dtype constraints and validation results
Example before dtype/backend resolution:

flowchart TD
    A[input: F16] --> B[rms_norm]
    B --> C[linear_project]
    W1[weight: Q4_K] --> C
    C --> D[activation]
    D --> E[linear_project]
    W2[weight: Q8_0] --> E
    E --> F[output: F16]
Example after resolution for a backend requiring native int8 weights:

flowchart TD
    A[input: F16] --> B[rms_norm]
    B --> C[linear_project]

    W1[weight: Q4_K] --> L1[lower Q4_K to int8]
    L1 --> C

    C --> D[activation]
    D --> E[linear_project]

    W2[weight: Q8_0] --> L2[lower Q8_0 to int8]
    L2 --> E

    E --> F[output: F16]
The compiled path has a different purpose from the adaptive runtime path.

The adaptive path is optimized for flexibility:

load many GGUF variants with the same DSL
preserve source dtypes where possible
dispatch dynamically based on resolved tensor dtype
support mixed quantization without requiring a separate model definition
The compiled path is optimized for stable, specialized execution:

freeze dtype and layout decisions before execution
lower the graph into a resolved DAG
schedule the DAG into an executable plan
optionally lower the plan into StableHLO, MLIR, or native optimized code
allow aggressive fusion, layout planning, memory planning, and static validation
flowchart TD
    A[DSL graph] --> B[Constraint resolution]
    B --> C[Resolved DAG with dtypes and shapes]
    C --> D[Executable plan]
    D --> E{Optional external compiler path}
    E -- No --> F[Tensor-engine execution]
    E -- Yes --> G[StableHLO / MLIR]
    G --> H[Backend optimization]
    H --> I[Native optimized artifact]
The compiled execution path may produce several levels of artifact:

Resolved DAG: normalized graph with explicit tensor flow and dtype/layout metadata.
Executable tensor-engine plan: scheduled graph with selected kernels and planned buffers.
StableHLO / MLIR module: compiler IR for external optimization and backend lowering.
Native backend artifact: JIT function, shared library, command buffer, serialized runtime module, or backend-specific executable blob.
This means the system can support two complementary modes:

flowchart LR
    A[DSL + model file] --> B{Execution mode}

    B -- Adaptive runtime --> C[Tensor engine]
    C --> D[Dynamic dtype/backend dispatch]

    B -- Compiled path --> E[Resolved DAG]
    E --> F[Executable plan]
    F --> G[Optional StableHLO / MLIR / native code]
The compiled path should be treated as an explicit lowering target, not as the default interpretation of the DSL. This keeps the normal GGUF path flexible while still allowing high-performance deployment when dtype, shape, layout, and backend contracts are stable enough to compile.

Lowering Phase Placement
Lowering should belong primarily to load/compile-time graph preparation, not ordinary forward execution.

The recommended split is:

Loading:
  read the file and create logical tensors

Compilation / graph preparation:
  resolve constraints, insert conversions, select layouts, select kernels

Lowering:
  convert resolved graph/tensors/ops into the representation required by the selected backend

Execution:
  run the already-lowered executable plan
flowchart TD
    A[Load model file] --> B[Create logical tensors]
    B --> C[Build DSL DAG]
    C --> D[Resolve dtype/backend constraints]
    D --> E[Lower tensors / ops / layouts]
    E --> F[Create executable plan]
    F --> G[Forward execution]
Lowering may happen at load time when the target backend is already known.

load GGUF for NPU
→ immediately convert required tensors to int8/native layout
→ store lowered tensors
This provides early failure and simple execution, but is less flexible if the same loaded model should target multiple backends.

Lowering may also happen during an explicit compile step.

load GGUF once
→ keep source tensors
→ compile for CPU or NPU later
→ lower only for the selected target
This is more flexible and allows multiple lowered variants to be cached.

Execution-time lowering should be avoided for hard requirements. If it is used, it should be treated as lazy or deferred compilation, not as normal forward execution. It must produce the same result as the explicit compile path and should cache the lowered result for subsequent executions.

DType Safety
This design provides dtype safety by turning dtype compatibility into a graph preparation invariant.

The prepared DAG or executable plan should contain only tensors, conversions, and ops whose dtype and layout contracts have been resolved and validated against the selected backend.

flowchart LR
    A[Source tensors] --> B[Build DAG]
    B --> C[Resolve dtype constraints]
    C --> D[Insert conversions/lowering]
    D --> E[Validate backend kernels]
    E --> F[DType-safe executable plan]
    F --> G[Forward execution]
After graph preparation, forward execution should not perform dtype discovery. It should execute a plan that is already known to be valid.

The safety guarantees are:

every tensor has explicit dtype metadata
every op declares accepted input and output dtype contracts
every backend declares supported dtype/layout/kernel combinations
constraint resolution validates the graph before execution
required conversions are inserted explicitly
unsupported dtype combinations fail before forward execution
kernel dispatch uses resolved dtype, not wrapper-class identity
A valid executable node must satisfy all relevant contracts:

flowchart TD
    A[Node: linearProject] --> B{Input dtype valid?}
    A --> C{Weight dtype valid?}
    A --> D{Output dtype valid?}
    A --> E{Backend kernel exists?}

    B --> F[Valid executable node]
    C --> F
    D --> F
    E --> F
DType safety does not automatically imply precision safety.

A conversion such as Q4_K to int8, FP16 to int8, or Q8 to int8 may be valid according to dtype rules while still being lossy. Lossy conversion, calibration, scale handling, and acceptable accuracy loss should be controlled by separate conversion and precision policies.

The complete safety model includes:

dtype safety: can the graph execute with these dtypes?
layout safety: does the backend understand this memory layout?
shape safety: do tensor dimensions match op contracts?
conversion safety: is this conversion allowed, calibrated, cached, and valid?
precision safety: is the accuracy loss acceptable?
Error Handling
Errors should occur as early as possible.

Load/compile-time errors
These should occur when:

a hard dtype constraint cannot be satisfied
no conversion path exists
the selected backend does not support the required dtype
required layout lowering is unavailable
logical tensor shape is incompatible with the target kernel
quantization metadata is insufficient for conversion
Forward-time errors
Forward-time errors should be limited to unexpected execution failures.

They should not be used for ordinary dtype compatibility discovery.

Forward execution should only operate on resolved tensors.

Kernel Dispatch
Kernel dispatch should use explicit tensor metadata.

Dispatch should be based on:

operation kind
input dtype
weight dtype
output dtype
backend
layout
possibly quantization parameters
Dispatch should not depend on:

marker classes
wrapper class identity
raw storage array type
physical byte-count shape
flowchart LR
    A[Resolved tensor metadata] --> B[Kernel key]
    B --> C[Dispatch table]
    C --> D[Selected backend kernel]
Benefits
This design provides:

one DSL definition for many quantized variants
clean support for mixed-precision GGUF
explicit dtype semantics
early failure for impossible backend constraints
backend-specific lowering without polluting the architecture DSL
cleaner shape inference
no dtype marker-class hacks
less ambiguity between packed quantized formats and native execution formats
a natural path for CPU, SIMD, and NPU backends
Tradeoffs
More complex constraint resolution
The loader or compiler must understand dtype policies and backend capabilities.

More explicit dtype model
The tensor engine must represent dtype and layout as first-class metadata.

Conversion cost
When strict constraints require conversion, load or compile time may increase.

For example, Q4_K to native int8 may require dequantization and requantization.

Potential precision loss
Some conversions are lossy.

The system may need policy controls for whether lossy conversion is allowed.

More backend capability metadata
Backends need to declare which dtypes, layouts, and conversions they support.

Open Questions
Should dtype constraints live in the DSL, backend profile, or both?
Should lossy conversion require an explicit opt-in policy?
Should lowering happen at load time, compile time, or lazily before first execution?
Should lowered tensors be cached?
How should per-layer and per-tensor constraints be represented?
How should backend-specific layouts be named and versioned?
How should quantization metadata be preserved during lowering?
Should unsupported soft preferences warn or silently fall back?
Should graph optimization occur before or after dtype lowering?
How should mixed backend execution be represented?
Recommended Direction
The recommended direction is:

adaptive by default
explicit dtype constraints when needed
DSL converted into a resolved DAG for prepared execution
constraints resolved before execution
conversion/lowering handled during load or compile preparation
execution-time lowering only as lazy/deferred compilation
hard requirements fail early
kernel dispatch uses real tensor dtype
logical shape belongs to the tensor contract
physical storage remains an implementation detail
compiled path may emit tensor-engine plans, StableHLO, MLIR, or native artifacts
In short:

The DSL should define architecture first, while allowing explicit dtype requirements only where execution needs them.

This gives the flexibility needed for GGUF and mixed quantization, while still supporting strict execution environments such as NPUs and explicit compiled targets such as StableHLO, MLIR, or native optimized code.

Final Summary
A strict dtype DSL is clean for fixed execution environments, but too rigid for general GGUF loading.

A fully adaptive DSL fits GGUF better, but needs explicit dtype metadata and a principled way to handle strict backend requirements.

The proposed hybrid model keeps the DSL adaptive by default and adds dtype constraints as execution policies. This allows source tensors to follow the model file unless an op or backend requires otherwise. When constraints exist, the loader or compiler either lowers the tensor into the required dtype/layout or fails before execution.

The compiled execution path should be understood as DSL-to-DAG preparation. The DSL is converted into a resolved DAG with explicit tensor flow, dtype/layout metadata, backend assignments, and conversion/lowering nodes. That DAG may then become an executable tensor-engine plan or be lowered further into StableHLO, MLIR, or native backend artifacts.

This gives the system dtype safety by making dtype compatibility a graph preparation invariant. Forward execution consumes a resolved plan rather than discovering dtype compatibility dynamically.

This avoids confusing source storage with executable representation and provides a cleaner foundation for CPU, SIMD, NPU, and compiled StableHLO/MLIR/native execution targets.
