package sk.ainet.bench.publish.scenarios

import sk.ainet.bench.publish.runner.Scenario

public object ScenarioRegistry {

    public fun ids(): List<String> = listOf(
        "engine-fp32-gemm",
        "engine-q4-gemm",
        "engine-kernel-matmul",
        "engine-elementwise-add",
        "engine-reductions-sum",
        "engine-reductions-mean",
    )

    public fun byId(id: String, provider: String, smoke: Boolean): Scenario? = when (id) {
        "engine-fp32-gemm" -> Fp32GemmScenario(smoke = smoke, providerName = provider)
        "engine-q4-gemm" -> Q4GemmScenario(smoke = smoke, providerName = provider)
        "engine-kernel-matmul" -> KernelMatmulScenario(smoke = smoke, providerName = provider)
        "engine-elementwise-add" -> ElementwiseAddScenario(smoke = smoke, providerName = provider)
        "engine-reductions-sum" -> ReductionsScenario(op = ReductionOp.SUM, smoke = smoke, providerName = provider)
        "engine-reductions-mean" -> ReductionsScenario(op = ReductionOp.MEAN, smoke = smoke, providerName = provider)
        else -> null
    }
}
