package sk.ainet.bench.publish.runner

public interface Scenario {
    public val id: String
    public val suite: String
    public val primaryMetric: String
    public val unit: String
    public val higherIsBetter: Boolean
    public val kernelProvider: String

    /**
     * Default parameter map describing the workload shape (e.g. `size=1024`).
     * Used to populate the [sk.ainet.bench.publish.schema.RunConfig.parameters]
     * field; pure metadata, no runtime knobs.
     */
    public val parameters: Map<String, String>

    /** Allocate inputs/outputs. Called once before warmups. */
    public fun setup()

    /** Single invocation that returns the primary metric value for this run. */
    public fun runOnce(): Double

    /** Release any large allocations. Called once after all measured runs. */
    public fun teardown() {}
}
