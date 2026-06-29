package sk.ainet.data.source

/** A named suspendable data processing stage. */
public interface PipelineStage<I, O> {
    public val name: String

    public fun validate(input: I): Boolean = true

    public suspend fun process(input: I): O
}

/** A schema-aware stage for data preprocessing pipelines. */
public interface DataTransformer<I, O> : PipelineStage<I, O> {
    public suspend fun transform(input: I): O

    public fun getOutputSchema(inputSchema: DataSchema): DataSchema = inputSchema

    override suspend fun process(input: I): O = transform(input)
}

/** Thrown when a data pipeline cannot execute a stage. */
public class DataPipelineException(
    message: String,
    cause: Throwable? = null
) : DataSourceException(message, cause)

/** A type-safe sequential data pipeline. */
public class DataPipeline<I, O> internal constructor(
    private val stages: List<PipelineStage<*, *>>
) {
    public val stageNames: List<String> = stages.map { stage -> stage.name }

    /** Adds [stage] to the end of this pipeline. */
    public fun <N> stage(stage: PipelineStage<O, N>): DataPipeline<I, N> =
        DataPipeline(stages + stage)

    /** Adds [stage] to the end of this pipeline. */
    public infix fun <N> then(stage: PipelineStage<O, N>): DataPipeline<I, N> =
        this.stage(stage)

    /** Executes each stage in order. */
    @Suppress("UNCHECKED_CAST")
    public suspend fun execute(input: I): O {
        var current: Any? = input
        for (stage in stages) {
            val typedStage = stage as PipelineStage<Any?, Any?>
            if (!typedStage.validate(current)) {
                throw DataPipelineException("Stage '${stage.name}' rejected its input")
            }
            current = typedStage.process(current)
        }
        return current as O
    }

    /** Returns a human-readable stage chain. */
    public fun describe(): String = stageNames.joinToString(" -> ")
}

/** Starts an identity data pipeline. */
public fun <I> dataPipeline(): DataPipeline<I, I> = DataPipeline(emptyList())

/** Creates a named suspendable pipeline stage. */
public fun <I, O> pipelineStage(
    name: String,
    validate: (I) -> Boolean = { true },
    process: suspend (I) -> O
): PipelineStage<I, O> {
    require(name.isNotBlank()) { "Pipeline stage name must not be blank" }
    return FunctionPipelineStage(name, validate, process)
}

/** Kotlinish alias for [pipelineStage]. */
public fun <I, O> stage(
    name: String,
    validate: (I) -> Boolean = { true },
    process: suspend (I) -> O
): PipelineStage<I, O> = pipelineStage(name, validate, process)

/** Creates a named schema-aware data transformer. */
public fun <I, O> dataTransformer(
    name: String,
    outputSchema: (DataSchema) -> DataSchema = { it },
    validate: (I) -> Boolean = { true },
    transform: suspend (I) -> O
): DataTransformer<I, O> {
    require(name.isNotBlank()) { "Data transformer name must not be blank" }
    return FunctionDataTransformer(name, validate, outputSchema, transform)
}

private class FunctionPipelineStage<I, O>(
    override val name: String,
    private val validateInput: (I) -> Boolean,
    private val processInput: suspend (I) -> O
) : PipelineStage<I, O> {
    override fun validate(input: I): Boolean = validateInput(input)

    override suspend fun process(input: I): O = processInput(input)
}

private class FunctionDataTransformer<I, O>(
    override val name: String,
    private val validateInput: (I) -> Boolean,
    private val outputSchema: (DataSchema) -> DataSchema,
    private val transformInput: suspend (I) -> O
) : DataTransformer<I, O> {
    override fun validate(input: I): Boolean = validateInput(input)

    override suspend fun transform(input: I): O = transformInput(input)

    override fun getOutputSchema(inputSchema: DataSchema): DataSchema = outputSchema(inputSchema)
}
