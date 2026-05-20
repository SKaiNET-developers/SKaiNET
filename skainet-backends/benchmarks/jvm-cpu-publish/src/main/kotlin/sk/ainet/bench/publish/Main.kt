package sk.ainet.bench.publish

import kotlinx.cli.ArgParser
import kotlinx.cli.ArgType
import kotlinx.cli.Subcommand
import kotlinx.cli.default
import kotlinx.cli.required
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import sk.ainet.backend.api.kernel.KernelServiceLoader
import sk.ainet.bench.publish.env.RuntimeInfoProvider
import sk.ainet.bench.publish.env.SystemInfoProvider
import sk.ainet.bench.publish.runner.BenchmarkRunner
import sk.ainet.bench.publish.scenarios.ScenarioRegistry
import java.io.File
import kotlin.system.exitProcess

private const val SCHEMA_VERSION = "1.0.0"

private val json = Json {
    prettyPrint = true
    encodeDefaults = true
}

public fun main(args: Array<String>) {
    KernelServiceLoader.installAll()
    val parser = ArgParser("skainet-engine-publish")
    parser.subcommands(RunCmd(), ListCmd(), PrintSystemCmd())
    parser.parse(args)
}

private class RunCmd : Subcommand("run", "Run a single engine benchmark scenario") {
    val scenario by option(ArgType.String, shortName = "s", description = "Scenario id").required()
    val out by option(ArgType.String, shortName = "o", description = "Output JSON file (- for stdout)").required()
    val warmups by option(ArgType.Int, description = "Warmup runs").default(8)
    val measured by option(ArgType.Int, description = "Measured runs").default(5)
    val seed by option(ArgType.String, description = "RNG seed").default("42")
    val smoke by option(ArgType.Boolean, description = "Use smoke parameters (smallest shape, 1+1 runs)").default(false)
    val provider by option(ArgType.String, description = "Kernel provider name").default("panama")
    val parseableLine by option(ArgType.Boolean, description = "Print PTS-parseable result line to stdout").default(true)

    override fun execute() {
        val scenarioObj = ScenarioRegistry.byId(scenario, provider = provider, smoke = smoke)
            ?: error("unknown scenario: $scenario; known: ${ScenarioRegistry.ids().joinToString(",")}")

        val effectiveWarmups = if (smoke) 1 else warmups
        val effectiveMeasured = if (smoke) 1 else measured
        val runner = BenchmarkRunner(
            warmupRuns = effectiveWarmups,
            measuredRuns = effectiveMeasured,
            seed = seed.toLong(),
            smokeMode = smoke,
            schemaVersion = SCHEMA_VERSION,
        )
        val record = runner.run(scenarioObj)
        val encoded = json.encodeToString(record)
        if (out == "-") {
            println(encoded)
        } else {
            val outFile = File(out)
            outFile.parentFile?.mkdirs()
            outFile.writeText(encoded)
        }
        if (parseableLine) {
            // PTS results parser line: scenario-specific key consumed by results-definition.xml.
            println("${scenarioObj.id}_${scenarioObj.unit}: ${"%.6f".format(record.metrics.valueMean)}")
        }
    }
}

private class ListCmd : Subcommand("list-scenarios", "List registered scenarios") {
    override fun execute() {
        for (id in ScenarioRegistry.ids()) println(id)
    }
}

private class PrintSystemCmd : Subcommand("print-system", "Print collected system and runtime info as JSON") {
    override fun execute() {
        val system = SystemInfoProvider.collect()
        val runtime = RuntimeInfoProvider.collect(selectedProvider = "n/a")
        println(json.encodeToString(SystemAndRuntime(system, runtime)))
    }
}

@kotlinx.serialization.Serializable
private data class SystemAndRuntime(
    val system: sk.ainet.bench.publish.schema.SystemInfo,
    val runtime: sk.ainet.bench.publish.schema.RuntimeInfo,
)

@Suppress("unused")
private fun exitWith(message: String, code: Int = 1): Nothing {
    System.err.println(message)
    exitProcess(code)
}
