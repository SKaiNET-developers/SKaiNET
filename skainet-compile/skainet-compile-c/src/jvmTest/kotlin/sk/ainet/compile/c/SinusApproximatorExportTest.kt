package sk.ainet.compile.c

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.model.dnn.mlp.SinusApproximator
import kotlin.test.Test
import kotlin.test.assertTrue
import java.io.File

class SinusApproximatorExportTest {

    @Test
    fun testExportSinusApproximatorToArduino() {
        val model = SinusApproximator()
        val facade = CCodegenFacade()
        val outputPath = File("build/generated/arduino").absolutePath
        val libraryName = "SinusApproximatorLib"
        
        // Ensure output directory exists
        File(outputPath).mkdirs()
        
        val ctx = DirectCpuExecutionContext()
        // DO NOT create module outside of the forwardPass lambda if you want it to record correctly
        // Or ensure that the module's operations are recording-aware.
        
        try {
            // The most simple way: use the facade with a forward pass lambda
            println("Starting exportToArduinoLibrary")
            val result = facade.exportToArduinoLibrary(
                model = model,
                forwardPass = { recordingCtx ->
                    println("Running forward pass")
                    // We create the module using the recording context
                    val module = model.create(recordingCtx)
                    
                    // We need to run a forward pass to record the operations
                    val input = sk.ainet.context.data<sk.ainet.lang.types.FP32, Float>(recordingCtx) {
                        tensor {
                            shape(1, 1) {
                                fromArray(floatArrayOf(0.0f))
                            }
                        }
                    }
                    module.forward(input, recordingCtx)
                    println("Forward pass completed")
                },
                outputPath = outputPath,
                libraryName = libraryName
            )
            
            println("Library generated at: ${result.libraryPath}")
            println("Generated files: ${result.generatedFiles}")
            
            val libDir = File(result.libraryPath)
            if (!libDir.exists()) {
                println("Library directory NOT FOUND at: ${libDir.absolutePath}")
                println("Current working directory: ${File(".").absolutePath}")
                println("Contents of build/generated/arduino:")
                File("build/generated/arduino").listFiles()?.forEach { println(" - ${it.name}") }
            }
            assertTrue(libDir.exists(), "Library directory should exist at ${libDir.absolutePath}")
            assertTrue(File(libDir, "src/${libraryName.lowercase()}.h").exists(), "Header file should exist")
            val sourceFile = File(libDir, "src/${libraryName.lowercase()}.c")
            // The file extension might be .c or .cpp depending on version, let's be flexible
            val finalSourceFile = if (sourceFile.exists()) sourceFile else File(libDir, "src/${libraryName.lowercase()}.cpp")
            assertTrue(finalSourceFile.exists(), "Source file should exist")
            
            // Check for non-zero bias values
            val sourceContent = finalSourceFile.readText()
            println("Generated source content length: ${sourceContent.length}")
            
            val lines = sourceContent.lines()
            
            if (sourceContent.contains("_bias")) {
                println("Source preview (bias part):")
                val biasStartLines = lines.filter { it.contains("_bias") && it.contains("{") }
                biasStartLines.forEach { println(it) }
                
                // Extract values more robustly
                val biasValues = mutableListOf<Double>()
                for (i in lines.indices) {
                    if (lines[i].contains("_bias") && lines[i].contains("{")) {
                        var j = i
                        var content = ""
                        while (j < lines.size && !lines[j].contains("}")) {
                            content += lines[j]
                            j++
                        }
                        if (j < lines.size) content += lines[j]
                        
                        val valuesStr = content.substringAfter("{").substringBefore("}")
                        valuesStr.split(",").forEach {
                            it.split(Regex("\\s+")).forEach { sub ->
                                val v = sub.trim().removeSuffix("f").trim()
                                if (v.isNotEmpty()) biasValues.add(v.toDouble())
                            }
                        }
                    }
                }
                
                println("Found ${biasValues.size} bias values")
                val hasNonZeroBias = biasValues.any { it != 0.0 }
                
                // Detailed check for dense_0_bias
                if (sourceContent.contains("dense_0_bias")) {
                    val d0bias = mutableListOf<Double>()
                    val linesList = sourceContent.lines()
                    for (i in linesList.indices) {
                        if (linesList[i].contains("dense_0_bias") && linesList[i].contains("{")) {
                            var j = i
                            var content = ""
                            while (j < linesList.size && !linesList[j].contains("}")) {
                                content += linesList[j]
                                j++
                            }
                            if (j < linesList.size) content += linesList[j]
                            val valuesStr = content.substringAfter("{").substringBefore("}")
                            valuesStr.split(",").forEach {
                                it.split(Regex("\\s+")).forEach { sub ->
                                    val v = sub.trim().removeSuffix("f").trim()
                                    if (v.isNotEmpty()) d0bias.add(v.toDouble())
                                }
                            }
                        }
                    }
                    println("dense_0_bias values: $d0bias")
                    val d0NonZero = d0bias.any { it != 0.0 }
                    assertTrue(d0NonZero, "dense_0_bias should have non-zero values")
                }
                
                assertTrue(hasNonZeroBias, "Should have at least one non-zero bias value in generated code")
            }

            if (sourceContent.contains("_weights")) {
                println("Source preview (weights part):")
                val weightStartLines = lines.filter { it.contains("_weights") && it.contains("{") }
                weightStartLines.take(5).forEach { println(it) }
                
                // Extract values more robustly
                val weightValues = mutableListOf<Double>()
                for (i in lines.indices) {
                    if (lines[i].contains("_weights") && lines[i].contains("{")) {
                        var j = i
                        var content = ""
                        while (j < lines.size && !lines[j].contains("}")) {
                            content += lines[j]
                            j++
                        }
                        if (j < lines.size) content += lines[j]
                        
                        val valuesStr = content.substringAfter("{").substringBefore("}")
                        valuesStr.split(",").forEach {
                            it.split(Regex("\\s+")).forEach { sub ->
                                val v = sub.trim().removeSuffix("f").trim()
                                if (v.isNotEmpty()) weightValues.add(v.toDouble())
                            }
                        }
                    }
                }
                
                println("Found ${weightValues.size} weight values")
                val hasNonZeroWeight = weightValues.any { it != 0.0 }
                assertTrue(hasNonZeroWeight, "Should have at least one non-zero weight value in generated code")
            }
        } catch (e: Exception) {
            println("EXCEPTION CAUGHT: ${e.message}")
            e.printStackTrace()
            throw e
        }
    }
}
