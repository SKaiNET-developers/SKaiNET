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
            assertTrue(File(libDir, "src/${libraryName.lowercase()}.c").exists(), "Source file should exist")
        } catch (e: Exception) {
            println("EXCEPTION CAUGHT: ${e.message}")
            e.printStackTrace()
            throw e
        }
    }
}
