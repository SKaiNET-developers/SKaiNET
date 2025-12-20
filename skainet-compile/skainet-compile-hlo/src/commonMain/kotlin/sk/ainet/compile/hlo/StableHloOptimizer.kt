package sk.ainet.compile.hlo

import sk.ainet.compile.hlo.validation.MlirParser
import sk.ainet.compile.hlo.validation.ParsedOperation

/**
 * Optimizer for StableHLO MLIR modules.
 * 
 * This class provides optimization passes for StableHLO code,
 * including operation fusion, constant folding, and other transformations.
 */
public class StableHloOptimizer {
    
    private val optimizationPasses = mutableListOf<OptimizationPass>()
    
    /**
     * Add an optimization pass
     */
    public fun addPass(pass: OptimizationPass) {
        optimizationPasses.add(pass)
    }
    
    /**
     * Optimize a StableHLO module by applying all registered passes
     */
    public fun optimize(module: StableHloModule): StableHloModule {
        var currentModule = module
        
        for (pass in optimizationPasses) {
            currentModule = pass.apply(currentModule)
        }
        
        return currentModule
    }
    
    /**
     * Create a default optimizer with common optimization passes
     */
    public companion object {
        public fun createDefault(): StableHloOptimizer {
            val optimizer = StableHloOptimizer()
            optimizer.addPass(ConstantFoldingPass())
            optimizer.addPass(OperationFusionPass())
            optimizer.addPass(DeadCodeEliminationPass())
            return optimizer
        }
        
        /**
         * Create an optimizer with aggressive optimizations
         */
        public fun createAggressive(): StableHloOptimizer {
            val optimizer = StableHloOptimizer()
            optimizer.addPass(ConstantFoldingPass())
            optimizer.addPass(OperationFusionPass())
            optimizer.addPass(DeadCodeEliminationPass())
            optimizer.addPass(ConstantFoldingPass()) // Run again after fusion
            return optimizer
        }
    }
}

/**
 * Interface for optimization passes
 */
public interface OptimizationPass {
    /**
     * Apply this optimization pass to a module
     */
    public fun apply(module: StableHloModule): StableHloModule
    
    /**
     * Name of this optimization pass
     */
    public val name: String
}

/**
 * Constant folding optimization pass
 * 
 * This pass identifies and folds constant expressions at compile time,
 * reducing runtime computation and enabling further optimizations.
 */
public class ConstantFoldingPass : OptimizationPass {
    override val name: String = "constant-folding"
    
    override fun apply(module: StableHloModule): StableHloModule {
        var content = module.content
        var changed = true
        
        // Keep applying optimizations until no more changes
        while (changed) {
            val originalContent = content
            content = foldConstants(content)
            changed = content != originalContent
        }
        
        return module.copy(
            content = content,
            metadata = module.metadata + ("optimizations" to (module.metadata["optimizations"] as? List<String> ?: emptyList()) + name)
        )
    }
    
    private fun foldConstants(content: String): String {
        val lines = content.lines().toMutableList()
        val constants = mutableMapOf<String, Float>()
        
        // First pass: extract constants
        for (i in lines.indices) {
            val line = lines[i].trim()
            val constantMatch = Regex("""(%\w+)\s*=\s*stablehlo\.constant\s+dense<([^>]+)>\s*:\s*tensor<[^>]*>""").find(line)
            if (constantMatch != null) {
                val varName = constantMatch.groupValues[1]
                val valueStr = constantMatch.groupValues[2]
                valueStr.toFloatOrNull()?.let { value ->
                    constants[varName] = value
                }
            }
        }
        
        // Second pass: fold arithmetic operations
        for (i in lines.indices) {
            val line = lines[i].trim()
            
            // Match arithmetic operations
            val addMatch = Regex("""(%\w+)\s*=\s*stablehlo\.add\s+(%\w+),\s*(%\w+)\s*:\s*(tensor<[^>]*>)""").find(line)
            if (addMatch != null) {
                val result = addMatch.groupValues[1]
                val left = addMatch.groupValues[2]
                val right = addMatch.groupValues[3]
                val type = addMatch.groupValues[4]
                
                val leftVal = constants[left]
                val rightVal = constants[right]
                
                if (leftVal != null && rightVal != null) {
                    val sum = leftVal + rightVal
                    lines[i] = "    $result = stablehlo.constant dense<$sum> : $type"
                    constants[result] = sum
                }
            }
            
            val mulMatch = Regex("""(%\w+)\s*=\s*stablehlo\.multiply\s+(%\w+),\s*(%\w+)\s*:\s*(tensor<[^>]*>)""").find(line)
            if (mulMatch != null) {
                val result = mulMatch.groupValues[1]
                val left = mulMatch.groupValues[2]
                val right = mulMatch.groupValues[3]
                val type = mulMatch.groupValues[4]
                
                val leftVal = constants[left]
                val rightVal = constants[right]
                
                if (leftVal != null && rightVal != null) {
                    val product = leftVal * rightVal
                    lines[i] = "    $result = stablehlo.constant dense<$product> : $type"
                    constants[result] = product
                }
            }
            
            val subMatch = Regex("""(%\w+)\s*=\s*stablehlo\.subtract\s+(%\w+),\s*(%\w+)\s*:\s*(tensor<[^>]*>)""").find(line)
            if (subMatch != null) {
                val result = subMatch.groupValues[1]
                val left = subMatch.groupValues[2]
                val right = subMatch.groupValues[3]
                val type = subMatch.groupValues[4]
                
                val leftVal = constants[left]
                val rightVal = constants[right]
                
                if (leftVal != null && rightVal != null) {
                    val difference = leftVal - rightVal
                    lines[i] = "    $result = stablehlo.constant dense<$difference> : $type"
                    constants[result] = difference
                }
            }
            
            val divMatch = Regex("""(%\w+)\s*=\s*stablehlo\.divide\s+(%\w+),\s*(%\w+)\s*:\s*(tensor<[^>]*>)""").find(line)
            if (divMatch != null) {
                val result = divMatch.groupValues[1]
                val left = divMatch.groupValues[2]
                val right = divMatch.groupValues[3]
                val type = divMatch.groupValues[4]
                
                val leftVal = constants[left]
                val rightVal = constants[right]
                
                if (leftVal != null && rightVal != null && rightVal != 0.0f) {
                    val quotient = leftVal / rightVal
                    lines[i] = "    $result = stablehlo.constant dense<$quotient> : $type"
                    constants[result] = quotient
                }
            }
        }
        
        return lines.joinToString("\n")
    }
}

/**
 * Dead code elimination pass
 * 
 * This pass removes operations whose results are never used,
 * reducing the size and complexity of the generated code.
 */
public class DeadCodeEliminationPass : OptimizationPass {
    override val name: String = "dead-code-elimination"
    
    override fun apply(module: StableHloModule): StableHloModule {
        val content = eliminateDeadCode(module.content)
        
        return module.copy(
            content = content,
            metadata = module.metadata + ("optimizations" to (module.metadata["optimizations"] as? List<String> ?: emptyList()) + name)
        )
    }
    
    private fun eliminateDeadCode(content: String): String {
        val lines = content.lines().toMutableList()
        val usedValues = mutableSetOf<String>()
        val definedValues = mutableMapOf<String, Int>() // variable -> line index
        
        // First pass: collect all uses (excluding definitions)
        for (line in lines) {
            val trimmedLine = line.trim()
            val uses = Regex("""%\w+""").findAll(trimmedLine)
            for (use in uses) {
                val varName = use.value
                // Don't count the definition itself as a use
                if (!trimmedLine.startsWith("$varName =")) {
                    usedValues.add(varName)
                }
            }
        }
        
        // Second pass: collect definitions
        for (i in lines.indices) {
            val line = lines[i].trim()
            val defMatch = Regex("""(%\w+)\s*=""").find(line)
            if (defMatch != null) {
                val varName = defMatch.groupValues[1]
                definedValues[varName] = i
            }
        }
        
        // Third pass: mark unused definitions for removal
        val linesToRemove = mutableSetOf<Int>()
        for ((varName, lineIndex) in definedValues) {
            if (varName !in usedValues) {
                // Check if this is not a return statement or other side-effect operation
                val line = lines[lineIndex].trim()
                if (!line.contains("return") && !line.contains("func.func") && !line.contains("//")) {
                    linesToRemove.add(lineIndex)
                }
            }
        }
        
        // Remove unused lines
        val filteredLines = lines.filterIndexed { index, _ -> index !in linesToRemove }
        
        return filteredLines.joinToString("\n")
    }
}



/**
 * Operation fusion optimization pass
 * 
 * This pass identifies opportunities to fuse multiple operations into
 * more efficient compound operations, reducing memory traffic and improving performance.
 */
public class OperationFusionPass : OptimizationPass {
    override val name: String = "operation-fusion"
    
    override fun apply(module: StableHloModule): StableHloModule {
        val content = fuseOperations(module.content)
        
        return module.copy(
            content = content,
            metadata = module.metadata + ("optimizations" to (module.metadata["optimizations"] as? List<String> ?: emptyList()) + name)
        )
    }
    
    private fun fuseOperations(content: String): String {
        val lines = content.lines().toMutableList()
        
        // Pattern 1: Fuse Add + ReLU (add followed by maximum with zero constant)
        var i = 0
        while (i < lines.size) {
            val currentLine = lines[i].trim()
            
            // Look for add operations
            val addMatch = Regex("""(%\w+)\s*=\s*stablehlo\.add\s+(%\w+),\s*(%\w+)\s*:\s*(tensor<[^>]*>)""").find(currentLine)
            if (addMatch != null) {
                val addResult = addMatch.groupValues[1]
                val addType = addMatch.groupValues[4]
                
                // Look for maximum operations that use this add result (not necessarily on the next line)
                for (j in i + 1 until lines.size) {
                    val laterLine = lines[j].trim()
                    val maxMatch = Regex("""(%\w+)\s*=\s*stablehlo\.maximum\s+$addResult,\s*(%\w+)\s*:\s*$addType""").find(laterLine)
                    if (maxMatch != null) {
                        val maxResult = maxMatch.groupValues[1]
                        val zeroOperand = maxMatch.groupValues[2]
                        
                        // Check if the other operand is a zero constant
                        if (isZeroConstant(zeroOperand, lines)) {
                            // Fuse into a single add with comment
                            lines[i] = "    $maxResult = stablehlo.add ${addMatch.groupValues[2]}, ${addMatch.groupValues[3]} : $addType // fused with relu"
                            lines.removeAt(j)
                            break
                        }
                    }
                    
                    // Pattern 2: Fuse Add + Multiply
                    val mulMatch = Regex("""(%\w+)\s*=\s*stablehlo\.multiply\s+$addResult,\s*(%\w+)\s*:\s*(tensor<[^>]*>)""").find(laterLine)
                    if (mulMatch != null) {
                        val mulResult = mulMatch.groupValues[1]
                        val mulOperand = mulMatch.groupValues[2]
                        val mulType = mulMatch.groupValues[3]
                        
                        // Fuse into a custom operation
                        lines[i] = "    $mulResult = stablehlo.add ${addMatch.groupValues[2]}, ${addMatch.groupValues[3]} : ${addMatch.groupValues[4]} // fused with multiply $mulOperand"
                        lines.removeAt(j)
                        break
                    }
                }
            }
            
            i++
        }
        
        return lines.joinToString("\n")
    }
    
    private fun isZeroConstant(operand: String, lines: List<String>): Boolean {
        for (line in lines) {
            val constMatch = Regex("""$operand\s*=\s*stablehlo\.constant\s+dense<([^>]+)>""").find(line)
            if (constMatch != null) {
                val value = constMatch.groupValues[1]
                return value == "0.0" || value == "0"
            }
        }
        return false
    }
}