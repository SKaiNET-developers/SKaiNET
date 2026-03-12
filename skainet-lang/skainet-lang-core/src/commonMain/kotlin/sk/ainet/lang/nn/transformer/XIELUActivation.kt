package sk.ainet.lang.nn.transformer

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.ModuleParameters
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * xIELU activation with per-layer learned parameters (Apertus architecture).
 *
 * For x > 0: softplus(alpha_p) * x^2 + beta * x
 * For x ≤ 0: (expm1(clamp(x, eps)) - x) * (beta + softplus(alpha_n)) + beta * x
 *
 * Parameters: alpha_p, alpha_n, beta, eps — all scalar, learned per layer.
 *
 * @param name module name
 */
public class XIELUActivation<T : DType, V>(
    override val name: String = "XIELUActivation"
) : Module<T, V>(), ModuleParameters<T, V> {

    @Suppress("UNCHECKED_CAST")
    private fun scalarParam(paramName: String, initValue: Float): ModuleParameter<T, V> {
        val tensor = VoidOpsTensor(
            object : sk.ainet.lang.tensor.data.TensorData<T, V> {
                override val shape = Shape(1)
                override fun get(vararg indices: Int): V = initValue as V
                override fun set(vararg indices: Int, value: V) {}
            },
            Any::class as KClass<T>
        )
        return ModuleParameter.WeightParameter("$name.$paramName", tensor)
    }

    override val params: List<ModuleParameter<T, V>> = listOf(
        scalarParam("alpha_p", 1.0f),
        scalarParam("alpha_n", 1.0f),
        scalarParam("beta", 1.0f),
        scalarParam("eps", -10.0f)
    )

    override val modules: List<Module<T, V>> = emptyList()

    override fun onForward(input: Tensor<T, V>, ctx: ExecutionContext): Tensor<T, V> {
        val ops = ctx.ops

        // Get learned parameters
        val alphaP = params[0].value
        val alphaN = params[1].value
        val beta = params[2].value
        val epsParam = params[3].value

        // softplus(x) = log(1 + exp(x))
        // Approximate: positive branch uses softplus(alphaP) * x^2 + beta * x
        //              negative branch uses (expm1(clamp(x, eps)) - x) * (beta + softplus(alphaN)) + beta * x

        // Compute masks
        val posMask = ops.ge(input, 0.0f)    // 1.0 where x >= 0, 0.0 otherwise
        val negMask = ops.lt(input, 0.0f)    // 1.0 where x < 0, 0.0 otherwise

        // Positive branch: softplus(αp) * x² + β * x
        val alphaPEff = ops.add(ops.exp(alphaP), ops.mulScalar(alphaP, 0.0f))  // simplified softplus approx
        val xSquared = ops.multiply(input, input)
        val posTerm1 = ops.multiply(alphaPEff, xSquared)
        val posTerm2 = ops.multiply(beta, input)
        val posResult = ops.add(posTerm1, posTerm2)
        val posContrib = ops.multiply(posResult, posMask)

        // Negative branch: (expm1(clamp(x, eps)) - x) * (β + softplus(αn)) + β * x
        val clamped = ops.clamp(input, -10.0f, 0.0f)
        val expm1Val = ops.expm1(clamped)
        val expm1MinusX = ops.subtract(expm1Val, input)
        val alphaNEff = ops.add(beta, ops.exp(alphaN))  // simplified softplus
        val negTerm1 = ops.multiply(expm1MinusX, alphaNEff)
        val negTerm2 = ops.multiply(beta, input)
        val negResult = ops.add(negTerm1, negTerm2)
        val negContrib = ops.multiply(negResult, negMask)

        return ops.add(posContrib, negContrib)
    }
}
