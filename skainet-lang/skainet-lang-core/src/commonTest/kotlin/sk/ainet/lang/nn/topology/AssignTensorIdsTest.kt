package sk.ainet.lang.nn.topology

import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.GradState
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseFloatArrayTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.operators.OpsBoundTensor
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.tensor.ops.VoidTensorOps
import sk.ainet.lang.types.FP32
import kotlin.reflect.KClass
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertSame
import kotlin.test.assertTrue

class AssignTensorIdsTest {

    private fun tensor(): VoidOpsTensor<FP32, Float> =
        VoidOpsTensor(DenseFloatArrayTensorData<FP32>(Shape(2, 2), FloatArray(4)), FP32::class)

    /** A tensor that cannot carry an id (not a TensorIdBearer). */
    private fun bareTensor(): Tensor<FP32, Float> = object : Tensor<FP32, Float> {
        override val data: TensorData<FP32, Float> = DenseFloatArrayTensorData(Shape(1), FloatArray(1))
        override val dtype: KClass<FP32> = FP32::class
        override val ops: TensorOps = VoidTensorOps()
        override val gradState: GradState<FP32, Float> = GradState()
    }

    private class Leaf(
        override val name: String,
        weight: Tensor<FP32, Float>,
        bias: Tensor<FP32, Float>? = null,
    ) : Module<FP32, Float>(), ModuleParameters<FP32, Float> {
        override val modules: List<Module<FP32, Float>> = emptyList()
        override val params: List<ModuleParameter<FP32, Float>> = buildList {
            add(ModuleParameter.WeightParameter("$name.weight", weight))
            if (bias != null) add(ModuleParameter.BiasParameter("$name.bias", bias))
        }
    }

    private class Block(override val name: String, override val modules: List<Module<FP32, Float>>) : Module<FP32, Float>()

    @Test
    fun idsFollowTheModuleTreeAndParameterShortNames() {
        val qW = tensor(); val qB = tensor(); val mlpW = tensor()
        val model = Block("model", listOf(
            Block("layers", listOf(
                Block("blk.0", listOf(Leaf("attn", qW, qB), Leaf("mlp", mlpW))),
            )),
        ))
        val ids = model.assignTensorIds()

        assertEquals(3, ids.tensors.size)
        assertTrue(ids.notCarried.isEmpty())
        assertSame(qW, ids["model.layers.blk.0.attn.weight"])
        assertSame(qB, ids["model.layers.blk.0.attn.bias"])
        assertSame(mlpW, ids["model.layers.blk.0.mlp.weight"])
        assertEquals(TensorId(listOf("model", "layers", "blk.0", "attn"), "weight"), qW.id)
        assertEquals("model/layers/blk.0/attn", qW.id!!.legacyPath())

        // same segments as bindPaths
        bindPaths(model)
        model.walkDepthFirst { node -> node.params.forEach { p -> assertEquals(node.path, p.value.id!!.legacyPath()) } }
    }

    @Test
    fun idempotentAndRootOverridable() {
        val w = tensor()
        val model = Block("net", listOf(Leaf("fc", w)))
        val a = model.assignTensorIds(); val b = model.assignTensorIds()
        assertEquals(a, b)
        assertEquals("net.fc.weight", w.id!!.canonical)
        model.assignTensorIds(root = "")
        assertEquals("fc.weight", w.id!!.canonical)
        assertEquals(TensorId(listOf("fc"), "weight"), model.assignTensorIds("").tensors.keys.single())
    }

    @Test
    fun tensorsThatCannotCarryAnIdAreReported() {
        val bare = bareTensor()
        val model = Block("m", listOf(Leaf("fc", bare)))
        val ids = model.assignTensorIds()
        assertEquals(listOf(TensorId(listOf("m", "fc"), "weight")), ids.notCarried)
        assertSame(bare, ids["m.fc.weight"])
        assertNull(bare.id)
    }

    @Test
    fun idsSurviveRebindingToAnotherContext() {
        val w = tensor()
        Block("m", listOf(Leaf("fc", w))).assignTensorIds()
        val bound = OpsBoundTensor(w, VoidTensorOps())
        assertEquals(w.id, bound.id)
        // setting through the bound wrapper writes through to the origin
        bound.id = TensorId(listOf("m2"), "weight")
        assertEquals("m2.weight", w.id!!.canonical)
        // a bound wrapper over a bare tensor keeps the id locally
        val wrapped = OpsBoundTensor(bareTensor(), VoidTensorOps())
        wrapped.id = TensorId(listOf("x"), "w")
        assertEquals("x.w", wrapped.id!!.canonical)
    }

    @Test
    fun parameterShortNames() {
        val node = Block("attn", emptyList())
        assertEquals("weight", parameterShortName(node, "attn.weight"))
        assertEquals("q_proj.weight", parameterShortName(node, "attn.q_proj.weight"))
        assertEquals("weight_ih", parameterShortName(node, "weight_ih"))
        assertEquals("attn.", parameterShortName(node, "attn."))
    }
}
