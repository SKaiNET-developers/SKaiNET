package sk.ainet.lang.dag

import sk.ainet.context.schedule.SCHEDULE_ATTRIBUTE_KEY
import sk.ainet.context.schedule.ScheduleHint
import sk.ainet.lang.tensor.ops.MatmulOperation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

/** SKEEP-005: schedule hints attach to `dag { }` nodes under the shared attribute key. */
class ScheduleDslTest {

    @Test
    fun perOpScheduleLandsUnderTheKey() {
        val program = dag {
            val x = input<FP32>("x", TensorSpec("x", listOf(1, 4), "Float32"))
            val w = parameter<FP32, Float>("w") { shape(4, 4) { ones() } }
            op(MatmulOperation<FP32, Float>(), listOf(x, w), schedule = parallel("rows", parallelism = 8))
        }
        val mm = program.nodes.last { it.operation is MatmulOperation<*, *> }
        assertEquals(ScheduleHint(listOf("rows"), 8), mm.scheduleHint())
        assertEquals(mm.scheduleHint(), mm.attributes[SCHEDULE_ATTRIBUTE_KEY])
    }

    @Test
    fun ambientScheduleAppliesToEveryOpInTheBlockAndExplicitWins() {
        val program = dag {
            val x = input<FP32>("x", TensorSpec("x", listOf(1, 4), "Float32"))
            val w = parameter<FP32, Float>("w") { shape(4, 4) { ones() } }
            schedule(parallel("rows")) {
                op(MatmulOperation<FP32, Float>(), listOf(x, w), id = "ambient")
                op(MatmulOperation<FP32, Float>(), listOf(x, w), schedule = parallel("rows", parallelism = 2), id = "explicit")
            }
            op(MatmulOperation<FP32, Float>(), listOf(x, w), id = "outside")
        }
        val byId = program.nodes.associateBy { it.id }
        assertEquals(ScheduleHint(listOf("rows")), byId.getValue("ambient").scheduleHint())
        assertEquals(ScheduleHint(listOf("rows"), 2), byId.getValue("explicit").scheduleHint(), "explicit per-op hint wins")
        assertNull(byId.getValue("outside").scheduleHint(), "the block's hint does not leak")
    }
}
