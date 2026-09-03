package sk.ainet.exec.tensor.ops

import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertTrue
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import kotlin.test.Ignore

class PlatformCpuOpsFactoryJvmTest {

    @AfterTest
    fun clearFlags() {
        System.clearProperty("skainet.cpu.vector.enabled")
        System.clearProperty("skainet.cpu.blas.enabled")
    }

    @Test
    fun returnsJvmOpsWhenVectorFlagEnabled() {
        System.setProperty("skainet.cpu.vector.enabled", "true")
        val factory = platformDefaultCpuOpsFactory()
        val ops = factory(DenseTensorDataFactory(), sk.ainet.context.schedule.Schedule.Sequential)
        assertTrue(ops is DefaultCpuOpsJvm)
    }

    @Test
    fun fallsBackToScalarOpsWhenFlagDisabled() {
        System.setProperty("skainet.cpu.vector.enabled", "false")
        val factory = platformDefaultCpuOpsFactory()
        val ops = factory(DenseTensorDataFactory(), sk.ainet.context.schedule.Schedule.Sequential)
        assertTrue(ops !is DefaultCpuOpsJvm)
    }
}
