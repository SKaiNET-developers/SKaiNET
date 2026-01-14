package sk.ainet.tape

import sk.ainet.lang.trace.OpTrace

/**
 * Strategy for recording operations and traces onto an execution tape.
 */
public interface TapeRecordingStrategy {
    /**
     * Records a [RecordedOperation] into the provided target list.
     */
    public fun recordOperation(
        op: RecordedOperation,
        targetList: MutableList<RecordedOperation>
    )

    /**
     * Records an [OpTrace] into the provided target list.
     */
    public fun recordTrace(
        trace: OpTrace,
        targetList: MutableList<OpTrace>
    )
}

/**
 * A strategy that actively records operations and traces.
 */
public class ActiveRecordingStrategy : TapeRecordingStrategy {
    override fun recordOperation(
        op: RecordedOperation,
        targetList: MutableList<RecordedOperation>
    ) {
        targetList.add(op)
    }

    override fun recordTrace(
        trace: OpTrace,
        targetList: MutableList<OpTrace>
    ) {
        targetList.add(trace)
    }
}

/**
 * A strategy that ignores all recording requests.
 */
public class NoOpRecordingStrategy : TapeRecordingStrategy {
    override fun recordOperation(
        op: RecordedOperation,
        targetList: MutableList<RecordedOperation>
    ) {
        // Do nothing
    }

    override fun recordTrace(
        trace: OpTrace,
        targetList: MutableList<OpTrace>
    ) {
        // Do nothing
    }
}
