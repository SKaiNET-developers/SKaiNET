package sk.ainet.lang.tensor.ops

/**
 * Single source of truth for convolution output-shape arithmetic.
 *
 * Both eager execution (`VoidTensorOps`) and graph emission
 * (`Conv{1,2,3}dOperation.inferOutputs`) must produce identical shapes;
 * keeping the formula here prevents the two paths from diverging.
 */
public object ConvShapeUtils {

    /**
     * Conv1d output shape: input `(batch, in_channels, length)`,
     * weight `(out_channels, in_channels_per_group, kernel_length)`,
     * result `(batch, out_channels, out_length)`.
     */
    public fun conv1dOutputShape(
        inputShape: IntArray,
        weightShape: IntArray,
        stride: Int,
        padding: Int,
        dilation: Int
    ): IntArray {
        require(inputShape.size == 3) {
            "Conv1d input must be rank 3 (batch, channels, length), got rank ${inputShape.size}"
        }
        require(weightShape.size == 3) {
            "Conv1d weight must be rank 3 (out_channels, in_channels, kernel_length), got rank ${weightShape.size}"
        }
        val batch = inputShape[0]
        val outChannels = weightShape[0]
        val inLength = inputShape[2]
        val kernel = weightShape[2]
        val outLength = (inLength + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
        return intArrayOf(batch, outChannels, outLength)
    }

    /**
     * Conv2d output shape: input `(batch, in_channels, height, width)`,
     * weight `(out_channels, in_channels_per_group, kernel_h, kernel_w)`,
     * result `(batch, out_channels, out_h, out_w)`.
     */
    public fun conv2dOutputShape(
        inputShape: IntArray,
        weightShape: IntArray,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        dilation: Pair<Int, Int>
    ): IntArray {
        require(inputShape.size == 4) {
            "Conv2d input must be rank 4 (batch, channels, height, width), got rank ${inputShape.size}"
        }
        require(weightShape.size == 4) {
            "Conv2d weight must be rank 4 (out_channels, in_channels, kernel_h, kernel_w), got rank ${weightShape.size}"
        }
        val batch = inputShape[0]
        val outChannels = weightShape[0]
        val inH = inputShape[2]
        val inW = inputShape[3]
        val kH = weightShape[2]
        val kW = weightShape[3]
        val outH = (inH + 2 * padding.first - dilation.first * (kH - 1) - 1) / stride.first + 1
        val outW = (inW + 2 * padding.second - dilation.second * (kW - 1) - 1) / stride.second + 1
        return intArrayOf(batch, outChannels, outH, outW)
    }

    /**
     * Conv3d output shape: input `(batch, in_channels, depth, height, width)`,
     * weight `(out_channels, in_channels_per_group, kernel_d, kernel_h, kernel_w)`,
     * result `(batch, out_channels, out_d, out_h, out_w)`.
     */
    public fun conv3dOutputShape(
        inputShape: IntArray,
        weightShape: IntArray,
        stride: Triple<Int, Int, Int>,
        padding: Triple<Int, Int, Int>,
        dilation: Triple<Int, Int, Int>
    ): IntArray {
        require(inputShape.size == 5) {
            "Conv3d input must be rank 5 (batch, channels, depth, height, width), got rank ${inputShape.size}"
        }
        require(weightShape.size == 5) {
            "Conv3d weight must be rank 5 (out_channels, in_channels, kernel_d, kernel_h, kernel_w), got rank ${weightShape.size}"
        }
        val batch = inputShape[0]
        val outChannels = weightShape[0]
        val inD = inputShape[2]
        val inH = inputShape[3]
        val inW = inputShape[4]
        val kD = weightShape[2]
        val kH = weightShape[3]
        val kW = weightShape[4]
        val outD = (inD + 2 * padding.first - dilation.first * (kD - 1) - 1) / stride.first + 1
        val outH = (inH + 2 * padding.second - dilation.second * (kH - 1) - 1) / stride.second + 1
        val outW = (inW + 2 * padding.third - dilation.third * (kW - 1) - 1) / stride.third + 1
        return intArrayOf(batch, outChannels, outD, outH, outW)
    }
}
