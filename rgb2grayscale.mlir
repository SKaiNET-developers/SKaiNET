module {
  func.func @rgb2grayscale(%arg0: tensor<1x3x4x4xf32>) -> (tensor<1x1x4x4xf32>) {
    // input n1_input: t0 : tensor<1x3x4x4xf32>
    // weight n2_weight: frozen parameter
    %v0 = stablehlo.constant dense<[[[[0.2989]], [[0.587]], [[0.114]]]]> : tensor<1x3x1x1xf32>
    %v1 = stablehlo.convolution(%arg0, %v0) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x4x4xf32>, tensor<1x3x1x1xf32>) -> tensor<1x1x4x4xf32>
    return %v1 : tensor<1x1x4x4xf32>
  }
}
