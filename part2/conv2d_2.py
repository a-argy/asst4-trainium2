import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal



"""
This kernel implements a simple 2D matrix transpose.
It uses a tile-based approach along with NKI's built-in transpose kernel,
which only works on tiles of size <= 128x128.
"""
@nki.compiler.skip_middle_end_transformations
@nki.jit
def matrix_transpose(a_tensor):
    M, N = a_tensor.shape
    out = nl.ndarray((N, M), dtype=a_tensor.dtype, buffer=nl.hbm)
    tile_dim = nl.tile_size.pmax  # this should be 128
    
    TILE_DIMENSION = 128
    TILES_PER_BATCH = 1
    OVERSIZED_HEIGHT = TILE_DIMENSION * TILES_PER_BATCH
    
    assert M % tile_dim == N % tile_dim == 0, "Matrix dimensions not divisible by tile dimension!"
    assert M % OVERSIZED_HEIGHT == N % OVERSIZED_HEIGHT == 0, "Matrix dimensions not divisible by tile dimension!"
    
    # Loop over batches of tiles
    # iterate over width of the matrix
    for m in nl.affine_range(M // TILE_DIMENSION):
        # iterate over height of the matrix
        for n in nl.affine_range(N // OVERSIZED_HEIGHT):
            # Load a 128x512 tile
            a_tile = nl.ndarray((TILE_DIMENSION, OVERSIZED_HEIGHT), dtype=a_tensor.dtype, buffer=nl.sbuf)
            nisa.dma_copy(src=a_tensor[m * TILE_DIMENSION : (m + 1) * TILE_DIMENSION, n * OVERSIZED_HEIGHT : (n + 1) * OVERSIZED_HEIGHT], dst=a_tile)
            # Transpose each 128x128 sub-tile
            for i in nl.affine_range(TILES_PER_BATCH):
                sub_tile = a_tile[:, i * TILE_DIMENSION : (i + 1) * TILE_DIMENSION]
                result = nisa.nc_transpose(data=sub_tile)
                result_copy = nl.copy(result, dtype=a_tensor.dtype)
                nisa.dma_copy(src=result_copy, dst=out[(n * TILES_PER_BATCH + i) * TILE_DIMENSION : (n * TILES_PER_BATCH + i + 1) * TILE_DIMENSION, m * TILE_DIMENSION : (m + 1) * TILE_DIMENSION])
    
    return out


@nki.compiler.skip_middle_end_transformations
@nki.jit
def nki_matmul_tiled_(lhsT, rhs, result):
  """NKI kernel to compute a matrix multiplication operation in a tiled manner"""

  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"

  # Maximum free dimension of the stationary operand of general matrix multiplication on tensor engine
  TILE_M = nl.tile_size.gemm_stationary_fmax  # 128

  # Maximum partition dimension of a tile
  TILE_K = nl.tile_size.pmax  # 128

  # Maximum free dimension of the moving operand of general matrix multiplication on tensor engine
  TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  # Use affine_range to loop over tiles
  for m in nl.affine_range(M // TILE_M):
    for n in nl.affine_range(N // TILE_N):
      # Allocate a tensor in PSUM
      res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

      for k in nl.affine_range(K // TILE_K):
        # Declare the tiles on SBUF
        lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
        rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)

        # Load tiles from lhsT and rhs
        nisa.dma_copy(dst=lhsT_tile, src=lhsT[k * TILE_K:(k + 1) * TILE_K, m * TILE_M:(m + 1) * TILE_M])
        nisa.dma_copy(dst=rhs_tile, src=rhs[k * TILE_K:(k + 1) * TILE_K, n * TILE_N:(n + 1) * TILE_N])

        # Accumulate partial-sums into PSUM
        res_psum += nisa.nc_matmul(lhsT_tile[...], rhs_tile[...])

      # Copy the result from PSUM back to SBUF, and cast to expected output data-type
      res_sb = nl.copy(res_psum, dtype=result.dtype)
      nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N], src=res_sb)

"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""
@nki.compiler.skip_middle_end_transformations
@nki.jit
def fused_conv2d_maxpool_2(X, W, bias, pool_size=1):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]
    # Constants
    NUM_FILTERS = out_channels


    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == out_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )
    # print("X_out shape:")
    # print(X_out.shape)

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = nl.tile_size.pmax
    n_tiles_c_in = in_channels // c_in_pmax
    k = out_channels // (filter_height * filter_height)
    # Process the images in batches
   
    # # flatten image
    # print("x shape:")
    # print(X.shape)
    # x_re = X.reshape(shape=(batch_size, (input_height * input_width), in_channels))
    # print("x_re shape:")
    # print(x_re.shape)

    # # flatten filters
    # print("w shape:")
    # print(W.shape)
    # w_re = W.reshape(shape=(out_channels, in_channels, filter_height * filter_width))
    # print("w_re shape:")
    # print(w_re.shape)

    # x_re_T = nl.ndarray(shape=(batch_size, (input_height * input_width), in_channels), dtype=X.dtype, buffer=nl.hbm)
    # for b in nl.affine_range(batch_size):
    #     out = matrix_transpose(x_re[b])
    #     nisa.dma_copy(src=out, dst=x_re_T[b])
    # print("x_re_T shape:")
    # print(x_re_T.shape)

    for b in nl.affine_range(batch_size):
        for i in nl.affine_range(filter_height):
            for j in nl.affine_range(filter_width):
                result = nl.ndarray(shape = ((input_height - filter_height + 1) * (input_width - filter_width + 1), out_channels), dtype = X.dtype, buffer = nl.hbm)
                
                
                # input_tile = nl.ndarray(shape=((input_height - filter_height + 1) * (input_width - filter_width + 1), in_channels), dtype=X.dtype, buffer=nl.hbm)
                # NOTE: EVEN if you reshape the entire input image, the indexing is too complicate (actually i dont think it is possible)
                # instead, we can just copy the slice of the input image and THEN reshape
                # nisa.dma_copy(src= X[b, : , i : i + (input_height - filter_height + 1), j : j + (input_width - filter_width + 1)], dst = input_tile)
                
                # this didnt work because you need to copy the data to an intermediate buffer then reshape that buffer
                # input_tile = nl.ndarray(shape=((input_height - filter_height + 1) * (input_width - filter_width + 1), in_channels), dtype=X.dtype, buffer=nl.hbm)
                # src_slice = X[b, :, i : i + (input_height - filter_height + 1), j : j + (input_width - filter_width + 1)]
                # src_reshaped = src_slice.reshape(shape=((input_height - filter_height + 1) * (input_width - filter_width + 1), in_channels))
                # nisa.dma_copy(src=src_reshaped, dst=input_tile)
                
                # First, allocate and copy with the original shape
                input_tile_temp = nl.ndarray(shape=(in_channels, (input_height - filter_height + 1), (input_width - filter_width + 1)), dtype=X.dtype, buffer=nl.sbuf)
                nisa.dma_copy(src=X[b, :, i : i + (input_height - filter_height + 1), j : j + (input_width - filter_width + 1)], dst=input_tile_temp)

                # Then reshape the copied buffer
                input_tile = input_tile_temp.reshape(shape=((input_height - filter_height + 1) * (input_width - filter_width + 1), in_channels))

                filter_tile = nl.ndarray(shape=(out_channels, in_channels), dtype=W.dtype, buffer=nl.hbm)
                nisa.dma_copy(src= W[ : , : , i : i + 1, j : j+ 1], dst = filter_tile)
                print("filter_tile shape:")
                print(filter_tile.shape)
                print("input_tile shape:")
                print(input_tile.shape)
                nki_matmul_tiled_(filter_tile, input_tile, result)
                # add the result to the output
                nisa.dma_copy(src=result, dst=X_out[b, :, i : i + (input_height - filter_height + 1), j : j + (input_width - filter_width + 1)], dst_rmw_op=np.add)

    return X_out

