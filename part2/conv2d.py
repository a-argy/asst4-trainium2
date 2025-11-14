import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal

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
def fused_conv2d_maxpool(X, W, bias, pool_size=1):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]
    # Constants
    NUM_FILTERS = out_channels
    MAX_WIDTH = nl.tile_size.gemm_moving_fmax
    # this is maxxed out, there might be an intelligent way to set this dynamically
    FILTER_CHUNK = 128


    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size

    # we need to be smart about picking these constants 
    # the product of the last two dimensions in the matmul can't exceed nl.tile_size.gemm_moving_fmax
    # HEIGHT_CHUNK = min((MAX_WIDTH // out_width), out_height)
    
    
    # if (out_height // 4) > HEIGHT_CHUNK:
    #     BIG_HEIGHT_CHUNK = out_height // 4
    #     SMALL_HEIGHT_CHUNK = HEIGHT_CHUNK
    # else:
    #     BIG_HEIGHT_CHUNK  = out_height
    #     SMALL_HEIGHT_CHUNK = HEIGHT_CHUNK

    # SMALL_HEIGHT_CHUNK = min(MAX_WIDTH // out_width, out_height)
    # BIG_HEIGHT_CHUNK   = min(out_height, SMALL_HEIGHT_CHUNK * 2)

    if input_width * input_height <= 512:
        SMALL_HEIGHT_CHUNK = out_height
        BIG_HEIGHT_CHUNK   = out_height
    else:
        SMALL_HEIGHT_CHUNK = 2
        BIG_HEIGHT_CHUNK   = 6
    POOL_HEIGHT = pool_size
    POOL_WIDTH = pool_size

    # this dimensinon can be at most 128 in the matmul
    IN_CHANNEL_CHUNK = min(128, in_channels)

    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == out_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = nl.tile_size.pmax
    n_tiles_c_in = in_channels // c_in_pmax
    k = out_channels // (filter_height * filter_height)


    # Preload weights (with transpose) and bias into SBUF for easy use fo tensor_copy later
    W_sbuf = nl.ndarray(shape = (FILTER_CHUNK, IN_CHANNEL_CHUNK, in_channels // IN_CHANNEL_CHUNK, out_channels // FILTER_CHUNK, filter_height, filter_width), dtype = X.dtype, buffer = nl.sbuf)
    bias_sbuf = nl.ndarray(shape = (FILTER_CHUNK, out_channels // FILTER_CHUNK), dtype = nl.float32, buffer = nl.sbuf)
    # copy weights and transpose into sbuf
    for o_ch in nl.affine_range(out_channels // FILTER_CHUNK):
        W_temp = nl.ndarray(shape = (FILTER_CHUNK, in_channels, filter_height, filter_width), dtype = X.dtype, buffer = nl.sbuf)
        for in_ch in nl.affine_range(in_channels // IN_CHANNEL_CHUNK):
            nisa.dma_copy(dst=W_temp[ : , in_ch * IN_CHANNEL_CHUNK : (in_ch + 1) * IN_CHANNEL_CHUNK, : , : ], src = W[o_ch * FILTER_CHUNK : (o_ch + 1) * FILTER_CHUNK, in_ch * IN_CHANNEL_CHUNK : (in_ch + 1) * IN_CHANNEL_CHUNK, : , : ])
            for i in nl.affine_range(filter_height):
                for j in nl.affine_range(filter_width):
                    # filter_tile = nl.ndarray(shape = (FILTER_CHUNK, IN_CHANNEL_CHUNK), dtype = X.dtype, buffer = nl.sbuf)
                    # filter_tile = nisa.tensor_copy(W_temp[ : , in_ch * IN_CHANNEL_CHUNK : (in_ch + 1) * IN_CHANNEL_CHUNK, i, j])

                    filter_tile_transpose_psum = nisa.nc_transpose(data=W_temp[ : , in_ch * IN_CHANNEL_CHUNK : (in_ch + 1) * IN_CHANNEL_CHUNK, i, j])
                    W_sbuf[ : , : , in_ch, o_ch, i, j] = nisa.tensor_copy(filter_tile_transpose_psum, dtype=W_temp.dtype)
        # copy bias into sbuf
        nisa.dma_copy(dst=bias_sbuf[ : , o_ch], src = bias[o_ch * FILTER_CHUNK : (o_ch + 1) * FILTER_CHUNK])
        
    
    # Main computation 

    for b in nl.affine_range(batch_size):
        for o_ch in nl.affine_range(out_channels // FILTER_CHUNK): 
            for big_row in nl.affine_range(out_height // BIG_HEIGHT_CHUNK): 
                # i really want this loop inside of channels because then we can apply bias 
                # and maxpooling for each output channel once only. This may create some data
                # transferring in efficienceis

                oversized_res_sbuf = nl.ndarray(shape = (FILTER_CHUNK, BIG_HEIGHT_CHUNK, out_width), dtype = X.dtype, buffer = nl.sbuf)

                if pool_size == 2:
                    oversized_res_sbuf = nl.ndarray(shape = (FILTER_CHUNK, BIG_HEIGHT_CHUNK // POOL_HEIGHT, out_pool_width), dtype = X.dtype, buffer = nl.sbuf)
                        # res_psum = res_psum.reshape(shape = (FILTER_CHUNK, SMALL_HEIGHT_CHUNK // POOL_HEIGHT, POOL_HEIGHT, out_pool_width, POOL_WIDTH))



                # copy the bias into sbuf to be used broadcast through the sbuf result at the end of this iteration
                # bias_tile = nl.ndarray(shape = (FILTER_CHUNK, 1), dtype=nl.float32, buffer = nl.sbuf)
                #bias_tile = nisa.tensor_copy(bias_sbuf[ : , o_ch])
               

                for small_row in nl.affine_range(BIG_HEIGHT_CHUNK // SMALL_HEIGHT_CHUNK): 

                    # initalize with zeros in psum
                    res_psum = nl.zeros(shape = (FILTER_CHUNK, SMALL_HEIGHT_CHUNK, out_width), dtype = nl.float32, buffer = nl.psum)

                    # chunk input channels after defining res_psum because we are contracting on this dimension
                    for in_ch in nl.affine_range(in_channels // IN_CHANNEL_CHUNK):
                        input_tile = nl.ndarray(shape = (IN_CHANNEL_CHUNK, SMALL_HEIGHT_CHUNK + filter_height - 1, input_width), dtype = X.dtype, buffer = nl.sbuf)
                        nisa.dma_copy(dst=input_tile, src=X[b, in_ch * IN_CHANNEL_CHUNK : (in_ch + 1) * IN_CHANNEL_CHUNK, (big_row * BIG_HEIGHT_CHUNK) + (small_row * SMALL_HEIGHT_CHUNK) : (big_row * BIG_HEIGHT_CHUNK) + ((small_row + 1) * SMALL_HEIGHT_CHUNK) + filter_height - 1, : ])
                        for i in nl.affine_range(filter_height):
                            for j in nl.affine_range(filter_width):
                                # create the filter tile and copy in data
                                # filter tile is split by output channel & input channel
                                #sub_input_tile = nisa.tensor_copy(input_tile[ : , i : SMALL_HEIGHT_CHUNK + i, j : out_width + j])
                                
                    
                                # filter_tile_transpose = nl.ndarray(shape = (IN_CHANNEL_CHUNK, FILTER_CHUNK), dtype = X.dtype, buffer = nl.sbuf)
                                #filter_tile_transpose = nisa.tensor_copy(W_sbuf[ : , : , in_ch, o_ch, i, j])
                                
                                # # create the input tile and copy in data
                                # # input tile is split by input channel & height
                                # input_tile = nl.ndarray(shape = (IN_CHANNEL_CHUNK, SMALL_HEIGHT_CHUNK, out_width), dtype = X.dtype, buffer = nl.sbuf)
                                # nisa.dma_copy(dst=input_tile, src=X[b, in_ch * IN_CHANNEL_CHUNK : (in_ch + 1) * IN_CHANNEL_CHUNK, i + (big_row * BIG_HEIGHT_CHUNK) + (small_row * SMALL_HEIGHT_CHUNK) : i + (big_row * BIG_HEIGHT_CHUNK) + ((small_row + 1) * SMALL_HEIGHT_CHUNK), j: j + out_width])

                                # 2d x 3d mat mul
                                res_psum += nisa.nc_matmul(W_sbuf[ : , : , in_ch, o_ch, i, j], input_tile[ : , i : SMALL_HEIGHT_CHUNK + i, j : out_width + j])
                                
                    
                    # move the accumulation from psum to oversized sbuf
                    if pool_size == 2:
                        res_psum = res_psum.reshape(shape = (FILTER_CHUNK, SMALL_HEIGHT_CHUNK // POOL_HEIGHT, POOL_HEIGHT, out_pool_width, POOL_WIDTH))
                        res_psum = nisa.tensor_reduce(op=nki.language.maximum, data=res_psum, axis=(2,4))
                        oversized_res_sbuf[ : , small_row * SMALL_HEIGHT_CHUNK // 2 : (small_row + 1) * SMALL_HEIGHT_CHUNK // 2, 0 : out_pool_width] = nisa.tensor_copy(res_psum)
                    else:
                        oversized_res_sbuf[ : , small_row * SMALL_HEIGHT_CHUNK : (small_row + 1) * SMALL_HEIGHT_CHUNK, : ] = nisa.tensor_copy(res_psum)

                # the free dimensino of sbuf (single bias) is stretched acorss the free dimension of result (hieght x width)
                # (this note was before the move) : possible inefficiency here because this touches each bias more than once, given we are chunking
                # by HEIGHT_CHUNK on the outside of this loop, BUT we can't fit a tile into psum without chunking...
                oversized_res_sbuf = nisa.tensor_tensor(oversized_res_sbuf, bias_sbuf[ : , o_ch], nki.language.add)
                # maxpool happens here 
                if pool_size == 2:
                    # # oversized_res_sbuf = nl.ndarray(shape = (FILTER_CHUNK, BIG_HEIGHT_CHUNK, out_width), dtype = X.dtype, buffer = nl.sbuf)
                    
                    # oversized_res_sbuf = oversized_res_sbuf.reshape(shape = (FILTER_CHUNK, (BIG_HEIGHT_CHUNK // POOL_HEIGHT), POOL_HEIGHT, (out_width // POOL_WIDTH), POOL_WIDTH))
                    # oversized_res_sbuf = nisa.tensor_reduce(op=nki.language.maximum, data=oversized_res_sbuf, axis=(2,4))
                    # # oversized_res_sbuf = oversized_res_sbuf.reshape(shape = (FILTER_CHUNK, BIG_HEIGHT_CHUNK, out_width))
                    nisa.dma_copy(dst=X_out[b, o_ch * FILTER_CHUNK : (o_ch + 1) * FILTER_CHUNK, big_row * (BIG_HEIGHT_CHUNK // POOL_HEIGHT) : (1 + big_row) * (BIG_HEIGHT_CHUNK // POOL_HEIGHT), 0 : (out_width // POOL_WIDTH)], src=oversized_res_sbuf)
                   
                else: 
                    nisa.dma_copy(dst=X_out[b, o_ch * FILTER_CHUNK : (o_ch + 1) * FILTER_CHUNK, big_row * BIG_HEIGHT_CHUNK : (1 + big_row) * BIG_HEIGHT_CHUNK, : ], src=oversized_res_sbuf)


    return X_out
    