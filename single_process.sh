#!/bin/bash
export LD_LIBRARY_PATH=/public/home/qinxy/anaconda3/envs/avse/lib:$LD_LIBRARY_PATH

#export MIOPEN_DEBUG_DISABLE_FIND_DB=1
#export MIOPEN_DEBUG_CONV_WINOGRAD=0  
#export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1 
#export HSA_USERPTR_FOR_PAGED_MEM=0

#expor tMIOPEN_DISABLE_CACHE=true
export MIOPEN_FIND_MODE=3
export MIOPEN_USER_DB_PATH=/tmp/miopen-udb
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen-cache

export GLOO_SOCKET_IFNAME=ib0
export HSA_USERPTR_FOR_PAGED_MEM=0
export HSA_FORCE_FINE_GRAIN_PRICE=1
export NCCL_IB_HCA=mlx5_0
#export NCCL_SOCKET_IFNAME=eno1
#export NCCL_IB_DISABLE=1
#export NCCL_DEBUG=info
export ROCBLAS_TENSILE_LIBPATH=/public/software/compiler/rocm/dtk-23.04.1/lib/rocblas/library_dcu2
export PATH=/public/software/compiler/rocm/dtk-23.04.1/llvm/bin/:$PATH
export MIOPEN_ASSIGN_AMDGCN_ASSEMBLER=/public/software/compiler/rocm/dtk-23.04.1/llvm/bin/clang

lrank=$OMPI_COMM_WORLD_LOCAL_RANK
comm_rank=$OMPI_COMM_WORLD_RANK
comm_size=$OMPI_COMM_WORLD_SIZE

myhost=$(hostname)
echo lrank is $lrank at $myhost
echo comm_rank is $comm_rank at $myhost
echo comm_size is $comm_size at $myhost

echo disturl is tcp://${1}:34567

# APP="python -u train.py --name 25_1217_NOT_product_NEW_random_repeat_conv_before_LSTM_front_product_two_masked_3_random_view_mead_add_after_vconv_new_exceptM019_9 --dist-url tcp://${1}:34567 --dist-backend nccl --world-size=${comm_size} --rank=${comm_rank} --continue_from ./ckpt/25_NEW_random_repeat_conv_before_LSTM_front_product_two_masked_3_random_view_mead_add_after_vconv_new_exceptM019_4/epoch39.pth.tar"
APP="python -u train.py --name 25_wave --continue_from ./ckpt/25_NEW_random_repeat_conv_before_LSTM_front_product_two_masked_3_random_view_mead_add_after_vconv_new_exceptM019_3/epoch42.pth.tar --dist-url tcp://${1}:34567 --dist-backend nccl --world-size=${comm_size} --rank=${comm_rank}"


case ${lrank} in
[0])
  export HIP_VISIBLE_DEVICES=0
  export UCX_NET_DEVICES=mlx5_0:1
  export UCX_IB_PCI_BW=mlx5_0:50Gbs
  NCCL_SOCKET_IFNAME=ib0 numactl --cpunodebind=0 --membind=0 ${APP}
;;
[1])
  export HIP_VISIBLE_DEVICES=1
  export UCX_NET_DEVICES=mlx5_1:1
  export UCX_IB_PCI_BW=mlx5_1:50Gbs
  NCCL_SOCKET_IFNAME=ib0 numactl --cpunodebind=1 --membind=1 ${APP}
;;
[2])
  export HIP_VISIBLE_DEVICES=2
  export UCX_NET_DEVICES=mlx5_2:1
  export UCX_IB_PCI_BW=mlx5_2:50Gbs
  NCCL_SOCKET_IFNAME=ib0 numactl --cpunodebind=2 --membind=2 ${APP}
;;
[3])
  export HIP_VISIBLE_DEVICES=3
  export UCX_NET_DEVICES=mlx5_3:1
  export UCX_IB_PCI_BW=mlx5_3:50Gbs
  NCCL_SOCKET_IFNAME=ib0 numactl --cpunodebind=3 --membind=3 ${APP}
;;
esac

