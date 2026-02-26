#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/glm4-9B.sh"

CKPT_ARGS=(
   --hf-checkpoint /apdcephfs_tj4/share_303922427/altmanyang/cache/huggingface/hub/models--zai-org--GLM-Z1-9B-0414/snapshots/b221b06fefb23ca320922cf6e68ab5f2fb82de81
   --ref-load /apdcephfs_tj4/share_303922427/altmanyang/cache/huggingface/GLM-Z1-9B-0414_torch_dist
   --load /apdcephfs_tj4/share_303922427/altmanyang/GLM-Z1-9B-0414_slime/
   --save /apdcephfs_tj4/share_303922427/altmanyang/GLM-Z1-9B-0414_slime/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data /apdcephfs_tj4/share_303922427/altmanyang/cache/huggingface/hub/datasets--zhuzilin--dapo-math-17k/snapshots/2e65612930298bde4c5d58fd97b3f23a483aaff9/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   --rm-type deepscaler

   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1

   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime /apdcephfs_tj4/share_303922427/altmanyang/cache/huggingface/hub/datasets--zhuzilin--aime-2024/snapshots/1c625e328db94ec7ef7ff169016b097c468d60b9/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 2
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4608
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_KEY=${WANDB_KEY:-""}
WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-test
   --wandb-group glm-4-9B
   --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8080
ulimit -n 1048576 || ulimit -n 65536 || true
ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8080
sleep 5

# Build the runtime environment JSON with proper variable substitution
# RUNTIME_ENV_JSON="{
#   \"env_vars\": {
#     \"PYTHONPATH\": \"/root/Megatron-LM/\",
#     \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
#     \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
#   }
# }"

ray job submit --address="http://127.0.0.1:8080" \
   --runtime-env-json='{
      "env_vars": {
         "PYTHONPATH": "/root/Megatron-LM/",
         "CUDA_DEVICE_MAX_CONNECTIONS": "1",
         "NCCL_IP_LOCAL_INTERFACE_TYPE": "IPv4",
         "NCCL_IB_GID_INDEX": "3",
         "NCCL_CHECK_DISABLE": "1",
         "NCCL_IB_SL": "3",
         "NCCL_P2P_DISABLE": "0",
         "NCCL_IB_DISABLE": "1",
         "NCCL_SOCKET_IFNAME": "bond1",
         "NCCL_LL_THRESHOLD": "16384",
         "NCCL_IB_CUDA_SUPPORT": "1",
         "UCX_NET_DEVICES": "bond1",
         "NCCL_IB_HCA": "mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6",
         "NCCL_COLLNET_ENABLE": "0",
         "SHARP_COLL_ENABLE_SAT": "0",
         "NCCL_NET_GDR_LEVEL": "2",
         "NCCL_IB_QPS_PER_CONNECTION": "4",
         "NCCL_IB_TC": "160",
         "NCCL_PXN_DISABLE": "0",
         "NCCL_NVLS_ENABLE": "1",
         "NCCL_PROFILE_PRIMS_ENABLE": "1",
         "HTTP_PROXY": "star-proxy.oa.com:3128",
         "HTTPS_PROXY": "star-proxy.oa.com:3128",
         "NO_PROXY": ".woa.com,mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,tlinux-mirrorlist.tencent-cloud.com,localhost,127.0.0.1,mirrors-tlinux.tencentyun.com,.oa.com,.local,.3gqq.com,.7700.org,.ad.com,.ada_sixjoy.com,.addev.com,.app.local,.apps.local,.aurora.com,.autotest123.com,.bocaiwawa.com,.boss.com,.cdc.com,.cdn.com,.cds.com,.cf.com,.cjgc.local,.cm.com,.code.com,.datamine.com,.dvas.com,.dyndns.tv,.ecc.com,.expochart.cn,.expovideo.cn,.fms.com,.great.com,.hadoop.sec,.heme.com,.home.com,.hotbar.com,.ibg.com,.ied.com,.ieg.local,.ierd.com,.imd.com,.imoss.com,.isd.com,.isoso.com,.itil.com,.kao5.com,.kf.com,.kitty.com,.lpptp.com,.m.com,.matrix.cloud,.matrix.net,.mickey.com,.mig.local,.mqq.com,.oiweb.com,.okbuy.isddev.com,.oss.com,.otaworld.com,.paipaioa.com,.qqbrowser.local,.qqinternal.com,.qqwork.com,.rtpre.com,.sc.oa.com,.sec.com,.server.com,.service.com,.sjkxinternal.com,.sllwrnm5.cn,.sng.local,.soc.com,.t.km,.tcna.com,.teg.local,.tencentvoip.com,.tenpayoa.com,.test.air.tenpay.com,.tr.com,.tr_autotest123.com,.vpn.com,.wb.local,.webdev.com,.webdev2.com,.wizard.com,.wqq.com,.wsd.com,.sng.com,.music.lan,.mnet2.com,.tencentb2.com,.tmeoa.com,.pcg.com,www.wip3.adobe.com,www-mm.wip3.adobe.com,mirrors.tencent.com,csighub.tencentyun.com,127.0.0.1,localhost"
      }
   }' \
   -- python train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
