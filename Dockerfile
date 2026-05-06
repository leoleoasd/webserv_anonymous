ARG REPO=248189905876.dkr.ecr.us-east-1.amazonaws.com/greenland
ARG BASE_TAG=base
FROM ${REPO}:${BASE_TAG}

SHELL ["/bin/bash", "-c"]

RUN condax install uv
RUN condax install s5cmd

ENV LANG=en_US.UTF-8
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install python3-dev libnuma1 cmake -y && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY pyproject.toml /workdir/pyproject.toml
COPY uv.lock /workdir/uv.lock
COPY .python-version /workdir/.python-version
COPY thirdparty/slime/pyproject.toml /workdir/thirdparty/slime/pyproject.toml
COPY thirdparty/rl_web_agent/pyproject.toml /workdir/thirdparty/rl_web_agent/pyproject.toml
COPY thirdparty/bfcl/pyproject.toml /workdir/thirdparty/bfcl/pyproject.toml
RUN mkdir -p /root/.ssh && \
    ssh-keyscan github.com >> /root/.ssh/known_hosts

RUN --mount=type=cache,id=uv-cache2,target=/root/.cache/uv <<'SH'
source /root/miniforge3/etc/profile.d/conda.sh
conda activate base
cd /workdir
uv sync --only-group build
# WTF nvshmem
cd /workdir/.venv/lib/python3.12/site-packages/nvidia/nvshmem/lib
ln -sf libnvshmem_host.so.3 libnvshmem_host.so
ln -sf nvshmem_bootstrap_mpi.so.3 nvshmem_bootstrap_mpi.so
ln -sf nvshmem_bootstrap_pmi2.so.3 nvshmem_bootstrap_pmi2.so
ln -sf nvshmem_bootstrap_pmi.so.3 nvshmem_bootstrap_pmi.so
ln -sf nvshmem_bootstrap_pmix.so.3 nvshmem_bootstrap_pmix.so
ln -sf nvshmem_bootstrap_shmem.so.3 nvshmem_bootstrap_shmem.so
ln -sf nvshmem_bootstrap_uid.so.3 nvshmem_bootstrap_uid.so
ln -sf nvshmem_transport_ibdevx.so.3 nvshmem_transport_ibdevx.so
ln -sf nvshmem_transport_ibgda.so.3 nvshmem_transport_ibgda.so
ln -sf nvshmem_transport_ibrc.so.3 nvshmem_transport_ibrc.so
ln -sf nvshmem_transport_libfabric.so.3 nvshmem_transport_libfabric.so
ln -sf nvshmem_transport_ucx.so.3 nvshmem_transport_ucx.so
SH

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv <<'SH'
source /root/miniforge3/etc/profile.d/conda.sh
J="$(nproc)"
export CMAKE_BUILD_PARALLEL_LEVEL=$J CTEST_PARALLEL_LEVEL=$J NPY_NUM_BUILD_JOBS=$J \
       CARGO_BUILD_JOBS=$J MAX_JOBS=$J MAKEFLAGS="-j$J -l$J" NINJAFLAGS="-j $J" \
       NVCC_APPEND_FLAGS="--threads 4" APEX_PARALLEL_BUILD=8 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 \
       NVSHMEM_DIR=/workdir/.venv/lib/python3.12/site-packages/nvidia/nvshmem
echo "$MAX_JOBS"
cd /workdir
uv sync
SH



RUN cd /workdir && uv run wandb login 5f979adf061882b2252d23ea8472a6fb3c492565
ENV HF_TOKEN=hf_oLmBzlEDHPbWXsMkLGoNhWbo
RUN curl -fsSL https://opencode.ai/install | bash
RUN cat <<EOF > /root/.opencode/config.json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "amazon-bedrock": {
      "options": {
        "region": "ap-south-1",
        "profile": "xianft"
      }
    }
  }
}
EOF
RUN <<'EOF' cat >> /root/.aws/config
[profile xianft]
role_arn = arn:aws:iam::801953956576:role/crossaccountbedrock
source_profile=default
max_attempts=100
retry_mode=adaptive
EOF

RUN <<'EOF' cat >> /root/.bashrc
source /workdir/.venv/bin/activate
EOF

COPY . /workdir


WORKDIR /workdir
