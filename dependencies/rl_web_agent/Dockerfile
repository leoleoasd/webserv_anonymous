ARG REPO=YOUR_ECR_REPO
ARG BASE_TAG=base
FROM ${REPO}:${BASE_TAG}

SHELL ["/bin/bash", "-c"]

RUN condax install uv

ENV LANG=en_US.UTF-8
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install python3-dev cmake -y && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY pyproject.toml /workdir/pyproject.toml
COPY uv.lock /workdir/uv.lock
COPY .python-version /workdir/.python-version
COPY dependencies/verl/pyproject.toml /workdir/dependencies/verl/pyproject.toml
COPY dependencies/verl/uv.lock /workdir/dependencies/verl/uv.lock
COPY dependencies/webarena/setup.py /workdir/dependencies/webarena/setup.py
COPY dependencies/webarena/setup.cfg /workdir/dependencies/webarena/setup.cfg
COPY dependencies/webarena/requirements.txt /workdir/dependencies/webarena/requirements.txt

RUN <<'SH'
source /root/miniforge3/etc/profile.d/conda.sh
conda activate base
cd /workdir
uv sync
SH
ENV CUDNN_PATH=/workdir/.venv/lib/python3.12/site-packages/nvidia/cudnn

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv <<'SH'
source /root/miniforge3/etc/profile.d/conda.sh
J="$(nproc)"
export CMAKE_BUILD_PARALLEL_LEVEL=$J CTEST_PARALLEL_LEVEL=$J NPY_NUM_BUILD_JOBS=$J \
       CARGO_BUILD_JOBS=$J MAX_JOBS=$J MAKEFLAGS="-j$J -l$J" NINJAFLAGS="-j $J"
echo "$MAX_JOBS"
cd /workdir
uv sync --extra compile --extra webarena
SH
RUN /workdir/.venv/bin/playwright install-deps
RUN /workdir/.venv/bin/playwright install


RUN cd /workdir && uv run wandb login YOUR_WANDB_TOKEN


COPY . /workdir


WORKDIR /workdir
