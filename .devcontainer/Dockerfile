FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

# 基本パッケージのインストール（cmakeを除外）
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    pkg-config \
    libssl-dev \
    ripgrep \
    graphviz \
    unzip \
    ninja-build \
    ffmpeg \
    python3-tk

# Node.js 18.x のインストール（Claude Code に必要）
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y nodejs 

# npmのグローバルディレクトリを設定（権限問題を回避）
RUN mkdir -p /usr/local/lib/npm-global \
    && npm config set prefix /usr/local/lib/npm-global
ENV PATH="/usr/local/lib/npm-global/bin:${PATH}"

# Claude Code のインストール
RUN npm install -g @anthropic-ai/claude-code

# 最新のCMakeをソースからインストール
RUN wget https://cmake.org/files/v3.30/cmake-3.30.5.tar.gz \
    && tar -xzf cmake-3.30.5.tar.gz \
    && cd cmake-3.30.5 \
    && ./bootstrap --parallel=$(nproc) \
    && make -j$(nproc) \
    && make install \
    && cd .. \
    && rm -rf cmake-3.30.5 cmake-3.30.5.tar.gz

# Rustのインストール
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Rust用のCUDAツールチェーンを追加
RUN rustup target add x86_64-unknown-linux-gnu

# CUDAの環境変数設定
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# LLVM公式リポジトリを追加してclang-19をインストール
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - \
    && echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-19 main" >> /etc/apt/sources.list \
    && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    llvm-19 \
    llvm-19-dev \
    libclang-19-dev \
    clang-19 \
    clang-tidy-19 \
    clang-format-19

# clang-19をデフォルトのclangとして設定
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-19 100 \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-19 100 \
    && update-alternatives --install /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-19 100 \
    && update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-19 100

# OpenGL and graphics libraries
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libglew-dev \
    libglfw3-dev \
    libglm-dev \
    mesa-utils \
    xorg-dev \
    libx11-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    vim \
    libgtest-dev \
    && cd /usr/src/gtest \
    && cmake CMakeLists.txt \
    && make -j$(nproc) \
    && cp lib/*.a /usr/lib \
    && rm -rf /var/lib/apt/lists/*

# 開発用ツール
RUN cargo install cargo-watch cargo-edit

# Taskfileのインストール
RUN sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b /usr/local/bin

# uv のインストール
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc

# 作業ディレクトリの設定
WORKDIR /workspace

# Claude Code の動作確認（オプション - 認証が必要なのでコメントアウト）
# RUN claude --version
