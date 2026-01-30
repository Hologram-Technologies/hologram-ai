# Use official Microsoft Rust devcontainer as base
FROM mcr.microsoft.com/devcontainers/rust:1-bookworm

# Set environment variables
ENV RUST_BACKTRACE=1 \
    CARGO_TERM_COLOR=always \
    RUST_LOG=info

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    pkg-config \
    # Protobuf compiler (for prost-build)
    protobuf-compiler \
    # Tools
    curl \
    wget \
    git \
    zsh \
    && rm -rf /var/lib/apt/lists/*

# Install mold linker (faster linking - 5-10x faster than default ld)
ARG MOLD_VERSION=2.34.1
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then MOLD_ARCH="x86_64"; else MOLD_ARCH="aarch64"; fi && \
    curl -sL "https://github.com/rui314/mold/releases/download/v${MOLD_VERSION}/mold-${MOLD_VERSION}-${MOLD_ARCH}-linux.tar.gz" | tar xz -C /tmp && \
    mv /tmp/mold-${MOLD_VERSION}-${MOLD_ARCH}-linux/bin/mold /usr/local/bin/ && \
    mv /tmp/mold-${MOLD_VERSION}-${MOLD_ARCH}-linux/lib/mold /usr/local/lib/ && \
    rm -rf /tmp/mold-${MOLD_VERSION}-${MOLD_ARCH}-linux && \
    ln -sf /usr/local/bin/mold /usr/local/lib/mold/ld

# Install GitHub CLI
RUN mkdir -p -m 755 /etc/apt/keyrings && \
    wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null && \
    chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null && \
    apt-get update && \
    apt-get install -y gh && \
    rm -rf /var/lib/apt/lists/*

# Switch to vscode user for cargo installations
USER vscode

# Install Rust development tools
ARG JUST_VERSION=1.38.0
RUN ARCH=$(uname -m) && \
    # Install just (command runner)
    curl -sL "https://github.com/casey/just/releases/download/${JUST_VERSION}/just-${JUST_VERSION}-${ARCH}-unknown-linux-musl.tar.gz" | tar xz -C /tmp && \
    mkdir -p ~/.cargo/bin && \
    mv /tmp/just ~/.cargo/bin/ && \
    rm -f /tmp/just.1 /tmp/CHANGELOG.md /tmp/LICENSE /tmp/README.md && \
    # Install cargo-nextest (faster test runner)
    cargo install cargo-nextest --locked

# Install Claude Code CLI
RUN curl -fsSL https://claude.ai/install.sh | bash

# Set working directory
WORKDIR /workspace
