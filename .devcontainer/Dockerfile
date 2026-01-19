FROM auser/hologram-devcontainer:latest as dev

# Set environment variables for the container
ENV RUST_BACKTRACE=1

# Install mold linker for faster linking (up to 5-10x faster than default ld),
# sccache for caching compiled crates across clean builds, and just command runner
USER root
ARG MOLD_VERSION=2.34.1
ARG SCCACHE_VERSION=0.12.0
ARG JUST_VERSION=1.38.0
RUN ARCH=$(uname -m) && \
    # Install mold
    if [ "$ARCH" = "x86_64" ]; then MOLD_ARCH="x86_64"; else MOLD_ARCH="aarch64"; fi && \
    curl -sL "https://github.com/rui314/mold/releases/download/v${MOLD_VERSION}/mold-${MOLD_VERSION}-${MOLD_ARCH}-linux.tar.gz" | tar xz -C /tmp && \
    mv /tmp/mold-${MOLD_VERSION}-${MOLD_ARCH}-linux/bin/mold /usr/local/bin/ && \
    mv /tmp/mold-${MOLD_VERSION}-${MOLD_ARCH}-linux/lib/mold /usr/local/lib/ && \
    rm -rf /tmp/mold-${MOLD_VERSION}-${MOLD_ARCH}-linux && \
    ln -sf /usr/local/bin/mold /usr/local/lib/mold/ld && \
    # Install sccache
    curl -sL "https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/sccache-v${SCCACHE_VERSION}-${ARCH}-unknown-linux-musl.tar.gz" | tar xz -C /tmp && \
    mv /tmp/sccache-v${SCCACHE_VERSION}-${ARCH}-unknown-linux-musl/sccache /usr/local/bin/ && \
    rm -rf /tmp/sccache-v${SCCACHE_VERSION}-${ARCH}-unknown-linux-musl && \
    # Install just
    curl -sL "https://github.com/casey/just/releases/download/${JUST_VERSION}/just-${JUST_VERSION}-${ARCH}-unknown-linux-musl.tar.gz" | tar xz -C /tmp && \
    mv /tmp/just /usr/local/bin/ && \
    rm -f /tmp/just.1 /tmp/CHANGELOG.md /tmp/LICENSE /tmp/README.md && \
    # Install cargo-nextest (faster test runner) - compile from source for correct architecture
    su - vscode -c "cargo install cargo-nextest --locked"

# Install GitHub CLI (gh) - https://github.com/cli/cli/blob/trunk/docs/install_linux.md#debian
RUN mkdir -p -m 755 /etc/apt/keyrings && \
    wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null && \
    chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null && \
    apt-get update && \
    apt-get install -y gh && \
    rm -rf /var/lib/apt/lists/*

# Set the default user (if not already set in base image)
USER vscode

# Install Claude Code CLI
RUN curl -fsSL https://claude.ai/install.sh | bash

# Set the working directory
WORKDIR /workspace
