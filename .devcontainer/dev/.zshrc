# Load nvm
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion

# Path to your oh-my-zsh installation.
export ZSH="$HOME/.oh-my-zsh"

# Set name of the theme to load
ZSH_THEME="robbyrussell"

# Which plugins would you like to load?
plugins=(
  git
  node
  npm
  yarn
  golang
  rust
  pip
  python
  docker
  docker-compose
  kubectl
  vscode
  zsh-autosuggestions
  zsh-syntax-highlighting
  zsh-completions
)

# Load Oh My Zsh
source $ZSH/oh-my-zsh.sh

alias ll='ls -la'
alias ct='cargo test -- --nocapture --color=always'
alias cb='cargo build'
alias cbr='cargo build --release'
alias cf='cargo fmt'
alias cfc='cargo fmt --check'
alias cw='cargo watch'
alias cwtest='cargo watch -x test'
alias cwcheck='cargo watch -x check'
alias cwclippy='cargo watch -x clippy'
alias cwfmt='cargo watch -x fmt'
alias cwfmtcheck='cargo watch -x fmt -- --check'
alias cwdoc='cargo watch -x doc'

alias g='git'