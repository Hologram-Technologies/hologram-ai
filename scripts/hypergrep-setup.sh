#!/bin/bash
# Set up hypergrep for AI agents in your project.
#
# Usage: hypergrep-setup.sh [project-dir]
#
# Detects which AI tools you use and creates the right config files.

set -e

DIR="${1:-.}"

if [ ! -d "$DIR" ]; then
    echo "Error: $DIR is not a directory"
    exit 1
fi

BOLD="\033[1m"
GREEN="\033[32m"
RESET="\033[0m"

info() { echo -e "${GREEN}${BOLD}$1${RESET}"; }

INSTRUCTIONS='This project has `hypergrep` installed globally. Use it instead of grep/ripgrep for ALL code search. It returns structural results (function bodies, signatures, call graphs) instead of raw lines, using 87% fewer tokens.

Finding code (replaces grep/rg):
- `hypergrep --layer 1 --budget 800 --json "pattern" .` -- signatures + call graph, capped at 800 tokens
- `hypergrep -s "pattern" .` -- full function bodies (when you need to edit)
- `hypergrep -l "pattern" .` -- file paths only

Understanding code:
- `hypergrep --model "" .` -- codebase overview (run ONCE at session start)
- `hypergrep --callers "function_name" .` -- who calls this?
- `hypergrep --callees "function_name" .` -- what does this call?

Before making changes:
- `hypergrep --impact "function_name" .` -- what breaks if I change this? ALWAYS run before editing.

Quick checks:
- `hypergrep --exists "redis" .` -- does this project use X? (NO = guaranteed, YES = likely)

Rules:
1. Run `hypergrep --model "" .` at session start to understand the codebase
2. Use `hypergrep --layer 1 --budget 800 --json` instead of grep/rg for finding code
3. Run `hypergrep --impact` before changing any function
4. Use `hypergrep -s` instead of reading whole files when you need function bodies
5. Use `hypergrep --exists` to check for technologies instead of searching'

COUNT=0

# Claude Code
if [ -f "$DIR/CLAUDE.md" ]; then
    echo "$INSTRUCTIONS" >> "$DIR/CLAUDE.md"
    info "Updated: CLAUDE.md (Claude Code)"
    COUNT=$((COUNT+1))
else
    echo "# Code Search" > "$DIR/CLAUDE.md"
    echo "" >> "$DIR/CLAUDE.md"
    echo "$INSTRUCTIONS" >> "$DIR/CLAUDE.md"
    info "Created: CLAUDE.md (Claude Code)"
    COUNT=$((COUNT+1))
fi

# Cursor
if [ -f "$DIR/.cursorrules" ]; then
    echo "" >> "$DIR/.cursorrules"
    echo "$INSTRUCTIONS" >> "$DIR/.cursorrules"
    info "Updated: .cursorrules (Cursor)"
else
    echo "$INSTRUCTIONS" > "$DIR/.cursorrules"
    info "Created: .cursorrules (Cursor)"
fi
COUNT=$((COUNT+1))

# GitHub Copilot
mkdir -p "$DIR/.github"
if [ -f "$DIR/.github/copilot-instructions.md" ]; then
    echo "" >> "$DIR/.github/copilot-instructions.md"
    echo "$INSTRUCTIONS" >> "$DIR/.github/copilot-instructions.md"
    info "Updated: .github/copilot-instructions.md (Copilot)"
else
    echo "$INSTRUCTIONS" > "$DIR/.github/copilot-instructions.md"
    info "Created: .github/copilot-instructions.md (Copilot)"
fi
COUNT=$((COUNT+1))

# Windsurf
echo "$INSTRUCTIONS" > "$DIR/.windsurfrules"
info "Created: .windsurfrules (Windsurf)"
COUNT=$((COUNT+1))

echo ""
info "Done! Configured $COUNT AI tools to use hypergrep in $DIR"
echo ""
echo "Your AI agents will now use hypergrep for code search automatically."
