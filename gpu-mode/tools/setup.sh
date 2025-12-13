#!/bin/bash
# Setup script for GPU MODE kernel development environment
# Run this on your local PC to set up the environment

set -e

# Allow user to override workspace directory
WORKSPACE="${WORKSPACE:-gpu-mode}"
WORKSPACE_DIR="$HOME/$WORKSPACE"

echo "========================================="
echo "GPU MODE Kernel Development Setup"
echo "========================================="
echo ""
echo "Workspace directory: $WORKSPACE_DIR"
echo ""

# Create workspace
echo "Creating workspace at: $WORKSPACE_DIR"
mkdir -p "$WORKSPACE_DIR"
cd "$WORKSPACE_DIR"

# Check for Rust
echo ""
echo "Checking for Rust installation..."
if ! command -v cargo &> /dev/null; then
    echo "Rust not found. Installing..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    source "$HOME/.cargo/env"
else
    echo "✓ Rust is installed ($(rustc --version))"
fi

# Clone Popcorn CLI
echo ""
echo "Setting up Popcorn CLI..."
if [ ! -d "popcorn-cli" ]; then
    echo "Cloning popcorn-cli repository..."
    git clone https://github.com/gpu-mode/popcorn-cli.git
else
    echo "✓ popcorn-cli already cloned"
fi

cd popcorn-cli
echo "Building popcorn-cli..."
./build.sh

if [ -f "target/release/popcorn-cli" ]; then
    echo "✓ Popcorn CLI built successfully"
else
    echo "✗ Build failed"
    exit 1
fi

# Clone reference kernels
cd "$WORKSPACE_DIR"
echo ""
echo "Cloning reference kernels..."
if [ ! -d "reference-kernels" ]; then
    git clone https://github.com/gpu-mode/reference-kernels.git
else
    echo "✓ reference-kernels already cloned"
fi

# Create submissions directory
echo ""
echo "Creating submissions directory..."
mkdir -p "$WORKSPACE_DIR/submissions"

# Create a simple .envrc file for convenience
echo ""
echo "Creating environment configuration..."
cat > "$WORKSPACE_DIR/.envrc" << EOF
# GPU MODE Environment Configuration
# Source this file: source ~/.envrc or add to your ~/.bashrc

# Workspace directory
export WORKSPACE="$WORKSPACE"

# Add popcorn-cli to PATH (optional)
# export PATH="\$HOME/$WORKSPACE/popcorn-cli/target/release:\$PATH"

# Set your API URL (get from Discord: /get-api-url)
# export POPCORN_API_URL="<url_from_discord>"
EOF

# Setup complete
echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Join GPU MODE Discord:"
echo "   https://discord.gg/gpumode"
echo ""
echo "2. Get API URL from Discord:"
echo "   In Discord, type: /get-api-url"
echo ""
echo "3. Set API URL in your environment:"
echo "   export POPCORN_API_URL=\"<url_from_discord>\""
echo "   (or add to $WORKSPACE_DIR/.envrc)"
echo ""
echo "4. Authenticate with Popcorn CLI:"
echo "   cd $WORKSPACE_DIR/popcorn-cli"
echo "   ./target/release/popcorn-cli register discord"
echo ""
echo "5. Browse available problems:"
echo "   cd $WORKSPACE_DIR/reference-kernels/problems"
echo "   ls"
echo ""
echo "6. Start coding!"
echo "   cd $WORKSPACE_DIR/submissions"
echo ""
echo "For detailed instructions, see SETUP_GUIDE.md"
echo ""
echo "Optional: Add to your ~/.bashrc for persistent environment:"
echo "  echo 'export WORKSPACE=$WORKSPACE' >> ~/.bashrc"
echo "  echo 'source ~/$WORKSPACE/.envrc' >> ~/.bashrc"
echo ""
