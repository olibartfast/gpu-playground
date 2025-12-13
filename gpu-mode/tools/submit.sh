#!/bin/bash
# Submit a kernel to a GPU MODE leaderboard
# Usage: ./submit.sh [OPTIONS] <path_to_kernel.py>
#
# Options:
#   -g, --gpu         GPU model (e.g., B200, A100, H100)
#   -l, --leaderboard Leaderboard name (e.g., nvfp4_gemm)
#   -m, --mode        Submission mode: leaderboard or dev (default: leaderboard)
#   -w, --workspace   Workspace directory name (default: gpu-mode)
#   -h, --help        Show this help message

set -e

# Default configuration (can be overridden by environment variables or command line)
WORKSPACE="${WORKSPACE:-gpu-mode}"
POPCORN_CLI="${POPCORN_CLI:-$HOME/$WORKSPACE/popcorn-cli/target/release/popcorn-cli}"
GPU="${GPU:-}"
LEADERBOARD="${LEADERBOARD:-}"
MODE="${MODE:-leaderboard}"

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] <kernel.py>"
    echo ""
    echo "Submit a kernel to a GPU MODE leaderboard"
    echo ""
    echo "Options:"
    echo "  -g, --gpu <model>         GPU model (e.g., B200, A100, H100)"
    echo "  -l, --leaderboard <name>  Leaderboard name (e.g., nvfp4_gemm)"
    echo "  -m, --mode <mode>         Submission mode: leaderboard or dev (default: leaderboard)"
    echo "  -w, --workspace <dir>     Workspace directory name (default: gpu-mode)"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -g B200 -l nvfp4_gemm submissions/my_kernel.py"
    echo "  $0 --gpu A100 --leaderboard grayscale --mode dev test_kernel.py"
    echo ""
    echo "Environment variables:"
    echo "  WORKSPACE         - Workspace directory name"
    echo "  POPCORN_API_URL   - API URL (get from Discord: /get-api-url)"
    echo "  GPU               - Default GPU model"
    echo "  LEADERBOARD       - Default leaderboard name"
    echo "  MODE              - Default submission mode"
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--gpu)
            GPU="$2"
            shift 2
            ;;
        -l|--leaderboard)
            LEADERBOARD="$2"
            shift 2
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -w|--workspace)
            WORKSPACE="$2"
            POPCORN_CLI="$HOME/$WORKSPACE/popcorn-cli/target/release/popcorn-cli"
            shift 2
            ;;
        -h|--help)
            show_usage
            ;;
        -*)
            echo "Error: Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
        *)
            KERNEL_FILE="$1"
            shift
            ;;
    esac
done

# Check if popcorn-cli exists
if [ ! -f "$POPCORN_CLI" ]; then
    echo "Error: Popcorn CLI not found at $POPCORN_CLI"
    echo ""
    echo "Please build it first:"
    echo "  cd ~/$WORKSPACE/popcorn-cli && ./build.sh"
    echo ""
    echo "Or set POPCORN_CLI environment variable to the correct path."
    exit 1
fi

# Check if kernel file provided
if [ -z "$KERNEL_FILE" ]; then
    echo "Error: No kernel file specified"
    echo ""
    show_usage
fi

# Check if kernel file exists
if [ ! -f "$KERNEL_FILE" ]; then
    echo "Error: Kernel file not found: $KERNEL_FILE"
    exit 1
fi

# Check if GPU and LEADERBOARD are set
if [ -z "$GPU" ] || [ -z "$LEADERBOARD" ]; then
    echo "Error: GPU model and leaderboard name are required"
    echo ""
    echo "Specify them via command line:"
    echo "  $0 -g <GPU_MODEL> -l <LEADERBOARD_NAME> $KERNEL_FILE"
    echo ""
    echo "Or set environment variables:"
    echo "  export GPU=<GPU_MODEL>"
    echo "  export LEADERBOARD=<LEADERBOARD_NAME>"
    echo ""
    echo "To browse available leaderboards:"
    echo "  $POPCORN_CLI submit"
    exit 1
fi

# Check API URL is set
if [ -z "$POPCORN_API_URL" ]; then
    echo "Warning: POPCORN_API_URL not set"
    echo ""
    echo "Get your API URL from Discord:"
    echo "  1. Join https://discord.gg/gpumode"
    echo "  2. In Discord, type: /get-api-url"
    echo "  3. Export the URL: export POPCORN_API_URL=\"<url>\""
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check authentication
if [ ! -f "$HOME/.popcorn.yaml" ]; then
    echo "Error: Not authenticated with Popcorn CLI"
    echo ""
    echo "Please authenticate first:"
    echo "  $POPCORN_CLI register discord"
    echo ""
    echo "Make sure POPCORN_API_URL is set first."
    exit 1
fi

# Display submission info
echo "========================================="
echo "Submitting Kernel to GPU MODE"
echo "========================================="
echo "GPU:         $GPU"
echo "Leaderboard: $LEADERBOARD"
echo "Mode:        $MODE"
echo "Kernel:      $KERNEL_FILE"
echo "CLI:         $POPCORN_CLI"
echo "========================================="
echo ""

# Submit
"$POPCORN_CLI" submit \
    --gpu "$GPU" \
    --leaderboard "$LEADERBOARD" \
    --mode "$MODE" \
    "$KERNEL_FILE"

# Show results
echo ""
echo "========================================="
echo "Submission complete!"
echo "========================================="
echo ""
echo "Check your position:"
echo "  1. Discord: /leaderboard $LEADERBOARD"
echo "  2. Web: https://www.gpumode.com"
echo ""
echo "To browse all leaderboards:"
echo "  $POPCORN_CLI submit"
echo ""
