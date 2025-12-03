#!/bin/bash
# Quick setup script for perf on Linux
# Run this once on your Linux host to ensure perf is installed and configured

set -e

echo "=== Setting up perf for cache profiling ==="
echo ""

# Detect Linux distribution
if [ -f /etc/debian_version ]; then
    echo "Detected Debian/Ubuntu"
    if ! command -v perf &> /dev/null; then
        echo "Installing linux-perf..."
        sudo apt-get update
        sudo apt-get install -y linux-perf
    else
        echo "perf is already installed"
    fi
elif [ -f /etc/redhat-release ]; then
    echo "Detected RHEL/CentOS/Fedora"
    if ! command -v perf &> /dev/null; then
        echo "Installing perf..."
        sudo yum install -y perf || sudo dnf install -y perf
    else
        echo "perf is already installed"
    fi
elif [ -f /etc/arch-release ]; then
    echo "Detected Arch Linux"
    if ! command -v perf &> /dev/null; then
        echo "Installing perf..."
        sudo pacman -S --noconfirm perf
    else
        echo "perf is already installed"
    fi
else
    echo "Unknown Linux distribution. Please install perf manually."
    exit 1
fi

echo ""
echo "Configuring perf permissions..."

# Set perf_event_paranoid to allow user-level profiling
# -1 = allow all users
#  0 = allow user + kernel profiling
#  1 = kernel only
#  2 = kernel only (no raw tracepoints)
sudo sysctl -w kernel.perf_event_paranoid=-1

# Make it persistent across reboots
if ! grep -q "kernel.perf_event_paranoid" /etc/sysctl.conf 2>/dev/null; then
    echo "kernel.perf_event_paranoid=-1" | sudo tee -a /etc/sysctl.conf
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "You can now run:"
echo "  ./scripts/perf_cache_profile.sh lookup_kernel_bench single_table_lookup 10"
echo "  ./scripts/perf_record_cache.sh lookup_kernel_bench single_table_lookup 10"
echo ""
echo "Test perf with:"
echo "  perf --version"


