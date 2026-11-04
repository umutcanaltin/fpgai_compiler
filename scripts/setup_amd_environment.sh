#!/usr/bin/env bash
# Source in an interactive Linux bash shell; do not execute directly.
# Example:
#   export FPGAI_AMD_ROOT=/tools/Xilinx
#   export FPGAI_AMD_VERSION=2023.2
#   source scripts/setup_amd_environment.sh
#
# This only loads vendor-provided shell settings and selects executable
# locations. It never runs synthesis, modifies an installation, or
# automatically downloads tools or proprietary platforms.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  printf '%s\n' 'ERROR: Source this file: source scripts/setup_amd_environment.sh' >&2
  exit 2
fi

_fpgai_amd_error() {
  printf 'FPGAI AMD environment: %s\n' "$*" >&2
  return 1
}

if [[ -z "${FPGAI_AMD_ROOT:-}" ]]; then
  _fpgai_amd_error 'Set FPGAI_AMD_ROOT to the installation parent (e.g. /tools/Xilinx).'
  return 2
fi
if [[ ! -d "$FPGAI_AMD_ROOT" ]]; then
  _fpgai_amd_error "Installation parent not found: $FPGAI_AMD_ROOT"
  return 2
fi

export FPGAI_AMD_VERSION="${FPGAI_AMD_VERSION:-2023.2}"
_fpgai_vitis_settings="$FPGAI_AMD_ROOT/Vitis/$FPGAI_AMD_VERSION/settings64.sh"
_fpgai_vivado_settings="$FPGAI_AMD_ROOT/Vivado/$FPGAI_AMD_VERSION/settings64.sh"

# Official AMD Vitis settings script normally sets XILINX_VITIS,
# XILINX_VIVADO and tool locations for the session.
if [[ ! -f "$_fpgai_vitis_settings" ]]; then
  _fpgai_amd_error "Vitis settings file missing: $_fpgai_vitis_settings"
  return 2
fi
# shellcheck disable=SC1090
source "$_fpgai_vitis_settings" || {
  _fpgai_amd_error 'AMD Vitis settings script failed'
  return 2
}

# In split or partial installations the Vivado path may be separate.
# Require a real Vivado executable only when requested by the user.
if [[ -f "$_fpgai_vivado_settings" ]]; then
  # shellcheck disable=SC1090
  source "$_fpgai_vivado_settings" || {
    _fpgai_amd_error 'AMD Vivado settings script failed'
    return 2
  }
else
  printf 'FPGAI AMD environment: No Vivado settings at %s (HLS-only may work).\n' \
    "$_fpgai_vivado_settings" >&2
fi

# AMD uses PLATFORM_REPO_PATHS for external .xpfm platforms. Keep the
# variable untouched; this repository does not supply a board platform.
# Optional XRT setup is separate. Sourcing XRT is relevant for datacenter
# acceleration and some runtime workflows, not source-only HLS synthesis.
if [[ "${FPGAI_SOURCE_XRT:-0}" == "1" ]]; then
  _fpgai_xrt_settings="${FPGAI_XRT_SETUP:-/opt/xilinx/xrt/setup.sh}"
  if [[ ! -f "$_fpgai_xrt_settings" ]]; then
    _fpgai_amd_error "Explicitly requested XRT file missing: $_fpgai_xrt_settings"
    return 2
  fi
  # shellcheck disable=SC1090
  source "$_fpgai_xrt_settings" || return 2
fi

# Respect explicitly supplied executable locations; otherwise rely on PATH.
export FPGAI_VITIS_HLS_EXECUTABLE="${FPGAI_VITIS_HLS_EXECUTABLE:-vitis_hls}"
export FPGAI_VIVADO_EXECUTABLE="${FPGAI_VIVADO_EXECUTABLE:-vivado}"

printf '%s\n' "AMD settings loaded for $FPGAI_AMD_VERSION (shell only)."
printf '  vitis_hls: %s\n' "$(command -v vitis_hls || printf 'NOT FOUND')"
printf '  vivado:    %s\n' "$(command -v vivado || printf 'NOT FOUND')"
printf '%s\n' 'No compilation or FPGA execution has been performed.'

unset _fpgai_vitis_settings _fpgai_vivado_settings _fpgai_xrt_settings
unset -f _fpgai_amd_error
