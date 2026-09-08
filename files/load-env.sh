# Shared environment selection. Profiles deliberately override stale .env values.
source "${QWEN_ENV_FILE:-$SCRIPT_DIR/.env}" || return 1
if [[ -n "${QWEN_PROFILE:-}" ]]; then
    case "$QWEN_PROFILE" in nvfp4-native|nvfp4-long) ;; *) echo "Unknown QWEN_PROFILE" >&2; return 1 ;; esac
    if [[ -n "${OVERRIDE_MODEL_ID:-}${OVERRIDE_SERVED_MODEL_NAME:-}${OVERRIDE_MAX_MODEL_LEN:-}${OVERRIDE_YARN_ENABLE:-}" ]]; then
        echo "A named profile cannot be combined with FP8-wrapper overrides" >&2
        return 1
    fi
    source "$SCRIPT_DIR/deploy/profiles/nvfp4-common.env" || return 1
    source "$SCRIPT_DIR/deploy/profiles/$QWEN_PROFILE.env" || return 1
    case "${QWEN_MTP_TOKENS:-3}" in 0|1|2|3) MTP_NUM_SPECULATIVE_TOKENS="${QWEN_MTP_TOKENS:-3}" ;; *) echo "MTP comparison supports 0, 1, 2, or 3" >&2; return 1 ;; esac
fi
