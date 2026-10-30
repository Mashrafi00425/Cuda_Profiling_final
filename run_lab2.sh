#!/usr/bin/env bash
# Run on the Jetson in the same folder as the three .cu files.
# Produces outputs.txt from REAL hardware results; never fabricates timings.
set -uo pipefail
cd "$(dirname "$0")" || exit 1
export PATH="/usr/local/cuda/bin:$PATH"
OUT="$PWD/outputs.txt"
TMP="$(mktemp -d)" || exit 1
trap 'rm -rf "$TMP"' EXIT

heading() { printf '\n\n===== %s =====\n' "$1" | tee -a "$OUT"; }
append_cmd() {
    printf '$ ' >> "$OUT"
    printf '%q ' "$@" >> "$OUT"
    printf '\n' >> "$OUT"
    "$@" >> "$OUT" 2>&1
    local rc=$?
    if (( rc != 0 )); then
        printf '[Command exit status: %d]\n' "$rc" >> "$OUT"
    fi
    return "$rc"
}

printf 'CUDA Lab 2 - Profiling Outputs\nCaptured on: %s\nMachine: %s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$(hostname)" > "$OUT"

heading 'CUDA compiler'
if ! command -v nvcc >/dev/null 2>&1; then
    echo 'ERROR: nvcc was not found; run this script on your CUDA-equipped Jetson.' | tee -a "$OUT"
    exit 1
fi
append_cmd nvcc --version || true

heading 'Compile device_query.cu'
if ! append_cmd nvcc -std=c++11 -O2 -lineinfo device_query.cu -o device_query; then
    echo 'Compilation failed. Correct the CUDA installation before continuing.' | tee -a "$OUT"
    exit 1
fi

heading 'Jetson device query'
if ! append_cmd ./device_query; then
    echo 'The Jetson GPU could not be queried.' | tee -a "$OUT"
    exit 1
fi
# Use the device's own architecture (e.g. sm_53 for Jetson Nano).
ARCH="$(./device_query 2>/dev/null | sed -n 's/^Compute capability: \([0-9][0-9]*\)\.\([0-9][0-9]*\)$/sm_\1\2/p' | head -n 1)"
ARCH_FLAG=()
if [[ -n "$ARCH" ]]; then ARCH_FLAG=("-arch=$ARCH"); fi
printf 'Detected GPU architecture: %s\n' "${ARCH:-toolkit default}" | tee -a "$OUT"

heading 'Compiler ptxas output: vector_add.cu'
if ! append_cmd nvcc -std=c++11 -O2 -lineinfo "${ARCH_FLAG[@]}" -Xptxas=-v vector_add.cu -o vector_add; then
    echo 'vector_add compilation failed.' | tee -a "$OUT"
    exit 1
fi

heading 'Compiler ptxas output: memory_stride.cu'
if ! append_cmd nvcc -std=c++11 -O2 -lineinfo "${ARCH_FLAG[@]}" -Xptxas=-v memory_stride.cu -o memory_stride; then
    echo 'memory_stride compilation failed.' | tee -a "$OUT"
    exit 1
fi

heading 'Vector add: measured kernel time and c[100]'
append_cmd ./vector_add || true

heading 'Memory stride: both timings, with one warm-up each'
append_cmd ./memory_stride || true

heading 'tegrastats while GPU kernel is running'
if command -v tegrastats >/dev/null 2>&1; then
    tegrastats --interval 200 > "$TMP/tegra.log" 2>&1 &
    TEGRA_PID=$!
    # Repeated kernels keep the GPU active long enough to capture a reading.
    ./memory_stride --stride 32 --stress-seconds 5 > "$TMP/stress.log" 2>&1
    STRESS_RC=$?
    kill "$TEGRA_PID" 2>/dev/null || true
    wait "$TEGRA_PID" 2>/dev/null || true
    grep -m 1 -E 'RAM|GR3D|GPU' "$TMP/tegra.log" >> "$OUT" || \
       echo '[No tegrastats sample was captured]' >> "$OUT"
    if (( STRESS_RC != 0 )); then
        printf '[Stress kernel failed: exit %d]\n' "$STRESS_RC" >> "$OUT"
        cat "$TMP/stress.log" >> "$OUT"
    fi
else
    echo '[tegrastats command is not available]' >> "$OUT"
fi

heading 'Nsight Systems: CUDA API Summary and CUDA GPU Kernel Summary'
NSYS_OK=0
if command -v nsys >/dev/null 2>&1; then
    printf '$ nsys profile --trace=cuda --sample=none -o <temporary-report> ./vector_add\n' >> "$OUT"
    nsys profile --trace=cuda --sample=none --force-overwrite=true \
        -o "$TMP/vector_add_profile" ./vector_add > "$TMP/nsys_profile.log" 2>&1
    NSYS_RC=$?
    if (( NSYS_RC != 0 )); then
        printf '[nsys profile failed: exit %d]\n' "$NSYS_RC" >> "$OUT"
        cat "$TMP/nsys_profile.log" >> "$OUT"
    else
        REP=''
        if [[ -f "$TMP/vector_add_profile.nsys-rep" ]]; then
            REP="$TMP/vector_add_profile.nsys-rep"
        elif [[ -f "$TMP/vector_add_profile.qdrep" ]]; then
            REP="$TMP/vector_add_profile.qdrep"
        fi
        if [[ -z "$REP" ]]; then
            echo '[nsys did not produce a usable report file]' >> "$OUT"
            cat "$TMP/nsys_profile.log" >> "$OUT"
        else
            printf '$ nsys stats --report cuda_api_sum --report cuda_gpu_kern_sum <report>\n' >> "$OUT"
            nsys stats --report cuda_api_sum --report cuda_gpu_kern_sum \
                "$REP" >> "$OUT" 2>&1 && NSYS_OK=1 || \
                echo '[Could not generate both nsys tables; see errors above]' >> "$OUT"
        fi
    fi
else
    echo '[nsys not installed / not on PATH]' >> "$OUT"
fi
if (( NSYS_OK == 0 )) && command -v nvprof >/dev/null 2>&1; then
    echo 'Fallback: nvprof CUDA API and GPU summaries (NOT nsys tables).' >> "$OUT"
    append_cmd nvprof --print-api-summary --print-gpu-summary ./vector_add || true
fi

heading 'Nsight Compute: stride 1 and stride 32'
for STRIDE in 1 32; do
    printf '\n--- Stride %d ---\n' "$STRIDE" >> "$OUT"
    NEED_FALLBACK=1
    if command -v ncu >/dev/null 2>&1; then
        printf '$ ncu --set full --launch-skip 1 --launch-count 1 ./memory_stride --stride %d --profile\n' "$STRIDE" >> "$OUT"
        ncu --set full --launch-skip 1 --launch-count 1 \
            ./memory_stride --stride "$STRIDE" --profile > "$TMP/ncu_$STRIDE.log" 2>&1
        NCU_RC=$?
        cat "$TMP/ncu_$STRIDE.log" >> "$OUT"
        if (( NCU_RC == 0 )); then
            NEED_FALLBACK=0
        else
            printf '[ncu failed for stride %d: exit %d]\n' "$STRIDE" "$NCU_RC" >> "$OUT"
        fi
    else
        echo '[ncu is not installed / not on PATH]' >> "$OUT"
    fi
    if (( NEED_FALLBACK == 1 )); then
        if command -v nvprof >/dev/null 2>&1; then
            echo 'Fallback: nvprof (older Jetsons / unsupported ncu GPU).' >> "$OUT"
            append_cmd nvprof --metrics gld_efficiency,gld_transactions_per_request \
                ./memory_stride --stride "$STRIDE" --profile || true
        else
            echo '[No nvprof fallback is installed either.]' >> "$OUT"
        fi
    fi
done

heading 'Required two-sentence explanation'
printf '%s\n' \
    "(1) Vector addition itself is a short GPU operation; most of the end-to-end execution time is generally spent in CUDA initialization, device allocations, and moving input/output data between host and GPU rather than adding numbers." \
    "(2) Stride 32 is slower because neighboring threads load elements far apart in memory, so the GPU needs less-efficient, noncoalesced memory transactions despite executing the same kernel instructions." >> "$OUT"

printf '\nDone. Review %s for actual timings and any profiling-tool errors.\n' "$OUT"
