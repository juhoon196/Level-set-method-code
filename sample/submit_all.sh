#!/usr/bin/env bash
set -e

# 5개 노드 * 노드당 6개 작업(8코어) = 총 30개 동시 실행
NUM_SLOTS=15
LAST_JOB_IDS=()
for (( i=0; i<NUM_SLOTS; i++ )); do LAST_JOB_IDS+=(""); done

TASKS=()
for f in $(seq 5 5 50);     do TASKS+=("./gsolver2d_A $f"); done
for f in $(seq 55 5 100);    do TASKS+=("./gsolver2d_B $f"); done
for f in $(seq 110 10 200);  do TASKS+=("./gsolver2d_C $f"); done

mkdir -p logs
echo "Submitting ${#TASKS[@]} jobs into $NUM_SLOTS pipelines..."

count=0
for task_info in "${TASKS[@]}"; do
    EXE=$(echo $task_info | awk '{print $1}')
    FREQ=$(echo $task_info | awk '{print $2}')
    
    SLOT_IDX=$(( count % NUM_SLOTS ))
    PREV_ID=${LAST_JOB_IDS[$SLOT_IDX]}

    if [ -z "$PREV_ID" ]; then
        OUTPUT=$(sbatch --job-name="F${FREQ}" task.sh "$EXE" "$FREQ")
    else
        OUTPUT=$(sbatch --job-name="F${FREQ}" --dependency=afterany:$PREV_ID task.sh "$EXE" "$FREQ")
    fi

    JOB_ID=$(echo "$OUTPUT" | awk '{print $4}')
    LAST_JOB_IDS[$SLOT_IDX]=$JOB_ID
    
    echo "[Slot $((SLOT_IDX+1))] Submitted $FREQ Hz (Job ID: $JOB_ID)"
    count=$((count + 1))
done
