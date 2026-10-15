#!/usr/bin/env bash
# Submit once from repository root. Each frame has one GPU training writer at a time.
set -euo pipefail
REPO=$(pwd)
META="$REPO/results/small_workflow"
if [ -s "$META/jobs.csv" ]; then
    echo "Existing submission manifest: $META/jobs.csv; refusing duplicate submissions" >&2
    exit 1
fi
mkdir -p "$META" "$REPO/results/small_pbs_logs"
git rev-parse HEAD > "$META/source_commit.txt"
git diff --binary > "$META/uncommitted_changes.patch"
cp tools/uts_small_workflow_job.sh "$META/job_script_snapshot.sh"
printf 'frame,stage,start_fold,end_fold,job_id,depends_on\n' > "$META/jobs.csv"
submit_stage() {
    local frame=$1 stage=$2 start=$3 end=$4 dependency=$5 label=$6 queue=$7 duration=$8
    local resources='select=1:ncpus=4:mem=32gb'
    local deps=()
    if [ "$queue" = small_gpuq ]; then resources+=':ngpus=1'; fi
    if [ -n "$dependency" ]; then deps=(-W "depend=afterok:$dependency"); fi
    mkdir -p "$REPO/results/small_pbs_logs/frame$frame"
    local job
    job=$(qsub -N "$label" -q "$queue" -l "$resources" -l "walltime=$duration" \
        "${deps[@]}" -v "FRAME=$frame,STAGE=$stage,START=$start,END=$end" \
        -o "$REPO/results/small_pbs_logs/frame$frame/$label.out" \
        -e "$REPO/results/small_pbs_logs/frame$frame/$label.err" \
        tools/uts_small_workflow_job.sh) || return 1
    [[ "$job" =~ ^[0-9]+\. ]] || { echo "Invalid PBS job ID: $job" >&2; return 1; }
    printf '%s,%s,%s,%s,%s,%s\n' "$frame" "$stage" "$start" "$end" "$job" "$dependency" >> "$META/jobs.csv"
    printf '%s' "$job"
}
for frame in 160 220; do
    prep=$(submit_stage "$frame" prepare 0 0 '' "small${frame}_prep" smallq 04:00:00)
    previous=$(submit_stage "$frame" train 0 0 "$prep" "small${frame}_check" small_gpuq 06:00:00)
    for start in 1 25; do
        end=$((start+23))
        if [ "$end" -gt 48 ]; then end=48; fi
        previous=$(submit_stage "$frame" train "$start" "$end" "$previous" "small${frame}_${start}_${end}" small_gpuq 48:00:00)
    done
    post=$(submit_stage "$frame" post 0 0 "$previous" "small${frame}_post" smallq 02:00:00)
    if [ "$frame" = 160 ]; then post160=$post; else post220=$post; fi
done
combined=$(submit_stage 160 combined 0 0 "$post160:$post220" small_both_figs smallq 02:00:00)
cat "$META/jobs.csv"
echo "Final combined figures job: $combined"
