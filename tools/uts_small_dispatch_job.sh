#!/bin/bash -l
# Submit later batches only after earlier GPU jobs finish, respecting max_queued=6.
set -euo pipefail
cd "${PBS_O_WORKDIR:?}"
REPO=$(pwd)
META="$REPO/results/small_workflow"
QSUB=/opt/pbs/bin/qsub
test -x "$QSUB"
# Resolve this recovery chain from its submission manifest, not fixed job IDs.
PREP220=${PREP220:-$(awk -F, '$1 == 220 && $2 == "prepare" { id=$5 } END { if (!id) exit 1; print id }' "$META/jobs.csv")}
LAST160=${LAST160:-$(awk -F, '$1 == 160 && $2 == "train" && $4 < 36 { if ($4+0 >= last+0) { last=$4; id=$5 } } END { if (!id) exit 1; print id }' "$META/jobs.csv")}
submit() {
    local frame=$1 stage=$2 start=$3 end=$4 dependency=$5 label=$6 queue=$7 duration=$8 script=${9:-tools/uts_small_workflow_job.sh}
    local resources='select=1:ncpus=4:mem=32gb'
    if [ "$queue" = small_gpuq ]; then resources+=':ngpus=1'; fi
    mkdir -p "$REPO/results/small_pbs_logs/frame$frame"
    local variables="FRAME=$frame,STAGE=$stage,START=$start,END=$end"
    if [ "$stage" = tail_dispatch ]; then variables+=",POST220=$POST220"; fi
    local job
    job=$("$QSUB" -N "$label" -q "$queue" -l "$resources" -l "walltime=$duration" \
        -W "depend=afterok:$dependency" -v "$variables" \
        -o "$REPO/results/small_pbs_logs/frame$frame/$label.out" \
        -e "$REPO/results/small_pbs_logs/frame$frame/$label.err" "$script") || return 1
    [[ "$job" =~ ^[0-9]+\. ]] || return 1
    printf '%s,%s,%s,%s,%s,%s\n' "$frame" "$stage" "$start" "$end" "$job" "$dependency" >> "$META/jobs.csv"
    printf '%s' "$job"
}
if [ "${STAGE:-kickoff}" = kickoff ]; then
    # fold_00 of frame160 has finished, so there are at most five waiting GPU jobs.
    train220=$(submit 220 train 0 48 "$PREP220" small220_all small_gpuq 48:00:00)
    POST220=$(submit 220 post 0 0 "$train220" small220_post_ok smallq 02:00:00)
    export POST220
    later=$(submit 160 tail_dispatch 36 48 "$LAST160" small160_dispatch smallq 00:10:00 tools/uts_small_dispatch_job.sh)
    echo "Frame220 training=$train220 post=$POST220; frame160 tail dispatcher=$later"
elif [ "$STAGE" = tail_dispatch ]; then
    train160=$(submit 160 train 36 48 "$LAST160" small160_tail small_gpuq 48:00:00)
    post160=$(submit 160 post 0 0 "$train160" small160_post_ok smallq 02:00:00)
    combined=$(submit 160 combined 0 0 "$post160:${POST220:?}" small_both_ok smallq 02:00:00)
    echo "Frame160 remaining=$train160 post=$post160; combined figures=$combined"
else
    echo "Unknown dispatch stage $STAGE" >&2; exit 2
fi
