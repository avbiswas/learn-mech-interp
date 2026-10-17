#!/bin/sh
# Runs the experiment scripts in parallel instead of one after another.
#
# Not everything can start at once. In 01-superposition, scripts 03, 07, 08 and
# 11 read JSON that 02 writes, 08 also reads what 06 writes, and 12_explainer
# reads 12_editing_a_fact and 13. Those stay behind their parents; everything
# else runs at the same time.
#
# Each worker gets a slice of the CPU (12 cores / jobs) so the processes do not
# fight over threads, which is what makes a naive "&" fan-out slower, not faster.
#
# Usage:
#   scripts/parallel.sh                          # both folders, 4 at a time
#   scripts/parallel.sh -j 6                     # six at a time
#   scripts/parallel.sh -j 3 -t 4                # 3 workers, 4 threads each
#   scripts/parallel.sh 02-inside-the-transformer # one folder only
#   scripts/parallel.sh -n                       # print the plan, run nothing
#
# Logs go to .parallel-logs/<folder>__<script>.log, one file per script.
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
LOGDIR="$ROOT/.parallel-logs"

JOBS=4
THREADS=""
DRY=0
FOLDERS=""

ARGS=""

while [ $# -gt 0 ]; do
  case "$1" in
    -j) JOBS=${2:?-j needs a number}; shift 2 ;;
    -t) THREADS=${2:?-t needs a number}; shift 2 ;;
    -n) DRY=1; shift ;;
    -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "unknown flag: $1" >&2; exit 2 ;;
    *) ARGS="$ARGS $1"; shift ;;
  esac
done

CPUS=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)
if [ -z "$THREADS" ]; then
  THREADS=$((CPUS / JOBS))
  [ "$THREADS" -ge 1 ] || THREADS=1
fi

# --- the dependency graph -----------------------------------------------------
# "Which scripts must already have finished before this one can start?"
# Empty means the script stands alone.
deps_for() {
  case "$1/$2" in
    01-superposition/12_explainer.py) echo "12_editing_a_fact.py 13_room_to_be_clean.py" ;;
    01-superposition/03_*.py)         echo "02_five_concepts_two_numbers.py" ;;
    01-superposition/07_*.py)         echo "02_five_concepts_two_numbers.py" ;;
    01-superposition/08_*.py)         echo "02_five_concepts_two_numbers.py 06_neurons_that_mean_two_things.py" ;;
    01-superposition/11_*.py)         echo "02_five_concepts_two_numbers.py" ;;
    *)                                echo "" ;;
  esac
}

# --- bookkeeping --------------------------------------------------------------
STATE=$(mktemp -d "${TMPDIR:-/tmp}/mech-parallel.XXXXXX") || exit 1
trap 'rm -rf "$STATE"' EXIT INT TERM
mkdir -p "$STATE/pid" "$STATE/code" "$STATE/log" "$LOGDIR"

key() { echo "$1__$2"; }

RUNNING=""
PENDING=""
DONE_OK=""
DONE_BAD=""
SKIPPED=""
NRUN=0

words() { set -- $1; echo $#; }

for a in $ARGS; do
  case "$a" in
    */[0-9][0-9]_*.py)
      f=${a%%/*}; s=${a#*/}
      if [ ! -e "$ROOT/$f/$s" ]; then echo "no such script: $a" >&2; exit 2; fi
      PENDING="$PENDING $f/$s" ;;
    *)
      if [ ! -d "$ROOT/$a" ]; then echo "no such folder: $a" >&2; exit 2; fi
      FOLDERS="$FOLDERS $a" ;;
  esac
done

[ -n "$FOLDERS" ] || [ -n "$PENDING" ] || FOLDERS="01-superposition 02-inside-the-transformer"

for folder in $FOLDERS; do
  for path in "$ROOT/$folder"/[0-9][0-9]_*.py; do
    [ -e "$path" ] || continue
    PENDING="$PENDING $folder/$(basename "$path")"
  done
done

if [ -z "$PENDING" ]; then echo "nothing to run"; exit 0; fi

if [ "$DRY" = 1 ]; then
  echo "plan: $(words "$PENDING") scripts, $JOBS at a time, $THREADS threads each ($CPUS cores)"
  for p in $PENDING; do
    d=$(deps_for "${p%%/*}" "${p#*/}")
    if [ -n "$d" ]; then echo "  $p   after: $d"; else echo "  $p   (independent)"; fi
  done
  exit 0
fi

launch() {
  k=$(key "$1" "$2")
  (
    cd "$ROOT/$1" || exit 97
    OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
    VECLIB_MAXIMUM_THREADS=$THREADS NUMEXPR_MAX_THREADS=$THREADS \
      uv run python "$2" > "$STATE/log/$k.log" 2>&1
    echo $? > "$STATE/code/$k"
  ) &
  echo $! > "$STATE/pid/$k"
  RUNNING="$RUNNING $k"
  NRUN=$((NRUN + 1))
  printf 'start   %s\n' "$k"
}

reap() {
  still=""
  for k in $RUNNING; do
    pid=$(cat "$STATE/pid/$k")
    if kill -0 "$pid" 2>/dev/null; then
      still="$still $k"
      continue
    fi
    wait "$pid" 2>/dev/null
    code=$(cat "$STATE/code/$k" 2>/dev/null || echo 1)
    cp "$STATE/log/$k.log" "$LOGDIR/$k.log" 2>/dev/null
    NRUN=$((NRUN - 1))
    if [ "$code" = 0 ]; then
      DONE_OK="$DONE_OK $k"
      printf 'ok      %s\n' "$k"
    else
      DONE_BAD="$DONE_BAD $k"
      printf 'FAIL    %s (exit %s)  log: .parallel-logs/%s.log\n' "$k" "$code" "$k"
    fi
  done
  RUNNING="$still"
}

deps_ok() {
  for d in $(deps_for "$1" "$2"); do
    case " $DONE_OK " in *" $(key "$1" "$d") "*) ;; *) return 1 ;; esac
  done
  return 0
}

deps_broken() {
  for d in $(deps_for "$1" "$2"); do
    case " $DONE_BAD " in *" $(key "$1" "$d") "*) return 0 ;; esac
  done
  return 1
}

echo "running $(words "$PENDING") scripts, $JOBS at a time, $THREADS threads each ($CPUS cores)"

while [ -n "$PENDING" ] || [ "$NRUN" -gt 0 ]; do
  reap
  started=0
  still=""
  for p in $PENDING; do
    f=${p%%/*}; s=${p#*/}
    if deps_broken "$f" "$s"; then
      SKIPPED="$SKIPPED $p"
      printf 'skip    %s  (a script it reads failed)\n' "$p"
      continue
    fi
    if [ "$NRUN" -lt "$JOBS" ] && deps_ok "$f" "$s"; then
      launch "$f" "$s"
      started=$((started + 1))
      continue
    fi
    still="$still $p"
  done
  PENDING="$still"

  # Nothing running and nothing launchable means a script is waiting on a
  # parent that is not part of this run. Say so instead of hanging forever.
  if [ "$NRUN" -eq 0 ] && [ -n "$PENDING" ] && [ "$started" -eq 0 ]; then
    echo "stuck: these need a script that is not in this run:" >&2
    for p in $PENDING; do
      printf '  %s needs: %s\n' "$p" "$(deps_for "${p%%/*}" "${p#*/}")" >&2
    done
    exit 2
  fi
  if [ "$NRUN" -gt 0 ] || [ -n "$PENDING" ]; then sleep 0.4; fi
done

echo
echo "$(words "$DONE_OK") ok, $(words "$DONE_BAD") failed, $(words "$SKIPPED") skipped.  logs in .parallel-logs/"
[ -z "$DONE_BAD" ] || exit 1
