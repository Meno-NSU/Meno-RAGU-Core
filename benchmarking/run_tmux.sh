#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
session=meno-benchmark
log_file=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --session) session="$2"; shift 2 ;;
        --log-file) log_file="$2"; shift 2 ;;
        *) break ;;
    esac
done
if [[ $# -eq 0 || -z "$log_file" ]]; then
    echo 'Использование: benchmarking/run_tmux.sh --session ИМЯ --log-file ПУТЬ collect|evaluate [аргументы]' >&2
    exit 2
fi
uv_executable="$(command -v uv)"
command -v tmux >/dev/null
if tmux has-session -t "=$session" 2>/dev/null; then
    echo "Сессия уже существует: $session. Подключитесь через tmux attach -t $session" >&2
    exit 2
fi
mkdir -p -- "$(dirname -- "$log_file")"
log_file="$(cd -- "$(dirname -- "$log_file")" && pwd)/$(basename -- "$log_file")"
extra=()
if [[ "$1" == evaluate ]]; then extra=(--extra evaluate); fi
printf -v command_line '%q ' "$uv_executable" run --project benchmarking "${extra[@]}" --frozen python -u -m benchmarking "$@"
printf -v quoted_log '%q' "$log_file"
runner="set -o pipefail; $command_line 2>&1 | tee -a $quoted_log; status=\${PIPESTATUS[0]}; printf '\\nКод завершения: %s\\n' \"\$status\" | tee -a $quoted_log; exec bash"
tmux new-session -d -s "$session" -c "$repo_root" bash -c "$runner"
echo "Запущена сессия $session; журнал $log_file"
echo "Подключение: tmux attach -t $session"
