#!/usr/bin/env bash
set -euo pipefail

app_identifier="com.jakes.sprite-maker"
default_tail_lines=50

usage() {
  cat <<'EOF'
Usage: show-ollama-logs.sh [OPTION]

Display the Sprite Studio Ollama log files.

Options:
  -t, --tail [LINES]  Show the last LINES of the newest log (default: 50)
      --path          Print the log directory path
  -h, --help          Show this help
EOF
}

case "$(uname -s)" in
  Darwin)
    app_data_directory="${HOME:?HOME must be set}/Library/Application Support/$app_identifier"
    ;;
  MINGW*|MSYS*|CYGWIN*)
    app_data_directory="${APPDATA:-${HOME:?HOME or APPDATA must be set}/AppData/Roaming}/$app_identifier"
    ;;
  *)
    app_data_directory="${XDG_DATA_HOME:-${HOME:?HOME must be set}/.local/share}/$app_identifier"
    ;;
esac

log_directory="$app_data_directory/logs"
tail_lines="$default_tail_lines"
mode="list"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -t|--tail)
      mode="tail"
      if [[ $# -gt 1 && "$2" != -* ]]; then
        tail_lines="$2"
        shift
      fi
      ;;
    --tail=*)
      mode="tail"
      tail_lines="${1#*=}"
      ;;
    --path)
      printf '%s\n' "$log_directory"
      exit 0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ "$mode" == "tail" ]]; then
  if ! [[ "$tail_lines" =~ ^[0-9]+$ ]] || (( tail_lines < 1 )); then
    printf 'Tail line count must be a positive integer: %s\n' "$tail_lines" >&2
    exit 2
  fi
fi

if [[ ! -d "$log_directory" ]]; then
  printf 'No Ollama log directory yet: %s\n' "$log_directory"
  exit 0
fi

log_files="$(
  for log_file in "$log_directory"/ollama.log.*; do
    [[ -f "$log_file" ]] && printf '%s\n' "$log_file"
  done | sort
)"
if [[ -z "$log_files" ]]; then
  printf 'No Ollama log files found in: %s\n' "$log_directory"
  exit 0
fi

if [[ "$mode" == "tail" ]]; then
  latest_log="$(printf '%s\n' "$log_files" | tail -n 1)"
  printf 'Showing the last %s lines of %s\n\n' "$tail_lines" "$latest_log"
  tail -n "$tail_lines" "$latest_log"
  exit 0
fi

printf 'Ollama log directory: %s\n\n' "$log_directory"
printf '%s\n' "$log_files"
