#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/create_github_release.sh <artifact> [version] [options]

Artifacts:
  estimators
  partition_tree
  pyo3
  partition_tree_py
  python

Options:
  --target <git-ref>         Git ref to release from. Default: main
  --title <title>            Override the generated release title
  --notes-file <path>        Use a custom release notes file instead of GitHub-generated notes
  --notes-start-tag <tag>    Start GitHub-generated notes after the given tag
  --wait                     Wait for the matching release workflow to finish
  --wait-timeout <seconds>   Maximum time to wait for the workflow. Default: 7200
  --poll-interval <seconds>  Poll interval while waiting. Default: 15
  --draft                    Create the release as a draft
  --prerelease               Mark the release as a prerelease
  --dry-run                  Print the gh command without executing it
  -h, --help                 Show this help message

Examples:
  scripts/create_github_release.sh estimators --wait
  scripts/create_github_release.sh pyo3 --target main --wait
  scripts/create_github_release.sh python --notes-file /tmp/release-notes.md
  scripts/create_github_release.sh partition_tree 0.1.0 --wait
EOF
}

require_command() {
  local command_name="$1"

  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Error: required command '$command_name' was not found in PATH." >&2
    exit 1
  fi
}

read_manifest_version() {
  local manifest_path="$1"
  local manifest_version

  if [[ ! -f "$manifest_path" ]]; then
    echo "Error: manifest '$manifest_path' does not exist." >&2
    exit 1
  fi

  manifest_version="$(awk -F'"' '/^version[[:space:]]*=[[:space:]]*"/ { print $2; exit }' "$manifest_path")"

  if [[ -z "$manifest_version" ]]; then
    echo "Error: could not determine a top-level version from '$manifest_path'." >&2
    exit 1
  fi

  printf '%s\n' "$manifest_version"
}

infer_version_from_manifests() {
  local manifest_path
  local manifest_version
  local inferred_version=""

  for manifest_path in "$@"; do
    manifest_version="$(read_manifest_version "$manifest_path")"

    if [[ -z "$inferred_version" ]]; then
      inferred_version="$manifest_version"
      continue
    fi

    if [[ "$inferred_version" != "$manifest_version" ]]; then
      echo "Error: version mismatch across manifests. Expected '$inferred_version' but '$manifest_path' declares '$manifest_version'." >&2
      exit 1
    fi
  done

  if [[ -z "$inferred_version" ]]; then
    echo "Error: no manifests were provided for version inference." >&2
    exit 1
  fi

  printf '%s\n' "$inferred_version"
}

wait_for_workflow_run() {
  local workflow_name="$1"
  local triggered_at="$2"
  local wait_timeout="$3"
  local poll_interval="$4"

  local started_at
  started_at="$(date +%s)"

  printf 'Waiting for workflow "%s" to start\n' "$workflow_name"

  while true; do
    local run_id
    run_id="$({
      gh run list \
        --workflow "$workflow_name" \
        --event release \
        --limit 20 \
        --json databaseId,createdAt,status,conclusion,url \
        --jq 'map(select(.createdAt >= "'"$triggered_at"'")) | sort_by(.createdAt) | .[0].databaseId // empty'
    } 2>/dev/null)"

    if [[ -n "$run_id" ]]; then
      printf 'Watching workflow run %s\n' "$run_id"

      while true; do
        local run_state
        run_state="$({
          gh run view "$run_id" \
            --json status,conclusion,url \
            --jq '[.status, (.conclusion // ""), .url] | @tsv'
        } 2>/dev/null)"

        if [[ -z "$run_state" ]]; then
          echo "Error: unable to inspect workflow run '$run_id'." >&2
          exit 1
        fi

        local status conclusion run_url
        IFS=$'\t' read -r status conclusion run_url <<<"$run_state"

        if [[ "$status" == "completed" ]]; then
          if [[ "$conclusion" == "success" ]]; then
            printf 'Workflow completed successfully: %s\n' "$run_url"
            return 0
          fi

          printf 'Workflow failed with conclusion %s: %s\n' "$conclusion" "$run_url" >&2
          exit 1
        fi

        if (( $(date +%s) - started_at >= wait_timeout )); then
          echo "Error: timed out waiting for workflow '$workflow_name' to complete." >&2
          exit 1
        fi

        sleep "$poll_interval"
      done
    fi

    if (( $(date +%s) - started_at >= wait_timeout )); then
      echo "Error: timed out waiting for workflow '$workflow_name' to start." >&2
      exit 1
    fi

    sleep "$poll_interval"
  done
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

artifact="$1"
shift

raw_version=""

if [[ $# -gt 0 && "$1" != --* ]]; then
  raw_version="$1"
  shift
fi

target_ref="main"
release_title=""
notes_file=""
notes_start_tag=""
wait_for_completion=false
wait_timeout=7200
poll_interval=15
draft=false
prerelease=false
dry_run=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      target_ref="$2"
      shift 2
      ;;
    --title)
      release_title="$2"
      shift 2
      ;;
    --notes-file)
      notes_file="$2"
      shift 2
      ;;
    --notes-start-tag)
      notes_start_tag="$2"
      shift 2
      ;;
    --wait)
      wait_for_completion=true
      shift
      ;;
    --wait-timeout)
      wait_timeout="$2"
      shift 2
      ;;
    --poll-interval)
      poll_interval="$2"
      shift 2
      ;;
    --draft)
      draft=true
      shift
      ;;
    --prerelease)
      prerelease=true
      shift
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown option '$1'." >&2
      usage
      exit 1
      ;;
  esac
done

require_command git
require_command gh

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

case "$artifact" in
  estimators)
    manifest_paths=("crates/estimators/Cargo.toml")
    tag_prefix="estimators-v"
    title_prefix="estimators v"
    workflow_name="Rust Crates CI"
    ;;
  partition_tree)
    manifest_paths=("crates/partition_tree/Cargo.toml")
    tag_prefix="partition_tree-v"
    title_prefix="partition_tree v"
    workflow_name="Rust Crates CI"
    ;;
  pyo3|pyo3_partition_tree)
    manifest_paths=("pyo3_partition_tree/Cargo.toml" "pyo3_partition_tree/pyproject.toml")
    tag_prefix="pyo3-v"
    title_prefix="pyo3_partition_tree v"
    workflow_name="PyO3 CI"
    ;;
  partition_tree_py|python)
    manifest_paths=("partition_tree/pyproject.toml")
    tag_prefix="partition_tree-py-v"
    title_prefix="partition_tree Python package v"
    workflow_name="Python Package CI"
    ;;
  *)
    echo "Error: unsupported artifact '$artifact'." >&2
    usage
    exit 1
    ;;
esac

if [[ -n "$raw_version" ]]; then
  version="${raw_version#v}"
else
  version="$(infer_version_from_manifests "${manifest_paths[@]}")"
  printf 'Inferred version %s from %s\n' "$version" "${manifest_paths[*]}"
fi

tag="${tag_prefix}${version}"
default_title="${title_prefix}${version}"

if [[ -n "$notes_file" && -n "$notes_start_tag" ]]; then
  echo "Error: --notes-file and --notes-start-tag cannot be used together." >&2
  exit 1
fi

if [[ -n "$notes_file" && ! -f "$notes_file" ]]; then
  echo "Error: notes file '$notes_file' does not exist." >&2
  exit 1
fi

if ! [[ "$wait_timeout" =~ ^[0-9]+$ ]] || (( wait_timeout <= 0 )); then
  echo "Error: --wait-timeout must be a positive integer." >&2
  exit 1
fi

if ! [[ "$poll_interval" =~ ^[0-9]+$ ]] || (( poll_interval <= 0 )); then
  echo "Error: --poll-interval must be a positive integer." >&2
  exit 1
fi

if [[ "$dry_run" != true ]]; then
  if ! gh auth status >/dev/null 2>&1; then
    echo "Error: gh is not authenticated. Run 'gh auth login' first." >&2
    exit 1
  fi

  if gh release view "$tag" >/dev/null 2>&1; then
    echo "Error: release '$tag' already exists on GitHub." >&2
    exit 1
  fi
fi

if [[ -z "$release_title" ]]; then
  release_title="$default_title"
fi

gh_command=(gh release create "$tag" --target "$target_ref" --title "$release_title")

if [[ -n "$notes_file" ]]; then
  gh_command+=(--notes-file "$notes_file")
else
  gh_command+=(--generate-notes)
  if [[ -n "$notes_start_tag" ]]; then
    gh_command+=(--notes-start-tag "$notes_start_tag")
  fi
fi

if [[ "$draft" == true ]]; then
  gh_command+=(--draft)
fi

if [[ "$prerelease" == true ]]; then
  gh_command+=(--prerelease)
fi

printf 'Creating GitHub release for %s with tag %s\n' "$artifact" "$tag"

if [[ "$dry_run" == true ]]; then
  printf 'Dry run:'
  printf ' %q' "${gh_command[@]}"
  printf '\n'

  if [[ "$wait_for_completion" == true ]]; then
    printf 'Dry run: would wait for workflow "%s" to complete\n' "$workflow_name"
  fi

  exit 0
fi

triggered_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

"${gh_command[@]}"

printf 'Created release %s\n' "$tag"

if [[ "$wait_for_completion" == true ]]; then
  wait_for_workflow_run "$workflow_name" "$triggered_at" "$wait_timeout" "$poll_interval"
fi