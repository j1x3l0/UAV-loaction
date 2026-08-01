#!/usr/bin/env bash

set -euo pipefail

PX4_ROOT="${PX4_ROOT:-/root/px4-deploy/PX4-Autopilot-v1.17.0-sih}"
PX4_BUILD="$PX4_ROOT/build/px4_sitl_default"
PX4_BIN="$PX4_BUILD/bin/px4"
PX4_WORKDIR="$PX4_BUILD/src/modules/simulation/simulator_sih"
PX4_PID_FILE="${PX4_PID_FILE:-/root/px4-deploy/px4-sih.pid}"
PX4_LOG_FILE="${PX4_LOG_FILE:-/root/px4-deploy/px4-sih.log}"

usage() {
	printf 'Usage: %s {start|stop|restart|status|check|foreground}\n' "$0"
}

read_pid() {
	if [[ -r "$PX4_PID_FILE" ]]; then
		cat "$PX4_PID_FILE"
	fi
}

is_running() {
	local px4_pid
	px4_pid="$(read_pid)"
	[[ -n "$px4_pid" ]] && kill -0 "$px4_pid" 2>/dev/null
}

require_build() {
	if [[ ! -x "$PX4_BIN" ]]; then
		printf 'PX4 binary not found or not executable: %s\n' "$PX4_BIN" >&2
		exit 1
	fi
}

start_px4() {
	require_build
	if is_running; then
		printf 'PX4 SIH is already running (pid=%s).\n' "$(read_pid)"
		return 0
	fi

	mkdir -p "$PX4_WORKDIR"
	cd "$PX4_WORKDIR"
	nohup env PX4_SIM_MODEL=sihsim_quadx PX4_SIMULATOR=sihsim \
		"$PX4_BIN" -d >"$PX4_LOG_FILE" 2>&1 </dev/null &
	local px4_pid=$!
	printf '%s\n' "$px4_pid" >"$PX4_PID_FILE"

	for _ in {1..30}; do
		if ! kill -0 "$px4_pid" 2>/dev/null; then
			printf 'PX4 SIH exited during startup. See %s\n' "$PX4_LOG_FILE" >&2
			exit 1
		fi
		if grep -q 'Startup script returned successfully' "$PX4_LOG_FILE"; then
			printf 'PX4 SIH started (pid=%s, log=%s).\n' "$px4_pid" "$PX4_LOG_FILE"
			return 0
		fi
		sleep 1
	done

	printf 'PX4 SIH startup timed out. See %s\n' "$PX4_LOG_FILE" >&2
	exit 1
}

stop_px4() {
	if ! is_running; then
		printf 'PX4 SIH is not running.\n'
		rm -f "$PX4_PID_FILE"
		return 0
	fi

	local px4_pid
	px4_pid="$(read_pid)"
	cd "$PX4_WORKDIR"
	"$PX4_BUILD/bin/px4-shutdown" >/dev/null 2>&1 || kill -TERM "$px4_pid" 2>/dev/null || true

	for _ in {1..10}; do
		if ! kill -0 "$px4_pid" 2>/dev/null; then
			rm -f "$PX4_PID_FILE"
			printf 'PX4 SIH stopped.\n'
			return 0
		fi
		sleep 1
	done

	kill -KILL "$px4_pid" 2>/dev/null || true
	rm -f "$PX4_PID_FILE"
	printf 'PX4 SIH required forced cleanup after 10 seconds.\n' >&2
}

status_px4() {
	if is_running; then
		printf 'PX4 SIH is running (pid=%s).\n' "$(read_pid)"
	else
		printf 'PX4 SIH is stopped.\n'
		return 1
	fi
}

check_mavlink() {
	status_px4 >/dev/null
	python3 -c "from pymavlink import mavutil; m=mavutil.mavlink_connection('udpin:0.0.0.0:14550'); h=m.wait_heartbeat(timeout=8); assert h is not None, 'MAVLink heartbeat timeout'; print('MAVLink heartbeat OK: system=%d component=%d type=%d autopilot=%d' % (m.target_system, m.target_component, h.type, h.autopilot))"
}

foreground_px4() {
	require_build
	cd "$PX4_WORKDIR"
	exec env PX4_SIM_MODEL=sihsim_quadx PX4_SIMULATOR=sihsim "$PX4_BIN"
}

case "${1:-}" in
	start) start_px4 ;;
	stop) stop_px4 ;;
	restart) stop_px4; start_px4 ;;
	status) status_px4 ;;
	check) check_mavlink ;;
	foreground) foreground_px4 ;;
	*) usage; exit 2 ;;
esac
