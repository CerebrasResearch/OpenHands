#!/bin/bash

# Docker Cleanup Script - Runs continuously every 30 minutes
# Usage: sudo bash docker-cleanup.sh [log_file]
# Example: sudo bash docker-cleanup.sh /path/to/custom.log
# Or use screen/tmux for better control

# Configuration
MINUTES_THRESHOLD=60    # Keep resources created in the last 60 minutes (1 hour)
INTERVAL_SECONDS=1800   # 30 minutes

# Set log file - use argument if provided, otherwise use current directory
if [ -n "$1" ]; then
    LOG_FILE="$1"
else
    LOG_FILE="$(pwd)/docker-cleanup.log"
fi

# Convert minutes to hours for Docker filter
HOURS_THRESHOLD=$((MINUTES_THRESHOLD / 60))

# Create log file if it doesn't exist
touch "$LOG_FILE" 2>/dev/null || {
    echo "Warning: Cannot create log file at $LOG_FILE"
    echo "Logs will only be displayed on console"
    LOG_FILE="/dev/null"
}

# Function to log to both console and file
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "================================================"
log "Docker Cleanup Service Started"
log "Started at: $(date)"
log "Cleanup interval: 30 minutes"
log "Keeps resources newer than: ${MINUTES_THRESHOLD} minutes (${HOURS_THRESHOLD} hours)"
log "Log file: $LOG_FILE"
log "================================================"
log ""

# Function to perform cleanup
cleanup_docker() {
    log "========================================"
    log "Starting Docker cleanup at $(date)"
    log "Threshold: ${MINUTES_THRESHOLD} minutes (${HOURS_THRESHOLD} hours)"
    log "========================================"

    # Remove stopped containers older than threshold
    log "[$(date +%H:%M:%S)] Cleaning old stopped containers..."
    docker container prune -f --filter "until=${HOURS_THRESHOLD}h" 2>&1 | tee -a "$LOG_FILE"

    # Remove dangling images (not tagged and not used)
    log "[$(date +%H:%M:%S)] Cleaning dangling images..."
    docker image prune -f 2>&1 | tee -a "$LOG_FILE"

    # Remove unused images older than threshold
    log "[$(date +%H:%M:%S)] Cleaning old unused images..."
    docker image prune -a -f --filter "until=${HOURS_THRESHOLD}h" 2>&1 | tee -a "$LOG_FILE"

    # Remove unused volumes
    log "[$(date +%H:%M:%S)] Cleaning unused volumes..."
    docker volume prune -f 2>&1 | tee -a "$LOG_FILE"

    # Remove unused networks
    log "[$(date +%H:%M:%S)] Cleaning unused networks..."
    docker network prune -f 2>&1 | tee -a "$LOG_FILE"

    # Remove build cache older than threshold
    log "[$(date +%H:%M:%S)] Cleaning old build cache..."
    docker builder prune -f --filter "until=${HOURS_THRESHOLD}h" 2>&1 | tee -a "$LOG_FILE"

    log "[$(date +%H:%M:%S)] Docker cleanup completed"
    log ""
}

# Trap to handle script termination
trap 'log ""; log "Docker cleanup service stopped at $(date)"; exit 0' SIGINT SIGTERM

# Main loop
while true; do
    cleanup_docker

    log "Next cleanup in 30 minutes (at $(date -d "+30 minutes" 2>/dev/null || date -v+30M 2>/dev/null || echo "30 minutes"))"
    log "Press Ctrl+C to stop"
    log ""

    sleep $INTERVAL_SECONDS
done
