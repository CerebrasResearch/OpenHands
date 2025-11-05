#!/bin/bash

# Docker Cleanup Script - Runs continuously every 30 minutes
# Protects images/containers based on LAST ACCESS time (not creation time)
# Usage: sudo bash docker-cleanup.sh [OPTIONS] [log_file]
# Options:
#   --dry-run    Show what would be deleted without actually deleting
# Examples: 
#   sudo bash docker-cleanup.sh /path/to/custom.log
#   sudo bash docker-cleanup.sh --dry-run
#   sudo bash docker-cleanup.sh --dry-run /path/to/custom.log

# Configuration
MINUTES_THRESHOLD=60    # Keep resources accessed in the last 60 minutes (1 hour)
INTERVAL_SECONDS=1800   # 30 minutes
DRY_RUN=false           # Dry run mode flag

# Parse arguments
LOG_FILE=""
for arg in "$@"; do
    if [ "$arg" = "--dry-run" ]; then
        DRY_RUN=true
    elif [ -z "$LOG_FILE" ]; then
        LOG_FILE="$arg"
    fi
done

# Set log file - use argument if provided, otherwise use current directory
if [ -z "$LOG_FILE" ]; then
    LOG_FILE="$(pwd)/docker-cleanup.log"
fi

# Convert minutes to hours for Docker filter
HOURS_THRESHOLD=$((MINUTES_THRESHOLD / 60))
if [ $HOURS_THRESHOLD -lt 1 ]; then
    HOURS_THRESHOLD=1
fi

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

# Function to get the most recent timestamp from multiple values
get_most_recent() {
    local max_time=0
    for time in "$@"; do
        if [ -n "$time" ] && [ "$time" -gt "$max_time" ]; then
            max_time=$time
        fi
    done
    echo "$max_time"
}

# Function to parse Docker timestamp to unix timestamp
parse_timestamp() {
    local timestamp="$1"
    if [ -z "$timestamp" ] || [ "$timestamp" = "0001-01-01T00:00:00Z" ]; then
        echo "0"
        return
    fi
    date -d "$timestamp" +%s 2>/dev/null || date -j -f "%Y-%m-%dT%H:%M:%S" "${timestamp%%.*}" +%s 2>/dev/null || echo "0"
}

log "================================================"
log "Docker Cleanup Service Started"
log "Started at: $(date)"
log "Mode: $([ "$DRY_RUN" = true ] && echo "DRY RUN (no deletions)" || echo "ACTIVE (will delete)")"
log "Cleanup interval: 30 minutes"
log "Keeps resources LAST ACCESSED in: ${MINUTES_THRESHOLD} minutes"
log "Log file: $LOG_FILE"
log "================================================"
log ""

# Function to perform cleanup
cleanup_docker() {
    log "========================================"
    log "Starting Docker cleanup at $(date)"
    log "Mode: $([ "$DRY_RUN" = true ] && echo "DRY RUN" || echo "ACTIVE")"
    log "Protection threshold: ${MINUTES_THRESHOLD} minutes (LAST ACCESS)"
    log "========================================"

    # Calculate cutoff timestamp
    CUTOFF_TIME=$(($(date +%s) - MINUTES_THRESHOLD * 60))
    log "Cutoff timestamp: $CUTOFF_TIME ($(date -d @$CUTOFF_TIME 2>/dev/null || date -r $CUTOFF_TIME 2>/dev/null))"

    # Get images that are parents of other images
    log "[$(date +%H:%M:%S)] Identifying parent images..."
    PARENT_IMAGES=$(docker images -q | while read child_img; do
        docker inspect "$child_img" --format '{{.Parent}}' 2>/dev/null
    done | grep -v '^$' | sort -u)

    # Get images currently in use by RUNNING containers
    log "[$(date +%H:%M:%S)] Identifying images in use by running containers..."
    RUNNING_IMAGES=$(docker ps --format '{{.Image}}' | while read img; do
        docker inspect "$img" --format='{{.Id}}' 2>/dev/null
    done | sort -u)

    # Clean stopped containers based on last access (Started time or Created time)
    log "[$(date +%H:%M:%S)] Cleaning old stopped containers..."
    CONTAINERS_REMOVED=0
    docker ps -a --format '{{.ID}}' | while read container_id; do
        # Get container info
        CONTAINER_DATA=$(docker inspect "$container_id" --format '{{.State.Status}}|{{.State.StartedAt}}|{{.Created}}|{{.Name}}' 2>/dev/null)
        
        if [ -z "$CONTAINER_DATA" ]; then
            continue
        fi
        
        STATUS=$(echo "$CONTAINER_DATA" | cut -d'|' -f1)
        STARTED_AT=$(echo "$CONTAINER_DATA" | cut -d'|' -f2)
        CREATED=$(echo "$CONTAINER_DATA" | cut -d'|' -f3)
        NAME=$(echo "$CONTAINER_DATA" | cut -d'|' -f4)
        
        # Skip running containers
        if [ "$STATUS" = "running" ]; then
            continue
        fi

        # Get timestamps
        STARTED_TIMESTAMP=$(parse_timestamp "$STARTED_AT")
        CREATED_TIMESTAMP=$(parse_timestamp "$CREATED")
        
        # Use the most recent timestamp (last access = last started or created)
        LAST_ACCESS=$(get_most_recent "$STARTED_TIMESTAMP" "$CREATED_TIMESTAMP")

        if [ "$LAST_ACCESS" -gt 0 ] && [ "$LAST_ACCESS" -lt "$CUTOFF_TIME" ]; then
            LAST_ACCESS_DATE=$(date -d "@$LAST_ACCESS" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -r "$LAST_ACCESS" '+%Y-%m-%d %H:%M:%S' 2>/dev/null)
            if [ "$DRY_RUN" = true ]; then
                log "  [DRY RUN] Would remove container: $NAME ($container_id) - last accessed: $LAST_ACCESS_DATE"
                CONTAINERS_REMOVED=$((CONTAINERS_REMOVED + 1))
            else
                log "  Removing container: $NAME ($container_id) - last accessed: $LAST_ACCESS_DATE"
                if docker rm "$container_id" 2>&1 | tee -a "$LOG_FILE"; then
                    CONTAINERS_REMOVED=$((CONTAINERS_REMOVED + 1))
                fi
            fi
        else
            LAST_ACCESS_DATE=$(date -d "@$LAST_ACCESS" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -r "$LAST_ACCESS" '+%Y-%m-%d %H:%M:%S' 2>/dev/null)
            log "  Keeping container: $NAME ($container_id) - last accessed: $LAST_ACCESS_DATE"
        fi
    done
    if [ "$DRY_RUN" = true ]; then
        log "  [DRY RUN] Would remove $CONTAINERS_REMOVED old stopped containers"
    else
        log "  Removed $CONTAINERS_REMOVED old stopped containers"
    fi

    # Clean dangling images
    log "[$(date +%H:%M:%S)] Cleaning dangling images..."
    if [ "$DRY_RUN" = true ]; then
        DANGLING_COUNT=$(docker images -f "dangling=true" -q | wc -l)
        log "  [DRY RUN] Would remove $DANGLING_COUNT dangling images"
    else
        docker image prune -f 2>&1 | tee -a "$LOG_FILE"
    fi

    # Clean unused images based on last access time
    log "[$(date +%H:%M:%S)] Analyzing image access times..."
    IMAGES_REMOVED=0
    
    # First, build a map of image last access times
    declare -A IMAGE_LAST_ACCESS
    
    # Check all containers (including stopped) to find when each image was last accessed
    docker ps -a --format '{{.ID}}' | while read container_id; do
        CONTAINER_DATA=$(docker inspect "$container_id" --format '{{.Image}}|{{.State.StartedAt}}|{{.Created}}' 2>/dev/null)
        
        if [ -z "$CONTAINER_DATA" ]; then
            continue
        fi
        
        IMAGE_ID=$(echo "$CONTAINER_DATA" | cut -d'|' -f1)
        STARTED_AT=$(echo "$CONTAINER_DATA" | cut -d'|' -f2)
        CREATED=$(echo "$CONTAINER_DATA" | cut -d'|' -f3)
        
        STARTED_TIMESTAMP=$(parse_timestamp "$STARTED_AT")
        CREATED_TIMESTAMP=$(parse_timestamp "$CREATED")
        
        # Most recent access for this container
        CONTAINER_ACCESS=$(get_most_recent "$STARTED_TIMESTAMP" "$CREATED_TIMESTAMP")
        
        # Update image's last access if this is more recent
        CURRENT_ACCESS="${IMAGE_LAST_ACCESS[$IMAGE_ID]:-0}"
        if [ "$CONTAINER_ACCESS" -gt "$CURRENT_ACCESS" ]; then
            IMAGE_LAST_ACCESS[$IMAGE_ID]=$CONTAINER_ACCESS
        fi
    done
    
    # Export the associative array for subshell access
    for key in "${!IMAGE_LAST_ACCESS[@]}"; do
        echo "$key ${IMAGE_LAST_ACCESS[$key]}"
    done > /tmp/image_access_times_$$.txt
    
    # Now check each image
    docker images --format '{{.ID}}' | while read img_id; do
        # Get image details
        IMG_INFO=$(docker inspect "$img_id" --format '{{.RepoTags}}|{{.Created}}' 2>/dev/null)
        
        if [ -z "$IMG_INFO" ]; then
            continue
        fi
        
        IMG_NAME=$(echo "$IMG_INFO" | cut -d'|' -f1 | tr -d '[]' | tr ' ' ',' | cut -d',' -f1)
        IMG_CREATED=$(echo "$IMG_INFO" | cut -d'|' -f2)
        
        # Use <none> if no tag
        if [ -z "$IMG_NAME" ] || [ "$IMG_NAME" = "null" ]; then
            IMG_NAME="<none>"
        fi

        # Skip if image is a parent of another image
        if echo "$PARENT_IMAGES" | grep -q "$img_id"; then
            log "  Keeping image: $IMG_NAME ($img_id) - parent of other images"
            continue
        fi

        # Skip if image is in use by RUNNING container
        if echo "$RUNNING_IMAGES" | grep -q "$img_id"; then
            log "  Keeping image: $IMG_NAME ($img_id) - in use by running container"
            continue
        fi

        # Get last access time from our map
        LAST_ACCESS=$(grep "^$img_id " /tmp/image_access_times_$$.txt 2>/dev/null | awk '{print $2}')
        
        # If image was never used (no containers), use image creation time
        if [ -z "$LAST_ACCESS" ] || [ "$LAST_ACCESS" = "0" ]; then
            LAST_ACCESS=$(parse_timestamp "$IMG_CREATED")
        fi

        if [ "$LAST_ACCESS" -gt 0 ] && [ "$LAST_ACCESS" -lt "$CUTOFF_TIME" ]; then
            LAST_ACCESS_DATE=$(date -d "@$LAST_ACCESS" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -r "$LAST_ACCESS" '+%Y-%m-%d %H:%M:%S' 2>/dev/null)
            if [ "$DRY_RUN" = true ]; then
                log "  [DRY RUN] Would remove image: $IMG_NAME ($img_id) - last accessed: $LAST_ACCESS_DATE"
                IMAGES_REMOVED=$((IMAGES_REMOVED + 1))
            else
                log "  Removing image: $IMG_NAME ($img_id) - last accessed: $LAST_ACCESS_DATE"
                if docker rmi "$img_id" 2>&1 | tee -a "$LOG_FILE"; then
                    IMAGES_REMOVED=$((IMAGES_REMOVED + 1))
                fi
            fi
        else
            LAST_ACCESS_DATE=$(date -d "@$LAST_ACCESS" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -r "$LAST_ACCESS" '+%Y-%m-%d %H:%M:%S' 2>/dev/null)
            log "  Keeping image: $IMG_NAME ($img_id) - last accessed: $LAST_ACCESS_DATE"
        fi
    done
    
    # Cleanup temp file
    rm -f /tmp/image_access_times_$.txt
    
    if [ "$DRY_RUN" = true ]; then
        log "  [DRY RUN] Would remove $IMAGES_REMOVED old unused images"
    else
        log "  Removed $IMAGES_REMOVED old unused images"
    fi

    # Clean unused volumes
    log "[$(date +%H:%M:%S)] Cleaning unused volumes..."
    if [ "$DRY_RUN" = true ]; then
        UNUSED_VOLUMES=$(docker volume ls -qf dangling=true | wc -l)
        log "  [DRY RUN] Would remove $UNUSED_VOLUMES unused volumes"
    else
        docker volume prune -f 2>&1 | tee -a "$LOG_FILE"
    fi

    # Clean unused networks
    log "[$(date +%H:%M:%S)] Cleaning unused networks..."
    if [ "$DRY_RUN" = true ]; then
        UNUSED_NETWORKS=$(docker network ls --filter "type=custom" -q | while read net; do
            if [ "$(docker network inspect "$net" --format='{{len .Containers}}')" = "0" ]; then
                echo "$net"
            fi
        done | wc -l)
        log "  [DRY RUN] Would remove $UNUSED_NETWORKS unused networks"
    else
        docker network prune -f 2>&1 | tee -a "$LOG_FILE"
    fi

    # Clean build cache
    log "[$(date +%H:%M:%S)] Cleaning old build cache..."
    if [ "$DRY_RUN" = true ]; then
        log "  [DRY RUN] Would clean build cache older than ${HOURS_THRESHOLD}h"
    else
        docker builder prune -f --filter "until=${HOURS_THRESHOLD}h" 2>&1 | tee -a "$LOG_FILE"
    fi

    log "[$(date +%H:%M:%S)] Docker cleanup completed"
    log ""
}

# Trap to handle script termination
trap 'log ""; log "Docker cleanup service stopped at $(date)"; exit 0' SIGINT SIGTERM

# Main loop
while true; do
    cleanup_docker

    log "Next cleanup in 30 minutes (at $(date -d "+30 minutes" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -v+30M '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "30 minutes"))"
    log "Press Ctrl+C to stop"
    log ""

    sleep $INTERVAL_SECONDS
done