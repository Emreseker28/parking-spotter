#!/bin/bash
# usage: decode_cost.sh <channel> <subtype> <seconds>  -- decode to /dev/null, report CPU seconds. No files written.
set -a; . /home/likewhosloss/parking-spotter/.env; set +a
PASS=$(python3 -c "import os,urllib.parse as u;print(u.quote(os.environ['RTSP_PASS'],safe=''))")
URL="rtsp://$RTSP_USER:$PASS@$NVR_IP:554/cam/realmonitor?channel=$1&subtype=$2"
/usr/bin/time -f "ch$1 sub$2 wall=%es user=%Us sys=%Ss cpu=%P maxrss=%MKB" \
  taskset -c 0,2 ffmpeg -nostdin -loglevel error -rtsp_transport tcp -fflags nobuffer -flags low_delay -i "$URL" -t $3 -f rawvideo -pix_fmt bgr24 pipe:1 2>&1 >/dev/null | sed "s/$PASS/***/g"
