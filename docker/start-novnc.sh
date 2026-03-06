#!/bin/bash
export DISPLAY=:10

# Clean up old sockets
rm -f /tmp/.X11-unix/X10 || true
rm -f /tmp/.X10-lock || true

# 1) Start virtual display
Xvfb :10 -screen 0 1280x800x16 -nolisten tcp &
sleep 2

# 2) Start lightweight window manager
fluxbox &

# 3) Start VNC server (포트 명시 권장)
x11vnc -display :10 -forever -nopw -shared -rfbport 5900 &
sleep 2

# 4) Start noVNC
websockify --web=/usr/share/novnc/ 6080 localhost:5900 &
echo "noVNC running at: http://localhost:6080/vnc.html"

# 5) Keep container alive
tail -f /dev/null