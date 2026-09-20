@echo off
ffmpeg -f image2 -r 60 -i "frame_%%d.jpg" -vcodec libx264 -profile:v high444 -refs 16 -crf 18 -preset ultrafast out_x.mp4
PAUSE