#!/bin/bash

count=0
total=0
total_files=$(find /Users/a1/Project/AUDIO/neutral -name "*.wav" | wc -l)
echo "Total files to process: $total_files"
find /Users/a1/Project/AUDIO/neutral -name "*.wav" | while read -r file; do
    dur=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$file")
    total=$(echo "$total + $dur" | bc)
    count=$((count+1))
    echo "$count/$total_files files processed, current total: $(echo "$total/60" | bc -l) minutes"
    if (( count % 100 == 0 )); then
        echo "--- $count/$total_files files processed, current total: $(echo "$total/60" | bc -l) minutes ---"
    fi
done
echo "Final: $count/$total_files files processed, total duration: $(echo "$total/60" | bc -l) minutes"