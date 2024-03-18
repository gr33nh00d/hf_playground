ffmpeg_command="ffmpeg "

# Iterate over the file names and construct the input list
for i in {0..15}; do
    input_file="frame_$i.mp4"
    ffmpeg_command="$ffmpeg_command -i \"$input_file\""
done

# Add the filter_complex and output file parameters
ffmpeg_command="$ffmpeg_command -filter_complex \"[0:v][1:v][2:v][3:v][4:v][5:v][6:v][7:v][8:v][9:v][10:v][11:v][12:v][13:v][14:v][15:v]concat=n=16:v=1:a=0[outv]\" -map \"[outv]\" \"combined.mp4\""

# Execute the ffmpeg command
eval $ffmpeg_command