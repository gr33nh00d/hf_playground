import os
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from moviepy.editor import ImageSequenceClip


pipe = DiffusionPipeline.from_pretrained("strangeman3107/animov-512x")
print("---------------------------------------------")
print("MADE DIFFUSION_PIPELINE")
print("---------------------------------------------")


pipe = pipe.to("cuda")
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.enable_model_cpu_offload()

torch.cuda.empty_cache()
print("---------------------------------------------")
print("MADE A SCHEDULER")
print(pipe.device)
print("---------------------------------------------")

exit(0)

prompt = "A guy with a sword"
video_frames = pipe(prompt, num_inference_steps=15, num_frames=25).frames
print("---------------------------------------------")
print("Ran the pipe")
print("---------------------------------------------")



print(video_frames.shape)

video_path = export_to_video(video_frames[0], "sword.mp4")


# ------------------------

# Create a directory to store individual frame videos
# os.makedirs("frame_videos", exist_ok=True)

# # Iterate over the frames in the batch and export each frame individually
# for i, frame in enumerate(video_frames[0]):
#     frame_path = os.path.join("frame_videos", f"frame_{i}.mp4")
#     export_to_video([frame], frame_path)  # Convert frame tensor to numpy array
#     print(f"Exported frame {i} to {frame_path}")