import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from moviepy.editor import ImageSequenceClip

pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
# pipe = pipe.to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

prompt = "A anime guy with a sword"
video_frames = pipe(prompt, num_inference_steps=40, num_frames=50).frames

video_path = export_to_video(video_frames[0], "gundum.mp4")


# ------------------------

# Create a directory to store individual frame videos
# os.makedirs("frame_videos", exist_ok=True)

# # Iterate over the frames in the batch and export each frame individually
# for i, frame in enumerate(video_frames[0]):
#     frame_path = os.path.join("frame_videos", f"frame_{i}.mp4")
#     export_to_video([frame], frame_path)  # Convert frame tensor to numpy array
#     print(f"Exported frame {i} to {frame_path}")