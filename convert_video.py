from moviepy import VideoFileClip 
video = VideoFileClip("tracked_output.mp4") 
video.write_videofile( 
    "streamlit_video.mp4", 
    codec="libx264" 
)