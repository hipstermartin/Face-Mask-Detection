import streamlit as st
from moviepy.editor import*

vid_file=open("./out_video.mp4","rb").read()
st.video(vid_file,start_time=0)


'''
def image_to_video():
        my_clip = VideoFileClip("./out_video.mp4")
        duration = int(my_clip.duration)
        clips=[]
        for i in range(0, duration):
            my_clip.save_frame("./images/picture" + str(i) + ".jpg", i)
            clips.append(ImageClip("./images/picture"+str(i)+".jpg").set_duration(1))
        video=concatenate_videoclips(clips,method="compose")
        video.write_videofile('./images/result.mp4',fps=120)
'''
