import os
k_ = os.listdir("198")
for i in k_:
    os.system("ffmpeg -i 198/"+i+" 198/"+i.split(".")[0]+".wav")