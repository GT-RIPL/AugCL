import numpy as np

video_easy = [846.388, 786.2969, 449.679, 459.6468]
color_hard = [942.1688, 902.559, 748.475, 882.3651]
video_hard = [245.054, 113.675, 125.347, 83.1718]

print(np.mean(video_easy))
print(np.std(video_easy))
print(np.mean(color_hard))
print(np.std(color_hard))
print(np.mean(video_hard))
print(np.std(video_hard))