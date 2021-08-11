from model import MobileNet
from utils.youtube_downloader import YoutubeDownloader

# example video (traffic video)
url = 'https://www.youtube.com/watch?v=jjlBnrzSGjc&ab_channel=PanasonicSecurity'
youtube_downloader = YoutubeDownloader(url=url)
# download the video from youtube
video_name = youtube_downloader.download_video()

# ----------------------------------------------------------------------------------

# init the MobileNet model
net = MobileNet(weights_path='models/frozen_inference_graph.pb',
                pbtext_path='models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt',
                classnames_path="./models/coco.names")

# ---------------------------------------------------------------------------------
# test the model
net.live_test(camera=False, video=video_name)
