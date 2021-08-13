from model import MobileNet
from utils.youtube_downloader import YoutubeDownloader
from models import DetectionModels

# example video (traffic video)
url = 'https://www.youtube.com/watch?v=jjlBnrzSGjc&ab_channel=PanasonicSecurity'
youtube_downloader = YoutubeDownloader(url=url)
# download the video from youtube
video_name = youtube_downloader.download_video()

# ----------------------------------------------------------------------------------

MODEL = DetectionModels.MOBILENET_V2

# init the MobileNet model
net = MobileNet(weights_path=MODEL["weights"],
                pbtext_path=MODEL["pbtxt"],
                classnames_path=MODEL["classnames"])

# ---------------------------------------------------------------------------------
# test the model
net.live_test(camera=False, video=video_name)
