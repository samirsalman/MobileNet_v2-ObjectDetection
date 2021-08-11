import pytube


class YoutubeDownloader:
    def __init__(self, url):
        self.url = url
        self.youtube = pytube.YouTube(url)

    def download_video(self, res: str = "720p"):
        # download the video from YouTube
        video = self.youtube.streams.filter(res=res).first()
        video.download("videos/")
        return f"videos/{video.default_filename}"
