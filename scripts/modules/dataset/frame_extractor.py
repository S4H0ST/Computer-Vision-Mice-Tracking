import cv2
from pathlib import Path
from helpers.interfaces import BaseModule
from helpers.config import paths, DatasetParams

class FrameExtractor(BaseModule):
    """Extrae frames de videos crudos para ser etiquetados."""

    def __init__(self, config: DatasetParams):
        self.fps_target = config.fps_extract
        self.input_dir = paths.raw_videos
        self.output_dir = paths.raw_images / "extracted_frames"

    def run(self) -> None:
        if not self.validate_file(self.input_dir): return
        self.output_dir.mkdir(parents=True, exist_ok=True)

        videos = list(self.input_dir.glob("*.mp4"))
        if not videos:
            print(f"⚠️ No hay videos .mp4 en {self.input_dir}")
            return

        print(f"🎞️ Extrayendo frames a {self.fps_target} FPS...")
        total_saved = 0

        for video_path in videos:
            cap = cv2.VideoCapture(str(video_path))
            fps_video = cap.get(cv2.CAP_PROP_FPS)
            if fps_video == 0: continue

            interval = max(1, int(fps_video / self.fps_target))
            name = video_path.stem
            count = 0
            saved = 0

            while True:
                success, frame = cap.read()
                if not success: break

                if count % interval == 0:
                    out_name = f"{name}_{saved:04d}.jpg"
                    cv2.imwrite(str(self.output_dir / out_name), frame)
                    saved += 1
                    total_saved += 1
                count += 1
            cap.release()
            print(f"   -> {name}: {saved} imágenes.")