import sys
from pathlib import Path
sys.path.insert(0, str(Path(r"C:\Users\sohai\Documents\1_TFG\Github_Proyect\Computer-Vision-Mice-Tracking\scripts")))
from helpers.configuracion import paths, train_cfg
from modules.core_yolo.trainer import YOLOTrainer
paths.check_dirs()
trainer = YOLOTrainer(train_cfg)
trainer.run()
