import sys
from helpers.config import paths, train_cfg, detect_cfg

# --- CAMBIO IMPORTANTE: Apuntamos a modules.core ---
from modules.core.calibrator import ZoneCalibrator
from modules.core.trainer import YOLOTrainer
from modules.core.detector import RatDetector
from modules.brain.trainer_manager import RNNTrainer


def main():
    paths.check_dirs()

    while True:
        print("\n" + "=" * 40)
        print(" [(;)] RAT MODEL MANAGER (Entrenamiento & IA)")
        print("=" * 40)
        print("1. Calibrar Zonas (Paredes/Agujeros)")
        print("     1.1 Primero coordenadas del borde EXTERIOR")
        print("     1.2 Luego coordenadas del borde INTERIOR")
        print("     1.3 Luego coordenadas de los AGUJEROS (4)")
        print("2. Entrenar Modelo YOLO")
        print("3. Ejecutar Deteccion y Analisis")
        print("4. Entrenar Cerebro RNN")
        print("5. Salir")

        opt = input("\n[?] Opción: ")

        if opt == "1":
            calib = ZoneCalibrator(paths.video_source)
            calib.run()

        elif opt == "2":
            trainer = YOLOTrainer(train_cfg)
            trainer.run()

        elif opt == "3":
            if not paths.coords_json.exists():
                print("[¿?] ERROR: Primero debes calibrar las zonas (Opción 1).")
                continue

            detector = RatDetector(detect_cfg)
            detector.run()

        elif opt == "4":
            print("\n[Brain] Iniciando entrenamiento de la red temporal (LSTM)...")
            brain_trainer = RNNTrainer()
            brain_trainer.train()

        elif opt == "5":
            print("[*] Saliendo...")
            break

        else:
            print("[!] Opción no válida.")


if __name__ == "__main__":
    main()