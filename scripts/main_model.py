import sys
from helpers.config import paths, train_cfg, detect_cfg

# Módulos del Core
from modules.core.calibrator import ZoneCalibrator
from modules.core.trainer import YOLOTrainer
from modules.core.detector import RatDetector

# Módulo del Cerebro (NUEVO)
from modules.brain.trainer_manager import RNNTrainer


def main():
    paths.check_dirs()

    while True:
        print("\n" + "=" * 50)
        print(" [(;)] RAT MODEL MANAGER (Sistema Híbrido)")
        print("=" * 50)
        print("1. Calibrar Zonas (Paredes/Agujeros)")
        print("2. Entrenar Modelo YOLO (Visual)")
        print("3. Ejecutar Detección (Genera datos y usa RNN si existe)")
        print("4. Entrenar Cerebro RNN (Temporal)")
        print("5. Salir")

        opt = input("\n[?] Elige una opción: ")

        if opt == "1":
            calib = ZoneCalibrator(paths.img_source)
            calib.run()

        elif opt == "2":
            trainer = YOLOTrainer(train_cfg)
            trainer.run()

        elif opt == "3":
            # Verificación de seguridad
            if not paths.coords_json.exists():
                print("[!] ERROR: Primero debes calibrar las zonas (Opción 1).")
                continue

            # Ejecutar el detector
            detector = RatDetector(detect_cfg)
            detector.run()

        elif opt == "4":
            print("\n[Brain] Iniciando entrenamiento de la red temporal (LSTM)...")
            # Entrenamos usando los CSVs generados por la opción 3
            brain_trainer = RNNTrainer()
            brain_trainer.train()

        elif opt == "5":
            print("[*] Saliendo...")
            break

        else:
            print("[!] Opción no válida.")


if __name__ == "__main__":
    main()