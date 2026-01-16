import sys
from helpers.config import paths, data_cfg

# --- CAMBIO IMPORTANTE: Apuntamos a modules.dataset ---
from modules.dataset.frame_extractor import FrameExtractor
from modules.dataset.dataset_builder import DatasetBuilder

def main():
    paths.check_dirs()

    while True:
        print("\n" + "=" * 40)
        print(" ;) RAT DATASET TOOLKIT (Etiquetado & Limpieza)")
        print("=" * 40)
        print("1. Extraer Frames de Videos Crudos (para etiquetar)")
        print("2. Compilar Dataset Final (Juntar + Split Train/Valid)")
        print("3. Salir")

        opt = input("\n[?] Opción: ")

        if opt == "1":
            print(f"[i]  Buscando videos en: {paths.raw_videos}")
            extractor = FrameExtractor(data_cfg)
            extractor.run()

        elif opt == "2":
            print(f"[i]  Recopilando desde: {paths.raw_images}")
            builder = DatasetBuilder(data_cfg)
            builder.run()

        elif opt == "3":
            sys.exit()
        else:
            print("[X] Opción inválida")

if __name__ == "__main__":
    main()