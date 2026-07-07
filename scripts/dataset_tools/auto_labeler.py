# scripts/dataset_tools/auto_labeler.py
"""
Generador automático de labels YOLO-Pose para el dataset del Open Field Test.

Estrategia:
  1. ROI fija (interior de la caja) para ignorar el fondo exterior brillante.
  2. Umbral de Otsu sobre canal de brillo → blob blanco = ratón.
  3. Contorno más grande dentro del ROI → bounding box YOLO.
  4. PCA del contorno → eje mayor del cuerpo → extremos = snout / tail base.
  5. Centroide del contorno → spine center.

Precisión esperada:
  - Bounding box:  ~90 %  (el ratón contrasta mucho con el fondo oscuro)
  - Keypoints:     ~65-70 %  (posturas compactas como grooming/rearing son menos precisas)

Uso:
    cd scripts
    python dataset_tools/auto_labeler.py --src ../media_original/roboflow_testRata1/seleccion
    python dataset_tools/auto_labeler.py --src ... --debug  # guarda imagen de verificación
"""

import argparse
import cv2
import numpy as np
from pathlib import Path

# Orden de clases según data.yaml
CLASS_MAP = {
    "climbing":    0,
    "grooming":    1,
    "head_dipping": 2,
    "horizontal":  3,
    "rearing":     4,
}

# ROI interior de la caja (píxeles, imagen 1920×1080).
# Excluye las patas del trípode (izq/dcha) y el borde exterior de la caja.
# Ajustar con --calibrate si la cámara cambia de posición.
ROI_X1, ROI_Y1 = 545, 110
ROI_X2, ROI_Y2 = 1385, 890

# Rango de área del blob del ratón (px² dentro del ROI)
MIN_BLOB_AREA = 500
MAX_BLOB_AREA = 55_000

# El blob del ratón no puede abarcar más de este % del ROI en ninguna dimensión
MAX_BLOB_RATIO = 0.88

# Sigma sobre la media del ROI para el umbral adaptativo
SIGMA_LEVELS = (2.2, 1.7, 1.3, 1.0, 0.8, 0.6)


def _class_from_name(filename: str):
    name = filename.lower()
    for key, cid in CLASS_MAP.items():
        if name.startswith(key):
            return cid
    return None


def _mean_brightness(cnt, gray_roi):
    """Brillo medio de los píxeles dentro del contorno."""
    mask = np.zeros(gray_roi.shape, np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    return float(cv2.mean(gray_roi, mask=mask)[0])


def _rat_candidates(thresh, roi_w, roi_h):
    """
    Filtra contornos por área y tamaño relativo.
    NO filtra por posición del centroide para no excluir climbing en bordes.
    """
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    valid = []
    for c in cnts:
        area = cv2.contourArea(c)
        if not (MIN_BLOB_AREA <= area <= MAX_BLOB_AREA):
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w > MAX_BLOB_RATIO * roi_w or h > MAX_BLOB_RATIO * roi_h:
            continue   # blob demasiado grande → trípode o borde completo
        valid.append(c)
    return valid


def _best_rat_blob(gray_roi):
    """
    Entre los candidatos válidos elige el de MAYOR BRILLO MEDIO.
    El ratón es blanco uniforme (brillo ~180-240).
    El trípode/equipo metálico es gris (~80-140) → queda excluido.
    Reintenta bajando el umbral adaptativo hasta encontrar candidatos.
    """
    roi_h, roi_w = gray_roi.shape
    k = np.ones((5, 5), np.uint8)

    mean_v = float(np.mean(gray_roi))
    std_v  = float(np.std(gray_roi))

    last_thresh = None
    for sigma in SIGMA_LEVELS:
        thr = int(np.clip(mean_v + sigma * std_v, 35, 230))
        _, thresh = cv2.threshold(gray_roi, thr, 255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k)
        candidates = _rat_candidates(thresh, roi_w, roi_h)
        last_thresh = thresh
        if candidates:
            # Elegir el blob con mayor brillo medio, no el más grande
            best = max(candidates,
                       key=lambda c: _mean_brightness(c, gray_roi))
            return best, thresh

    return None, last_thresh


def _keypoints_from_contour(cnt):
    """
    Devuelve (snout, spine_center, tail_base) como arrays [x, y] en coords del ROI.
    Usa PCA para encontrar el eje mayor del cuerpo.
    Heurística: el extremo más alto (y menor) del eje mayor = snout.
    """
    pts = cnt.reshape(-1, 2).astype(np.float64)
    mean, eigvecs = cv2.PCACompute(pts, mean=np.array([]))
    major = eigvecs[0]  # dirección del eje mayor

    # Proyección escalar de cada punto sobre el eje mayor
    centered = pts - mean
    proj = centered @ major

    idx_a = int(np.argmin(proj))
    idx_b = int(np.argmax(proj))
    pt_a = pts[idx_a]
    pt_b = pts[idx_b]
    centroid = mean[0]

    # Snout = extremo con y más pequeño (más arriba en imagen)
    if pt_a[1] <= pt_b[1]:
        snout, tail = pt_a, pt_b
    else:
        snout, tail = pt_b, pt_a

    return snout, centroid, tail


def label_image(img_path: Path, debug: bool = False):
    """
    Procesa una imagen y devuelve la línea YOLO-Pose o None si falla.
    Si debug=True devuelve también la imagen anotada.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None

    H, W = img.shape[:2]
    class_id = _class_from_name(img_path.name)
    if class_id is None:
        return None, None

    # Recorte al interior de la caja
    roi = img[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    cnt, thresh = _best_rat_blob(gray_roi)
    if cnt is None:
        return None, None

    # Bounding box en coords del ROI → coords absolutas → normalizadas
    rx, ry, rw, rh = cv2.boundingRect(cnt)
    pad = 8
    rx = max(0, rx - pad);  ry = max(0, ry - pad)
    rw = min(roi.shape[1] - rx, rw + 2 * pad)
    rh = min(roi.shape[0] - ry, rh + 2 * pad)

    abs_cx = (ROI_X1 + rx + rw / 2) / W
    abs_cy = (ROI_Y1 + ry + rh / 2) / H
    abs_bw  = rw / W
    abs_bh  = rh / H

    # Keypoints
    snout, spine, tail = _keypoints_from_contour(cnt)

    def kp_norm(pt):
        return (ROI_X1 + pt[0]) / W, (ROI_Y1 + pt[1]) / H

    kp1 = kp_norm(snout)   # snout
    kp2 = kp_norm(spine)   # spine center
    kp3 = kp_norm(tail)    # tail base

    line = (
        f"{class_id} "
        f"{abs_cx:.6f} {abs_cy:.6f} {abs_bw:.6f} {abs_bh:.6f} "
        f"{kp1[0]:.6f} {kp1[1]:.6f} 2 "
        f"{kp2[0]:.6f} {kp2[1]:.6f} 2 "
        f"{kp3[0]:.6f} {kp3[1]:.6f} 2"
    )

    debug_img = None
    if debug:
        debug_img = img.copy()
        # Bounding box
        bx1 = int((abs_cx - abs_bw / 2) * W)
        by1 = int((abs_cy - abs_bh / 2) * H)
        bx2 = int((abs_cx + abs_bw / 2) * W)
        by2 = int((abs_cy + abs_bh / 2) * H)
        cv2.rectangle(debug_img, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
        # Keypoints
        colors = [(0, 0, 255), (0, 255, 255), (255, 0, 0)]
        labels_kp = ["snout", "spine", "tail"]
        for (kx, ky), col, lbl in zip([kp1, kp2, kp3], colors, labels_kp):
            px, py = int(kx * W), int(ky * H)
            cv2.circle(debug_img, (px, py), 6, col, -1)
            cv2.putText(debug_img, lbl, (px + 8, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        # ROI border
        cv2.rectangle(debug_img,
                      (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2),
                      (128, 128, 0), 1)
        # Class label
        cls_name = [k for k, v in CLASS_MAP.items() if v == class_id][0]
        cv2.putText(debug_img, cls_name, (bx1, by1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return line, debug_img


def run(src_dir: Path, debug: bool):
    images_dir = src_dir / "images"
    labels_dir = src_dir / "labels"
    debug_dir  = src_dir / "debug_labels"

    # Si no hay subcarpeta images/ usa src_dir directamente
    if not images_dir.exists():
        images_dir = src_dir

    labels_dir.mkdir(exist_ok=True)
    if debug:
        debug_dir.mkdir(exist_ok=True)

    imgs = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    ok = err = skip = 0

    for img_path in imgs:
        if _class_from_name(img_path.name) is None:
            skip += 1
            continue

        line, debug_img = label_image(img_path, debug=debug)

        if line is None:
            print(f"[!] sin blob: {img_path.name}")
            err += 1
            continue

        txt_path = labels_dir / f"{img_path.stem}.txt"
        txt_path.write_text(line)
        ok += 1

        if debug and debug_img is not None:
            cv2.imwrite(str(debug_dir / img_path.name), debug_img)

    print(f"\n[OK] Labels generados: {ok}  |  sin blob: {err}  |  sin clase: {skip}")
    print(f"     Labels en: {labels_dir}")
    if debug:
        print(f"     Debug imgs en: {debug_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True,
                        help="Carpeta con las imágenes (o con subcarpeta images/)")
    parser.add_argument("--debug", action="store_true",
                        help="Guarda imágenes con bbox y keypoints dibujados para verificar")
    args = parser.parse_args()
    run(Path(args.src), args.debug)


if __name__ == "__main__":
    main()
