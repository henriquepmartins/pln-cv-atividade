"""
Segmentação Watershed — separação de objetos sobrepostos

Técnica: Segmentação por região (Watershed) — Visão Computacional
Biblioteca: OpenCV + scikit-image

Pipeline:
    1. Imagem em tons de cinza
    2. Binarização Otsu (inversa) → silhueta dos objetos
    3. Operação morfológica (opening) → remove ruído
    4. Distance transform → mapa de distância até o fundo
    5. Threshold no mapa de distância → "sure foreground" (centros)
    6. connectedComponents → markers únicos por objeto
    7. cv2.watershed → linha de separação entre objetos colados
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import data

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def gerar_imagem_moedas() -> np.ndarray:
    """skimage.data.coins() é o dataset clássico para watershed."""
    cinza = data.coins()
    cinza = cv2.resize(cinza, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(cinza, cv2.COLOR_GRAY2BGR)


def segmentar(bgr: np.ndarray):
    cinza = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cinza_eq = clahe.apply(cinza)
    cinza_blur = cv2.GaussianBlur(cinza_eq, (5, 5), 0)

    _, binario = cv2.threshold(
        cinza_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binario, cv2.MORPH_OPEN, kernel, iterations=2)
    opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=3)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.45 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    desconhecido = cv2.subtract(sure_bg, sure_fg)

    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[desconhecido == 255] = 0

    markers = cv2.watershed(bgr.copy(), markers)

    resultado = bgr.copy()
    resultado[markers == -1] = [0, 0, 255]

    colorido = colorir_segmentos(markers)
    n_objetos = num_markers - 1

    return {
        "original": bgr,
        "binario": binario,
        "opening": opening,
        "dist": dist,
        "sure_fg": sure_fg,
        "resultado": resultado,
        "colorido": colorido,
        "n_objetos": n_objetos,
    }


def colorir_segmentos(markers: np.ndarray) -> np.ndarray:
    """Atribui uma cor aleatória para cada rótulo > 1."""
    h, w = markers.shape
    saida = np.zeros((h, w, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed=42)
    for label in np.unique(markers):
        if label <= 1:
            continue
        cor = rng.integers(60, 255, size=3, dtype=np.uint8)
        saida[markers == label] = cor
    saida[markers == -1] = [255, 255, 255]
    return saida


def plotar_pipeline(r: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(
        f"Pipeline Watershed — {r['n_objetos']} objetos detectados",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )

    paineis = [
        ("1. Imagem original", cv2.cvtColor(r["original"], cv2.COLOR_BGR2RGB), None),
        ("2. Binarização (Otsu)", r["binario"], "gray"),
        ("3. Opening morfológico", r["opening"], "gray"),
        ("4. Distance transform", r["dist"], "magma"),
        ("5. Sure foreground (markers)", r["sure_fg"], "gray"),
        ("6. Watershed (bordas em vermelho)", cv2.cvtColor(r["resultado"], cv2.COLOR_BGR2RGB), None),
    ]

    for ax, (titulo, img, cmap) in zip(axes.flat, paineis):
        ax.imshow(img, cmap=cmap)
        ax.set_title(titulo, fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    p = OUTPUT_DIR / "pipeline.png"
    plt.savefig(p, dpi=140, bbox_inches="tight")
    print(f"Pipeline salvo em {p}")


def plotar_antes_depois(r: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].imshow(cv2.cvtColor(r["original"], cv2.COLOR_BGR2RGB))
    axes[0].set_title("Antes", fontsize=13)
    axes[0].axis("off")

    overlay = cv2.cvtColor(r["original"], cv2.COLOR_BGR2RGB).copy()
    mask = r["colorido"].sum(axis=2) > 0
    overlay[mask] = (0.55 * overlay[mask] + 0.45 * r["colorido"][mask]).astype(np.uint8)
    axes[1].imshow(overlay)
    axes[1].set_title(
        f"Depois — {r['n_objetos']} objetos segmentados",
        fontsize=13,
    )
    axes[1].axis("off")

    plt.tight_layout()
    p = OUTPUT_DIR / "resultado_final.png"
    plt.savefig(p, dpi=140, bbox_inches="tight")
    print(f"Antes/Depois salvo em {p}")


def main() -> None:
    bgr = gerar_imagem_moedas()
    print("Executando pipeline watershed…")
    r = segmentar(bgr)
    print(f"\n{r['n_objetos']} objetos detectados.")
    plotar_pipeline(r)
    plotar_antes_depois(r)


if __name__ == "__main__":
    main()
