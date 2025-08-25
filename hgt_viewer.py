#!/usr/bin/env python3
"""
Visualizzatore di file SRTM .hgt

Funzionalità:
- Carica un file .hgt (SRTM1/3) come matrice di elevazione.
- Gestione valori nodata (-32768) come NaN.
- Visualizzazione con colormap (default: 'terrain') e opzionale hillshade.
- Estensione geografica stimata dal nome file (es. N46E009) per assi in gradi.
- Opzioni CLI: output PNG, downsample, crop e colormap.

Uso rapido:
  python hgt_viewer.py N46E009.hgt --output preview.png --hillshade --downsample 2
"""
from __future__ import annotations

import argparse
import math
import os
import re
from typing import Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource


NODATA = -32768


def parse_tile_name(path: str) -> Optional[Tuple[float, float]]:
    """Estrae (lat0, lon0) dal nome file HGT, es. N46E009.hgt -> (46, 9).

    Ritorna None se parsing non riuscito.
    """
    name = os.path.basename(path)
    m = re.match(r"^([NS])(\d{1,2})([EW])(\d{1,3})\.hgt$", name, re.IGNORECASE)
    if not m:
        return None
    ns, lat, ew, lon = m.groups()
    lat_v = int(lat) * (1 if ns.upper() == 'N' else -1)
    lon_v = int(lon) * (1 if ew.upper() == 'E' else -1)
    return float(lat_v), float(lon_v)


def load_hgt(path: str) -> np.ndarray:
    """Carica un file .hgt in un array 2D (float32), sostituendo NODATA con NaN.

    - I dati sono big-endian int16.
    - La dimensione viene dedotta dalla lunghezza del file (quadrata: 1201 o 3601 per tipico SRTM).
    - La prima riga corrisponde al bordo Nord (origine in alto).
    """
    data = np.fromfile(path, dtype=">i2")
    if data.size == 0:
        raise ValueError(f"File vuoto o non leggibile: {path}")

    size = int(round(math.sqrt(data.size)))
    if size * size != data.size:
        raise ValueError(
            f"Dimensione non quadrata: elementi={data.size}, radice={size}. File corrotto o non HGT?"
        )
    arr = data.reshape((size, size))
    arr = arr.astype(np.float32)
    arr[arr == NODATA] = np.nan
    return arr


def downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return arr
    return arr[::factor, ::factor]


def crop_bounds(arr: np.ndarray, top: int = 0, left: int = 0, height: Optional[int] = None, width: Optional[int] = None) -> np.ndarray:
    h, w = arr.shape
    if height is None:
        height = h - top
    if width is None:
        width = w - left
    bottom = min(h, top + height)
    right = min(w, left + width)
    if top < 0 or left < 0 or bottom <= top or right <= left:
        raise ValueError("Parametri crop non validi")
    return arr[top:bottom, left:right]


def hillshade(arr: np.ndarray, azdeg: float = 315, altdeg: float = 45, vert_exag: float = 1.0) -> np.ndarray:
    """Crea una mappa hillshade normalizzata [0,1] usando LightSource di Matplotlib."""
    ls = LightSource(azdeg=azdeg, altdeg=altdeg)
    # Sostituisci NaN con media locale per evitare artefatti nella pendenza.
    filled = np.where(np.isnan(arr), np.nanmean(arr), arr)
    shaded = ls.hillshade(filled, vert_exag=vert_exag)
    return shaded


def plot_elevation(
    arr: np.ndarray,
    extent: Optional[Tuple[float, float, float, float]] = None,
    cmap: str = "terrain",
    use_hillshade: bool = False,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = (8, 8),
    interpolation: str = "nearest",
):
    # Gestione figsize None con default sicuro
    fig, ax = plt.subplots(figsize=figsize or (8, 8))

    if use_hillshade:
        hs = hillshade(arr)
        ax.imshow(hs, cmap="gray", extent=extent, origin="upper", interpolation=interpolation)
        im = ax.imshow(
            arr,
            cmap=cmap,
            extent=extent,
            origin="upper",
            alpha=0.6,
            interpolation=interpolation,
        )
    else:
        im = ax.imshow(arr, cmap=cmap, extent=extent, origin="upper", interpolation=interpolation)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Elevazione (m)")

    if extent is not None:
        ax.set_xlabel("Longitudine (°)")
        ax.set_ylabel("Latitudine (°)")
    else:
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")

    if title:
        ax.set_title(title)

    ax.set_aspect('equal')
    fig.tight_layout()
    return fig, ax


def main():
    p = argparse.ArgumentParser(description="Visualizza un file SRTM .hgt")
    p.add_argument("hgt_path", nargs='?', help="Percorso al file .hgt (es. N46E009.hgt)")
    p.add_argument("--output", "-o", help="Salva l'immagine in PNG/SVG invece di mostrare la finestra")
    p.add_argument("--cmap", default="terrain", help="Colormap Matplotlib (default: terrain)")
    p.add_argument("--hillshade", action="store_true", help="Sovrappone hillshade per effetto rilievo")
    p.add_argument("--downsample", type=int, default=1, help="Fattore di campionamento (>=1)")
    p.add_argument("--crop", type=str, help="Ritaglio top,left,height,width (px), es. 0,0,1200,1200")
    p.add_argument("--no-extent", action="store_true", help="Non usare estensione geografica (usa pixel)")
    # Opzioni qualità/resa
    p.add_argument("--dpi", type=int, default=200, help="DPI per il salvataggio (default: 200)")
    p.add_argument(
        "--figsize",
        type=str,
        help="Dimensioni figura in pollici larghezza,altezza (es. 10,10)",
    )
    p.add_argument(
        "--interpolation",
        default="nearest",
        choices=["nearest", "none", "bilinear", "bicubic", "lanczos"],
        help="Interpolazione per l'immagine (default: nearest)",
    )
    # Opzioni batch
    p.add_argument("--all", action="store_true", help="Elabora tutti i file .hgt nella cartella di input")
    p.add_argument("--input-dir", type=str, help="Cartella di input da cui leggere i .hgt (default: .)")
    p.add_argument("--recursive", action="store_true", help="Cerca .hgt ricorsivamente nelle sottocartelle")
    p.add_argument("--output-dir", type=str, help="Cartella in cui salvare le immagini (default: accanto al .hgt)")
    p.add_argument("--suffix", type=str, default="", help="Suffisso da aggiungere al nome file in output (prima dell'estensione)")
    p.add_argument("--mosaic", action="store_true", help="Crea un'unica immagine con tutti i .hgt trovati")
    # Logging
    p.add_argument("--verbose", action="store_true", help="Mostra messaggi dettagliati di avanzamento")
    p.add_argument("--quiet", action="store_true", help="Riduce al minimo l'output (sovrascrive --verbose)")

    args = p.parse_args()

    def list_hgt_files(base_dir: str, recursive: bool) -> List[str]:
        base_dir = base_dir or "."
        files: List[str] = []
        if recursive:
            for root, _, fnames in os.walk(base_dir):
                for f in fnames:
                    if f.lower().endswith('.hgt'):
                        files.append(os.path.join(root, f))
        else:
            for f in os.listdir(base_dir):
                if f.lower().endswith('.hgt'):
                    files.append(os.path.join(base_dir, f))
        return sorted(files)

    def build_output_path(hgt_path: str) -> str:
        if args.output and not args.all and not args.input_dir:
            # Caso singolo: usa il path esplicito passato con --output
            return args.output
        out_dir = args.output_dir or os.path.dirname(hgt_path)
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(hgt_path))[0]
        return os.path.join(out_dir, f"{base}{args.suffix}.png")

    def resample_nearest(arr: np.ndarray, new_shape: Tuple[int, int]) -> np.ndarray:
        """Ridimensiona con nearest-neighbor a new_shape (rows, cols) senza dipendenze extra."""
        r, c = arr.shape
        nr, nc = new_shape
        if (r, c) == (nr, nc):
            return arr
        row_idx = np.clip(np.rint(np.linspace(0, r - 1, nr)).astype(int), 0, r - 1)
        col_idx = np.clip(np.rint(np.linspace(0, c - 1, nc)).astype(int), 0, c - 1)
        return arr[row_idx][:, col_idx]

    def log(msg: str, level: int = 1):
        if not args.quiet and (args.verbose or level <= 1):
            print(msg, flush=True)

    def process_mosaic(files: List[str]) -> str:
        # Filtra files con nomi parsabili
        tiles: List[Tuple[str, float, float]] = []  # (path, lat0, lon0)
        log(f"Analizzo {len(files)} file per mosaico...")
        for f in files:
            tl = parse_tile_name(f)
            if tl is None:
                log(f"[Avviso] Salto file senza nome standard HGT: {f}")
                continue
            lat0, lon0 = tl
            tiles.append((f, lat0, lon0))
        if not tiles:
            raise ValueError("Nessun file HGT con nome parsabile per mosaico")

        # Carica una volta per ottenere dimensioni e target comune
        sizes: List[Tuple[int, int]] = []
        arrays: List[np.ndarray] = []
        log(f"Carico {len(tiles)} tile...")
        for i, (path, _, _) in enumerate(tiles):
            log(f"[{i+1}/{len(tiles)}] Carico {os.path.basename(path)}", level=2)
            a = load_hgt(path)
            if args.downsample and args.downsample > 1:
                a = downsample(a, args.downsample)
            arrays.append(a)
            sizes.append(a.shape)

        # Target shape per tile (usa la più piccola per evitare upsampling)
        min_rows = min(s[0] for s in sizes)
        min_cols = min(s[1] for s in sizes)
        target_shape = (min_rows, min_cols)
        log(f"Allineo i tile alla dimensione comune {target_shape}...")
        arrays = [resample_nearest(a, target_shape) for a in arrays]

        # Bounding box in gradi interi
        lats = sorted({int(lat) for _, lat, _ in tiles})
        lons = sorted({int(lon) for _, _, lon in tiles})
        lat_min, lat_max = min(lats), max(lats)
        lon_min, lon_max = min(lons), max(lons)
        n_lat = (lat_max - lat_min) + 1
        n_lon = (lon_max - lon_min) + 1

        tile_rows, tile_cols = target_shape
        mosaic = np.full((n_lat * tile_rows, n_lon * tile_cols), np.nan, dtype=np.float32)

        # Posiziona ogni tile. Riga 0 = nord (lat più alta)
        log("Compongo il mosaico...")
        for (path, lat0, lon0), arr_local in zip(tiles, arrays):
            row_tile = int(round(lat_max - int(lat0)))  # 0 in alto per lat_max
            col_tile = int(round(int(lon0) - lon_min))
            r0 = row_tile * tile_rows
            c0 = col_tile * tile_cols
            mosaic[r0:r0 + tile_rows, c0:c0 + tile_cols] = arr_local

        extent = (lon_min, lon_max + 1, lat_min, lat_max + 1)

        # Figura e interpolazione
        figsize_local = None
        if args.figsize:
            try:
                w, h = map(float, re.split(r"[xX,]", args.figsize))
                figsize_local = (w, h)
            except Exception:
                p.error("--figsize deve essere nel formato L,H (es. 10,10)")

        fig, _ = plot_elevation(
            mosaic,
            extent=extent,
            cmap=args.cmap,
            use_hillshade=args.hillshade,
            title=f"Mosaico HGT: {n_lat}x{n_lon} tiles",
            figsize=figsize_local,
            interpolation=args.interpolation,
        )

        # Path output
        if args.output:
            out_path = args.output
        else:
            out_dir = args.output_dir or (args.input_dir or ".")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"mosaic{args.suffix}.png")

        log("Renderizzo e salvo il mosaico...")
        fig.savefig(out_path, dpi=args.dpi)
        plt.close(fig)
        return out_path

    def process_one(hgt_path: str, idx: Optional[int] = None, total: Optional[int] = None) -> str:
        if not os.path.isfile(hgt_path):
            raise FileNotFoundError(f"File non trovato: {hgt_path}")

        tag = f"[{idx}/{total}] " if (idx is not None and total is not None) else ""
        log(f"{tag}Carico {os.path.basename(hgt_path)}...")
        arr_local = load_hgt(hgt_path)

        if args.crop:
            try:
                top, left, height, width = map(int, args.crop.split(','))
            except Exception:
                p.error("--crop deve essere nel formato top,left,height,width")
            arr_local = crop_bounds(arr_local, top=top, left=left, height=height, width=width)

        if args.downsample and args.downsample > 1:
            arr_local = downsample(arr_local, args.downsample)

        extent_local = None
        title_local = os.path.basename(hgt_path)
        if not args.no_extent:
            tl = parse_tile_name(hgt_path)
            if tl is not None:
                lat0, lon0 = tl
                extent_local = (lon0, lon0 + 1, lat0, lat0 + 1)
                title_local += f"  [{lat0:.0f}° to {lat0+1:.0f}°, {lon0:.0f}° to {lon0+1:.0f}°]"

        # Figura e interpolazione
        figsize_local = None
        if args.figsize:
            try:
                w, h = map(float, re.split(r"[xX,]", args.figsize))
                figsize_local = (w, h)
            except Exception:
                p.error("--figsize deve essere nel formato L,H (es. 10,10)")

        fig, _ = plot_elevation(
            arr_local,
            extent=extent_local,
            cmap=args.cmap,
            use_hillshade=args.hillshade,
            title=title_local,
            figsize=figsize_local,
            interpolation=args.interpolation,
        )

        out_path = build_output_path(hgt_path)
        log(f"{tag}Salvo in {out_path}...")
        fig.savefig(out_path, dpi=args.dpi)
        plt.close(fig)
        return out_path

    # Modalità batch o mosaico
    if args.all or args.input_dir or args.mosaic:
        base = args.input_dir or "."
        files = list_hgt_files(base, args.recursive)
        if not files:
            p.error(f"Nessun .hgt trovato in {base} {'(ricorsivo)' if args.recursive else ''}")
        log(f"Trovati {len(files)} file .hgt in {base} {'(ricorsivo)' if args.recursive else ''}")
        if args.mosaic:
            out = process_mosaic(files)
            print(f"Salvato mosaico: {out}")
            return
        else:
            if args.output and files:
                print("[Avviso] Ignoro --output in modalità batch; usare --output-dir per specificare la cartella di destinazione.")
            outputs = []
            total = len(files)
            for i, f in enumerate(files, start=1):
                out = process_one(f, idx=i, total=total)
                print(f"Salvato: {out}")
                outputs.append(out)
            print(f"Completato: {len(outputs)} immagini generate.")
            return

    # Modalità singolo file (compatibile con comportamento precedente)
    hgt_path = args.hgt_path
    if not hgt_path:
        candidates = [f for f in os.listdir('.') if f.lower().endswith('.hgt')]
        if not candidates:
            p.error("Specifica un file .hgt o posiziona un .hgt nella cartella corrente")
        hgt_path = candidates[0]

    out = process_one(hgt_path)
    if args.output:
        print(f"Salvato: {out}")
    else:
        # In assenza di --output, mostra a schermo la figura del singolo file
        plt.show()


if __name__ == "__main__":
    main()
