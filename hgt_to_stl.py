#!/usr/bin/env python3
"""
Convertitore HGT -> STL (superficie o solido opzionale)

Esempi:
  python hgt_to_stl.py N46E009.hgt --downsample 20 --geo-scale --units mm --z-exaggeration 1.5 --output N46E009_ds20.stl
  python hgt_to_stl.py --all --input-dir . --downsample 30 --geo-scale --units mm --output-dir exports

Note:
- Le dimensioni delle mesh HGT possono essere enormi (milioni di triangoli). Per sicurezza,
  lo script blocca l'export se > 500k triangoli, a meno di usare --allow-large.
"""
from __future__ import annotations

import argparse
import math
import os
import re
from typing import List, Optional, Tuple

import numpy as np

try:
    from stl import mesh as stl_mesh
except Exception as e:
    raise SystemExit("Dipendenza mancante: installa 'numpy-stl' (pip install numpy-stl)")


NODATA = -32768


def parse_tile_name(path: str) -> Optional[Tuple[float, float]]:
    name = os.path.basename(path)
    m = re.match(r"^([NS])(\d{1,2})([EW])(\d{1,3})\.hgt$", name, re.IGNORECASE)
    if not m:
        return None
    ns, lat, ew, lon = m.groups()
    lat_v = int(lat) * (1 if ns.upper() == 'N' else -1)
    lon_v = int(lon) * (1 if ew.upper() == 'E' else -1)
    return float(lat_v), float(lon_v)


def load_hgt(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=">i2")
    if data.size == 0:
        raise ValueError(f"File vuoto o non leggibile: {path}")
    size = int(round(math.sqrt(data.size)))
    if size * size != data.size:
        raise ValueError(
            f"Dimensione non quadrata: elementi={data.size}, radice={size}. File corrotto o non HGT?"
        )
    arr = data.reshape((size, size)).astype(np.float32)
    arr[arr == NODATA] = np.nan
    return arr


def downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor is None or factor <= 1:
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


def meters_per_degree(lat_deg: float) -> Tuple[float, float]:
    lat_rad = math.radians(lat_deg)
    m_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * lat_rad) + 1.175 * math.cos(4 * lat_rad)
    m_per_deg_lon = 111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3 * lat_rad)
    return m_per_deg_lat, m_per_deg_lon


def build_vertices(arr: np.ndarray, use_geo: bool, units: str, z_exagg: float, tile_origin: Optional[Tuple[float, float]]) -> Tuple[np.ndarray, int, int]:
    """Crea la griglia di vertici (X,Y,Z) in unità richieste.

    units: 'mm' | 'm' | 'unit'
    use_geo: se True, scala XY usando gradi -> metri usando lat media del tile.
    """
    rows, cols = arr.shape
    # Z in metri di default; converti in unità scelte
    z = arr.copy()
    # Converti XY scale
    if use_geo and tile_origin is not None:
        lat0, lon0 = tile_origin
        lat_mid = lat0 + 0.5
        m_per_deg_lat, m_per_deg_lon = meters_per_degree(lat_mid)
        deg_per_px = 1.0 / (rows - 1)  # 1 grado per tile
        dy_m = m_per_deg_lat * deg_per_px
        dx_m = m_per_deg_lon * deg_per_px
        x = np.arange(cols, dtype=np.float64) * dx_m
        y = np.arange(rows, dtype=np.float64) * dy_m
    else:
        x = np.arange(cols, dtype=np.float64)
        y = np.arange(rows, dtype=np.float64)

    X, Y = np.meshgrid(x, y)

    # Converti unità
    if units == 'mm':
        X *= 1000.0 if use_geo else 1.0
        Y *= 1000.0 if use_geo else 1.0
        z *= 1000.0  # m -> mm
    elif units == 'm':
        # X,Y già in metri se geo, altrimenti unità arbitrarie
        pass
    elif units == 'unit':
        # X,Y in px; Z in m -> rendi coerente: usa z così com'è (m) o scala a 1 unit = 1 px? Si lascia m per z
        pass
    else:
        raise ValueError("units sconosciute: usa mm|m|unit")

    z *= z_exagg

    # Sposta NaN a valori adiacenti per evitare fori isolati nella triangolazione (ma lasceremo saltare celle con NaN)
    return np.dstack((X, Y, z)).astype(np.float64), rows, cols


def grid_triangles(rows: int, cols: int, valid_mask: np.ndarray) -> np.ndarray:
    """Crea triangoli 2 per cella dove i 4 vertici sono validi (non NaN). Ritorna (n_faces, 3) indici."""
    # Celle valide
    v = valid_mask
    cell_valid = v[:-1, :-1] & v[1:, :-1] & v[:-1, 1:] & v[1:, 1:]
    ii, jj = np.where(cell_valid)
    if ii.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    # Indici dei vertici
    idx = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols)
    i0 = idx[ii, jj]
    i1 = idx[ii + 1, jj]
    i2 = idx[ii, jj + 1]
    i3 = idx[ii + 1, jj + 1]
    # Triangoli per cella: (i0, i1, i2) e (i3, i2, i1)
    tris1 = np.stack([i0, i1, i2], axis=1)
    tris2 = np.stack([i3, i2, i1], axis=1)
    return np.concatenate([tris1, tris2], axis=0)


def export_stl(vertices_xyz: np.ndarray, faces: np.ndarray, out_path: str):
    n_faces = faces.shape[0]
    m = stl_mesh.Mesh(np.zeros(n_faces, dtype=stl_mesh.Mesh.dtype))
    # numpy-stl vuole coordinate per faccia; duplichiamo i vertici per semplicità
    for i in range(n_faces):
        a, b, c = faces[i]
        m.vectors[i] = np.array([
            vertices_xyz[a], vertices_xyz[b], vertices_xyz[c]
        ], dtype=np.float64)
    m.save(out_path)


def ensure_outward_normals(vertices_xyz: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Se il mesh è orientato all'interno (volume firmato negativo), inverte il winding di tutte le facce.
    Usa la somma del prodotto misto: sum(v0 · ((v1 - v0) x (v2 - v0))) ~ 6 * Volume.
    Ritorna l'array facce (eventualmente modificato)."""
    if faces.size == 0:
        return faces
    v0 = vertices_xyz[faces[:, 0]]
    v1 = vertices_xyz[faces[:, 1]]
    v2 = vertices_xyz[faces[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    cross = np.cross(e1, e2)
    signed = (v0 * cross).sum(axis=1).sum()
    if signed < 0:
        # Flip tutte le facce
        faces = faces[:, [0, 2, 1]]
        try:
            print("[Info] Normali invertite per orientamento verso l'esterno.")
        except Exception:
            pass
    return faces


def main():
    p = argparse.ArgumentParser(description="Converte HGT in STL")
    p.add_argument("hgt_path", nargs='?', help="Percorso del file .hgt")
    p.add_argument("--output", "-o", help="Path file STL in output")
    p.add_argument("--downsample", type=int, default=20, help="Fattore di campionamento (default: 20)")
    p.add_argument("--crop", type=str, help="Ritaglio top,left,height,width in px")
    p.add_argument("--z-exaggeration", type=float, default=1.0, help="Esagerazione verticale (default 1.0)")
    p.add_argument("--geo-scale", action="store_true", help="Scala XY usando metri/px, basato su lat del tile")
    p.add_argument("--units", choices=["mm", "m", "unit"], default="mm", help="Unità STL (default: mm)")
    p.add_argument("--close", action="store_true", help="Chiudi con pareti verticali e base piana")
    p.add_argument("--base-offset", type=float, default=10.0, help="Distanza sotto la quota minima per la base (stesse unità di Z). Default: 10")
    p.add_argument("--allow-large", action="store_true", help="Permetti export > 500k triangoli")
    # Batch
    p.add_argument("--all", action="store_true", help="Elabora tutti i .hgt nella cartella")
    p.add_argument("--input-dir", type=str, help="Cartella di input (default: .)")
    p.add_argument("--recursive", action="store_true", help="Cerca ricorsivamente")
    p.add_argument("--output-dir", type=str, help="Cartella in cui salvare gli STL")
    p.add_argument("--mosaic", action="store_true", help="Crea un unico STL mosaico con tutti i .hgt trovati")
    p.add_argument("--half-merge", nargs=2, metavar=("LEFT_HGT", "RIGHT_HGT"), help="Unisci metà destra del primo e metà sinistra del secondo in un unico STL")
    p.add_argument("--bottom-half", action="store_true", help="Con --half-merge, usa solo la metà inferiore (sud) dei due tile")

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

    def build_vertices_extent(arr: np.ndarray, extent: Optional[Tuple[float, float, float, float]], use_geo: bool, units: str, z_exagg: float) -> Tuple[np.ndarray, int, int]:
        rows, cols = arr.shape
        Z = arr.copy()
        # Coordinate X,Y in gradi o pixel
        if extent is not None:
            lon_min, lon_max, lat_min, lat_max = extent
            deg_per_px_lon = (lon_max - lon_min) / max(1, (cols - 1))
            deg_per_px_lat = (lat_max - lat_min) / max(1, (rows - 1))
            # Costruisci griglie in gradi
            x_deg = lon_min + np.arange(cols, dtype=np.float64) * deg_per_px_lon
            # Y: riga 0 = lat_max, cresce verso nord (right-handed)
            y_idx = np.arange(rows, dtype=np.float64)
            y_deg = lat_min + (rows - 1 - y_idx) * deg_per_px_lat
            Xd, Yd = np.meshgrid(x_deg, y_deg)
            if use_geo:
                # Approssimazione: usa fattori alla latitudine media
                lat_mid = (lat_min + lat_max) * 0.5
                m_per_deg_lat, m_per_deg_lon = meters_per_degree(lat_mid)
                X = (Xd - lon_min) * m_per_deg_lon
                Y = (Yd - lat_min) * m_per_deg_lat
            else:
                X = np.arange(cols, dtype=np.float64)
                y_pix = (rows - 1 - np.arange(rows, dtype=np.float64))
                X, Y = np.meshgrid(X, y_pix)
        else:
            # fallback a pixel
            X = np.arange(cols, dtype=np.float64)
            Y = np.arange(rows, dtype=np.float64)
            X, Y = np.meshgrid(X, Y)

        # Unità
        if units == 'mm':
            if use_geo:
                X *= 1000.0
                Y *= 1000.0
            Z *= 1000.0
        elif units == 'm':
            # X,Y già in m se use_geo
            pass
        elif units == 'unit':
            pass
        else:
            raise ValueError("units sconosciute")
        Z *= z_exagg
        V = np.dstack((X, Y, Z)).astype(np.float64)
        return V, rows, cols

    def build_closed_mesh(vertices_xyz: np.ndarray, faces: np.ndarray, rows: int, cols: int, base_offset: float) -> Tuple[np.ndarray, np.ndarray]:
        """Aggiunge base piana e pareti perimetrali.
        Base a z = min(Z) - base_offset.
        """
        Vgrid = vertices_xyz.reshape(rows, cols, 3)
        Z = Vgrid[..., 2]
        # Base
        base_z = np.nanmin(Z) - float(base_offset)
        Vbottom = Vgrid.copy()
        Vbottom[..., 2] = base_z
        N = rows * cols
        V_all = np.vstack([vertices_xyz, Vbottom.reshape(-1, 3)])

        # Bottom faces (tutti i pixel, anche se top ha NaN)
        valid_bottom = np.ones((rows, cols), dtype=bool)
        f_bottom = grid_triangles(rows, cols, valid_bottom)
        # Flip winding per normali verso il basso
        f_bottom = f_bottom[:, [0, 2, 1]] + N

        # Pareti perimetrali solo dove i vertici top non sono NaN
        idx = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols)
        wall_faces = []
        def add_quad(t0, t1):
            b0, b1 = t0 + N, t1 + N
            wall_faces.append([t0, t1, b0])
            wall_faces.append([b0, t1, b1])

        # Top edge (row 0)
        zrow = Z[0, :]
        for j in range(cols - 1):
            if np.isfinite(zrow[j]) and np.isfinite(zrow[j + 1]):
                t0 = idx[0, j]
                t1 = idx[0, j + 1]
                add_quad(t0, t1)
        # Bottom edge (row rows-1)
        zrow = Z[-1, :]
        for j in range(cols - 1):
            if np.isfinite(zrow[j]) and np.isfinite(zrow[j + 1]):
                t1 = idx[-1, j + 1]
                t0 = idx[-1, j]
                add_quad(t1, t0)
        # Left edge (col 0)
        zcol = Z[:, 0]
        for i in range(rows - 1):
            if np.isfinite(zcol[i]) and np.isfinite(zcol[i + 1]):
                t0 = idx[i, 0]
                t1 = idx[i + 1, 0]
                add_quad(t0, t1)
        # Right edge (col cols-1)
        zcol = Z[:, -1]
        for i in range(rows - 1):
            if np.isfinite(zcol[i]) and np.isfinite(zcol[i + 1]):
                t1 = idx[i + 1, -1]
                t0 = idx[i, -1]
                add_quad(t1, t0)

        f_walls = np.array(wall_faces, dtype=np.int64) if wall_faces else np.zeros((0, 3), dtype=np.int64)

        f_all = np.vstack([faces, f_bottom, f_walls]) if faces.size else np.vstack([f_bottom, f_walls])
        return V_all, f_all

    def process_one(hgt_path: str) -> str:
        arr = load_hgt(hgt_path)
        if args.crop:
            try:
                top, left, height, width = map(int, args.crop.split(','))
            except Exception:
                raise SystemExit("--crop deve essere top,left,height,width")
            arr = crop_bounds(arr, top=top, left=left, height=height, width=width)
        arr = downsample(arr, args.downsample)

        rows, cols = arr.shape
        approx_faces = 2 * max(0, rows - 1) * max(0, cols - 1)
        if approx_faces > 500_000 and not args.allow_large:
            raise SystemExit(
                f"Mesh troppo grande (~{approx_faces:,} triangoli). Usa --downsample maggiore o --allow-large."
            )

        tile_origin = parse_tile_name(hgt_path)
        if tile_origin is not None:
            lat0, lon0 = tile_origin
            extent = (lon0, lon0 + 1, lat0, lat0 + 1)
        else:
            extent = None
        V, r, c = build_vertices_extent(arr, extent, args.geo_scale, args.units, args.z_exaggeration)

        # Mask valid vertices (non NaN)
        valid = ~np.isnan(V[..., 2])
        faces = grid_triangles(r, c, valid)
        if args.close:
            V_flat = V.reshape(-1, 3)
            V_flat, faces = build_closed_mesh(V_flat, faces, r, c, args.base_offset)
        else:
            V_flat = V.reshape(-1, 3)

        # Export
        out_dir = args.output_dir or os.path.dirname(hgt_path)
        os.makedirs(out_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(hgt_path))[0]
        out_path = args.output or os.path.join(out_dir, f"{base_name}.stl")
        faces = ensure_outward_normals(V_flat, faces)
        export_stl(V_flat, faces, out_path)
        return out_path

    def resample_nearest(arr: np.ndarray, new_shape: Tuple[int, int]) -> np.ndarray:
        r, c = arr.shape
        nr, nc = new_shape
        if (r, c) == (nr, nc):
            return arr
        row_idx = np.clip(np.rint(np.linspace(0, r - 1, nr)).astype(int), 0, r - 1)
        col_idx = np.clip(np.rint(np.linspace(0, c - 1, nc)).astype(int), 0, c - 1)
        return arr[row_idx][:, col_idx]

    def process_mosaic(files: List[str]) -> str:
        # Seleziona solo file parsabili in griglia
        tiles: List[Tuple[str, float, float]] = []
        for f in files:
            tl = parse_tile_name(f)
            if tl is None:
                print(f"[Avviso] salto file non standard: {f}")
                continue
            lat0, lon0 = tl
            tiles.append((f, lat0, lon0))
        if not tiles:
            raise SystemExit("Nessun tile valido per mosaico")

        # Carica e downsample
        arrays: List[np.ndarray] = []
        shapes: List[Tuple[int, int]] = []
        for path, _, _ in tiles:
            a = load_hgt(path)
            a = downsample(a, args.downsample)
            arrays.append(a)
            shapes.append(a.shape)

        # Uniforma le dimensioni dei tile (usa minima)
        min_rows = min(r for r, _ in shapes)
        min_cols = min(c for _, c in shapes)
        target = (min_rows, min_cols)
        arrays = [resample_nearest(a, target) for a in arrays]

        # Griglia lat/lon
        lats = sorted({int(lat) for _, lat, _ in tiles})
        lons = sorted({int(lon) for _, _, lon in tiles})
        lat_min, lat_max = min(lats), max(lats)
        lon_min, lon_max = min(lons), max(lons)
        n_lat = (lat_max - lat_min) + 1
        n_lon = (lon_max - lon_min) + 1

        tile_r, tile_c = target
        mosaic = np.full((n_lat * tile_r, n_lon * tile_c), np.nan, dtype=np.float32)

        # Posiziona: riga 0 = lat_max (nord)
        for (path, lat0, lon0), arr in zip(tiles, arrays):
            row_tile = int(round(lat_max - int(lat0)))
            col_tile = int(round(int(lon0) - lon_min))
            r0 = row_tile * tile_r
            c0 = col_tile * tile_c
            mosaic[r0:r0 + tile_r, c0:c0 + tile_c] = arr

        rows, cols = mosaic.shape
        approx_faces = 2 * max(0, rows - 1) * max(0, cols - 1)
        if approx_faces > 1_000_000 and not args.allow_large:
            raise SystemExit(
                f"Mosaico troppo grande (~{approx_faces:,} triangoli). Aumenta --downsample o usa --allow-large."
            )

        extent = (lon_min, lon_max + 1, lat_min, lat_max + 1)
        V, r, c = build_vertices_extent(mosaic, extent if args.geo_scale else None, args.geo_scale, args.units, args.z_exaggeration)
        valid = ~np.isnan(V[..., 2])
        faces = grid_triangles(r, c, valid)
        if args.close:
            V_flat = V.reshape(-1, 3)
            V_flat, faces = build_closed_mesh(V_flat, faces, r, c, args.base_offset)
        else:
            V_flat = V.reshape(-1, 3)

        out_dir = args.output_dir or (args.input_dir or ".")
        os.makedirs(out_dir, exist_ok=True)
        out_path = args.output or os.path.join(out_dir, "mosaic.stl")
        faces = ensure_outward_normals(V_flat, faces)
        export_stl(V_flat, faces, out_path)
        return out_path

    def process_half_merge(left_hgt: str, right_hgt: str) -> str:
        """Combina la metà destra del tile sinistro con la metà sinistra del tile destro (stessa latitudine)."""
        A = load_hgt(left_hgt)
        B = load_hgt(right_hgt)
        # Downsample
        A = downsample(A, args.downsample)
        B = downsample(B, args.downsample)
        # Uniforma dimensioni
        r = min(A.shape[0], B.shape[0])
        c = min(A.shape[1], B.shape[1])
        A = A[:r, :c]
        B = B[:r, :c]
        # Eventuale metà inferiore
        if args.bottom_half:
            midr = r // 2
            A = A[midr:, :]
            B = B[midr:, :]
            r = A.shape[0]
        mid = c // 2
        left_half = A[:, mid:]
        right_half = B[:, :mid]
        mosaic = np.concatenate([left_half, right_half], axis=1)

        rows, cols = mosaic.shape
        approx_faces = 2 * max(0, rows - 1) * max(0, cols - 1)
        # Usa limite del mosaico
        if approx_faces > 1_000_000 and not args.allow_large:
            raise SystemExit(
                f"Mosaico metà+metà troppo grande (~{approx_faces:,} triangoli). Aumenta --downsample o usa --allow-large."
            )

        # Extent: lon 0.5° + 0.5° centrato tra i due tile, lat dal nome
        tl = parse_tile_name(left_hgt)
        tr = parse_tile_name(right_hgt)
        extent = None
        if tl and tr:
            lat_l, lon_l = tl
            lat_r, lon_r = tr
            if int(lat_l) == int(lat_r) and int(lon_r) == int(lon_l) + 1:
                lon_min = lon_l + 0.5
                lon_max = lon_r + 0.5
                lat_min = lat_l
                lat_max = lat_l + (0.5 if args.bottom_half else 1.0)
                extent = (lon_min, lon_max, lat_min, lat_max)

        V, r, c = build_vertices_extent(mosaic, extent if args.geo_scale else None, args.geo_scale, args.units, args.z_exaggeration)
        valid = ~np.isnan(V[..., 2])
        faces = grid_triangles(r, c, valid)
        V_flat = V.reshape(-1, 3)
        if args.close:
            V_flat, faces = build_closed_mesh(V_flat, faces, r, c, args.base_offset)

        out_dir = args.output_dir or (os.path.dirname(left_hgt) or ".")
        os.makedirs(out_dir, exist_ok=True)
        base_name = f"half_{os.path.splitext(os.path.basename(left_hgt))[0]}_{os.path.splitext(os.path.basename(right_hgt))[0]}"
        out_path = args.output or os.path.join(out_dir, f"{base_name}.stl")
        faces = ensure_outward_normals(V_flat, faces)
        export_stl(V_flat, faces, out_path)
        return out_path


    # Batch / Mosaico
    if args.half_merge:
        out = process_half_merge(args.half_merge[0], args.half_merge[1])
        print(f"Salvato half-merge STL: {out}")
        return

    if args.all or args.input_dir or args.mosaic:
        base = args.input_dir or "."
        files = list_hgt_files(base, args.recursive)
        if not files:
            raise SystemExit(f"Nessun .hgt trovato in {base}")
        if args.mosaic:
            out = process_mosaic(files)
            print(f"Salvato mosaico STL: {out}")
            return
        else:
            outs = []
            for f in files:
                out = process_one(f)
                print(f"Salvato: {out}")
                outs.append(out)
            print(f"Completato: {len(outs)} STL generati.")
            return

    # Singolo
    hgt_path = args.hgt_path
    if not hgt_path:
        cands = [f for f in os.listdir('.') if f.lower().endswith('.hgt')]
        if not cands:
            raise SystemExit("Specifica un file .hgt o posiziona un .hgt nella cartella corrente")
        hgt_path = cands[0]
    out = process_one(hgt_path)
    print(f"Salvato: {out}")


if __name__ == "__main__":
    main()
