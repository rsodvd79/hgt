# Strumenti HGT: visualizzazione e conversione STL

Raccolta di script per lavorare con file SRTM `.hgt`:
- `hgt_viewer.py`: genera immagini PNG delle altimetrie, con colormap e hillshade.
- `hgt_to_stl.py`: converte le altimetrie in mesh STL (anche mosaico, con base chiusa e pareti).

## Requisiti

- Python 3.9+
- Librerie: `numpy`, `matplotlib`, `numpy-stl`
- Installa dipendenze: vedi `requirements.txt`.

## Installazione rapida

```
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Visualizzazione PNG — `hgt_viewer.py`

Esempi di base:
- Apri in finestra con hillshade:
	```
	python hgt_viewer.py N46E009.hgt --hillshade
	```
- Salva PNG ad alta definizione:
	```
	python hgt_viewer.py N46E009.hgt --hillshade --dpi 400 --figsize 16,12 --interpolation nearest --output exports/N46E009_hd.png
	```
- Ritaglio in pixel: `--crop top,left,height,width`:
	```
	python hgt_viewer.py N46E009.hgt --crop 0,0,1200,1200 --output exports/ritaglio.png
	```
- Disattiva assi geografici (usa pixel):
	```
	python hgt_viewer.py N46E009.hgt --no-extent
	```

Opzioni principali:
- `--hillshade` abilita ombreggiatura di rilievo.
- `--cmap <nome>` colormap (es. `terrain`, `viridis`).
- `--downsample <N>` campiona ogni N pixel (velocizza e riduce memoria).
- `--dpi <D>` e `--figsize L,H` controllano qualità e dimensione dell’output.
- `--interpolation nearest|none|bilinear|bicubic|lanczos` per il rendering dei pixel.

Batch su cartelle:
```
python hgt_viewer.py --all --input-dir HGT --hillshade --dpi 300 --output-dir exports
```

Mosaico unico (tutti i tile in un’unica immagine):
```
python hgt_viewer.py --mosaic --input-dir HGT --hillshade --dpi 400 --figsize 16,12 --output-dir exports
```

## Conversione in STL — `hgt_to_stl.py`

Singolo file, scala geografica in millimetri, esagerazione verticale e chiusura base/pareti:
```
python hgt_to_stl.py HGT/N46E009.hgt \
	--downsample 40 --geo-scale --units mm --z-exaggeration 1.2 \
	--close --base-offset 10 \
	--output exports/N46E009_closed_ds40.stl
```

Opzioni principali:
- `--downsample <N>`: riduce il numero di triangoli (2 per cella). Aumenta per mesh più leggera.
- `--geo-scale`: scala XY in metri (o mm) in base alla latitudine del tile (senza, XY sono in pixel).
- `--units mm|m|unit`: unità del modello; Z è sempre l’elevazione (m), scalata secondo `--units`.
- `--z-exaggeration <F>`: fattore di esagerazione verticale.
- `--crop top,left,height,width`: ritaglio in pixel.
- `--close`: chiude il modello con base piana e pareti verticali.
- `--base-offset 10`: pone la base a min(Z) - 10 (stesse unità di Z: es. mm se `--units mm`).
- `--allow-large`: permette l’export oltre le soglie di sicurezza (mesh molto pesanti).

Batch su cartelle (uno STL per file):
```
python hgt_to_stl.py --all --input-dir HGT --downsample 40 --geo-scale --units mm --output-dir exports
```

Mosaico STL unico (tutti i tile in una sola mesh):
```
python hgt_to_stl.py --mosaic --input-dir HGT \
	--downsample 50 --geo-scale --units mm --z-exaggeration 1.2 \
	--close --base-offset 10 \
	--output exports/mosaic_closed_ds50.stl
```

Half‑merge (mezzo + mezzo, 0.5° + 0.5°):
- Metà destra del tile sinistro + metà sinistra del tile destro (stessa latitudine, longitudes adiacenti):
```
python hgt_to_stl.py --half-merge HGT/N45E007.hgt HGT/N45E008.hgt \
	--downsample 8 --geo-scale --units mm --z-exaggeration 5.0 \
	--close --base-offset 10 \
	--output exports/half_N45E007_N45E008_ds8.stl
```
- Solo la metà inferiore (sud) dei due tile, poi unione metà destra/sinistra:
```
python hgt_to_stl.py --half-merge HGT/N45E007.hgt HGT/N45E008.hgt \
	--bottom-half \
	--downsample 8 --geo-scale --units mm --z-exaggeration 5.0 \
	--close --base-offset 10 \
	--output exports/half_bottom_N45E007_N45E008_ds8.stl
```
Suggerimenti:
- Riduci `--downsample` (es. 6, 4, 2, 1) per più dettaglio, tenendo conto delle soglie di sicurezza sui triangoli.
- Usa `--allow-large` solo se sei consapevole di tempi/memoria elevati.

Limiti di sicurezza e performance:
- Lo script blocca l’export se il numero stimato di triangoli supera: ~500k (singolo) / ~1M (mosaico).
- Usa `--downsample` per ridurre la complessità oppure `--allow-large` consapevolmente.

## Note

- I pixel con valore `-32768` (NoData) sono trattati come NaN e non triangolati.
- L’estensione geografica (assi in gradi) viene derivata dal nome del file standard (es. `N46E009.hgt` copre 46–47°N e 9–10°E).
- Con `--geo-scale`, XY sono in metri (o mm) secondo `--units`; senza, sono in pixel.
- Per immagini molto grandi, combina `--downsample`, `--dpi` e `--figsize` per bilanciare qualità e tempo/memoria.
 - Le mesh STL esportate applicano automaticamente l’orientamento delle normali verso l’esterno; l’asse Y è orientato a nord (X est, Y nord, Z su).

## Riferimenti

- Dataset e tile HGT (SRTM/DEM) reperibili su: https://sonny.4lima.de