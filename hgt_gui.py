#!/usr/bin/env python3
import os
import sys
import threading
import queue
import subprocess
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from tkinter import scrolledtext
    TK_AVAILABLE = True
    _TK_ERR = None
except Exception as e:
    TK_AVAILABLE = False
    _TK_ERR = e


HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, 'hgt_to_stl.py')


if TK_AVAILABLE:
    class HgtGui(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title('HGT → STL - GUI')
            self.geometry('920x720')
            self.minsize(860, 640)

            self.proc = None
            self.q = queue.Queue()

            self._build_ui()

        # ... unchanged rest of the class body ...
        def _build_ui(self):
            # Mode
            mode_frame = ttk.LabelFrame(self, text='Modalità')
            mode_frame.pack(fill='x', padx=10, pady=8)
            self.mode_var = tk.StringVar(value='single')
            for text, val in [('Singolo', 'single'), ('Batch', 'batch'), ('Mosaico', 'mosaic'), ('Half-merge', 'half')]:
                ttk.Radiobutton(mode_frame, text=text, value=val, variable=self.mode_var, command=self._on_mode_change).pack(side='left', padx=8, pady=6)

            # Paths
            paths = ttk.LabelFrame(self, text='Sorgenti e destinazioni')
            paths.pack(fill='x', padx=10, pady=8)

            # Single file
            self.hgt_path = tk.StringVar()
            row = ttk.Frame(paths)
            row.pack(fill='x', padx=6, pady=3)
            ttk.Label(row, text='File HGT:').pack(side='left', padx=(0,6))
            ttk.Entry(row, textvariable=self.hgt_path, width=70).pack(side='left', expand=True, fill='x')
            ttk.Button(row, text='Sfoglia…', command=lambda: self._pick_file(self.hgt_path)).pack(side='left', padx=6)

            # Batch / Mosaic dir
            self.input_dir = tk.StringVar()
            row = ttk.Frame(paths)
            row.pack(fill='x', padx=6, pady=3)
            ttk.Label(row, text='Cartella input:').pack(side='left', padx=(0,6))
            ttk.Entry(row, textvariable=self.input_dir, width=70).pack(side='left', expand=True, fill='x')
            ttk.Button(row, text='Sfoglia…', command=lambda: self._pick_dir(self.input_dir)).pack(side='left', padx=6)
            self.recursive_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(row, text='Ricorsivo', variable=self.recursive_var).pack(side='left', padx=8)

            # Half-merge files
            half = ttk.Frame(paths)
            half.pack(fill='x', padx=6, pady=3)
            self.left_hgt = tk.StringVar()
            self.right_hgt = tk.StringVar()
            ttk.Label(half, text='Half-merge sinistra:').pack(side='left', padx=(0,6))
            ttk.Entry(half, textvariable=self.left_hgt, width=32).pack(side='left', fill='x')
            ttk.Button(half, text='…', width=3, command=lambda: self._pick_file(self.left_hgt)).pack(side='left', padx=4)
            ttk.Label(half, text='destra:').pack(side='left', padx=(12,6))
            ttk.Entry(half, textvariable=self.right_hgt, width=32).pack(side='left', fill='x')
            ttk.Button(half, text='…', width=3, command=lambda: self._pick_file(self.right_hgt)).pack(side='left', padx=4)
            self.bottom_half = tk.BooleanVar(value=False)
            ttk.Checkbutton(half, text='Solo metà inferiore', variable=self.bottom_half).pack(side='left', padx=12)

            # Output
            out = ttk.Frame(paths)
            out.pack(fill='x', padx=6, pady=3)
            self.output_path = tk.StringVar()
            ttk.Label(out, text='Output file:').pack(side='left', padx=(0,6))
            ttk.Entry(out, textvariable=self.output_path, width=70).pack(side='left', expand=True, fill='x')
            ttk.Button(out, text='Sfoglia…', command=lambda: self._pick_save(self.output_path)).pack(side='left', padx=6)

            # Batch output dir
            outd = ttk.Frame(paths)
            outd.pack(fill='x', padx=6, pady=3)
            self.output_dir = tk.StringVar()
            ttk.Label(outd, text='Output dir (batch):').pack(side='left', padx=(0,6))
            ttk.Entry(outd, textvariable=self.output_dir, width=70).pack(side='left', expand=True, fill='x')
            ttk.Button(outd, text='Sfoglia…', command=lambda: self._pick_dir(self.output_dir)).pack(side='left', padx=6)

            # Options
            opts = ttk.LabelFrame(self, text='Opzioni')
            opts.pack(fill='x', padx=10, pady=8)

            grid = ttk.Frame(opts)
            grid.pack(fill='x', padx=6, pady=3)
            # downsample
            ttk.Label(grid, text='Downsample:').grid(row=0, column=0, sticky='w', padx=(0,6))
            self.downsample = tk.StringVar(value='10')
            ttk.Entry(grid, textvariable=self.downsample, width=8).grid(row=0, column=1, sticky='w')
            # geo-scale
            self.geo = tk.BooleanVar(value=True)
            ttk.Checkbutton(grid, text='Geo-scale', variable=self.geo).grid(row=0, column=2, sticky='w', padx=12)
            # units
            ttk.Label(grid, text='Units:').grid(row=0, column=3, sticky='w', padx=(12,6))
            self.units = tk.StringVar(value='m')
            ttk.Combobox(grid, textvariable=self.units, values=['mm', 'm', 'unit'], width=6, state='readonly').grid(row=0, column=4, sticky='w')
            # z-exag
            ttk.Label(grid, text='Z exaggeration:').grid(row=0, column=5, sticky='w', padx=(12,6))
            self.zex = tk.StringVar(value='1.0')
            ttk.Entry(grid, textvariable=self.zex, width=8).grid(row=0, column=6, sticky='w')
            # crop
            ttk.Label(grid, text='Crop (t,l,h,w):').grid(row=0, column=7, sticky='w', padx=(12,6))
            self.crop = tk.StringVar(value='')
            ttk.Entry(grid, textvariable=self.crop, width=16).grid(row=0, column=8, sticky='w')

            # closing and safety
            row2 = ttk.Frame(opts)
            row2.pack(fill='x', padx=6, pady=3)
            self.close = tk.BooleanVar(value=True)
            ttk.Checkbutton(row2, text='Chiudi base+pareti (--close)', variable=self.close).pack(side='left')
            ttk.Label(row2, text='Base offset:').pack(side='left', padx=(12,6))
            self.base_offset = tk.StringVar(value='10')
            ttk.Entry(row2, textvariable=self.base_offset, width=8).pack(side='left')
            ttk.Label(row2, text='Base Z (override):').pack(side='left', padx=(12,6))
            self.base_z = tk.StringVar(value='')
            ttk.Entry(row2, textvariable=self.base_z, width=10).pack(side='left')
            self.allow_large = tk.BooleanVar(value=False)
            ttk.Checkbutton(row2, text='Consenti mesh grandi (--allow-large)', variable=self.allow_large).pack(side='left', padx=12)

            # Mosaic-only
            mopts = ttk.LabelFrame(self, text='Opzioni Mosaico / Ritagli speciali')
            mopts.pack(fill='x', padx=10, pady=8)
            rowm = ttk.Frame(mopts)
            rowm.pack(fill='x', padx=6, pady=3)
            ttk.Label(rowm, text='Raggio circolare:').pack(side='left')
            self.circle_radius = tk.StringVar(value='')
            ttk.Entry(rowm, textvariable=self.circle_radius, width=12).pack(side='left', padx=6)
            self.circular_wall = tk.BooleanVar(value=False)
            ttk.Checkbutton(rowm, text='Parete circolare (richiede close)', variable=self.circular_wall).pack(side='left', padx=12)
            ttk.Label(rowm, text='Nota: con geo-scale, il raggio è nelle unità selezionate (m/mm); altrimenti in pixel.').pack(side='left', padx=12)

            # Actions
            actions = ttk.Frame(self)
            actions.pack(fill='x', padx=10, pady=8)
            ttk.Button(actions, text='Esegui', command=self.run).pack(side='left')
            ttk.Button(actions, text='Annulla', command=self.cancel).pack(side='left', padx=8)

            # Log
            logf = ttk.LabelFrame(self, text='Output')
            logf.pack(fill='both', expand=True, padx=10, pady=8)
            self.log = scrolledtext.ScrolledText(logf, height=20, wrap='word')
            self.log.pack(fill='both', expand=True)

            self._on_mode_change()

        def _pick_file(self, var: tk.StringVar):
            path = filedialog.askopenfilename(title='Seleziona file HGT', filetypes=[('HGT', '*.hgt'), ('Tutti', '*.*')])
            if path:
                var.set(path)

        def _pick_dir(self, var: tk.StringVar):
            path = filedialog.askdirectory(title='Seleziona cartella')
            if path:
                var.set(path)

        def _pick_save(self, var: tk.StringVar):
            path = filedialog.asksaveasfilename(title='Seleziona file di output', defaultextension='.stl', filetypes=[('STL', '*.stl'), ('Tutti', '*.*')])
            if path:
                var.set(path)

        def _on_mode_change(self):
            mode = self.mode_var.get()
            self.log.insert('end', f"Modalità selezionata: {mode}\n")
            self.log.see('end')

        def _build_cmd(self):
            cmd = [sys.executable, SCRIPT]
            mode = self.mode_var.get()
            def add_flag(flag, cond):
                if cond:
                    cmd.append(flag)
            def add_opt(flag, value):
                if value not in (None, ''):
                    cmd.extend([flag, str(value)])
            if mode == 'single':
                if not self.hgt_path.get():
                    raise ValueError('Seleziona un file HGT')
                cmd.append(self.hgt_path.get())
                add_opt('--output', self.output_path.get())
            elif mode == 'batch':
                add_flag('--all', True)
                add_opt('--input-dir', self.input_dir.get() or '.')
                add_flag('--recursive', self.recursive_var.get())
                add_opt('--output-dir', self.output_dir.get())
            elif mode == 'mosaic':
                add_flag('--mosaic', True)
                add_opt('--input-dir', self.input_dir.get() or '.')
                add_flag('--recursive', self.recursive_var.get())
                add_opt('--output', self.output_path.get())
                add_opt('--circle-radius', self.circle_radius.get())
                add_flag('--circular-wall', self.circular_wall.get())
            elif mode == 'half':
                if not self.left_hgt.get() or not self.right_hgt.get():
                    raise ValueError('Seleziona entrambi i file per half-merge')
                cmd.extend(['--half-merge', self.left_hgt.get(), self.right_hgt.get()])
                add_flag('--bottom-half', self.bottom_half.get())
                add_opt('--output', self.output_path.get())
            add_opt('--downsample', self.downsample.get())
            add_flag('--geo-scale', self.geo.get())
            add_opt('--units', self.units.get())
            add_opt('--z-exaggeration', self.zex.get())
            add_flag('--close', self.close.get())
            add_opt('--base-offset', self.base_offset.get())
            add_opt('--base-z', self.base_z.get())
            add_flag('--allow-large', self.allow_large.get())
            if self.crop.get():
                add_opt('--crop', self.crop.get())
            return cmd

        def run(self):
            if self.proc is not None:
                messagebox.showinfo('In esecuzione', 'Un processo è già in corso. Annulla prima di avviare un nuovo job.')
                return
            try:
                cmd = self._build_cmd()
            except Exception as e:
                messagebox.showerror('Parametri non validi', str(e))
                return
            self.log.insert('end', 'Comando: ' + ' '.join(f'"{c}"' if ' ' in c else c for c in cmd) + '\n')
            self.log.see('end')
            try:
                self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            except FileNotFoundError:
                messagebox.showerror('Errore', f'Impossibile trovare lo script: {SCRIPT}')
                self.proc = None
                return
            t = threading.Thread(target=self._pump_output, daemon=True)
            t.start()
            self.after(100, self._drain_queue)

        def _pump_output(self):
            assert self.proc is not None
            for line in self.proc.stdout:
                self.q.put(line)
            self.proc.wait()
            self.q.put(f"\n[Fine] Uscita: {self.proc.returncode}\n")

        def _drain_queue(self):
            while True:
                try:
                    line = self.q.get_nowait()
                except queue.Empty:
                    break
                self.log.insert('end', line)
                self.log.see('end')
            if self.proc and self.proc.poll() is None:
                self.after(100, self._drain_queue)
            else:
                self.proc = None

        def cancel(self):
            if self.proc and self.proc.poll() is None:
                try:
                    self.proc.terminate()
                except Exception:
                    pass
                self.log.insert('end', '\n[Annullato]\n')
                self.log.see('end')
            self.proc = None

def console_main():
    print("Tkinter non disponibile.")
    print("Avvio procedura guidata da terminale per hgt_to_stl.")
    def ask(prompt, default=None):
        s = input(f"{prompt}{' ['+str(default)+']' if default is not None else ''}: ").strip()
        return s if s else default
    def ask_bool(prompt, default=False):
        s = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
        if not s:
            return default
        return s in ('y','yes','s','si','true','1')

    modes = {'1':'single','2':'batch','3':'mosaic','4':'half'}
    print("Modalità: 1) singolo  2) batch  3) mosaic  4) half-merge")
    mode = modes.get(ask('Scegli modalità', '3'), 'mosaic')

    cmd = [sys.executable, SCRIPT]
    # Common
    downsample = ask('Downsample', '10')
    geo = ask_bool('Geo-scale', True)
    units = ask('Units (mm|m|unit)', 'm')
    zex = ask('Z exaggeration', '1.0')
    close = ask_bool('Chiudi base+pareti (--close)', True)
    base_off = ask('Base offset', '10')
    base_z = ask('Base Z (override, vuoto per calcolo da offset)', '')
    allow_large = ask_bool('Consenti mesh grandi (--allow-large)', False)
    crop = ask('Crop (t,l,h,w) oppure vuoto', '')

    if mode == 'single':
        hgt = ask('File HGT', '')
        if not hgt:
            print('File HGT richiesto.')
            return
        cmd.append(hgt)
        out = ask('Output file (.stl)', 'output.stl')
        cmd += ['--output', out]
    elif mode == 'batch':
        inp = ask('Cartella input', '.')
        cmd += ['--all', '--input-dir', inp]
        if ask_bool('Ricorsivo', True):
            cmd.append('--recursive')
        outd = ask('Cartella output', 'exports')
        cmd += ['--output-dir', outd]
    elif mode == 'mosaic':
        inp = ask('Cartella input', '.')
        cmd += ['--mosaic', '--input-dir', inp]
        if ask_bool('Ricorsivo', True):
            cmd.append('--recursive')
        out = ask('Output file (.stl)', 'mosaic.stl')
        cmd += ['--output', out]
        r = ask('Ritaglio circolare (raggio, vuoto per nessuno)', '')
        if r:
            cmd += ['--circle-radius', r]
            if ask_bool('Aggiungere parete circolare', False):
                cmd.append('--circular-wall')
    elif mode == 'half':
        left = ask('HGT sinistro', '')
        right = ask('HGT destro', '')
        if not left or not right:
            print('Entrambi i file sono richiesti.')
            return
        cmd += ['--half-merge', left, right]
        if ask_bool('Solo metà inferiore', False):
            cmd.append('--bottom-half')
        out = ask('Output file (.stl)', 'half.stl')
        cmd += ['--output', out]

    # Shared options
    if downsample:
        cmd += ['--downsample', downsample]
    if geo:
        cmd.append('--geo-scale')
    if units:
        cmd += ['--units', units]
    if zex:
        cmd += ['--z-exaggeration', zex]
    if close:
        cmd.append('--close')
    if base_off:
        cmd += ['--base-offset', base_off]
    if base_z:
        cmd += ['--base-z', base_z]
    if allow_large:
        cmd.append('--allow-large')
    if crop:
        cmd += ['--crop', crop]

    print('Comando: ' + ' '.join(cmd))
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except FileNotFoundError:
        print(f'Errore: non trovo {SCRIPT}')
        return
    for line in proc.stdout:
        print(line, end='')
    proc.wait()
    print(f"[Fine] Uscita: {proc.returncode}")


def main():
    if not os.path.exists(SCRIPT):
        print(f'Errore: non trovo hgt_to_stl.py in {HERE}', file=sys.stderr)
        sys.exit(1)
    if TK_AVAILABLE:
        app = HgtGui()
        app.mainloop()
    else:
        # Mostra una nota sintetica e avvia la console wizard
        sys.stderr.write(
            "[Avviso] Tkinter non disponibile (modulo _tkinter mancante).\n"
            "Avvio procedura guidata da terminale.\n"
            f"Dettagli: {repr(_TK_ERR)}\n\n"
        )
        console_main()


if __name__ == '__main__':
    main()
