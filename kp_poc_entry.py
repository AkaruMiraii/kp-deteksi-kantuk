"""Portable proof-of-concept entry point; detector source is unchanged."""
import sys,os,json,traceback
from pathlib import Path

def main():
    base=Path(sys.executable).parent if getattr(sys,'frozen',False) else Path(__file__).parent
    os.chdir(base)
    if '--self-test' in sys.argv:
        import cv2,mediapipe,numpy
        from mediapipe.tasks.python import BaseOptions,vision
        model=base/'models'/'face_landmarker.task'
        options=vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(model)),running_mode=vision.RunningMode.IMAGE)
        with vision.FaceLandmarker.create_from_options(options) as detector:
            frame=numpy.zeros((480,640,3),dtype=numpy.uint8)
            result=detector.detect(mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB,data=frame))
        recognizer=cv2.FaceRecognizerSF.create(str(base/'models'/'sface.onnx'),'')
        print(json.dumps({'ok':True,'landmark_loaded':True,'sface_loaded':recognizer is not None,'faces':len(result.face_landmarks),'camera_opened':False}))
        return
    from scripts.multi_live import parse_args,run
    if len(sys.argv)>1:
        run(parse_args(sys.argv[1:]));return
    import tkinter as tk
    from tkinter import ttk,messagebox
    window=tk.Tk();window.title('KP - Deteksi Indikasi Kantuk');window.geometry('470x380');window.resizable(False,False)
    frame=ttk.Frame(window,padding=22);frame.pack(fill='both',expand=True)
    ttk.Label(frame,text='Prototipe Deteksi Indikasi Kantuk',font=('Segoe UI',14,'bold')).pack(anchor='w',pady=(0,16))
    fields={}
    for label,key,default in [('Indeks kamera','camera','0'),('Kapasitas wajah (1-6)','max-people','1'),('Ambang EAR','ear-threshold','0.12'),('Durasi penutupan mata (detik)','closure-seconds','1.5'),('Ambang MAR','mar-threshold','0.50')]:
        row=ttk.Frame(frame);row.pack(fill='x',pady=4);ttk.Label(row,text=label).pack(side='left');var=tk.StringVar(value=default);ttk.Entry(row,textvariable=var,width=10).pack(side='right');fields[key]=var
    ttk.Label(frame,text='Identitas sesi aktif. Pemeriksaan keaslian tidak aktif.\nQ: keluar monitor. H: rincian. C: kalibrasi peserta.',wraplength=420).pack(anchor='w',pady=14)
    def start():
        argv=['--threshold-only','--session-identity','--enrollment-seconds','180']
        for key,var in fields.items():argv.extend(['--'+key,var.get()])
        try:args=parse_args(argv)
        except SystemExit:messagebox.showerror('Konfigurasi','Periksa nilai parameter.');return
        window.destroy();run(args)
    ttk.Button(frame,text='Mulai Kamera',command=start).pack(fill='x')
    window.mainloop()
if __name__=='__main__':
    try:main()
    except BaseException as error:
        if isinstance(error,SystemExit):raise
        base=Path(sys.executable).parent if getattr(sys,'frozen',False) else Path(__file__).parent
        (base/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
        print(traceback.format_exc(),file=sys.stderr);sys.exit(1)
