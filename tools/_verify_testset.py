import os, sys, soundfile as sf

root = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\jerry\Downloads\Testset\Open-Unmix_umxhq"
songs = sorted(os.listdir(root))
songs = [s for s in songs if os.path.isdir(os.path.join(root, s))]
print(f"{'song':<42}{'stem':<9}{'dur_s':>8}{'sr':>7}{'ch':>4}{'MB':>8}")
ok = True
for s in songs:
    for stem in ["vocals", "drums", "bass", "other"]:
        p = os.path.join(root, s, stem + ".wav")
        if not os.path.exists(p):
            print(f"{s[:40]:<42}{stem:<9}   MISSING"); ok = False; continue
        i = sf.info(p)
        print(f"{s[:40]:<42}{stem:<9}{i.duration:>8.1f}{i.samplerate:>7}{i.channels:>4}"
              f"{os.path.getsize(p)/1e6:>8.1f}")
    # 对比 mixture 时长
    mp = os.path.join(r"C:\Users\jerry\Downloads\Testset\DATA", s, "mixture.wav")
    if os.path.exists(mp):
        mi = sf.info(mp)
        vi = sf.info(os.path.join(root, s, "vocals.wav"))
        delta = abs(mi.duration - vi.duration)
        flag = "OK" if delta < 0.05 else f"MISMATCH {delta:.2f}s"
        print(f"{'':<42}vs mixture {mi.duration:.1f}s -> {flag}")
        if delta >= 0.05:
            ok = False
print("\nALL OK" if ok else "\n有异常")
