from pathlib import Path
from coin_counter_pipeline import list_image_files
files = list(list_image_files(Path("final_project/train_coins/50")))
print("抓到張數:", len(files))
for p in files[:5]:
    print(p)