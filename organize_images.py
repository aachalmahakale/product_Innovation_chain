import os, glob, shutil
os.makedirs("images", exist_ok=True)

for img in glob.glob("*.png"):
    shutil.move(img, os.path.join("images", img))

print(f"Moved {len(glob.glob('images/*.png'))} images into ./images/")
