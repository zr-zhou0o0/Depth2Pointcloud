from matplotlib import pyplot as plt
import numpy as np
import gzip


path = "data/raw/mask/000588_mask_003000.npy.gz"

with gzip.open(path, "rb") as f:
    image = np.load(f)
    
print(image.shape)
for i in range(image.shape[0]):
    # print(image[i][0:10])
    print(image[i])

plt.imshow(image, cmap='gray')
plt.savefig("depth_image.png")