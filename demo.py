import io
import cv2
from PIL import Image
from depth_refiner import DepthPredictor

pil_img = Image.open('raw_depth0195.png')
d_predictor = DepthPredictor(gpu_id=0)

e1 = cv2.getTickCount()
res = d_predictor(pil_img)
e2 = cv2.getTickCount()

t = (e2 - e1)/cv2.getTickFrequency()
print('time elapsed:', t)

cv2.imwrite('result.png', res)
