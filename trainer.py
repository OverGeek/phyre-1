import scene_extractor
import cv2

ext = scene_extractor.Extractor("rollouts/test")

data, target = ext.extract(n=10, n_in=3, n_out=3, stride=3)

#Data-Batch with Dimension n * n_in * 2
#Target-Batch with Dimension n * n_out

#Visualizing
alternate = True
for i in range(len(data)):
  for x in data[i]:
    cv2.imshow("Static", x)
    cv2.waitKey()
    
  for y in target[i]:
    cv2.imshow("Target", y)
    cv2.waitKey()

cv2.destroyAllWindows()