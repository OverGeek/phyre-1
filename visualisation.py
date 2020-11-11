import os as os
import numpy as np
import torch as T
import torch.nn.functional as F
from PIL import ImageDraw, Image, ImageFont

text = ['initial\nscene', 'base\npaths', 'target1\npaths', 'target2\npaths', 'action1\npaths',
        'action2\npaths', 'action\nballs', 'injected\nscene', 'GT\nscene']

for file_name in os.listdir('./image_matrices_pyramid'):
    data1 = T.load(os.path.join('./image_matrices_pyramid', file_name))
    data2 = T.load(os.path.join('./image_matrices_skip_pyramid', file_name))

    padded = T.cat([data1[:, :3], data2[:, 2].unsqueeze(1), data1[:, 3].unsqueeze(1),
                  data2[:, 3].unsqueeze(1), data1[:, 4:]], dim=1)
    reshaped = T.cat([T.cat([channels for channels in sample], dim=1) for sample in padded], dim=0)
    # print(reshaped.shape)
    if np.max(reshaped.numpy()) > 1.0:
        reshaped = reshaped / 256

    if text:
        if text:
            text_height = 40
        else:
            text_height = 0

        if len(reshaped.shape) == 2:
            reshaped = F.pad(reshaped, (0, 0, text_height, 0), value=1)
            img = Image.fromarray(np.uint8(reshaped.numpy() * 255), mode="L")
        elif len(reshaped.shape) == 3:
            reshaped = F.pad(reshaped, (0, 0, 0, 0, text_height, 0), value=1)
            img = Image.fromarray(np.uint8(reshaped.numpy() * 255))
        # font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", font_size)
        draw = ImageDraw.Draw(img)

        for i, words in enumerate(text):
            x, y = 0 + i * (reshaped.shape[1] - 0 - 0) // len(text), 0
            draw.text((x, y), words, fill=(0) if len(reshaped.shape) == 2 else (0, 0, 0))

        img.save(os.path.join('./visual_comparison', file_name[:-4]+'.png'))
