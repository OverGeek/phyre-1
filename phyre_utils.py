import numpy as np
import matplotlib.pyplot as plt
import torch as T
import torch.nn.functional as F
from phyre_rolllout_collector import load_phyre_rollouts, collect_solving_observations, collect_solving_dataset, \
    collect_solving_dataset_paths
import cv2
# import phyre
import os
import pickle
import random
import json
import gzip
from PIL import ImageDraw, Image, ImageFont


def make_dual_dataset(path, size=(32, 32), save=True):
    if os.path.exists(path + ".pickle"):
        with open(path + '.pickle', 'rb') as fhandle:
            X, Y = pickle.load(fhandle)
    else:
        X = load_phyre_rollouts(path)
        X, Y = prepare_data(X, size)
        X = T.tensor(X).float()
        Y = T.tensor(Y).float()
        if save:
            with open(path + '.pickle', 'wb') as fhandle:
                pickle.dump((X, Y), fhandle)
    dataloader = T.utils.data.DataLoader(T.utils.data.TensorDataset(X, Y), 32, shuffle=True)
    return dataloader


def make_mono_dataset_old(path, size=(32, 32), save=True, tasks=[], shuffle=True):
    if os.path.exists(path + ".pickle") and os.path.exists(path + "_index.pickle"):
        X = T.load(path + '.pickle')
        index = T.load(path + '_index.pickle')
        print(f"Loaded dataset from {path} with shape:", X.shape)
    else:
        if tasks:
            collect_solving_observations(path, tasks, n_per_task=1, stride=5, size=size)
        data_generator = load_phyre_rollout_data(path)
        data, index = format_raw_rollout_data(data_generator, size=size)
        X = T.tensor(data).float()
        if save:
            T.save(X, path + '.pickle')
            T.save(index, path + '_index.pickle')
    dataloader = T.utils.data.DataLoader(T.utils.data.TensorDataset(X), 32, shuffle=shuffle)
    return dataloader, index


def make_mono_dataset(path, size=(32, 32), tasks=[], batch_size=32, solving=True, n_per_task=1, shuffle=True,
                      proposal_dict=None, dijkstra=False):
    if os.path.exists(path + "/data.pickle") and os.path.exists(path + "/index.pickle"):
        try:
            with gzip.open(path + '/data.pickle', 'rb') as fp:
                data = pickle.load(fp)
                import random
                random.seed(42)
                random.shuffle(data)

                X = T.tensor(data).float()
        except OSError as e:
            print("WARNING still unzipped data file at", path)
            with open(path + '/data.pickle', 'rb') as fp:
                data = pickle.load(fp)
                X = T.tensor(data).float()
        with open(path + '/index.pickle', 'rb') as fp:
            index = pickle.load(fp)
        print(f"Loaded dataset from {path} with shape:", X.shape)
    else:
        if tasks:
            collect_solving_dataset_paths(path, tasks, n_per_task=n_per_task, stride=5, size=size, solving=solving,
                                          proposal_dict=proposal_dict, dijkstra=dijkstra)
        with gzip.open(path + '/data.pickle', 'rb') as fp:
            data = pickle.load(fp)
        with open(path + '/index.pickle', 'rb') as fp:
            index = pickle.load(fp)
        X = T.tensor(data).float()
        print(f"Loaded dataset from {path} with shape:", X.shape)

    X = X / 255  # correct for uint8 encoding
    dataloader = T.utils.data.DataLoader(T.utils.data.TensorDataset(X), batch_size, shuffle=False)
    return dataloader, index


def make_mono_dataset_2(path, size=(32, 32), tasks=[], batch_size=32, solving=True, n_per_task=1, shuffle=True,
                        proposal_dict=None, dijkstra=False, save=True):

    if os.path.exists(path+'./data.pickle'):
        with gzip.open(path + '/data.pickle', 'rb') as fp:
            data = pickle.load(fp)
    else:
        data = collect_solving_dataset_paths(path, tasks, n_per_task=n_per_task, stride=5, size=size, solving=solving,
                                          proposal_dict=proposal_dict, dijkstra=dijkstra, save=save)

    if not save:
        return data


def shrink_data(path):
    for folder in os.listdir(path):
        if folder.__contains__("64xy"):
            print("loading:", folder)
            try:
                with open(path + '/' + folder + '/data.pickle', 'rb') as fp:
                    data = pickle.load(fp)
                    data = (np.array(data) * 255).astype(np.uint8)
                with gzip.GzipFile(path + '/' + folder + '/data.pickle', 'wb') as fp:
                    pickle.dump(data, fp)
            except Exception as e:
                print(f"error loading {folder}:\n{e}")
            finally:
                print(folder, "finished")


def invert_bg(X, black_channel=None):
    white_bg = np.ones_like(X)
    white_bg[:, :, [0, 1]] -= np.repeat(X[:, :, None, 2], 2, -1)
    white_bg[:, :, [0, 2]] -= np.repeat(X[:, :, None, 1], 2, -1)
    white_bg[:, :, [1, 2]] -= np.repeat(X[:, :, None, 0], 2, -1)

    if black_channel is not None:
        for h in range(black_channel.shape[0]):
            for w in range(black_channel.shape[1]):
                if black_channel[h][w] == 1.:
                    white_bg[h, w, :] = 0.

    else:
        for h in range(white_bg.shape[0]):
            for w in range(white_bg.shape[1]):
                if X[h, w, 2] == 1.:
                    white_bg[h, w, :] = 0.

    return white_bg


def vis_pred_path_task(batch_images, save_dir, pic_id):
    num_rows = len(batch_images)
    h = batch_images[0][0].shape[0]
    w = batch_images[0][0].shape[1]
    sep = 2

    num_cols = 0
    for row in batch_images:
        num_cols = max(num_cols, len(row))

    grid = np.ones((h * num_rows + sep * (num_rows - 1), w * num_cols + sep * (num_cols - 1), 3)) / 2.

    h_start = 0
    h_end = h
    for row in batch_images:
        w_start = 0
        w_end = w
        for image in row:
            if image.shape[-1] == 3:
                image = invert_bg(image)
            else:
                image = invert_bg(image[:, :, :3], image[:, :, 3])

            grid[h_start: h_end, w_start: w_end, :] = image

            w_start += (w + sep)
            w_end += (w + sep)

        h_start += (h + sep)
        h_end += (h + sep)

    plt.imsave(os.path.join(save_dir, str(pic_id) + '.png'), grid)


def vis_pred_path(batch_images, save_dir, pic_id):
    num_pairs = batch_images[0].shape[0]
    h = batch_images[0][0].shape[0]
    w = batch_images[0][0].shape[1]
    sep = 2
    grid = np.zeros((h * num_pairs + sep * (num_pairs - 1), w * 2 + sep, 3))

    h_start = w_start = 0
    h_end = h
    w_end = w
    for gt, pred in zip(batch_images[0], batch_images[1]):
        gt = invert_bg(gt)
        pred = invert_bg(pred)
        grid[h_start: h_end, w_start: w_end, :] = gt
        grid[h_start: h_end, w_start + w + sep: w_end + w + sep] = pred
        h_start += (h + sep)
        h_end += (h + sep)

    img = Image.fromarray(np.uint8(grid * 255.))
    img.save(os.path.join(save_dir, pic_id + '.png'))


def vis_batch(batch, path, pic_id, text=[], rows=[], descr=[], save=True, font_size=11):
    # print(batch.shape)

    if len(batch.shape) == 4:
        padded = F.pad(batch, (1, 1, 1, 1), value=0.5)
    elif len(batch.shape) == 5:
        padded = F.pad(batch, (0, 0, 1, 1, 1, 1), value=0.5)
    else:
        print("Unknown shape:", batch.shape)

    # save image matrix
    T.save(padded, f'./image_matrices_skip_pyramid/' + pic_id + '.pt')

    # print(padded.shape)
    reshaped = T.cat([T.cat([channels for channels in sample], dim=1) for sample in padded], dim=0)
    # print(reshaped.shape)
    if np.max(reshaped.numpy()) > 1.0:
        reshaped = reshaped / 256
    os.makedirs(path, exist_ok=True)
    if text or rows or descr:
        if rows:
            row_width = 50
        else:
            row_width = 0

        if descr:
            descr_wid = 50
        else:
            descr_wid = 0
        if text:
            text_height = 40
        else:
            text_height = 0

        if len(reshaped.shape) == 2:
            reshaped = F.pad(reshaped, (row_width, descr_wid, text_height, 0), value=1)
            img = Image.fromarray(np.uint8(reshaped.numpy() * 255), mode="L")
        elif len(reshaped.shape) == 3:
            reshaped = F.pad(reshaped, (0, 0, row_width, descr_wid, text_height, 0), value=1)
            img = Image.fromarray(np.uint8(reshaped.numpy() * 255))
        # font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", font_size)
        draw = ImageDraw.Draw(img)

        for i, words in enumerate(text):
            x, y = row_width + i * (reshaped.shape[1] - row_width - descr_wid) // len(text), 0
            draw.text((x, y), words, fill=(0) if len(reshaped.shape) == 2 else (0, 0, 0))

        for j, words in enumerate(rows):
            x, y = 3, 10 + text_height + j * (reshaped.shape[0] - text_height) // len(rows)
            draw.text((x, y), words, fill=(0) if len(reshaped.shape) == 2 else (0, 0, 0))

        for j, words in enumerate(descr):
            x, y = 5 + reshaped.shape[1] - descr_wid, text_height + j * (reshaped.shape[0] - text_height) // len(descr)
            # print(x,y)
            draw.text((x, y), words, fill=(0) if len(reshaped.shape) == 2 else (0, 0, 0))

        if save:
            img.save(f'{path}/' + pic_id + '.png')
        else:
            return img
    else:
        if save:
            plt.imsave(f'{path}/' + pic_id + '.png', reshaped.numpy(), dpi=1000)
        else:
            return reshaped


def gifify(batch, path, pic_id, text=[], constant=None):
    # print(batch.shape)
    if np.max(batch.numpy()) > 1.0:
        batch = batch / 256

    if len(batch.shape) == 4:
        padded = F.pad(batch, (1, 1, 1, 1), value=0.5)
    elif len(batch.shape) == 5:
        padded = F.pad(batch, (0, 0, 1, 1, 1, 1), value=0.5)
    else:
        print("Unknown shape:", batch.shape)

    os.makedirs(path, exist_ok=True)

    frames = []
    for f_id in range(padded.shape[1]):
        frame = padded[:, f_id]
        frame = T.cat([sample for sample in frame], dim=1)
        if text:
            text_height = 30
            if len(frame.shape) == 2:
                # frame = F.pad(frame, (0,0,text_height,0), value=0.0)
                img = Image.fromarray(np.uint8(frame.numpy() * 255), mode="L")
            elif len(frame.shape) == 3:
                # frame = F.pad(frame, (0,0,0,0,text_height,0), value=0.0)
                img = Image.fromarray(np.uint8(frame.numpy() * 255))
            font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", 9)
            draw = ImageDraw.Draw(img)
            for i, words in enumerate(text):
                x, y = i * frame.shape[1] // len(text), 0
                draw.text((x, y), words, fill=(0) if len(frame.shape) == 2 else (0, 0, 0), font=font)
        else:
            if len(frame.shape) == 2:
                img = Image.fromarray(np.uint8(frame.numpy() * 255), mode="L")
            elif len(frame.shape) == 3:
                img = Image.fromarray(np.uint8(frame.numpy() * 255))

        if constant is not None:
            dst = Image.new('RGB', (img.width, img.height + constant.height), (255, 255, 255))
            dst.paste(constant, (0, 0))
            dst.paste(img, (0, constant.height))

            img = dst

        frames.append(img)

    frames[0].save(f'{path}/' + pic_id + '.gif', save_all=True, append_images=frames[1:], optimize=True, duration=300,
                   loop=0)


def make_visuals():
    sim = phyre.initialize_simulator(["00018:013", "00020:007", "00018:035"], 'ball')
    res = sim.simulate_action(0, sim.sample(0), stride=40)
    while not res.status.is_solved():
        res = sim.simulate_action(0, sim.sample(0), stride=40)
    # init.save("result/visuals/init1.png")

    obs = phyre.observations_to_uint8_rgb(sim.initial_scenes[0])
    obs = np.pad(obs, ((5, 5), (5, 5), (0, 0)))
    init1 = Image.fromarray(obs)
    init = init1.copy()
    for frame in res.images:
        obs = phyre.observations_to_uint8_rgb(frame)
        obs = np.pad(obs, ((5, 5), (5, 5), (0, 0)))
        frame = np.pad(frame, ((5, 5), (5, 5)))

        objects = Image.fromarray(np.flip((frame != 0), axis=0).astype(np.uint8) * 100)
        pic = Image.fromarray(obs)
        # pic.putalpha(0.5)
        init.paste(pic, (0, 0), objects)
    blended1 = init

    res = sim.simulate_action(1, sim.sample(1), stride=20)
    while not res.status.is_solved():
        res = sim.simulate_action(1, sim.sample(1), stride=20)
    # init.save("result/visuals/init1.png")

    obs = phyre.observations_to_uint8_rgb(sim.initial_scenes[1])
    obs = np.pad(obs, ((5, 5), (5, 5), (0, 0)))
    init2 = Image.fromarray(obs)
    init = init2.copy()
    for frame in res.images:
        obs = phyre.observations_to_uint8_rgb(frame)
        obs = np.pad(obs, ((5, 5), (5, 5), (0, 0)))
        frame = np.pad(frame, ((5, 5), (5, 5)))

        objects = Image.fromarray(np.flip((frame != 0), axis=0).astype(np.uint8) * 100)
        pic = Image.fromarray(obs)
        # pic.putalpha(0.5)
        init.paste(pic, (0, 0), objects)
    blended2 = init

    base = 256 + 10
    back = Image.new("RGB", (4 * base + 15, base))
    back.paste(init1, (0, 0))
    back.paste(blended1, (base + 5, 0))
    back.paste(init2, (2 * base + 10, 0))
    back.paste(blended2, (3 * base + 15, 0))
    os.makedirs("result/visuals", exist_ok=True)
    back.save("result/visuals/phyre.png")

    """
    obs = phyre.observations_to_uint8_rgb(sim.initial_scenes)
    print(obs.shape)
    padded = np.flip(np.pad(obs, ((0,0),(5,5),(5,5),(0,0))), axis=1)
    print(padded.shape)
    init = Image.fromarray(np.concatenate(padded, axis=1))
    """
    # init.save("result/visuals/blended1.png")
    # objects.save("result/visuals/red.png")"""


def prepare_data(data, size):
    targetchannel = 1
    X, Y = [], []
    print("Preparing dataset...")
    # x = np.zeros((X.shape[0], 7, size[0], size[1]))
    for variations in data:
        with_base = len(variations) > 1
        for (j, rollout) in enumerate(variations):
            if not isinstance(rollout, np.ndarray):
                break
            # length = (2*len(rollout))//3
            # rollout = rollout[:length]
            roll = np.zeros((len(rollout), 7, size[0], size[1]))
            for i, scene in enumerate(rollout):
                channels = [(scene == j).astype(float) for j in range(1, 8)]
                roll[i] = np.stack([(cv2.resize(c, size, cv2.INTER_MAX) > 0).astype(float) for c in channels])
            roll = np.flip(roll, axis=2)
            trajectory = (np.sum(roll[:, targetchannel], axis=0) > 0).astype(float)
            if not (with_base and j == 0):
                action = (np.sum(roll[:, 0], axis=0) > 0).astype(float)
            # goal_prior = dist_map(roll[0, 2] + roll[0, 3])
            # roll[0, 0] = goal_prior
            # TESTING ONLY
            # roll[0, 1] = roll[0, 0]
            if with_base and j == 0:
                base = trajectory
            else:
                action_ball = roll[0, 0].copy()
                roll[0, 0] = np.zeros_like(roll[0, 0])
                # print(goal_prior)
                # Contains the initial scene without action
                X.append(roll[0])
                # Contains goaltarget, actiontarget, basetrajectory
                Y.append(np.stack((trajectory, action, base if with_base else np.zeros_like(roll[0, 0]), action_ball)))
                # plt.imshow(trajectory)
                # plt.show()
    print("Finished preparing!")
    return X, Y


def extract_channels_and_paths(rollout, path_idxs=[1, 0], size=(32, 32), gamma=1):
    """
    returns init scenes from 'channels' followed by paths specified by 'path_idxs' 
    """
    paths = np.zeros((len(path_idxs), len(rollout), size[0], size[1]))
    alpha = 1
    for i, chans in enumerate(rollout):
        # extract color codings from channels
        # chans = np.array([(scene==ch).astype(float) for ch in channels])

        # if first frame extract init scene
        if not i:
            init_scene = np.array(
                [(cv2.resize(chans[ch], size, cv2.INTER_MAX) > 0).astype(float) for ch in range(len(chans))])

        # add path_idxs channels to paths
        for path_i, idx in enumerate(path_idxs):
            paths[path_i, i] = alpha * (cv2.resize(chans[idx], size, cv2.INTER_MAX) > 0).astype(float)
        alpha *= gamma

    # flip y axis and concat init scene with paths
    paths = np.flip(np.max(paths, axis=1).astype(float), axis=1)
    init_scene = np.flip(init_scene, axis=1)
    result = np.concatenate([init_scene, paths])
    return result


def format_raw_rollout_data(data, size=(32, 32)):
    targetchannel = 1
    data_bundle = []
    lib_dict = dict()
    print("Formating data...")
    # x = np.zeros((X.shape[0], 7, size[0], size[1]))
    for i, (base, trial, info) in enumerate(data):
        print(f"at sample {i}; {info}")
        # base_path = extract_channels_and_paths(base, channels=[1], path_idxs=[0], size=size)[1]
        # trial_channels = extract_channels_and_paths(trial, size=size)
        # sample = np.append(trial_channels, base_path[None], axis=0)
        try:
            task, subtask, number = info
            base_path = extract_channels_and_paths(base, path_idxs=[1], size=size)[-1]
            trial_channels = extract_channels_and_paths(trial, path_idxs=[1, 2, 0], size=size)
            sample = np.append(trial_channels, base_path[None], axis=0)
            # plt.imshow(np.concatenate(tuple(np.concatenate((sub, T.ones(32,1)*0.5), axis=1) for sub in sample), axis=1))
            # plt.show()
            data_bundle.append(sample)

            # Create indexing dict
            key = task + ':' + subtask
            if not key in lib_dict:
                lib_dict[key] = [i]
            else:
                lib_dict[key].append(i)
        except Exception as identifier:
            print(identifier)
    print("Finished preparing!")
    return data_bundle, lib_dict


def load_phyre_rollout_data(path, base=True):
    s = "/"
    fp = "observations.pickle"
    for task in os.listdir(path):
        for variation in os.listdir(path + s + task):
            if base:
                with open(path + s + task + s + variation + s + 'base' + s + fp, 'rb') as handle:
                    base_rollout = pickle.load(handle)
            for trialfolder in os.listdir(path + s + task + s + variation):
                final_path = path + s + task + s + variation + s + trialfolder + s + fp
                with open(final_path, 'rb') as handle:
                    trial_rollout = pickle.load(handle)
                if base:
                    yield (base_rollout, trial_rollout, (task, variation, trialfolder))
                else:
                    yield (trial_rollout)


def draw_ball(w, x, y, r, invert_y=False):
    """inverts y axis """
    x = int(w * x)
    y = int(w * (1 - y)) if invert_y else int(w * y)
    r = w * r
    X = T.arange(w).repeat((w, 1)).float()
    Y = T.arange(w).repeat((w, 1)).transpose(0, 1).float()
    X -= x  # X Distance
    Y -= y  # Y Distance
    dist = (X.pow(2) + Y.pow(2)).pow(0.5)
    return (dist < r).float()


def action_delta_generator(pure_noise=False):
    temp = 1
    radfac = 0.025
    coordfac = 0.1

    # for x,y,r in zip([0.05,-0.05,0.1,-0.1],[0,0,0,0],[-0.1,-0.2,-0.3,0]):
    # yield x,y,r

    if not pure_noise:
        for fac in [0.5, 1, 2]:
            for rad in [0, 1, -1]:
                for xd, yd in [(1, 0), (-1, 0), (2, 0), (-2, 0), (-1, 2), (1, 2), (-1, -2), (-1, -2)]:
                    # print((fac*np.array((coordfac*xd, coordfac*yd, rad*radfac))))
                    yield (fac * np.array((coordfac * xd, coordfac * yd, rad * radfac)))
    count = 0
    while True:
        count += 1
        action = ((np.random.randn(3)) * np.array([0.2, 0.1, 0.2]) * temp) * 0.1
        # print(count,"th", "ACTION:", action)
        if np.linalg.norm(action) < 0.05:
            continue
        yield action
        temp = 1.04 * temp if temp < 5 else temp


def pic_to_action_vector(pic, r_fac=1):
    X, Y = 0, 0
    for y in range(pic.shape[0]):
        for x in range(pic.shape[1]):
            if pic[y, x]:
                X += pic[y, x] * x
                Y += pic[y, x] * y
    summed = pic.sum()
    X /= pic.shape[0] * summed
    Y /= pic.shape[0] * summed
    r = np.sqrt(pic.sum() / (3.141592 * pic.shape[0] ** 2))
    return [X.item(), 1 - Y.item(), r_fac * r.item()]


def grow_action_vector(pic, r_fac=1, show=False, num_seeds=1, mask=None, check_border=False, updates=5):
    id = int((T.rand(1) * 100))
    # os.makedirs("result/flownet/solver/grower", exist_ok=True)
    # plt.imsave(f"result/flownet/solver/grower/{id}.png", pic)
    pic = pic * (pic > pic.mean())
    # plt.imsave(f"result/flownet/solver/grower/{id}_thresh.png", pic)

    wid = pic.shape[0]

    def get_value(x, y, r):
        ball = draw_ball(wid, x, y, r)
        potential = T.sum(ball)
        actual = T.sum(pic[ball.bool()])
        value = (actual ** 0.5) * actual / potential
        if mask is not None:
            overlap = mask[ball > 0].sum()
            if overlap > 0:
                return -overlap
        if check_border and ((x - r) < -0.00 or (y - r) < -0.00 or (x + r) > 1.00 or (y + r) > 1.00):
            return min([x - r, 1 - (x + r), y - r, 1 - (y + r)])
        return value

    def move_and_grow(x, y, r, v):
        delta = 0.7
        positions = [(x + dx, y + dy) for (dx, dy) in
                     [(-(0.3 + delta) / 30, 0), ((0.3 + delta) / 30, 0), (0, -(0.3 + delta) / 30),
                      (0, (0.3 + delta) / 30)] if (0 <= x + dx < 1) and (0 <= y + dy < 1)]
        bestpos = (x, y)
        bestrad = r
        bestv = v
        for pos in positions:
            value = get_value(*pos, r)
            rad, val = grow(*pos, r, value)
            if val > bestv:
                bestpos = pos
                bestrad = rad
                bestv = val
        return bestpos[0], bestpos[1], bestrad, bestv

    def grow(x, y, r, v):
        bestv = v
        bestrad = r
        for rad in [r + 0.005, r + 0.01, r + 0.03, r - 0.01]:
            if 0 < rad < 0.3:
                value = get_value(x, y, rad)
                if value > bestv:
                    bestv = value
                    bestrad = rad
        return bestrad, bestv

    seeds = []
    while len(seeds) < num_seeds:
        r = 0.04 + np.random.rand() * 0.05
        try:
            y, x = random.choice(T.nonzero((pic > 0.01))) + T.rand(2) * 0.05
            seeds.append((x.item() / wid, y.item() / wid, r))
        except Exception as e:
            print("EXCEPTION", e)
            y, x = wid // 2, wid // 2
            seeds.append((x / wid, y / wid, r))

    final_seeds = []
    for (x, y, r) in seeds:
        v = get_value(x, y, r)
        # plt.imshow(pic+draw_ball(wid,x,y,r))
        # plt.show()
        for i in range(updates):
            x, y, r, v = move_and_grow(x, y, r, v)
            # r, v = grow(x,y,r,v)
            if show:
                print(x, y, r, v)
                plt.imshow(pic + draw_ball(wid, x, y, r))
                plt.show()
        final_seeds.append(((x, y, r), v))

    action = np.array(max(final_seeds, key=lambda x: x[1])[0])
    action[1] = 1 - action[1]
    plt.imsave(f"result/flownet/solver/grower/{id}_drawn.png", draw_ball(wid, *action, invert_y=True))
    action[2] *= r_fac
    compare_action = action.copy()
    action[2] = action[2] if action[2] < 0.125 else 0.125
    action[0] = action[0] if action[0] > 0 else 0
    action[0] = action[0] if action[0] < 1 - 0 else 1 - 0
    action[1] = action[1] if action[1] > 0 else 0
    action[1] = action[1] if action[1] < 1 - 0 else 1 - 0
    if np.any(action != compare_action):
        print("something was out of bounce:", action, compare_action)
    return action


def pic_hist_to_action(pic, r_fac=3):
    # thresholding
    pic = pic * (pic > 0.2)
    # columns part of ball
    cols = [idx for (idx, val) in enumerate(np.sum(pic, axis=0)) if val > 2]
    start, end = min(cols), max(cols)
    x = (start + end) / 2
    x /= pic.shape[1]
    # rows part of ball
    rows = [idx for (idx, val) in enumerate(np.sum(pic, axis=1)) if val > 2]
    start, end = min(rows), max(rows)
    y = (start + end) / 2
    y /= pic.shape[0]
    # radius
    r = np.sqrt(pic.sum() / (3.141592 * pic.shape[0] ** 2))
    r = 0.1
    return x, y, r


def scenes_to_channels(X, size=(32, 32)):
    x = np.zeros((X.shape[0], 7, size[0], size[1]))
    for i, scene in enumerate(X):
        channels = [(scene == j).astype(float) for j in range(1, 8)]
        x[i] = np.flip(np.stack([(cv2.resize(c, size, cv2.INTER_MAX) > 0).astype(float) for c in channels]), axis=1)
    return x


def rollouts_to_specific_paths(batch, channel, size=(32, 32), gamma=1):
    trajectory = np.zeros((len(batch), size[0], size[1]))
    for j, r in enumerate(batch):
        path = np.zeros((len(r), size[0], size[1]))
        alpha = 1
        for i, scene in enumerate(r):
            chan = (scene == channel).astype(float)
            path[i] = alpha * (cv2.resize(chan, size, cv2.INTER_MAX) > 0).astype(float)
            alpha *= gamma
        path = np.flip(path, axis=1)
        base = np.max(path, axis=0).astype(float)
        trajectory[j] = base
    return trajectory


def extract_individual_auccess(path):
    with open(path + "/auccess-dict.json") as fp:
        dic = json.load(fp)

    w_res = dict((i, []) for i in range(25))
    c_res = dict((i, []) for i in range(25))
    keys = dic.keys()
    w_keys = [k for k in keys if k.__contains__("within")]
    c_keys = [k for k in keys if k.__contains__("cross")]
    print(c_keys)

    for k in w_keys:
        templ = int(k.split('_')[4][-2:])
        w_res[templ].append(dic[k])
    print(w_res)

    for k in c_keys:
        templ = int(k.split('_')[4][-2:])
        c_res[templ].append(dic[k])
    print(c_res)

    with open(path + "/average-auccess-horizontal.txt", "w") as fp:
        fp.write("within cross\n")
        within = [sum(templ) / len(templ) for templ in [w_res[i] for i in range(25)]]
        cross = [sum(templ) / len(templ) for templ in [(c_res[i] or [0]) for i in range(25)]]
        # fp.writelines([f"{('0000'+str(i))[-5:]} {w} {c}\n" for i,(w,c) in enumerate(zip(within, cross))])
        fp.writelines([str(round(item, 2))[-3:] + ' & ' for item in within] + ['\n'])
        fp.writelines([str(round(item, 2))[-3:] + ' & ' for item in cross] + ['\n'])
        fp.write(f"average {sum(within) / len(within)} {sum(cross) / (len(cross) - 1)}")


def collect_traj_lookup(tasks, save_path, number_per_task, show=False, stride=10):
    end_char = '\n'
    tries = 0
    max_tries = 100
    base_path = save_path
    cache = phyre.get_default_100k_cache('ball')
    actions = cache.action_array
    print("Amount per task", number_per_task)

    keys = []
    values = []

    sim = phyre.initialize_simulator(tasks, 'ball')
    for idx, task in enumerate(tasks):
        # COLLECT SOLVES
        n_collected = 0
        while n_collected < number_per_task:
            tries += 1

            # getting action
            action = actions[cache.load_simulation_states(task) == 1]
            print(f"collecting {n_collected + 1} interactions from {task} with {tries} tries", end=end_char)
            if len(action) == 0:
                print("no solution action in cache at task", task)
                action = [np.random.rand(3)]
            action = random.choice(action)

            # simulating action
            res = sim.simulate_action(idx, action,
                                      need_featurized_objects=True, stride=1)
            while res.status.is_invalid():
                action = np.random.rand(3)
                res = sim.simulate_action(idx, action,
                                          need_featurized_objects=True, stride=1)

            # checking result for contact
            def check_contact(res: phyre.Simulation):
                # print(res.images.shape)
                # print(len(res.bitmap_seq))
                # print(res.status.is_solved())
                idx1 = res.body_list.index('RedObject')
                idx2 = res.body_list.index('GreenObject')
                # print(idx1, idx2)
                # print(res.body_list)

                green_idx = res.featurized_objects.colors.index('GREEN')
                red_idx = res.featurized_objects.colors.index('RED')
                target_dist = sum(res.featurized_objects.diameters[[green_idx, red_idx]]) / 2
                for i, m in enumerate(res.bitmap_seq):
                    if m[idx1][idx2]:
                        pos = res.featurized_objects.features[i, [green_idx, red_idx], :2]
                        dist = np.linalg.norm(pos[1] - pos[0])
                        # print(dist, target_dist)
                        if not dist < target_dist + 0.005:
                            continue

                        red_radius = res.featurized_objects.diameters[red_idx] * 4
                        action_at_interaction = np.append(pos[1], red_radius)
                        return (True, i, pos[0], action_at_interaction, target_dist)

                return (False, 0, (0, 0), 0, 0)

            contact, i_step, green_pos, red_pos, summed_radii = check_contact(res)
            if contact:
                tries = 0

                step_n = 10
                # check whether contact happend too early
                if i_step - step_n < 0:
                    continue

                try:
                    green_idx = res.featurized_objects.colors.index('GREEN')
                    red_idx = res.featurized_objects.colors.index('RED')
                    green_minus, _ = res.featurized_objects.features[i_step - stride, [green_idx, red_idx], :2]
                    green_zero, _ = res.featurized_objects.features[i_step, [green_idx, red_idx], :2]
                    green_plus, _ = res.featurized_objects.features[i_step + stride, [green_idx, red_idx], :2]
                    green_key, _ = green_minus - green_zero, 0
                    green_value, _ = green_zero - green_plus, 0
                    keys.append((green_key[0], green_key[1]))
                    values.append((green_value[0], green_value[1]))
                except:
                    continue

                n_collected += 1

            if tries > max_tries:
                break

    keys = np.round(256 * np.array(keys))
    k_x_max = keys[np.argmax(np.abs(keys[:, 0])), 0]
    k_y_max = keys[np.argmax(np.abs(keys[:, 1])), 1]
    """keys[:,0] /= k_x_max/5
    keys[:,1] /= k_y_max/5
    k_x_max = np.max(np.abs(keys[:,0]))
    k_y_max = np.max(np.abs(keys[:,1]))"""
    values = np.round(256 * np.array(values))
    v_x_max = values[np.argmax(np.abs(values[:, 0])), 0]
    v_y_max = values[np.argmax(np.abs(values[:, 1])), 1]
    """values[:,0] /= v_x_max/5
    values[:,1] /= v_y_max/5
    v_x_max = np.max(np.abs(values[:,0]))
    v_y_max = np.max(np.abs(values[:,1]))"""

    table = dict()
    for i in range(len(keys)):
        k = tuple(keys[i])
        v = tuple(values[i])
        if k in table:
            table[k][v] = table[k][v] + 1 if v in table[k] else 1
        else:
            table[k] = {v: 1}

    # Save data to file
    os.makedirs(base_path, exist_ok=True)
    with open(f'{base_path}/lookup.pickle', 'wb') as fp:
        pickle.dump(table, fp)
    print(f"FINISH collecting trajectory lookup!")
    return keys, values, k_x_max, k_y_max, v_x_max, v_y_max, table


def visualize_actions_from_cache(amount):
    cache = phyre.get_default_100k_cache("ball")
    actions = cache.action_array[:amount]
    plt.scatter(actions[:, 0], actions[:, 1], alpha=0.3, s=1000 * actions[:, 2], c=actions[:, 2])
    plt.show()


def print_folds():
    eval_setup = 'ball_within_template'
    for fold_id in range(10):
        # print(phyre.get_fold(eval_setup, fold_id)[0][:10])
        print(phyre.get_fold(eval_setup, fold_id)[0][:10]
              == phyre.get_fold(eval_setup, fold_id)[0][:10])


def get_auccess_for_n_tries(n):
    eva = phyre.Evaluator(['00000:000'])
    for _ in range(n - 1):
        eva.maybe_log_attempt(0, phyre.SimulationStatus.NOT_SOLVED)
    for _ in range(101 - n):
        eva.maybe_log_attempt(0, phyre.SimulationStatus.SOLVED)
    return eva.get_auccess()


def get_auccess_for_n_tries_first_only(n):
    eva = phyre.Evaluator(['00000:000'])
    for i in range(1, 101):
        if n == i:
            eva.maybe_log_attempt(0, phyre.SimulationStatus.SOLVED)
        else:
            eva.maybe_log_attempt(0, phyre.SimulationStatus.NOT_SOLVED)
    return eva.get_auccess()


def add_dijkstra_to_data(path):
    if os.path.exists(path + "/data.pickle") and os.path.exists(path + "/index.pickle"):
        with open(path + '/data.pickle', 'rb') as fp:
            data = pickle.load(fp)
            X = T.tensor(data).float()
    else:
        print("Path not found")

    for scene in X:
        red = T.stack((X[:, 0], X[:, 0] * 0, X[:, 0] * 0), dim=-1)
        green = T.stack((X[:, 1], X[:, 0] * 0, X[:, 0] * 0), dim=-1)
        blues = T.stack((X[:, 2], X[:, 0] * 0, X[:, 0] * 0), dim=-1)
        blued = T.stack((X[:, 3], X[:, 0] * 0, X[:, 0] * 0), dim=-1)
        grey = T.stack((X[:, 4], X[:, 4], X[:, 4]), dim=-1)
        black = T.stack((X[:, 5], X[:, 0] * 0, X[:, 0] * 0), dim=-1)


if __name__ == "__main__":
    # visualize_actions_from_cache(1000)
    # print(get_auccess_for_n_tries(10))

    shrink_data("./data")
    exit()

    extract_individual_auccess("./result/solver/result/GEN-64-20e/individ-10")
    exit()

    for n in range(1, 20):
        print(get_auccess_for_n_tries_first_only(n))
    exit()

    make_visuals()
    exit()
    # Collecting trajectory lookup
    pic = draw_ball(32, 0.5, 0.2, 0.3) + draw_ball(32, 0.5, 0.5, 0.1)
    print(grow_action_vector(pic, check_border=True, mask=draw_ball(32, 0.5, 0.5, 0.1), show=True))
    # exit()
    fold_id = 0
    eval_setup = 'ball_within_template'
    train_ids, dev_ids, test_ids = phyre.get_fold(eval_setup, fold_id)
    all_tasks = train_ids + dev_ids + test_ids
    template13_tasks = [t for t in all_tasks if t.startswith('00013:')]
    template2_tasks = [t for t in all_tasks if t.startswith('00002:')]
    print(template2_tasks)
    # collect_specific_channel_paths(f'./data/template13_action_paths_10x', template13_tasks, 0)
    keys, values, kxm, kym, vxm, vym, table = collect_traj_lookup(all_tasks, 'result/traj_lookup/all_tasks', 10,
                                                                  stride=10)
    print(keys)
    print(values)
    print(kxm, kym, vxm, vym)
    print(table)
