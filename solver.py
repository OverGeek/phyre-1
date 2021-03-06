from phyre_utils import pic_to_action_vector, pic_hist_to_action, vis_batch, make_mono_dataset, grow_action_vector, \
    action_delta_generator
import torch as T
import phyre
import numpy as np
import cv2
import json
import itertools
from matplotlib import pyplot as plt
import os
from PIL import Image, ImageDraw, ImageFont
import sys
import json
import pickle


def get_auccess(solver, tasks, solve_noise=False, save_tries=False, brute=False):
    if save_tries:
        font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", 10)

    eval_setup = 'ball_within_template'
    sim = phyre.initialize_simulator(tasks, 'ball')
    init_scenes = T.tensor(
        [[cv2.resize((scene == channel).astype(float), (32, 32)) for channel in range(2, 7)] for scene in
         sim.initial_scenes]).float().flip(-2)
    eva = phyre.Evaluator(tasks)

    # Get Actions from solver:
    if brute:
        all_actions = solver.get_actions(tasks, init_scenes, brute=True)
    else:
        all_actions = solver.get_actions(tasks, init_scenes)
    # print(list(zip(tasks, all_actions)))
    # return 0

    # Loop through actions
    for t_idx, task in enumerate(tasks):
        # Get 100 actions from solver
        if solve_noise:
            # expects one action for task
            task_actions = [all_actions[t_idx]]
        else:
            # expects 100 actions for task
            task_actions = all_actions[t_idx]

        # Loop through actions
        for j, action in enumerate(task_actions):
            # Setting up visualization array
            vis_wid = 64
            vis_stack = T.zeros(6, 10, vis_wid, vis_wid, 3)
            vis_count = 1

            # Simulate action
            res = sim.simulate_action(t_idx, action, need_featurized_objects=False)

            # Refining if invalid Action
            t = 0
            temp = 1
            base_action = action.copy()
            print(base_action, 'base action')
            # Checking for valid action
            while res.status.is_invalid():
                t += 1
                action = base_action + (np.random.rand(3) - 0.5) * 0.05 * temp
                print(action, f"potential action for task {task}")
                res = sim.simulate_action(t_idx, action, need_featurized_objects=False)
                temp *= 1.01 if temp < 5 else 1
                # assert(t>500, "too many invalid tries")
            print(action, 'valid action')

            # Log first Attempt
            eva.maybe_log_attempt(t_idx, res.status)
            # Visualizing first attempt
            if save_tries:
                for i in range(min(len(res.images), 10)):
                    vis_stack[0, i] = T.tensor(
                        cv2.resize(phyre.observations_to_uint8_rgb(res.images[i]), (vis_wid, vis_wid)))

            # Collecting 100 Actions if solve noise
            warning_flag = False
            if solve_noise:
                base_action = action
                temp = 1
                error = False
                t = 0
                delta_generator = action_delta_generator()

                # Looping while less then 100 attempts
                while eva.attempts_per_task_index[t_idx] < 100:
                    # Searching for new action while not solved
                    if not res.status.is_solved():
                        """ OLD APPROACH
                        action = base_action + (np.random.rand(3)-0.5)*np.array([0.3,0.05,0.05])*temp
                        temp *= 1.01 if temp <5 else 1
                        """
                        if t < 1000:
                            action = base_action + delta_generator.__next__()
                            res = sim.simulate_action(t_idx, action, need_featurized_objects=False)
                            eva.maybe_log_attempt(t_idx, res.status)
                            t += 1
                        else:
                            if not warning_flag:
                                print(f"WARNING can't find valid action for {task}")
                                warning_flag = True
                                error = True
                            eva.maybe_log_attempt(t_idx, phyre.SimulationStatus.NOT_SOLVED)

                    # if solved -> repeating action
                    else:
                        if not warning_flag:
                            print(f"{task} solved after", eva.attempts_per_task_index[t_idx])

                            # Visualization
                            if save_tries and not error:
                                for i in range(min(len(res.images), 10)):
                                    vis_stack[5, i] = T.tensor(
                                        cv2.resize(phyre.observations_to_uint8_rgb(res.images[i]), (vis_wid, vis_wid)))
                        warning_flag = True
                        eva.maybe_log_attempt(t_idx, res.status)

                    # Visualization
                    if save_tries and not error and not res.status.is_invalid() and t and vis_count < 5:
                        for i in range(min(len(res.images), 10)):
                            vis_stack[vis_count, i] = T.tensor(
                                cv2.resize(phyre.observations_to_uint8_rgb(res.images[i]), (vis_wid, vis_wid)))
                        vis_count += 1

                if not warning_flag and not res.status.is_solved() and eva.attempts_per_task_index[t_idx] == 100:
                    print(f"{task} not solved")
                vis_batch(vis_stack, f'result/solver/pyramid', f"{task}_attempts")
            # Not Solve Noise Case
            else:
                # Visualization
                if save_tries and not res.status.is_invalid() and vis_count < 5:
                    for i in range(min(len(res.images), 10)):
                        vis_stack[vis_count, i] = T.tensor(
                            cv2.resize(phyre.observations_to_uint8_rgb(res.images[i]), (vis_wid, vis_wid)))
                    vis_count += 1
                if res.status.is_solved():
                    print(f"{task} solved after", eva.attempts_per_task_index[t_idx])
                    vis_batch(vis_stack, f'result/solver/pyramid', f"{task}_attempts")
                    while eva.attempts_per_task_index[t_idx] < 100:
                        eva.maybe_log_attempt(t_idx, res.status)
                    break

    return eva.get_auccess()


def extract_auc_dict(path):
    auc_dict = json.load(f'result/solver/result/{path}/auccess-dict.json')
    res = dict()
    for eval_setup in ['ball_within_template', 'ball_cross_template']:
        if not (eval_setup in res):
            res[eval_setup] = []

        for fold_id in range(10):
            if f"{eval_setup}_{fold_id}" in auc_dict:
                res[eval_setup].append(auc_dict[f"{eval_setup}_{fold_id}"])

            for template in [('0000' + str(i))[-5:] for i in range(25)]:
                if not (template in res):
                    res[template] = []
                if f"{eval_setup}_{fold_id}_{template}" in auc_dict:
                    res[template] = auc_dict[f"{eval_setup}_{fold_id}_{template}"]


if __name__ == "__main__":
    from flownet import *

    model_path = sys.argv[sys.argv.index("--path") + 1] if "--path" in sys.argv else "standard"
    type = sys.argv[sys.argv.index("--type") + 1] if "--type" in sys.argv else "pyramid"
    run = sys.argv[sys.argv.index("--run") + 1] if "--run" in sys.argv else "default"
    device = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda"
    pred_mode = sys.argv[sys.argv.index("--pred-mode") + 1] if "--pred-mode" in sys.argv else "CONS"
    train_mode = sys.argv[sys.argv.index("--train-mode") + 1] if "--train-mode" in sys.argv else "CONS"
    epochs = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 10
    pred_epochs = int(sys.argv[sys.argv.index("--pred-epochs") + 1]) if "--pred-epochs" in sys.argv else 5
    width = int(sys.argv[sys.argv.index("--width") + 1]) if "--width" in sys.argv else 64
    nper = int(sys.argv[sys.argv.index("--nper") + 1]) if "--nper" in sys.argv else 10
    folds = int(sys.argv[sys.argv.index("--folds") + 1]) if "--folds" in sys.argv else 1
    hidfac = float(sys.argv[sys.argv.index("--hidfac") + 1]) if "--hidfac" in sys.argv else 1
    seeds = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else 1
    foldstart = int(sys.argv[sys.argv.index("--foldstart") + 1]) if "--foldstart" in sys.argv else 0
    print("Model path:", model_path)
    gt_paths = "-gt-paths" in sys.argv
    shuffle = not ('-noshuff' in sys.argv)
    smart = '-smart' in sys.argv
    noise = not '-no-noise' in sys.argv
    proposal_data = '-proposal-data' in sys.argv
    no_scnd_stage = '-no-second-stage' in sys.argv
    dijkstra = '-dijkstra' in sys.argv

    auccess = []
    auc_dict = dict()
    for eval_setup in ['ball_within_template', 'ball_cross_template']:
        auccess.append(eval_setup)
        for fold_id in range(foldstart + folds):
            solver = FlownetSolver(model_path, type, width, smart=smart, run=run, num_seeds=seeds, device=device,
                                   hidfac=hidfac, dijkstra=dijkstra)

            # train_ids, dev_ids, test_ids = phyre.get_fold(eval_setup, fold_id)

            if "-get-all-data" in sys.argv:
                solver.load_data(setup=eval_setup, fold=fold_id, n_per_task=nper, brute_search=True, shuffle=shuffle)
                solver.load_data(setup=eval_setup, fold=fold_id, n_per_task=nper, shuffle=shuffle)
                solver.load_data(setup=eval_setup, fold=fold_id, n_per_task=nper, brute_search=True, shuffle=False,
                                 test=True)
                solver.load_data(setup=eval_setup, fold=fold_id, n_per_task=nper, shuffle=False, test=True)

            if "-load" in sys.argv:
                print(model_path, eval_setup, fold_id, "|| loading models...")
                solver.load_models(setup=eval_setup, fold=fold_id, no_second_stage=no_scnd_stage)

            if "-train" in sys.argv:
                if type == "brute":
                    print(model_path, eval_setup, fold_id, "|| loading data for brute training...")
                    solver.load_data(setup=eval_setup, fold=fold_id, n_per_task=nper, brute_search=True,
                                     shuffle=shuffle)
                    print(model_path, eval_setup, fold_id, "|| training 'brute search' models...")
                    solver.train_brute_search(epochs=epochs)
                elif type == "combi":
                    # print(model_path, eval_setup, fold_id, "|| loading data for combi first stage training...")
                    # solver.load_data(setup=eval_setup, fold=fold_id, n_per_task=nper, shuffle=shuffle)
                    # print(model_path, eval_setup, fold_id, "|| training generator models...")
                    # solver.train_supervised(epochs=epochs, train_mode=train_mode)
                    if proposal_data:
                        print(model_path, eval_setup, fold_id,
                              "|| loading/generating proposals for combi second stage data collection...")
                        proposals_path = f"./saves/{model_path}_{eval_setup}_{fold_id}_500_action_proposals"
                        if os.path.exists(proposals_path + "/proposal-dict.pickle"):
                            with open(proposals_path + "/proposal-dict.pickle", 'rb') as fp:
                                proposal_dict = pickle.load(fp)
                        else:
                            proposal_dict = solver.get_proposals(train_ids, "proposals")
                            os.makedirs(proposals_path, exist_ok=True)
                            with open(proposals_path + "/proposal-dict.pickle", 'wb') as fp:
                                pickle.dump(proposal_dict, fp)
                    print(model_path, eval_setup, fold_id, "|| loading data for combi second stage training...")
                    solver.load_data(setup=eval_setup, fold=fold_id, n_per_task=nper, brute_search=True,
                                     shuffle=shuffle, proposal_dict=proposal_dict if proposal_data else None)
                    print(model_path, eval_setup, fold_id, "|| training predictor models...")
                    solver.train_combi(epochs=pred_epochs, train_mode=pred_mode)
                else:
                    print(model_path, eval_setup, fold_id, "|| loading data for generative training...")
                    solver.load_data(setup=eval_setup, fold=fold_id, n_per_task=nper, shuffle=shuffle)
                    print(model_path, eval_setup, fold_id, "|| training 'generative' models...")
                    solver.train_supervised(epochs=epochs, train_mode=train_mode)

            if "-save" in sys.argv:
                print(model_path, eval_setup, fold_id, "|| saving models...")
                solver.save_models(setup=eval_setup, fold=fold_id)
                print("Models saved")

            if "-train_sfm1" in sys.argv:
                print(model_path, eval_setup, fold_id, "|| loading data for generative training...")
                solver.load_data(setup=eval_setup, fold=fold_id, n_per_task=nper, shuffle=shuffle)
                solver.load_processed_data_sfm1(setup=eval_setup, fold=fold_id, n_per_task=nper, shuffle=shuffle)
                print(model_path, eval_setup, fold_id, "|| training 'generative' models...")
                solver.train_sfm1(epochs=epochs, setup=eval_setup)

            if "-train_sfm_sequential" in sys.argv:
                print(model_path, eval_setup, fold_id, "|| loading data for generative training...")
                solver.load_data(setup=eval_setup, fold=fold_id, n_per_task=nper, shuffle=shuffle, sequential=True)
                print(model_path, eval_setup, fold_id, "|| training 'generative' models...")
                solver.train_sfm_sequential(epochs=epochs, setup=eval_setup, width=width, n_per_task=nper, fold=fold_id)

            if "-save_sfm1_paths" in sys.argv:
                solver.save_sfm1_paths(dir='./SfM1_paths', setup=eval_setup, fold=fold_id, n_per_task=nper)

            if "-save_sfm_sequential" in sys.argv:
                print(model_path, eval_setup, fold_id, "|| saving models...")
                solver.save_sfm_sequential(setup=eval_setup, fold=fold_id)
                print("Models saved")

            if "-train_sfm2" in sys.argv:
                solver.load_processed_data_sfm2(setup=eval_setup, fold=fold_id, n_per_task=nper, shuffle=shuffle)
                solver.train_sfm2(epochs=epochs, setup=eval_setup)

            if "-save_sfm1" in sys.argv:
                print(model_path, eval_setup, fold_id, "|| saving models...")
                solver.save_sfm1(setup=eval_setup, fold=fold_id)
                print("Models saved")

            if "-save_sfm2" in sys.argv:
                print(model_path, eval_setup, fold_id, "|| saving models...")
                solver.save_sfm2(setup=eval_setup, fold=fold_id)
                print("Models saved")

            if "-inspect" in sys.argv:
                if type == "brute":
                    print(model_path, eval_setup, fold_id, "|| loading data for brute testing...")
                    solver.load_data(setup=eval_setup, fold=fold_id, n_per_task=nper, brute_search=True,
                                     shuffle=shuffle, test=True)
                    print(model_path, eval_setup, fold_id, "|| inspecting brute performance...")
                    solver.inspect_brute_search(eval_setup, fold_id)
                elif type == "combi":
                    print(model_path, eval_setup, fold_id, "|| loading data for combi testing...")
                    solver.load_data(setup=eval_setup, fold=fold_id, n_per_task=nper, brute_search=True,
                                     shuffle=shuffle, test=True)
                    print(model_path, eval_setup, fold_id, "|| inspecting second stage performance...")
                    solver.inspect_combi(eval_setup, fold_id)
                elif type == "pyramid":
                    print(model_path, eval_setup, fold_id, "|| loading data for generative testing...")
                    solver.load_data(setup=eval_setup, fold=fold_id, n_per_task=nper, shuffle=shuffle, test=True)
                    print(model_path, eval_setup, fold_id, "|| inspecting generative performance...")
                    solver.inspect_supervised(eval_setup, fold_id)

            if "-make_visuals" in sys.argv:
                solver.make_visualisation(setup=eval_setup, fold=fold_id, n_per_task=nper,
                                          model="sfm1")
                solver.make_visualisation(setup=eval_setup, fold=fold_id, n_per_task=nper,
                                          model="sfm2")

            if "-solve" in sys.argv:
                print(model_path, eval_setup, fold_id, "|| getting auccess...")
                if type == "brute":
                    local_auccess = solver.brute_auccess(test_ids)
                    # auccess.append( get_auccess(solver, (test_ids+dev_ids)[:], solve_noise=False, save_tries=True, brute=True) )
                    os.makedirs(f'result/solver/result/{solver.path}/{run}', exist_ok=True)
                    with open(f'result/solver/result/{solver.path}/{run}/{eval_setup}_{fold_id}.txt', 'w') as handle:
                        handle.write(f"auccess: {local_auccess}")
                    auccess.append(local_auccess)
                elif type == "combi":
                    local_auccess = solver.combi_auccess(test_ids, f'{eval_setup}_{fold_id}', pure_noise=noise,
                                                         gt_paths=gt_paths)
                    # auccess.append( get_auccess(solver, (test_ids+dev_ids)[:], solve_noise=False, save_tries=True, brute=True) )
                    os.makedirs(f'result/solver/result/{solver.path}/{run}', exist_ok=True)
                    with open(f'result/solver/result/{solver.path}/{run}/{eval_setup}_{fold_id}.txt', 'w') as handle:
                        handle.write(f"auccess: {local_auccess}")
                    auccess.append(local_auccess)
                else:
                    local_auccess = solver.generative_auccess(test_ids, f'{eval_setup}_{fold_id}', pure_noise=noise)
                    # auccess.append( get_auccess(solver, (test_ids+dev_ids)[:], solve_noise=False, save_tries=True, brute=True) )
                    os.makedirs(f'result/solver/result/{solver.path}/{run}', exist_ok=True)
                    with open(f'result/solver/result/{solver.path}/{run}/{eval_setup}_{fold_id}.txt', 'w') as handle:
                        handle.write(f"auccess: {local_auccess}")
                    auccess.append(local_auccess)

            if "-solve-templ" in sys.argv:
                print(model_path, eval_setup, fold_id, "|| getting auccess per template...")
                for template in [('0000' + str(i))[-5:] for i in range(25)]:
                    ids = [id for id in test_ids + dev_ids if id.startswith(template)]
                    if not ids:
                        continue
                    print("solving", ids)
                    if type == "brute":
                        local_auccess = solver.brute_auccess(ids)
                        print(f"{template} auccess: {local_auccess}")
                        # auccess.append( get_auccess(solver, (test_ids+dev_ids)[:], solve_noise=False, save_tries=True, brute=True) )
                        os.makedirs(f'result/solver/result/{solver.path}/{run}', exist_ok=True)
                        with open(f'result/solver/result/{solver.path}/{run}/individual_{eval_setup}_{fold_id}.txt',
                                  'a') as handle:
                            handle.write(f"\n{template} auccess: {local_auccess}")
                        auccess.append(local_auccess)
                        auc_dict[f"{eval_setup}_{fold_id}_{template}"] = local_auccess
                        with open(f'result/solver/result/{solver.path}/{run}/auccess-dict.json', 'w') as fp:
                            json.dump(auc_dict, fp)
                    elif type == "combi":
                        local_auccess = solver.combi_auccess(ids, f'{eval_setup}_{fold_id}_{template}',
                                                             pure_noise=noise)
                        print(f"{template} auccess: {local_auccess}")
                        # auccess.append( get_auccess(solver, (test_ids+dev_ids)[:], solve_noise=False, save_tries=True, brute=True) )
                        os.makedirs(f'result/solver/result/{solver.path}/{run}', exist_ok=True)
                        with open(f'result/solver/result/{solver.path}/{run}/individual_{eval_setup}_{fold_id}.txt',
                                  'a') as handle:
                            handle.write(f"\n{template} auccess: {local_auccess}")
                        auccess.append(local_auccess)
                        auc_dict[f"{eval_setup}_{fold_id}_{template}"] = local_auccess
                        with open(f'result/solver/result/{solver.path}/{run}/auccess-dict.json', 'w') as fp:
                            json.dump(auc_dict, fp)
                    else:
                        local_auccess = solver.generative_auccess(ids, f'{eval_setup}_{fold_id}_{template}',
                                                                  pure_noise=noise)
                        print(f"{template} auccess: {local_auccess}")
                        # auccess.append( get_auccess(solver, (test_ids+dev_ids)[:], solve_noise=False, save_tries=True, brute=True) )
                        os.makedirs(f'result/solver/result/{solver.path}/{run}', exist_ok=True)
                        with open(f'result/solver/result/{solver.path}/{run}/individual_{eval_setup}_{fold_id}.txt',
                                  'a') as handle:
                            handle.write(f"\n{template} auccess: {local_auccess}")
                        auccess.append(local_auccess)
                        auc_dict[f"{eval_setup}_{fold_id}_{template}"] = local_auccess
                        with open(f'result/solver/result/{solver.path}/{run}/auccess-dict.json', 'w') as fp:
                            json.dump(auc_dict, fp)

                    """
                    auccess.append( get_auccess(solver, test_ids+dev_ids, solve_noise=True, save_tries=True) )
                    os.makedirs(f'result/solver/result/{solver.path}', exist_ok=True)
                    with open(f'result/solver/result/{solver.path}/{eval_setup}_{fold_id}.txt', 'w') as handle:
                        handle.write(f"auccess: {auccess[-1]}")
                    """

            """
            all_tasks = train_ids+dev_ids+test_ids
            filter_list = ['00000', '00001', '00002', '00003', '00004','00005', 
                            '00006','00007','00008','00009','00010','00011',
                            '00012','00013','00014','00015','00016','00017',
                            '00018','00019','00020','00021','00022','00023', '00024']
            """
            """
            auccess_per_task = dict()
            for filter_task in filter_list:
                selected_tasks = [t for t in all_tasks if t[:5]==filter_task]
                auccess = get_auccess(solver, selected_tasks, solve_noise=True, save_tries=True)
                auccess_per_task[filter_task] = auccess
            print(auccess_per_task)
            """
    print(auccess)
