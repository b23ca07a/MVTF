import argparse
import multiprocessing as mp
import os
import traceback
from collections import OrderedDict
from pathlib import Path

import torch
import yaml
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_VIEWS = ["fff", "ttt", "ddd", "3l30", "3r30", "3r60", "3l60"]
DEFAULT_META_ROOT = Path("/public/home/qinxy/photon/dataset/MEAD_SCP")
DEFAULT_CHECKPOINT_DIR = (
    PROJECT_DIR
    / "ckpt"
    / "25_NEW_random_repeat_conv_before_LSTM_front_product_two_masked_3_random_view_"
    "mead_add_after_vconv_new_exceptM019_9"
)
EPS = 1e-8


def parse_args():
    parser = argparse.ArgumentParser("Speech Separation")
    parser.add_argument(
        "--config",
        default="config/std_train.yml",
        type=str,
        help="config file path",
    )
    parser.add_argument(
        "--epoch",
        default="77",
        type=str,
        help="checkpoint epoch to evaluate",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        help="path to a checkpoint file; overrides --checkpoint-dir/--epoch",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(DEFAULT_CHECKPOINT_DIR),
        type=str,
        help="directory that stores epoch checkpoints",
    )
    parser.add_argument(
        "--meta-root",
        default=str(DEFAULT_META_ROOT),
        type=str,
        help="directory containing <view>_view_test.scp files",
    )
    parser.add_argument(
        "--noise-scp",
        default=None,
        type=str,
        help="optional override for the legacy inferset.args.noise_scp config entry",
    )
    parser.add_argument(
        "--gpu-ids",
        default="0,1,2,3",
        type=str,
        help="comma-separated GPU ids used for multiprocessing inference",
    )
    parser.add_argument(
        "--views",
        nargs="*",
        default=DEFAULT_VIEWS,
        help="views to evaluate",
    )
    return parser.parse_args()


def resolve_path(path_text):
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (PROJECT_DIR / path).resolve()


def load_config(config_path):
    with open(config_path) as rfile:
        return yaml.safe_load(rfile)


def resolve_checkpoint_path(args):
    if args.checkpoint:
        ckpt_path = resolve_path(args.checkpoint)
    else:
        ckpt_path = resolve_path(args.checkpoint_dir) / f"epoch{args.epoch}.pth.tar"

    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. "
            "Pass --checkpoint /path/to/epochXX.pth.tar "
            "or update --checkpoint-dir/--epoch."
        )
    return ckpt_path


def load_pretrained_modules(model, ckpt_path):
    model_info = torch.load(ckpt_path, map_location="cpu")
    state_dict = OrderedDict()
    for key, value in model_info["model_state_dict"].items():
        state_dict[key.replace("module.", "")] = value
    model.load_state_dict(state_dict)
    return model


def split_views_across_gpus(views, gpu_ids):
    assignments = {gpu_id: [] for gpu_id in gpu_ids}
    for index, view in enumerate(views):
        gpu_id = gpu_ids[index % len(gpu_ids)]
        assignments[gpu_id].append(view)
    return assignments


def evaluate_view(config, ckpt_path, meta_root, noise_scp, view, gpu_id, worker_slot):
    import dataloader
    import dataset
    import models.tfgridnet_separator as module_model
    from utils.loss.pit_criterion import sdr, sisnr
    from utils.utils import get_instance

    meta_file = meta_root / f"{view}_view_test.scp"
    if not meta_file.is_file():
        raise FileNotFoundError(f"Meta file not found for view {view}: {meta_file}")

    config["inferset"]["args"]["meta_file"] = str(meta_file)
    if noise_scp is not None:
        config["inferset"]["args"]["noise_scp"] = str(noise_scp)
    inferset = get_instance(dataset, config["inferset"])
    inferloader = get_instance(dataloader, config["inferloader"], inferset)

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(gpu_id)

    model = module_model.TFGridNet(**config["tfgridnet_kwargs"])
    model = load_pretrained_modules(model, str(ckpt_path))
    model.to(device)
    model.eval()

    total_sisdr = 0.0
    total_base_sisdr = 0.0
    total_sisdri = 0.0
    total_sdr = 0.0
    num_batches = 0

    prog_bar = tqdm(
        inferloader,
        desc=f"GPU{gpu_id}:{view}",
        position=worker_slot,
        leave=True,
    )
    with torch.no_grad():
        for batch in prog_bar:
            num_batches += 1
            mix = batch["mix"].to(device)
            tar = batch["tar"].to(device)
            ilens = batch["ilens"].squeeze(-1).to(device)
            lip_emb_0 = batch["lip_0"].to(device)
            lip_emb_1 = batch["lip_1"].to(device)
            lip_emb_2 = batch["lip_2"].to(device)
            masked = batch["view_mask"].to(device)

            est = -model(
                mix.transpose(1, -1),
                ilens,
                lip_emb_0,
                lip_emb_1,
                lip_emb_2,
                masked,
            )[0] * 0.2

            est_wav = est.squeeze(1)
            sisdr_value = torch.mean(sisnr(est_wav, tar, EPS)).item()
            base_sisdr_value = torch.mean(sisnr(mix, tar, EPS)).item()
            sisdri_value = sisdr_value - base_sisdr_value
            sdr_value = torch.mean(sdr(est_wav, tar, EPS)).item()

            total_sisdr += sisdr_value
            total_base_sisdr += base_sisdr_value
            total_sisdri += sisdri_value
            total_sdr += sdr_value

            prog_bar.set_postfix(
                sisdri=f"{total_sisdri / num_batches:.3f}",
                sdr=f"{total_sdr / num_batches:.3f}",
            )

    if num_batches == 0:
        raise RuntimeError(f"No inference samples were loaded for view {view}.")

    return {
        "view": view,
        "gpu_id": gpu_id,
        "num_batches": num_batches,
        "avg_sisdr": total_sisdr / num_batches,
        "avg_base_sisdr": total_base_sisdr / num_batches,
        "avg_sisdri": total_sisdri / num_batches,
        "avg_sdr": total_sdr / num_batches,
        "sum_sisdr": total_sisdr,
        "sum_base_sisdr": total_base_sisdr,
        "sum_sisdri": total_sisdri,
        "sum_sdr": total_sdr,
    }


def worker_main(
    gpu_id,
    assigned_views,
    config_path,
    ckpt_path,
    meta_root,
    noise_scp,
    worker_slot,
    result_queue,
):
    os.chdir(PROJECT_DIR)
    try:
        config_path = Path(config_path)
        ckpt_path = Path(ckpt_path)
        meta_root = Path(meta_root)
        noise_scp = Path(noise_scp) if noise_scp is not None else None
        worker_results = []
        for view in assigned_views:
            config = load_config(config_path)
            worker_results.append(
                evaluate_view(
                    config,
                    ckpt_path,
                    meta_root,
                    noise_scp,
                    view,
                    gpu_id,
                    worker_slot,
                )
            )
        result_queue.put({"gpu_id": gpu_id, "results": worker_results})
    except Exception:
        result_queue.put({"gpu_id": gpu_id, "error": traceback.format_exc()})


def print_summary(results):
    ordered_results = sorted(results, key=lambda item: DEFAULT_VIEWS.index(item["view"]) if item["view"] in DEFAULT_VIEWS else item["view"])

    print("\n[+] Per-view summary")
    for result in ordered_results:
        print(
            f"[+] view={result['view']} gpu={result['gpu_id']} "
            f"SiSDR={result['avg_sisdr']:.4f} "
            f"base_SiSDR={result['avg_base_sisdr']:.4f} "
            f"SiSDRi={result['avg_sisdri']:.4f} "
            f"SDR={result['avg_sdr']:.4f}"
        )

    total_batches = sum(result["num_batches"] for result in ordered_results)
    if total_batches == 0:
        raise RuntimeError("No results were collected from worker processes.")

    print("\n[+] Global summary")
    print(f"[+] avg SiSDR = {sum(result['sum_sisdr'] for result in ordered_results) / total_batches:.4f}")
    print(f"[+] avg base_SiSDR = {sum(result['sum_base_sisdr'] for result in ordered_results) / total_batches:.4f}")
    print(f"[+] avg SiSDRi = {sum(result['sum_sisdri'] for result in ordered_results) / total_batches:.4f}")
    print(f"[+] avg SDR = {sum(result['sum_sdr'] for result in ordered_results) / total_batches:.4f}")


def main():
    args = parse_args()
    config_path = resolve_path(args.config)
    assert config_path.is_file(), f"No such file: {config_path}"
    ckpt_path = resolve_checkpoint_path(args)
    meta_root = resolve_path(args.meta_root)
    if not meta_root.is_dir():
        raise NotADirectoryError(f"Meta root not found: {meta_root}")

    noise_scp = resolve_path(args.noise_scp) if args.noise_scp else None
    if noise_scp is not None and not noise_scp.is_file():
        raise FileNotFoundError(f"Noise scp not found: {noise_scp}")

    gpu_ids = [int(item.strip()) for item in args.gpu_ids.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("No valid GPU ids were provided.")

    views = args.views if args.views else DEFAULT_VIEWS
    assignments = split_views_across_gpus(views, gpu_ids)

    print(f"[+] Config: {config_path}")
    print(f"[+] Checkpoint: {ckpt_path}")
    print(f"[+] Meta root: {meta_root}")
    if noise_scp is not None:
        print(f"[+] Noise scp override: {noise_scp}")
    print(f"[+] Using GPUs: {gpu_ids}")
    for gpu_id in gpu_ids:
        print(f"[+] GPU {gpu_id} -> {assignments[gpu_id]}")

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []

    for worker_slot, gpu_id in enumerate(gpu_ids):
        assigned_views = assignments[gpu_id]
        if not assigned_views:
            continue
        process = ctx.Process(
            target=worker_main,
            args=(
                gpu_id,
                assigned_views,
                str(config_path),
                str(ckpt_path),
                str(meta_root),
                str(noise_scp) if noise_scp is not None else None,
                worker_slot,
                result_queue,
            ),
        )
        process.start()
        processes.append(process)

    results = []
    errors = []
    for _ in processes:
        message = result_queue.get()
        if "error" in message:
            errors.append(message)
        else:
            results.extend(message["results"])

    for process in processes:
        process.join()

    if errors:
        for error in errors:
            print(f"[!] Worker on GPU {error['gpu_id']} failed")
            print(error["error"])
        raise RuntimeError("At least one worker process failed.")

    print_summary(results)


if __name__ == "__main__":
    main()
