import datetime
import gc
from argparse import ArgumentParser
from math import log10
from pathlib import Path
from statistics import mean

import numpy as np
import pytorch_ssim
import torch
import torchvision.utils as utils
from lpips import lpips
from torch import optim
from torch.nn import BCELoss, MSELoss
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torch_fidelity import calculate_metrics
from tqdm import tqdm

from augment import AugmentPipe
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform, SingleTensorDataset, \
    HrValDatasetFromFolder
from model import Generator, Discriminator

NUM_ADV_BASELINE_EPOCHS = 30
NUM_BASELINE_PRETRAIN_EPOCHS = 5
PATCH_SIZE = 128
UPSCALE_FACTOR = 4
NUM_RESIDUAL_BLOCKS = 16
NUM_LOGGED_VALIDATION_IMAGES = 16  # Should be a multiple of 4
VAL_BATCH_SIZE = 32
AUGMENT_PROB_TARGET = 0.6
ADV_LOSS_BALANCER = 4e-5
BATCH_SIZE = 32
RT_BATCH_SMOOTHING_FACTOR = 8
AUGMENT_PROBABABILITY_STEP = 1e-3
CENTER_CROP_SIZE = 512

train_dataset_dir = 'data/ffhq/images512x512/train_set'
val_dataset_dir = 'data/ffhq/images512x512/val_set'

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")


def main():
    parser = ArgumentParser()
    parser.add_argument("--augmentation", action='store_true')
    parser.add_argument("--train-dataset-percentage", type=float, default=100)
    parser.add_argument("--val-dataset-percentage", type=int, default=100)
    parser.add_argument("--label-smoothing", type=float, default=0.9)
    parser.add_argument("--validation-frequency", type=int, default=1)
    args = parser.parse_args()

    ENABLE_AUGMENTATION = args.augmentation
    TRAIN_DATASET_PERCENTAGE = args.train_dataset_percentage
    VAL_DATASET_PERCENTAGE = args.val_dataset_percentage
    LABEL_SMOOTHING_FACTOR = args.label_smoothing
    VALIDATION_FREQUENCY = args.validation_frequency

    if ENABLE_AUGMENTATION:
        augment_batch = AugmentPipe()
        augment_batch.to(device)
    else:
        augment_batch = lambda x: x
        augment_batch.p = 0

    NUM_ADV_EPOCHS = round(NUM_ADV_BASELINE_EPOCHS / (TRAIN_DATASET_PERCENTAGE / 100))
    NUM_PRETRAIN_EPOCHS = round(NUM_BASELINE_PRETRAIN_EPOCHS / (TRAIN_DATASET_PERCENTAGE / 100))
    VALIDATION_FREQUENCY = round(VALIDATION_FREQUENCY / (TRAIN_DATASET_PERCENTAGE / 100))

    training_start = datetime.datetime.now().isoformat()

    train_set = TrainDatasetFromFolder(train_dataset_dir, patch_size=PATCH_SIZE,
                                       upscale_factor=UPSCALE_FACTOR)
    len_train_set = len(train_set)
    train_set = Subset(train_set, list(
        np.random.choice(np.arange(len_train_set), int(len_train_set * TRAIN_DATASET_PERCENTAGE / 100), False)))

    val_set = ValDatasetFromFolder(val_dataset_dir, upscale_factor=UPSCALE_FACTOR)
    len_val_set = len(val_set)
    val_set = Subset(val_set, list(
        np.random.choice(np.arange(len_val_set), int(len_val_set * VAL_DATASET_PERCENTAGE / 100), False)))

    train_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=BATCH_SIZE, shuffle=True,
                              pin_memory=True, prefetch_factor=8)
    val_loader = DataLoader(dataset=val_set, num_workers=2, batch_size=VAL_BATCH_SIZE, shuffle=False,
                            pin_memory=True, prefetch_factor=2)

    epoch_validation_hr_dataset = HrValDatasetFromFolder(val_dataset_dir)  # Useful to compute FID metric

    results_folder = Path(
        f"results_{training_start}_CS:{PATCH_SIZE}_US:{UPSCALE_FACTOR}x_TRAIN:{TRAIN_DATASET_PERCENTAGE}%_AUGMENTATION:{ENABLE_AUGMENTATION}")
    results_folder.mkdir(exist_ok=True)
    writer = SummaryWriter(str(results_folder / "tensorboard_log"))
    g_net = Generator(n_residual_blocks=NUM_RESIDUAL_BLOCKS, upsample_factor=UPSCALE_FACTOR)
    d_net = Discriminator(patch_size=PATCH_SIZE)
    lpips_metric = lpips.LPIPS(net='alex')

    g_net.to(device=device)
    d_net.to(device=device)
    lpips_metric.to(device=device)

    g_optimizer = optim.Adam(g_net.parameters(), lr=1e-4)
    d_optimizer = optim.Adam(d_net.parameters(), lr=1e-4)

    bce_loss = BCELoss()
    mse_loss = MSELoss()

    bce_loss.to(device=device)
    mse_loss.to(device=device)
    results = {'d_total_loss': [], 'g_total_loss': [], 'g_adv_loss': [], 'g_content_loss': [], 'd_real_mean': [],
               'd_fake_mean': [], 'psnr': [], 'ssim': [], 'lpips': [], 'fid': [], 'rt': [], 'augment_probability': []}

    augment_probability = 0
    num_images = len(train_set) * (NUM_PRETRAIN_EPOCHS + NUM_ADV_EPOCHS)
    prediction_list = []
    rt = 0

    for epoch in range(1, NUM_PRETRAIN_EPOCHS + NUM_ADV_EPOCHS + 1):
        train_bar = tqdm(train_loader, ncols=200)
        running_results = {'batch_sizes': 0, 'd_epoch_total_loss': 0, 'g_epoch_total_loss': 0, 'g_epoch_adv_loss': 0,
                           'g_epoch_content_loss': 0, 'd_epoch_real_mean': 0, 'd_epoch_fake_mean': 0, 'rt': 0,
                           'augment_probability': 0}
        image_percentage = epoch / (NUM_PRETRAIN_EPOCHS + NUM_ADV_EPOCHS) * 100
        g_net.train()
        d_net.train()

        for data, target in train_bar:
            augment_batch.p = torch.tensor([augment_probability], device=device)
            batch_size = data.size(0)
            running_results["batch_sizes"] += batch_size
            target = target.to(device)
            data = data.to(device)
            real_labels = torch.ones(batch_size, device=device)
            fake_labels = torch.zeros(batch_size, device=device)

            if epoch > NUM_PRETRAIN_EPOCHS:
                # Discriminator training
                d_optimizer.zero_grad(set_to_none=True)

                d_real_output = d_net(augment_batch(target))
                d_real_output_loss = bce_loss(d_real_output, real_labels * LABEL_SMOOTHING_FACTOR)

                fake_img = g_net(data)
                d_fake_output = d_net(augment_batch(fake_img))
                d_fake_output_loss = bce_loss(d_fake_output, fake_labels)

                d_total_loss = d_real_output_loss + d_fake_output_loss
                d_total_loss.backward()
                d_optimizer.step()

                d_real_mean = d_real_output.mean()
                d_fake_mean = d_fake_output.mean()

            # Generator training
            g_optimizer.zero_grad(set_to_none=True)

            fake_img = g_net(data)
            if epoch > NUM_PRETRAIN_EPOCHS:
                adversarial_loss = bce_loss(d_net(augment_batch(fake_img)),
                                            real_labels) * ADV_LOSS_BALANCER
                content_loss = mse_loss(fake_img, target)
                g_total_loss = content_loss + adversarial_loss
            else:
                adversarial_loss = mse_loss(torch.zeros(1, device=device),
                                            torch.zeros(1, device=device))  # Logging purposes, it is always zero
                content_loss = mse_loss(fake_img, target)
                g_total_loss = content_loss

            g_total_loss.backward()
            g_optimizer.step()

            if epoch > NUM_PRETRAIN_EPOCHS and ENABLE_AUGMENTATION:
                prediction_list.append((torch.sign(d_real_output - 0.5)).tolist())
                if len(prediction_list) == RT_BATCH_SMOOTHING_FACTOR:
                    rt_list = [prediction for sublist in prediction_list for prediction in sublist]
                    rt = mean(rt_list)
                    if mean(rt_list) > AUGMENT_PROB_TARGET:
                        augment_probability = min(0.85, augment_probability + AUGMENT_PROBABABILITY_STEP)
                    else:
                        augment_probability = max(0., augment_probability - AUGMENT_PROBABABILITY_STEP)
                    prediction_list.clear()

            running_results['g_epoch_total_loss'] += g_total_loss.to('cpu', non_blocking=True).detach() * batch_size
            running_results['g_epoch_adv_loss'] += adversarial_loss.to('cpu', non_blocking=True).detach() * batch_size
            running_results['g_epoch_content_loss'] += content_loss.to('cpu', non_blocking=True).detach() * batch_size
            if epoch > NUM_PRETRAIN_EPOCHS:
                running_results['d_epoch_total_loss'] += d_total_loss.to('cpu', non_blocking=True).detach() * batch_size
                running_results['d_epoch_real_mean'] += d_real_mean.to('cpu', non_blocking=True).detach() * batch_size
                running_results['d_epoch_fake_mean'] += d_fake_mean.to('cpu', non_blocking=True).detach() * batch_size
                running_results['rt'] += rt * batch_size
                running_results['augment_probability'] += augment_probability * batch_size

            train_bar.set_description(
                desc=f'[{epoch}/{NUM_ADV_EPOCHS + NUM_PRETRAIN_EPOCHS}] '
                     f'Loss_D: {running_results["d_epoch_total_loss"] / running_results["batch_sizes"]:.4f} '
                     f'Loss_G: {running_results["g_epoch_total_loss"] / running_results["batch_sizes"]:.4f} '
                     f'Loss_G_adv: {running_results["g_epoch_adv_loss"] / running_results["batch_sizes"]:.4f} '
                     f'Loss_G_content: {running_results["g_epoch_content_loss"] / running_results["batch_sizes"]:.4f} '
                     f'D(x): {running_results["d_epoch_real_mean"] / running_results["batch_sizes"]:.4f} '
                     f'D(G(z)): {running_results["d_epoch_fake_mean"] / running_results["batch_sizes"]:.4f} '
                     f'rt: {running_results["rt"] / running_results["batch_sizes"]:.4f} '
                     f'augment_probability: {running_results["augment_probability"] / running_results["batch_sizes"]:.4f}')

        if epoch == 1 or epoch == (
                NUM_PRETRAIN_EPOCHS + NUM_ADV_EPOCHS) or epoch % VALIDATION_FREQUENCY == 0 or VALIDATION_FREQUENCY == 1:
            torch.cuda.empty_cache()
            gc.collect()
            g_net.eval()
            # ...
            images_path = results_folder / Path(f'training_images_results')
            images_path.mkdir(exist_ok=True)

            with torch.no_grad():
                val_bar = tqdm(val_loader, ncols=160)
                val_results = {'epoch_mse': 0, 'epoch_ssim': 0, 'epoch_psnr': 0, 'epoch_avg_psnr': 0,
                               'epoch_avg_ssim': 0, 'epoch_lpips': 0, 'epoch_avg_lpips': 0, 'epoch_fid': 0,
                               'batch_sizes': 0}
                val_images = torch.empty((0, 0))
                epoch_validation_sr_dataset = None
                for lr, val_hr_restore, hr in val_bar:
                    batch_size = lr.size(0)
                    val_results['batch_sizes'] += batch_size
                    hr = hr.to(device=device)
                    lr = lr.to(device=device)

                    sr = g_net(lr)
                    sr = torch.clamp(sr, 0., 1.)
                    if not epoch_validation_sr_dataset:
                        epoch_validation_sr_dataset = SingleTensorDataset((sr.cpu() * 255).to(torch.uint8))

                    else:
                        epoch_validation_sr_dataset = ConcatDataset(
                            (epoch_validation_sr_dataset, SingleTensorDataset((sr.cpu() * 255).to(torch.uint8))))

                    batch_mse = ((sr - hr) ** 2).data.mean()  # Pixel-wise MSE
                    val_results['epoch_mse'] += batch_mse * batch_size
                    batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                    val_results['epoch_ssim'] += batch_ssim * batch_size
                    val_results['epoch_avg_ssim'] = val_results['epoch_ssim'] / val_results['batch_sizes']
                    val_results['epoch_psnr'] += 20 * log10(
                        hr.max() / (batch_mse / batch_size)) * batch_size
                    val_results['epoch_avg_psnr'] = val_results['epoch_psnr'] / val_results['batch_sizes']
                    val_results['epoch_lpips'] += torch.mean(lpips_metric(hr * 2 - 1, sr * 2 - 1)).to(
                        'cpu', non_blocking=True).detach() * batch_size
                    val_results['epoch_avg_lpips'] = val_results['epoch_lpips'] / val_results['batch_sizes']

                    val_bar.set_description(
                        desc=f"[converting LR images to SR images] PSNR: {val_results['epoch_avg_psnr']:4f} dB "
                             f"SSIM: {val_results['epoch_avg_ssim']:4f} "
                             f"LPIPS: {val_results['epoch_avg_lpips']:.4f} ")
                    if val_images.size(0) * val_images.size(1) < NUM_LOGGED_VALIDATION_IMAGES * 3:
                        if val_images.size(0) == 0:
                            val_images = torch.hstack(
                                (display_transform(CENTER_CROP_SIZE)(val_hr_restore).unsqueeze(0).transpose(0, 1),
                                 display_transform(CENTER_CROP_SIZE)(hr.data.cpu()).unsqueeze(0).transpose(0, 1),
                                 display_transform(CENTER_CROP_SIZE)(sr.data.cpu()).unsqueeze(0).transpose(0, 1)))
                        else:
                            val_images = torch.cat((val_images,
                                                    torch.hstack(
                                                        (display_transform(CENTER_CROP_SIZE)(val_hr_restore).unsqueeze(
                                                            0).transpose(0, 1),
                                                         display_transform(CENTER_CROP_SIZE)(hr.data.cpu()).unsqueeze(
                                                             0).transpose(0, 1),
                                                         display_transform(CENTER_CROP_SIZE)(sr.data.cpu()).unsqueeze(
                                                             0).transpose(0, 1)))))
                val_results['epoch_fid'] = calculate_metrics(
                    epoch_validation_sr_dataset, epoch_validation_hr_dataset,
                    cuda=True, fid=True, verbose=True)[
                    'frechet_inception_distance']  # Set batch_size=1 if you get memory error (inside calculate metric function)

                val_images = val_images.view(
                    (NUM_LOGGED_VALIDATION_IMAGES // 4, -1, 3, CENTER_CROP_SIZE, CENTER_CROP_SIZE))
                val_save_bar = tqdm(val_images, desc='[saving validation results]', ncols=160)

                for index, image_batch in enumerate(val_save_bar, start=1):
                    image_grid = utils.make_grid(image_batch, nrow=3, padding=5)
                    writer.add_image(f'progress{image_percentage:.1f}_index_{index}.png', image_grid)

        # save loss / scores / psnr /ssim
        results['d_total_loss'].append(running_results['d_epoch_total_loss'] / running_results['batch_sizes'])
        results['g_total_loss'].append(running_results['g_epoch_total_loss'] / running_results['batch_sizes'])
        results['g_adv_loss'].append(running_results['g_epoch_adv_loss'] / running_results['batch_sizes'])
        results['g_content_loss'].append(running_results['g_epoch_content_loss'] / running_results['batch_sizes'])
        results['d_real_mean'].append(running_results['d_epoch_real_mean'] / running_results['batch_sizes'])
        results['d_fake_mean'].append(running_results['d_epoch_fake_mean'] / running_results['batch_sizes'])
        results['rt'].append(running_results['rt'] / running_results['batch_sizes'])
        results['augment_probability'].append(running_results['augment_probability'] / running_results['batch_sizes'])
        if epoch == 1 or epoch == (
                NUM_PRETRAIN_EPOCHS + NUM_ADV_EPOCHS) or epoch % VALIDATION_FREQUENCY == 0 or VALIDATION_FREQUENCY == 1:
            results['psnr'].append(val_results['epoch_avg_psnr'])
            results['ssim'].append(val_results['epoch_avg_ssim'])
            results['lpips'].append(val_results['epoch_avg_lpips'])
            results['fid'].append(val_results['epoch_fid'])

        for metric, metric_values in results.items():
            if epoch == 1 or epoch == (
                    NUM_PRETRAIN_EPOCHS + NUM_ADV_EPOCHS) or epoch % VALIDATION_FREQUENCY == 0 or VALIDATION_FREQUENCY == 1 or \
                    metric not in ["psnr", "ssim", "lpips", "fid"]:
                writer.add_scalar(metric, metric_values[-1], int(image_percentage * num_images * 0.01))

        if epoch == 1 or epoch == (
                NUM_PRETRAIN_EPOCHS + NUM_ADV_EPOCHS) or epoch % VALIDATION_FREQUENCY == 0 or VALIDATION_FREQUENCY == 1:
            # save model parameters
            models_path = results_folder / "saved_models"
            models_path.mkdir(exist_ok=True)
            torch.save({
                'progress': image_percentage,
                'g_net': g_net.state_dict(),
                'd_net': g_net.state_dict(),
                # 'g_optimizer': g_optimizer.state_dict(), Uncomment this if you want resume training
                # 'd_optimizer': d_optimizer.state_dict(),
            }, str(models_path / f'progress_{image_percentage:.1f}.tar'))


if __name__ == '__main__':
    main()
