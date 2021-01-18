import datetime
from math import log10
from pathlib import Path

import pandas as pd
import pytorch_ssim
import torch
import torchvision.utils as utils
from torch import optim, nn
from torch.nn import BCELoss, MSELoss
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import vgg19
from torchvision.transforms import transforms
from tqdm import tqdm

from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from model import Generator, Discriminator

NUM_EPOCHS = int(2e5)
PRETRAIN_EPOCHS = 0
PATCH_SIZE = 96
UPSCALE_FACTOR = 4
VALIDATION_FREQUENCY = 50
NUM_LOGGED_VALIDATION_IMAGES = 30

if __name__ == '__main__':
    training_start = datetime.datetime.now().isoformat()
    train_set = TrainDatasetFromFolder('data/DIV2K_train_HR', patch_size=PATCH_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_set = ValDatasetFromFolder('data/DIV2K_valid_HR', upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False, pin_memory=True)

    results_folder = Path(f"results_{training_start}_CS:{PATCH_SIZE}_US:{UPSCALE_FACTOR}x")
    results_folder.mkdir(exist_ok=True)

    writer = SummaryWriter(str(results_folder / "tensorboard_log"))

    g_net = Generator(n_residual_blocks=16, upsample_factor=UPSCALE_FACTOR)
    d_net = Discriminator()

    if torch.cuda.is_available():
        g_net.cuda()
        d_net.cuda()

    g_optimizer = optim.Adam(g_net.parameters(), lr=1e-4)
    d_optimizer = optim.Adam(d_net.parameters(), lr=1e-4)

    g_scheduler = StepLR(g_optimizer, step_size=NUM_EPOCHS // 2, gamma=0.1)
    d_scheduler = StepLR(d_optimizer, step_size=NUM_EPOCHS // 2, gamma=0.1)

    bce_loss = BCELoss()
    mse_loss = MSELoss()

    vgg = vgg16(pretrained=True)
    feature_extractor = nn.Sequential(*list(vgg.features)[:31]).eval()
    for param in feature_extractor.parameters():
        param.requires_grad = False

    if torch.cuda.is_available():
        bce_loss.cuda()
        mse_loss.cuda()
        feature_extractor.cuda()

    results = {'d_total_loss': [], 'g_total_loss': [], 'd_real_mean': [], 'd_fake_mean': [], 'psnr': [], 'ssim': []}

    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader, ncols=160)
        running_results = {'batch_sizes': 0, 'd_epoch_total_loss': 0, 'g_epoch_total_loss': 0, 'd_epoch_real_mean': 0,
                           'd_epoch_fake_mean': 0}

        g_net.train()
        d_net.train()

        for data, target in train_bar:
            batch_size = data.size(0)
            real_labels = torch.ones(batch_size)
            fake_labels = torch.zeros(batch_size)
            running_results["batch_sizes"] += batch_size

            if torch.cuda.is_available():
                target = target.cuda()
                data = data.cuda()
                real_labels = real_labels.cuda()
                fake_labels = fake_labels.cuda()

            # Discriminator training
            d_optimizer.zero_grad()

            d_real_output = d_net(target)
            d_real_output_loss = bce_loss(d_real_output, real_labels)

            fake_img = g_net(data)
            d_fake_output = d_net(fake_img)
            d_fake_output_loss = bce_loss(d_fake_output, fake_labels)

            d_total_loss = d_real_output_loss + d_fake_output_loss
            d_total_loss.backward()
            d_optimizer.step()
            d_scheduler.step()

            d_real_mean = d_real_output.mean()
            d_fake_mean = d_fake_output.mean()

            # Generator training
            g_optimizer.zero_grad()

            fake_img = g_net(data)
            adversarial_loss = bce_loss(d_net(fake_img), real_labels)
            content_loss = mse_loss(feature_extractor(target), feature_extractor(fake_img))

            g_total_loss = content_loss + 1e-3 * adversarial_loss
            g_total_loss.backward()
            g_optimizer.step()
            g_scheduler.step()

            running_results['g_epoch_total_loss'] += g_total_loss.item() * batch_size
            running_results['d_epoch_total_loss'] += d_total_loss.item() * batch_size
            running_results['d_epoch_real_mean'] += d_real_mean.item() * batch_size
            running_results['d_epoch_fake_mean'] += d_fake_mean.item() * batch_size

            train_bar.set_description(
                desc=f'[{epoch}/{NUM_EPOCHS}] '
                     f'Loss_D: {running_results["d_epoch_total_loss"] / running_results["batch_sizes"]:.4f} '
                     f'Loss_G: {running_results["g_epoch_total_loss"] / running_results["batch_sizes"]:.4f} '
                     f'D(x): {running_results["d_epoch_real_mean"] / running_results["batch_sizes"]:.4f} '
                     f'D(G(z)): {running_results["d_epoch_fake_mean"] / running_results["batch_sizes"]:.4f}')

        if epoch % 50 == 1:
            g_net.eval()
            # ...
            images_path = results_folder / Path(f'training_images_results')
            images_path.mkdir(exist_ok=True)

            with torch.no_grad():
                val_bar = tqdm(val_loader, ncols=160)
                val_results = {'epoch_mse': 0, 'epoch_ssim': 0, 'epoch_psnr': 0, 'epoch_avg_psnr': 0,
                               'epoch_avg_ssim': 0,
                               'batch_sizes': 0, }
                val_images = []
                for lr, val_hr_restore, hr in val_bar:
                    batch_size = lr.size(0)
                    val_results['batch_sizes'] += batch_size
                    if torch.cuda.is_available():
                        hr = hr.cuda()
                        lr = lr.cuda()

                    sr = g_net(lr)
                    sr = ((sr + 1) / 2)  # rescale values from [-1,1] to [0,1]

                    batch_mse = ((sr - hr) ** 2).data.mean()  # Pixel-wise MSE
                    val_results['epoch_mse'] += batch_mse * batch_size
                    batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                    val_results['epoch_ssim'] += batch_ssim * batch_size
                    val_results['epoch_avg_ssim'] = val_results['epoch_ssim'] / val_results['batch_sizes']
                    val_results['epoch_psnr'] += 20 * log10(hr.max() / (batch_mse / batch_size)) * batch_size
                    val_results['epoch_avg_psnr'] = val_results['epoch_psnr'] / val_results['batch_sizes']

                    val_bar.set_description(
                        desc=f"[converting LR images to SR images] PSNR: {val_results['epoch_avg_psnr']:4f} dB SSIM: {val_results['epoch_avg_ssim']:4f}")

                    # This requires validation batch size = 1
                    val_images.extend(
                        [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                         display_transform()(sr.data.cpu().squeeze(0))])

                val_images = torch.stack(val_images)
                val_images = torch.chunk(val_images, val_images.size(0) // 15)
                val_save_bar = tqdm(val_images, desc='[saving training results]', ncols=160)

                for index, image_batch in enumerate(val_save_bar, start=1):
                    image_grid = utils.make_grid(image_batch, nrow=3, padding=5)
                    utils.save_image(image_grid, str(images_path / f'epoch_{epoch}_index_{index}.png'), padding=5)
                    writer.add_image(f'epoch_{epoch}_index_{index}.png', image_grid)

        # save model parameters
        models_path = results_folder / "saved_models"
        models_path.mkdir(exist_ok=True)
        torch.save(g_net.state_dict(), str(models_path / f'epoch_{epoch}_g_net.pth'))
        torch.save(d_net.state_dict(), str(models_path / f'epoch_{epoch}_d_net.pth'))

        # save loss / scores / psnr /ssim
        results['d_total_loss'].append(running_results['d_epoch_total_loss'] / running_results['batch_sizes'])
        results['g_total_loss'].append(running_results['g_epoch_total_loss'] / running_results['batch_sizes'])
        results['d_real_mean'].append(running_results['d_epoch_real_mean'] / running_results['batch_sizes'])
        results['d_fake_mean'].append(running_results['d_epoch_fake_mean'] / running_results['batch_sizes'])
        results['psnr'].append(val_results['epoch_avg_psnr'])
        results['ssim'].append(val_results['epoch_avg_ssim'])

        for metric, metric_values in results.items():
            writer.add_scalar(metric, metric_values[-1], epoch)

        if epoch % 50 == 1:
            data_frame = pd.DataFrame(
                data={'d_total_loss': results['d_total_loss'], 'g_total_loss': results['g_total_loss'],
                      'd_real_mean': results['d_real_mean'],
                      'd_fake_mean': results['d_fake_mean'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            data_frame.to_csv(str(results_folder / f"train_results.csv"), index_label='Epoch')
