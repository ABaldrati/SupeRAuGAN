from argparse import ArgumentParser
from math import log10
from pathlib import Path

import pytorch_ssim
import torch
from lpips import lpips
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from data_utils import ValDatasetFromFolder, SingleTensorDataset, test_display_transform, HrValDatasetFromFolder
from model import Generator
import torchvision.utils as utils
from torch_fidelity import calculate_metrics
import pandas as pd

UPSCALE_FACTOR = 4
NUM_RESIDUAL_BLOCKS = 16
BATCH_SIZE = 32
NUM_LOGGED_TEST_IMAGES = 32

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True, type=str)
    args = parser.parse_args()

    saved_model = torch.load(args.model, map_location=device)
    model_name = args.name

    g_net = Generator(n_residual_blocks=NUM_RESIDUAL_BLOCKS, upsample_factor=UPSCALE_FACTOR)
    lpips_metric = lpips.LPIPS(net='alex')

    g_net.to(device=device)
    lpips_metric.to(device=device)

    g_net.load_state_dict(saved_model['g_net'])

    test_folder = Path("test_results")
    test_folder.mkdir(exist_ok=True)

    results_folder = test_folder / Path(f"{model_name}")
    results_folder.mkdir(exist_ok=True)

    test_set = ValDatasetFromFolder('data/ffhq/images512x512/test_set', upscale_factor=UPSCALE_FACTOR)
    test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=False,
                             pin_memory=True)

    test_hr_dataset = HrValDatasetFromFolder('data/ffhq/images512x512/test_set')

    g_net.eval()
    images_path = results_folder / Path(f'test_images_results')
    images_path.mkdir(exist_ok=True)

    with torch.no_grad():
        test_bar = tqdm(test_loader, ncols=160)
        test_results = {'psnr': 0, 'ssim': 0, 'lpips': 0, 'fid': 0}
        accumulated_results = {'accumulated_mse': 0, 'accumulated_ssim': 0, 'accumulated_psnr': 0,
                               'accumulated_lpips': 0, 'batch_sizes': 0}
        test_images = torch.empty((0, 0))
        test_sr_dataset = None
        for lr, test_hr_restore, hr in test_bar:
            batch_size = lr.size(0)
            accumulated_results['batch_sizes'] += batch_size
            hr = hr.to(device=device)
            lr = lr.to(device=device)

            sr = g_net(lr)
            sr = torch.clamp(sr, 0., 1.)
            if not test_sr_dataset:
                test_sr_dataset = SingleTensorDataset((sr.cpu() * 255).to(torch.uint8))

            else:
                test_sr_dataset = ConcatDataset(
                    (test_sr_dataset, SingleTensorDataset((sr.cpu() * 255).to(torch.uint8))))

            batch_mse = ((sr - hr) ** 2).data.mean()  # Pixel-wise MSE
            accumulated_results['accumulated_mse'] += batch_mse * batch_size
            batch_ssim = pytorch_ssim.ssim(sr, hr).item()
            accumulated_results['accumulated_ssim'] += batch_ssim * batch_size
            test_results['ssim'] = accumulated_results['accumulated_ssim'] / accumulated_results['batch_sizes']
            accumulated_results['accumulated_psnr'] += 20 * log10(
                hr.max() / (batch_mse / batch_size)) * batch_size
            test_results['psnr'] = accumulated_results['accumulated_psnr'] / accumulated_results['batch_sizes']
            accumulated_results['accumulated_lpips'] += torch.mean(lpips_metric(hr * 2 - 1, sr * 2 - 1)).to(
                'cpu', non_blocking=True).detach() * batch_size
            test_results['lpips'] = accumulated_results['accumulated_lpips'] / accumulated_results['batch_sizes']

            test_bar.set_description(
                desc=f"[converting LR images to SR images] PSNR: {test_results['psnr']:4f} dB "
                     f"SSIM: {test_results['ssim']:4f} "
                     f"LPIPS: {test_results['lpips']:.4f} ")
            if test_images.size(0) * test_images.size(1) < NUM_LOGGED_TEST_IMAGES * 3:

                if test_images.size(0) == 0:
                    test_images = torch.hstack(
                        (test_display_transform()(test_hr_restore).unsqueeze(0).transpose(0, 1),
                         test_display_transform()(hr.data.cpu()).unsqueeze(0).transpose(0, 1),
                         test_display_transform()(sr.data.cpu()).unsqueeze(0).transpose(0, 1)))
                else:
                    test_images = torch.cat((test_images,
                                            torch.hstack(
                                                (test_display_transform()(test_hr_restore).unsqueeze(
                                                    0).transpose(0, 1),
                                                 test_display_transform()(hr.data.cpu()).unsqueeze(
                                                     0).transpose(0, 1),
                                                 test_display_transform()(sr.data.cpu()).unsqueeze(
                                                     0).transpose(0, 1)))))
        test_results['fid'] = calculate_metrics(test_sr_dataset, test_hr_dataset,
                                                cuda=True, fid=True, verbose=True)['frechet_inception_distance']

        test_images = test_images.view((NUM_LOGGED_TEST_IMAGES, -1, 3, 512, 512))
        test_save_bar = tqdm(test_images, desc='[saving test results]', ncols=160)

        for index, image_batch in enumerate(test_save_bar, start=1):
            image_grid = utils.make_grid(image_batch, nrow=3, padding=5)
            utils.save_image(image_grid, str(images_path / f"{index}.png"), padding=5)

    data_frame = pd.DataFrame(data=test_results, index=[model_name])
    data_frame.to_csv(str(test_folder / f"global_results.csv"), mode='a', index_label="Name")


if __name__ == '__main__':
    main()