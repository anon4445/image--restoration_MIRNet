from gooey import Gooey, GooeyParser
import os
import torch
import cv2
import matplotlib.pyplot as plt
from natsort import natsorted
from glob import glob
from runpy import run_path 
from skimage import img_as_ubyte
import torch.nn.functional as F
from tqdm import tqdm

@Gooey(program_name="Image Processing with MIRNet_v2", default_size=(700, 500))
def main():
    parser = GooeyParser(description="Choose Task and Input Image")
    parser.add_argument('task', metavar='Task', help='Choose the processing task', choices=['real_denoising', 'super_resolution', 'contrast_enhancement', 'lowlight_enhancement'])
    parser.add_argument('input_dir', metavar='Input Directory', help='Choose the directory containing input images', widget='DirChooser')
    parser.add_argument('out_dir', metavar='Output Directory', help='Choose the directory containing output images', widget='DirChooser')
    args = parser.parse_args()
    process_images(args.task, args.input_dir, args.out_dir)

def process_images(task, input_dir, out_dir):
    def get_weights_and_parameters(task):
        parameters = {
            'inp_channels': 3,
            'out_channels': 3,
            'n_feat': 80,
            'chan_factor': 1.5,
            'n_RRG': 4,
            'n_MRB': 2,
            'height': 3,
            'width': 2,
            'bias': False,
            'scale': 1,
            'task': task
        }
        if task == 'real_denoising':
            weights = os.path.join('model', 'real_denoising.pth')
        elif task == 'super_resolution':
            weights = os.path.join('model', 'sr_x2.pth')
            parameters['scale'] = 2
        elif task == 'contrast_enhancement':
            weights = os.path.join('model', 'enhancement_fivek.pth')
        elif task == 'lowlight_enhancement':
            weights = os.path.join('model', 'enhancement_lol.pth')
        return weights, parameters

    weights, parameters = get_weights_and_parameters(task)
    print(f"\n ==> Running {task} with weights {weights}\n ")

    load_arch = run_path(os.path.join('mirnet_v2_arch.py'))
    model = load_arch['MIRNet_v2'](**parameters)
    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint['params'])
    #model.eval()
    extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
    files = natsorted(glob(os.path.join(input_dir, '*')))
    img_multiple_of = 4
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        for filepath in tqdm(files):
            img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
            input_ = torch.from_numpy(img).float().div(255.).permute(2,0,1).unsqueeze(0)
            h,w = input_.shape[2], input_.shape[3]
            H,W = ((h+img_multiple_of)//img_multiple_of)*img_multiple_of, ((w+img_multiple_of)//img_multiple_of)*img_multiple_of
            padh = H-h if h%img_multiple_of!=0 else 0
            padw = W-w if w%img_multiple_of!=0 else 0
            input_ = F.pad(input_, (0,padw,0,padh), 'reflect')
            restored = model(input_)
            restored = torch.clamp(restored, 0, 1)
            restored = restored[:,:,:h,:w]
            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
            restored = img_as_ubyte(restored[0])
            filename = os.path.split(filepath)[-1]
            cv2.imwrite(os.path.join(out_dir, filename),cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))
    inp_filenames = natsorted(glob(os.path.join(input_dir, '*')))
    out_filenames = natsorted(glob(os.path.join(out_dir, '*')))
    num_display_images = 5
    if len(inp_filenames) > num_display_images:
        inp_filenames = inp_filenames[:num_display_images]
        out_filenames = out_filenames[:num_display_images]
    print(f"Results: {task}")
    for inp_file, out_file in zip(inp_filenames, out_filenames):
        degraded = cv2.cvtColor(cv2.imread(inp_file), cv2.COLOR_BGR2RGB)
        restored = cv2.cvtColor(cv2.imread(out_file), cv2.COLOR_BGR2RGB)
        fig, axes = plt.subplots(nrows=1, ncols=2)
        dpi = fig.get_dpi()
        fig.set_size_inches(900 / dpi, 448 / dpi)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        axes[0].axis('off')
        axes[0].imshow(degraded)
        axes[1].axis('off')
        axes[1].imshow(restored)
        plt.show()

if __name__ == '__main__':
    main()
