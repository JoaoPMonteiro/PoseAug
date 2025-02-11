from __future__ import print_function, absolute_import, division

import os
import os.path as path
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

#from function_baseline.config import get_parse_args
from function_baseline.data_preparation import data_preparation
from function_baseline.model_pos_preparation import model_pos_preparation
from function_poseaug.model_pos_eval import evaluate

import onnx
import onnxruntime

import argparse



def get_parse_args():
    parser = argparse.ArgumentParser(description='PyTorch training script')

    # General arguments
    parser.add_argument('--dataset', default='h36m', type=str, metavar='NAME', help='target dataset')
    parser.add_argument('--keypoints', default='gt', type=str, metavar='NAME', help='2D detections to use, \
    gt/hr/cpn_ft_h36m_dbb/detectron_ft_h36m')
    parser.add_argument('--actions', default='*', type=str, metavar='LIST',
                        help='actions to train/test on, separated by comma, or * for all')
    parser.add_argument('--checkpoint', default='checkpoint/debug', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('--snapshot', default=25, type=int, help='save models_baseline for every (default: 20)')
    parser.add_argument('--note', default='debug', type=str, help='additional name on checkpoint directory')

    # Evaluate choice
    parser.add_argument('--evaluate', default='', type=str, metavar='FILENAME',
                        help='checkpoint to evaluate (file name)')
    parser.add_argument('--action-wise', default=True, type=lambda x: (str(x).lower() == 'true'), help='train s1only')

    # Model arguments
    parser.add_argument('--posenet_name', default='videopose', type=str, help='posenet: gcn/stgcn/videopose/mlp')
    parser.add_argument('--output_onnx_name', default='onnx', type=str, help='name for output onnx') # wally
    parser.add_argument('--device_selected', default='cpu', type=str, help='option to select device') # wally
    parser.add_argument('--stages', default=4, type=int, metavar='N', help='stages of baseline model')
    parser.add_argument('--dropout', default=0.25, type=float, help='dropout rate')

    # Training detail
    parser.add_argument('--batch_size', default=1024, type=int, metavar='N',
                        help='batch size in terms of predicted frames')
    parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of training epochs')

    # Learning rate
    parser.add_argument('--lr', default=1.0e-3, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_decay', type=int, default=100000, help='num of steps of learning rate decay')
    parser.add_argument('--lr_gamma', type=float, default=0.96, help='gamma of learning rate decay')
    parser.add_argument('--no_max', dest='max_norm', action='store_false', help='if use max_norm clip on grad')
    parser.set_defaults(max_norm=True)

    # Experimental setting
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR', help='downsample frame rate by factor')
    parser.add_argument('--pretrain', default=False, type=lambda x: (str(x).lower() == 'true'), help='used in poseaug')
    parser.add_argument('--s1only', default=False, type=lambda x: (str(x).lower() == 'true'), help='train S1 only')
    parser.add_argument('--num_workers', default=2, type=int, metavar='N', help='num of workers for data loading')

    args = parser.parse_args()

    return args

def export_model(model_pos, posenet_name_output, device_selected):
    model_pos.eval()

    #dummy_input = torch.randn(1, 1, 16, 2, device='cuda')
    #dummy_input = torch.randn(1, 16, 2, device='cuda')
    #dummy_input = torch.randn(1, 16, 2, device='cpu')
    dummy_input = torch.randn(1, 16, 2, device=device_selected)
    #dummy_input = torch.randn(1, 1, 16, 2, device=device_selected)


    dummy_output = model_pos(dummy_input)

    torch.onnx.export(model_pos, dummy_input, posenet_name_output + '.onnx', opset_version=12, export_params=True,
                      verbose=True, input_names=['input'], output_names=['output'])

    print("done exporting")

    print("==> Checking model...")

    onnx_model = onnx.load(posenet_name_output + '.onnx')
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(posenet_name_output + '.onnx')

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(dummy_output), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def main(args):
    print('==> Using settings {}'.format(args))
    stride = args.downsample
    device_selected = args.device_selected
    posenet_name_output = args.output_onnx_name
    cudnn.benchmark = True
    #device = torch.device("cuda")
    #device = torch.device("cpu")
    device = torch.device(device_selected)

    print('==> Loading dataset...')
    data_dict = data_preparation(args)

    print("==> Creating model...")
    model_pos = model_pos_preparation(args, data_dict['dataset'], device)

    # Check if evaluate checkpoint file exist:
    assert path.isfile(args.evaluate), '==> No checkpoint found at {}'.format(args.evaluate)
    print("==> Loading checkpoint '{}'".format(args.evaluate))
    ckpt = torch.load(args.evaluate)
    try:
        model_pos.load_state_dict(ckpt['state_dict'])
    except:
        model_pos.load_state_dict(ckpt['model_pos'])

    print(posenet_name_output)
    print('--')
    wertkilo = model_pos.forward
    export_model(model_pos, posenet_name_output, device_selected)
    print('--')

    #print('==> Evaluating...')

    #error_h36m_p1, error_h36m_p2 = evaluate(data_dict['H36M_test'], model_pos, device)
    #print('H36M: Protocol #1   (MPJPE) overall average: {:.2f} (mm)'.format(error_h36m_p1))
    #print('H36M: Protocol #2 (P-MPJPE) overall average: {:.2f} (mm)'.format(error_h36m_p2))

    #error_3dhp_p1, error_3dhp_p2 = evaluate(data_dict['3DHP_test'], model_pos, device, flipaug='_flip')
    #print('3DHP: Protocol #1   (MPJPE) overall average: {:.2f} (mm)'.format(error_3dhp_p1))
    #print('3DHP: Protocol #2 (P-MPJPE) overall average: {:.2f} (mm)'.format(error_3dhp_p2))


if __name__ == '__main__':
    args = get_parse_args()
    # fix random
    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    # copy from #https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True

    main(args)
