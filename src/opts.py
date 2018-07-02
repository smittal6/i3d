import argparse

parser = argparse.ArgumentParser(description="Script for training i3d on UCF-101 with different options")

parser.add_argument("--trainlist", help="Training file list", type=str, default='../list/trainlist01.txt')
parser.add_argument("--testlist", help="Testing file list", type=str, default='../list/protestlist01.txt')
parser.add_argument('--nstr', type=str, default=None, help='string to help in logdir')

# ====================================== Model Configs =====================================
parser.add_argument('--modality', type=str, default='rgb', help='rgb / rgbdsc / flow / flowdsc / flyflow')
parser.add_argument('--wts', type=str, default='rgb', help='rgb/flow')
parser.add_argument('--mean', type=bool, default=False, help='While transforming the weights use mean or not')
parser.add_argument('--random', type=str, default=False, help='whether the first layer should have random weights')
parser.add_argument('--dog', type=bool, default=False, help='To apply a diff of gaussians wts as the first layer')


# ====================================== Learning configs ================================
parser.add_argument("--epochs", help="number of iterations", type=int, default=50)
parser.add_argument("--lr", help="Starting LR (would remain same in case of no sched)", type=float, default=1e-1)
parser.add_argument("--batch", help="batch-size", type=int, default=6)
parser.add_argument("--testbatch", help="test batch-size", type=int, default=6)
parser.add_argument('--sched', type=str, default=False, help='Use a scheduler or not')
parser.add_argument('--lr_steps', default=[2,6,10,13], type=int, nargs="+", help='epochs to decay learning rate by 10')


# ====================================== Runtime configs ================================
parser.add_argument("--gpus", help="Number of GPUs to train on", type=int, default=1)
parser.add_argument("--numw", help="number of workers on loading data", type=int, default=8)
parser.add_argument('--resume', type=str, default=None, help='Resume training from this file')
parser.add_argument('--ft', type=bool, default=False, help='Finetune the model or not')

# ===================================== USELESS ==========================================
parser.add_argument("--eval", help="Just put to keep constistency with original model (rgb)", type=str, default='rgb')

