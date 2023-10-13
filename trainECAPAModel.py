'''
This is the main code of the ECAPATDNN project, to define the parameters and build the construction
'''

import argparse, glob, os, torch, warnings, time
from tools import *
from dataLoader import train_loader, validate_loader
from ECAPAModel import ECAPAModel

parser = argparse.ArgumentParser(description = "ECAPA_trainer")
## Training Settings
parser.add_argument('--num_frames', type=int,   default=200,     help='Duration of the input segments, eg: 200 for 2 second')
parser.add_argument('--max_epoch',  type=int,   default=80,      help='Maximum number of epochs')
parser.add_argument('--batch_size', type=int,   default=16,     help='Batch size')
parser.add_argument('--n_cpu',      type=int,   default=0,       help='Number of loader threads')
parser.add_argument('--test_step',  type=int,   default=1,       help='Test and save every [test_step] epochs')
parser.add_argument('--lr',         type=float, default=0.001,   help='Learning rate')
parser.add_argument("--lr_decay",   type=float, default=0.97,    help='Learning rate decay every [test_step] epochs')

## Training and evaluation path/lists, save path
parser.add_argument('--train_list', type=str,   default="/home/user/sevenpointone/DAIC-WOZ/labels/train_split.txt",     help='The path of the training list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt')
parser.add_argument('--train_path', type=str,   default="/home/user/sevenpointone/DAIC-WOZ/audio_files/train_split",                    help='The path of the training data, eg:"/data08/VoxCeleb2/train/wav" in my case')
parser.add_argument('--validate_list',  type=str,   default="/home/user/sevenpointone/DAIC-WOZ/labels/dev_split.txt",              help='The path of the validation list')
parser.add_argument('--validate_path',  type=str,   default="/home/user/sevenpointone/DAIC-WOZ/audio_files/val_split",                    help='The path of the Validation data, eg:"/data08/VoxCeleb1/test/wav" in my case')
parser.add_argument('--test_list',  type=str,   default="/home/user/sevenpointone/DAIC-WOZ/labels/test_split.txt",              help='The path of the test list')
parser.add_argument('--test_path',  type=str,   default="/home/user/sevenpointone/DAIC-WOZ/audio_files/test_split",                    help='The path of the test data, eg:"/data08/VoxCeleb1/test/wav" in my case')
parser.add_argument('--musan_path', type=str,   default="/home/user/SJ/DATA/musan",                    help='The path to the MUSAN set, eg:"/data08/Others/musan_split" in my case')
parser.add_argument('--rir_path',   type=str,   default="/home/user/sevenpointone/RIRS_NOISES/simulated_rirs",     help='The path to the RIR set, eg:"/data08/Others/RIRS_NOISES/simulated_rirs" in my case');
parser.add_argument('--save_path',  type=str,   default="exps/exp1",                                     help='Path to save the score.txt and models')
parser.add_argument('--initial_model',  type=str,   default="",                                          help='Path of the initial_model')

## Model and Loss settings
parser.add_argument('--C',       type=int,   default=1024,   help='Channel size for the speaker encoder')
parser.add_argument('--threshold',       type=float,   default=0.5,   help='Threshold for prediction')

## Command
parser.add_argument('--eval',    dest='eval', action='store_true', help='Only do evaluation')

## Initialization
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()
args = init_args(args)

## Define the data loader
trainloader = train_loader(**vars(args))
validateloader = validate_loader(**vars(args))
trainLoader = torch.utils.data.DataLoader(trainloader, batch_size = args.batch_size, shuffle = True, num_workers = args.n_cpu, drop_last = True)
validateLoader = torch.utils.data.DataLoader(validateloader, batch_size = args.batch_size, shuffle = True, num_workers = args.n_cpu, drop_last = True)

## Search for the exist models
modelfiles = glob.glob('%s/model_0*.model'%args.model_save_path)
modelfiles.sort()

## Only do evaluation, the initial_model is necessary
if args.eval == True:
	s = ECAPAModel(**vars(args))
	print("Model %s loaded from previous state!"%args.initial_model)
	s.load_parameters(args.initial_model)
	EER, minDCF = s.eval_network(eval_list = args.eval_list, eval_path = args.eval_path)
	print("EER %2.2f%%, minDCF %.4f%%"%(EER, minDCF))
	quit()

## If initial_model is exist, system will train from the initial_model
if args.initial_model != "":
	print("Model %s loaded from previous state!"%args.initial_model)
	s = ECAPAModel(**vars(args))
	s.load_parameters(args.initial_model)
	epoch = 1

## Otherwise, system will try to start from the saved model&epoch
elif len(modelfiles) >= 1:
	print("Model %s loaded from previous state!"%modelfiles[-1])
	epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
	s = ECAPAModel(**vars(args))
	s.load_parameters(modelfiles[-1])
## Otherwise, system will train from scratch
else:
	epoch = 1
	s = ECAPAModel(**vars(args))

F1score = []
Acc = []
score_file = open(args.score_save_path, "a+")

# Score file log parameters
score_file.write("Num_frames %d, Batch %d, Channel_size %d, Threshold %2.2f\n"%(args.num_frames, args.batch_size, args.C, args.threshold))
while(1):
	## Training for one epoch
	train_loss, lr = s.train_network(epoch = epoch, loader = trainLoader)
	validate_loss = s.validate_network(loader = validateLoader)

	## Evaluation every [test_step] epochs
	if epoch % args.test_step == 0:
		s.save_parameters(args.model_save_path + "/model_%04d.model"%epoch)
		f1, acc = s.test_network(test_list = args.test_list, test_path = args.test_path)
		F1score.append(f1)
		Acc.append(acc)
		print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, ACC %2.2f%%, F1score %2.2f%%"%(epoch, Acc[-1], F1score[-1]))
		score_file.write("%d epoch, LR %f, Train_LOSS %f, Validate_LOSS %f, ACC %2.2f%%, F1score %2.2f%%\n"%(epoch, lr, train_loss, validate_loss, Acc[-1], F1score[-1]))
		score_file.flush()

	if epoch >= args.max_epoch:
		quit()

	epoch += 1
