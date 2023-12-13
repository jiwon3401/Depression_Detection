'''
This is the main code of the ECAPATDNN project, to define the parameters and build the construction
'''

import argparse, glob, os, torch, warnings, time
from tools import *
from dataLoader import train_loader, validate_loader
from ECAPAModel import ECAPAModel

parser = argparse.ArgumentParser(description = "ECAPA_trainer")
## Training Settings
parser.add_argument('--num_frames', type=int,   default=1000,     help='Duration of the input segments, eg: 200 for 2 second')
parser.add_argument('--max_epoch',  type=int,   default=80,      help='Maximum number of epochs')
parser.add_argument('--batch_size', type=int,   default=32,     help='Batch size')
parser.add_argument('--n_cpu',      type=int,   default=0,       help='Number of loader threads')
parser.add_argument('--test_step',  type=int,   default=1,       help='Test and save every [test_step] epochs')
parser.add_argument('--lr',         type=float, default=0.001,   help='Learning rate')
parser.add_argument("--lr_decay",   type=float, default=0.97,    help='Learning rate decay every [test_step] epochs')

## Training and evaluation path/lists, save path
parser.add_argument('--train_list', type=str,   default="/workspace/blocks1/nasw_bak_20231018/utils/train_split.txt",     
                    help='The path of the training list')
parser.add_argument('--validate_list',  type=str,   default="/workspace/blocks1/nasw_bak_20231018/utils/validation_split.txt",
                    help='The path of the validation list')
parser.add_argument('--test_list',  type=str,   default="/workspace/blocks1/nasw_bak_20231018/utils/test_split.txt",
                    help='The path of the test list')
# parser.add_argument('--musan_path', type=str,   default="/home/user/SJ/DATA/musan",                    
#                     help='The path to the MUSAN set, eg:"/data08/Others/musan_split" in my case')
parser.add_argument('--rir_path',   type=str,   default="/workspace/blocks1/nasw_bak_20231018/RIRS_NOISES/simulated_rirs",
                    help='The path to the RIR set, eg:"/data08/Others/RIRS_NOISES/simulated_rirs" in my case')
parser.add_argument('--save_path',  type=str,   default="exp/default",                                     
                    help='Path to save the score.txt and models')
parser.add_argument('--initial_model',  type=str,   
                    #default="/workspace/blocks1/nasw_bak_20231018/weights/th0.4_N1000_B32_acc0.62_f0.59.model",
                    help='Path of the initial_model')
parser.add_argument('--model_save_path',  type=str,   
                    #default="/workspace/blocks1/nasw_bak_20231018/weights/th0.4_N1000_B32_acc0.62_f0.59.model",
                    #/workspace/blocks1/nasw_bak_20231018/7.1/exp/1025_N3000_Scratch/model/model_0001.model
                    help='Path of the resume_model')


## Model and Loss settings
parser.add_argument('--C', type=int, default=1024, help='Channel size for the speaker encoder')
parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for prediction')
#focal loss parameters - alpha, gamma, pos_weight
parser.add_argument('--alpha', type=float, default=0.25, help='balancing parameter for class imbalance in Focal Loss')
parser.add_argument('--gamma', type=float, default=2, help='higher value gives more weight to misclassfied examples in Focal Loss')
parser.add_argument('--pos_weight', type=float, default=2.0, help='Weight for positive class in Focal Loss(or bceloss with weights)') #10528/2272


## Command
parser.add_argument('--eval', dest='eval', action='store_true', help='Only do evaluation')

## Initialization
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()
args = init_args(args)

## Define the data loader
trainloader = train_loader(train_list=args.train_list, rir_path=args.rir_path, num_frames=args.num_frames)
validateloader = validate_loader(validate_list=args.validate_list, num_frames=args.num_frames)
testloader = validate_loader(validate_list=args.test_list, num_frames=args.num_frames )

trainLoader = torch.utils.data.DataLoader(trainloader, batch_size = args.batch_size, shuffle = True, num_workers = args.n_cpu, drop_last = True)
validateLoader = torch.utils.data.DataLoader(validateloader, batch_size = args.batch_size, shuffle = False, num_workers = args.n_cpu, drop_last = False)
testLoader = torch.utils.data.DataLoader(testloader, batch_size = args.batch_size, shuffle = False, num_workers = args.n_cpu, drop_last = False)


## Search for the exist models
modelfiles = glob.glob('%s/model_0*.model'%args.model_save_path)
modelfiles.sort()

## Only do evaluation, the initial_model is necessary
if args.eval == True:
	s = ECAPAModel(**vars(args))
	print("Model %s loaded from previous state!"%args.initial_model)
	s.load_parameters(args.initial_model)
	f1, acc, confusion_matrix,precision, recall, specifity = s.test_network(test_list = args.test_list)
	print("acc: %2.2f, f1: %2.2f"%(acc, f1), "TP, FP, FN, TN :",confusion_matrix, "precision: %2.2f, recall: %2.2f, specifity: %2.2f"%(precision, recall,specifity))
	quit()

## If initial_model is exist, system will train from the initial_model
if args.initial_model != 'None':
	print("Model %s loaded from previous state!"%args.initial_model)
	s = ECAPAModel(**vars(args))
	s.load_parameters(args.initial_model)
	epoch = 1

## Otherwise, system will try to start from the saved model&epoch
elif len(modelfiles) >= 1:
	print("Model %s loaded from previous state!"%modelfiles[-1])
	epoch = str(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1 #int() -> error occured
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
score_file.write("Num_frames %d, Batch %d, Channel_size %d, Threshold %2.2f\n\n"%(args.num_frames, args.batch_size, args.C, args.threshold))


for epoch in range(1, args.max_epoch+1):
	## Training for one epoch
	train_loss, lr = s.train_network(epoch = epoch, loader = trainLoader)
	validate_loss = s.validate_network(loader = validateLoader) 
    
	## Evaluation every [test_step] epochs
	if epoch % args.test_step == 0:
		s.save_parameters(args.model_save_path + "/model_%04d.model"%epoch)
		f1, acc, confusion_matrix, precision, recall, specifity = s.test_network(test_list = args.test_list)
		F1score.append(f1)
		Acc.append(acc)
		print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, ACC: %2.2f, F1score %2.2f\n"%(epoch, Acc[-1], F1score[-1]))
		score_file.write("%d epoch, LR %f, Training_loss %f, Validation_LOSS %f, ACC %2.2f, F1score %2.2f, TP:%d FP:%d FN:%d TN:%d, Precision: %2.2f , Recall: %2.2f, Specifity %2.2f \n"%(epoch, lr, train_loss, validate_loss, Acc[-1], F1score[-1],confusion_matrix[0],confusion_matrix[1],confusion_matrix[2],confusion_matrix[3], precision, recall, specifity))
		score_file.flush()

