'''
This part is used to train the speaker model and evaluate the performances
'''

import torch, sys, os, tqdm, numpy, soundfile, time, pickle, random
import torch.nn as nn
from tools import *
from model import ECAPA_TDNN
import torch.nn.functional as F


class ECAPAModel(nn.Module):
	def __init__(self, lr, lr_decay, C , test_step, num_frames, threshold, alpha, gamma, pos_weight, **kwargs):
		super(ECAPAModel, self).__init__()
		## ECAPA-TDNN
		self.model = ECAPA_TDNN(C = C).cuda()
        #Classifier
		self.alpha = alpha
		self.gamma = gamma
		self.pos_weight = pos_weight
		self.optim           = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 2e-5)
		self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = test_step, gamma=lr_decay)
		print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.model.parameters()) / 1024 / 1024))
		self.num_frames = num_frames
		self.threshold = threshold


	def focal_loss(self, logits, labels):
		bce_loss = F.binary_cross_entropy_with_logits(logits, labels.to(dtype=torch.float), pos_weight = torch.tensor(self.pos_weight), reduction = 'none')
		p_t = torch.exp(-bce_loss)
		focal_loss = self.alpha * (1 - p_t)** self.gamma * bce_loss
		return torch.mean(focal_loss)


	def train_network(self, epoch, loader):
		self.train()
		## Update the learning rate based on the current epcoh
		self.scheduler.step(epoch - 1)
		loss = 0
		lr = self.optim.param_groups[0]['lr']
		for num, (data, labels) in enumerate(loader, start = 1):
			self.optim.zero_grad()
			labels            = torch.LongTensor(labels).cuda()
			logits = self.model.forward(data.cuda(), aug = True).squeeze()
			#nloss = self.loss(logits, labels.to(dtype=torch.float))		
			nloss = self.focal_loss(logits, labels.to(dtype=torch.float)) #handling imbalanced data
			nloss.backward()
			self.optim.step()
			#index += len(labels)
			#top1 += prec
			loss += nloss.detach().cpu().numpy()
			sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
			" [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
			" Train_Loss: %.5f"        %(loss/(num)) + \
            "\n")
			sys.stderr.flush()
		return loss/num, lr

	def validate_network(self, loader):
		self.eval()
		## Update the learning rate based on the current epcoh
		loss = 0
		for num, (data, labels) in enumerate(loader, start = 1):
			labels            = torch.LongTensor(labels).cuda()
			logits = self.model.forward(data.cuda(), aug = False).squeeze()
			nloss = self.focal_loss(logits, labels.to(dtype=torch.float))
			#nloss = self.loss(logits, labels.to(dtype=torch.float))			
			loss += nloss.detach().cpu().numpy()
			sys.stderr.write(" Validate_Loss: %.5f"        %(loss/(num)) + \
            "\n")
			sys.stderr.flush()
		sys.stdout.write("\n")
		return loss/num
	
	def test_network(self, test_list, test_path=None):
		self.eval()
		lines = open(test_list).read().splitlines()
		data_label=[]
		data_list=[]
		prediction=[]

		for index, line in enumerate(lines):
			speaker_label = int(line.split('-')[1]) # This is PHQ Binary {0,1}
			file_name     = line.split('-')[0] # Convert 301 > ~301_AUDIO.wav
			data_label.append(speaker_label)
			data_list.append(file_name)
		
		for i, data in enumerate(data_list):
			audio, sr = soundfile.read(data_list[i])

			if len(audio.shape) == 2 and audio.shape[1] > 1:
				audio = audio[:, 0]  # Extract the left channel
			length = self.num_frames * 160 + 240
			if audio.shape[0] <= length:
				shortage = length - audio.shape[0]
				audio = numpy.pad(audio, (0, shortage), 'wrap')
			start_frame = numpy.int64(random.random()*(audio.shape[0]-length))
			audio = audio[start_frame:start_frame + length]
			audio = numpy.stack([audio],axis=0)
			audio = torch.FloatTensor(audio[0])
			audio = audio.unsqueeze(0) # For Batch=1

			prediction.append(self.model(audio.cuda(), aug=False).item())
		
		# Choose a threshold (e.g., 0.5) to convert probabilities to binary predictions
		predicted_labels = numpy.array([1 if prob >= self.threshold else 0 for prob in prediction])
		true_labels = numpy.array([0 if label < 11 else 1 for label in data_label])
        
		# Calculate and print the accuracy
		acc = (true_labels == predicted_labels).mean()

		# Calculate and print the F1 score
		tp = ((true_labels == 1) & (predicted_labels == 1)).sum()
		fp = ((true_labels == 0) & (predicted_labels == 1)).sum()
		fn = ((true_labels == 1) & (predicted_labels == 0)).sum()
		tn = ((true_labels == 0) & (predicted_labels == 0)).sum()
        
		precision = tp / (tp + fp)
		recall = tp / (tp + fn)
		specifity = tn / (tn + fp)

		f1 = 2 * (precision * recall) / (precision + recall)
		return f1, acc, (tp, fp, fn, tn) , precision, recall, specifity
    #precision, recall

	def save_parameters(self, path):
		torch.save(self.state_dict(), path)

        
	def load_parameters(self, path):
		self_state = self.state_dict()
		loaded_state = torch.load(path)
		for name, param in loaded_state.items():
			origname = name
			if name not in self_state:
				name = name.replace("module.", "")
				if name not in self_state:
					print("%s is not in the model."%origname)
					continue
			if self_state[name].size() != loaded_state[origname].size():
				print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
				continue
			self_state[name].copy_(param)


'''
#bce loss with weights
class ECAPAModel(nn.Module):
	def __init__(self, lr, lr_decay, C , test_step, num_frames, threshold, pos_weight, **kwargs):
		super(ECAPAModel, self).__init__()
		## ECAPA-TDNN
		self.model = ECAPA_TDNN(C = C).cuda()
		## Classifier
		#self.loss	 = nn.BCELoss()
		self.loss	 = None
		self.pos_weight = pos_weight
		self.optim           = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 2e-5)
		self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = test_step, gamma=lr_decay)
		print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.model.parameters()) / 1024 / 1024))
		self.num_frames = num_frames
		self.threshold = threshold

		

	def train_network(self, epoch, loader):
		self.train()
		## Update the learning rate based on the current epcoh
		self.scheduler.step(epoch - 1)
		loss = 0
		lr = self.optim.param_groups[0]['lr']
		for num, (data, labels) in enumerate(loader, start = 1):
			self.zero_grad()
			labels            = torch.LongTensor(labels).cuda()
			logits = self.model.forward(data.cuda(), aug = True).squeeze()
			#nloss = self.loss(logits, labels.to(dtype=torch.float))		
			nloss = F.binary_cross_entropy_with_logits(logits, labels.to(dtype=torch.float), pos_weight = torch.tensor(self.pos_weight))
			nloss.backward()
			self.optim.step()
			#index += len(labels)
			#top1 += prec
			loss += nloss.detach().cpu().numpy()
			sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
			" [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
			" Train_Loss: %.5f"        %(loss/(num)) + \
            "\n")
			sys.stderr.flush()
		return loss/num, lr

	def validate_network(self, loader):
		self.eval()
		## Update the learning rate based on the current epcoh
		loss = 0
		for num, (data, labels) in enumerate(loader, start = 1):
			labels            = torch.LongTensor(labels).cuda()
			logits = self.model.forward(data.cuda(), aug = False).squeeze()
			nloss = F.binary_cross_entropy_with_logits(logits, labels.to(dtype=torch.float), pos_weight = torch.tensor(self.pos_weight))
			#nloss = self.loss(logits, labels.to(dtype=torch.float))			
			loss += nloss.detach().cpu().numpy()
			sys.stderr.write(" Validate_Loss: %.5f"        %(loss/(num)) + \
            "\n")
			sys.stderr.flush()
		sys.stdout.write("\n")
		return loss/num
	
	def test_network(self, test_list, test_path=None):
		self.eval()
		lines = open(test_list).read().splitlines()
		data_label=[]
		data_list=[]
		prediction=[]

		for index, line in enumerate(lines):
			speaker_label = int(line.split('-')[1]) # This is PHQ Binary {0,1}
			file_name     = line.split('-')[0] # Convert 301 > ~301_AUDIO.wav
			data_label.append(speaker_label)
			data_list.append(file_name)
		
		for i, data in enumerate(data_list):
			audio, sr = soundfile.read(data_list[i])

			if len(audio.shape) == 2 and audio.shape[1] > 1:
				audio = audio[:, 0]  # Extract the left channel
			length = self.num_frames * 160 + 240
			if audio.shape[0] <= length:
				shortage = length - audio.shape[0]
				audio = numpy.pad(audio, (0, shortage), 'wrap')
			start_frame = numpy.int64(random.random()*(audio.shape[0]-length))
			audio = audio[start_frame:start_frame + length]
			audio = numpy.stack([audio],axis=0)
			audio = torch.FloatTensor(audio[0])
			audio = audio.unsqueeze(0) # For Batch=1

			prediction.append(self.model(audio.cuda(), aug=False).item())
		
		# Choose a threshold (e.g., 0.5) to convert probabilities to binary predictions
		predicted_labels = numpy.array([1 if prob >= self.threshold else 0 for prob in prediction])
		true_labels = numpy.array([0 if label < 11 else 1 for label in data_label])
        
		# Calculate and print the accuracy
		acc = (true_labels == predicted_labels).mean()

		# Calculate and print the F1 score
		tp = ((true_labels == 1) & (predicted_labels == 1)).sum()
		fp = ((true_labels == 0) & (predicted_labels == 1)).sum()
		fn = ((true_labels == 1) & (predicted_labels == 0)).sum()
		tn = ((true_labels == 0) & (predicted_labels == 0)).sum()
        
		precision = tp / (tp + fp)
		recall = tp / (tp + fn)
		specifity = tn / (tn + fp)

		f1 = 2 * (precision * recall) / (precision + recall)
		return f1, acc, (tp, fp, fn, tn) , precision, recall, specifity
    #precision, recall

	def save_parameters(self, path):
		torch.save(self.state_dict(), path)

        
	def load_parameters(self, path):
		self_state = self.state_dict()
		loaded_state = torch.load(path)
		for name, param in loaded_state.items():
			origname = name
			if name not in self_state:
				name = name.replace("module.", "")
				if name not in self_state:
					print("%s is not in the model."%origname)
					continue
			if self_state[name].size() != loaded_state[origname].size():
				print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
				continue
			self_state[name].copy_(param)
            
'''
