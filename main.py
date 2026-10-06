import os
import sys
import json
import numpy as np
import torch, gc
import matplotlib.pyplot as plt

device = torch.device("cuda:0")
torch.cuda.set_device(device)
print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
print(f"torch.cuda.get_device_name(): {torch.cuda.get_device_name(0)}")

from model.pointtransformer.pointtransformerlayer import pt_repro as Model
from types import SimpleNamespace

c = 3
k = 3
B = 16

# Mean Configuration
cfg_mean = SimpleNamespace()
cfg_mean.num_encoder = 6
cfg_mean.planes = [16, 32, 64, 128, 256, 512]
cfg_mean.blocks = [2, 3, 4, 5, 6, 3]
cfg_mean.share_planes = 8
cfg_mean.stride = [1, 5, 4, 4, 4, 4]
cfg_mean.nsample = [8, 8, 16, 16, 16, 16]

# Body Configuration
cfg_body = SimpleNamespace()
cfg_body.num_encoder = 6
cfg_body.planes = [16, 32, 64, 128, 256, 512]
cfg_body.blocks = [2, 3, 4, 5, 6, 3]
cfg_body.share_planes = 8
cfg_body.stride = [1, 4, 4, 4, 4, 4]
cfg_body.nsample = [8, 8, 16, 16, 16, 16]

# Decoder Configuration
cfg_decoder = SimpleNamespace()
cfg_decoder.num_decoder = 6
cfg_decoder.planes = cfg_body.planes[::-1]
cfg_decoder.blocks = cfg_body.blocks[::-1]
cfg_decoder.share_planes = cfg_body.share_planes
cfg_decoder.nsample = cfg_body.nsample[::-1]


import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--training_list_file_path",type=str,required=True,default="",help="Training Set patient list file path")
parser.add_argument("--testing_list_file_path",type=str,required=True,default="",help="Testing Set patient file path")
parser.add_argument("--validation_list_file_path",type=str,default="",help="Validation Set patient file path (optional): the checkpoint is then chosen, and early stopping decided, on the validation loss")
parser.add_argument("--label_json_file_path",type=str,required=True,default="",help="JSON file path for label organs")
parser.add_argument("--data_root",type=str,required=True,default="",help="Data root path (For loading point clouds)")
parser.add_argument("--init_mean_shape_path",type=str,required=True,default="",help="Init mean shape file path")
parser.add_argument("--conversion_path",type=str,required=True,default="",help="Conversion file path for affine transformations")
parser.add_argument("--save_path", type=str, required=True, default="", help="Path to save results")
parser.add_argument("--which_label",type=str,required=True,default="",help="Which label set to use")
parser.add_argument("--batch_size",type=int,default=16,help="Batch size")
parser.add_argument("--epochs",type=int,default=100,help="Maximum number of epochs")
parser.add_argument("--lr",type=float,default=1e-3,help="Adam learning rate")
parser.add_argument("--milestones",type=str,default="25,50,75",help="Epochs at which the learning rate is multiplied by --gamma (comma separated, empty for none)")
parser.add_argument("--gamma",type=float,default=0.3,help="Learning-rate decay factor at each milestone")
parser.add_argument("--patience",type=int,default=6,help="Stop after this many epochs without a new best loss - validation loss if a validation set is given, else training loss (0 = never stop early)")
parser.add_argument("--monitor_test",action="store_true",help="Report test loss and test CD (mm, model vs template) after every epoch")
parser.add_argument("--residual",action="store_true",help="Predict template + residual_scale * correction instead of absolute coordinates")
parser.add_argument("--residual_scale",type=float,default=0.02,help="Size of a unit correction in normalised coordinates (--residual)")
parser.add_argument("--bn_recalibrate",action=argparse.BooleanOptionalAction,default=True,help="Recompute BatchNorm running statistics on the training set before evaluating")
parser.add_argument("--seed",type=int,default=0,help="Random seed (python, numpy, torch)")
parser.add_argument("--eval_only",action="store_true",help="Skip training; evaluate models/model_weights_best.pth in --save_path")

args = parser.parse_args()
# fixed seeds for repeatable runs (CUDA kernels such as pointops can still differ slightly between runs)
import random
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
B = args.batch_size


training_patient_list_file   = args.training_list_file_path
test_patient_list_file       = args.testing_list_file_path

# Load Patient Data
training_data = {}
train_patient_list = []
with open(training_patient_list_file, "r") as f:
	for line in f:
		train_patient_list.append(line.strip()) 
test_data = {}
test_patient_list = []
with open(test_patient_list_file, "r") as f:
	for line in f:
		test_patient_list.append(line.strip()) 
		
# Load Init Data
label_data_root = args.label_json_file_path
data_file = args.init_mean_shape_path
if data_file is not None:
	init_mean_data = np.load(data_file, allow_pickle=True)
	init_mean = init_mean_data["mean_pc_np"]
	init_mean_labels = init_mean_data["mean_label_np"]

which_label = args.which_label
with open(os.path.join(label_data_root, f"label_organs_{which_label}.json")) as f:
    label_organs = json.load(f)
	
# Select the given labels for init mean
keep = np.array([int(label) for label in label_organs.keys()])
mask = np.isin(init_mean_labels, keep)
init_mean_labels = init_mean_labels[mask]
init_mean = init_mean[mask]

data_root = args.data_root
if os.path.exists(data_root):
	loaded = np.load(data_root)
	data = {}
	for key in loaded.files:
		main, sub = key.split("__")
		data.setdefault(main, {})[sub] = loaded[key]
		
print(label_organs)

# sizes from the data: body points per subject, points per organ, number of organs
first_patient = data[train_patient_list[0]]
N_in = first_patient["input_points"].shape[0]
N_points = first_patient[next(iter(label_organs.values()))].shape[0]
N_classes = len(label_organs)
N_out = N_points * N_classes
assert len(init_mean) == N_out, f"mean shape has {len(init_mean)} points after label filtering, expected {N_out}"

# the two encoders are fused point-wise at the bottleneck, so both must reach the same
# number of points: N_in / prod(body strides) == N_out / prod(mean strides). The first
# downsampling stride of the mean encoder is set to make this hold.
bottleneck = N_in // int(np.prod(cfg_body.stride))
cfg_mean.stride[1] = N_out // (bottleneck * int(np.prod(cfg_mean.stride[2:])))
assert N_in == bottleneck * int(np.prod(cfg_body.stride)) and N_out == bottleneck * int(np.prod(cfg_mean.stride)), \
	f"cannot match encoder bottlenecks for N_in={N_in}, N_out={N_out}; adjust the strides"

print(cfg_mean)
print(cfg_body)
print(cfg_decoder)

model = Model(cfg_mean, cfg_body, cfg_decoder, c=c, k=k)
if args.residual:
	# zero-initialise the last layer: an untrained model returns the template exactly
	torch.nn.init.zeros_(model.cls[-1].weight)
	torch.nn.init.zeros_(model.cls[-1].bias)

def predict(body_coord, body_offset, mean_coord, mean_offset):
	"""Model output; with --residual a correction of each template point, so output point i stays
	template point i moved (needed e.g. for tools/volume_eval.py)."""
	out = model([body_coord, body_coord, body_offset], [mean_coord, mean_coord, mean_offset])
	return mean_coord + args.residual_scale * out if args.residual else out

train_input_points = torch.zeros((len(train_patient_list), N_in, 3), dtype=torch.float32)
train_target_points = torch.zeros((len(train_patient_list), N_out, 3), dtype=torch.float32)
train_target_labels = torch.zeros((len(train_patient_list), N_out), dtype=torch.int64)

for patient_id in train_patient_list:
	
	patient_data = data[patient_id]
	patient_target_points = torch.zeros((N_out,3), dtype=torch.float32)
	patient_target_labels = torch.zeros((N_out,), dtype=torch.int64)
	
	for idx, (label, organ) in enumerate(label_organs.items()):
		patient_target_points[idx * N_points:(idx + 1) * N_points] = torch.from_numpy(patient_data[organ])
		patient_target_labels[idx * N_points:(idx + 1) * N_points] = int(label)
	
	train_input_points[train_patient_list.index(patient_id)] = torch.from_numpy(patient_data['input_points'])
	train_target_points[train_patient_list.index(patient_id)] = patient_target_points
	train_target_labels[train_patient_list.index(patient_id)] = patient_target_labels
	
test_input_points = torch.zeros((len(test_patient_list), N_in, 3), dtype=torch.float32)
test_target_points = torch.zeros((len(test_patient_list), N_out, 3), dtype=torch.float32)
test_target_labels = torch.zeros((len(test_patient_list), N_out), dtype=torch.int64)

for patient_id in test_patient_list:
	
	patient_data = data[patient_id]
	patient_target_points = torch.zeros((N_out,3), dtype=torch.float32)
	patient_target_labels = torch.zeros((N_out,), dtype=torch.int64)
	
	for idx, (label, organ) in enumerate(label_organs.items()):
		patient_target_points[idx * N_points:(idx + 1) * N_points] = torch.from_numpy(patient_data[organ])
		patient_target_labels[idx * N_points:(idx + 1) * N_points] = int(label)
	
	test_input_points[test_patient_list.index(patient_id)] = torch.from_numpy(patient_data['input_points'])
	test_target_points[test_patient_list.index(patient_id)] = patient_target_points
	test_target_labels[test_patient_list.index(patient_id)] = patient_target_labels
	
# optional validation set (checkpoint selection / early stopping), built like the test set
val_patient_list = []
if args.validation_list_file_path:
	with open(args.validation_list_file_path) as f:
		val_patient_list = [line.strip() for line in f if line.strip()]
val_input_points = torch.zeros((len(val_patient_list), N_in, 3), dtype=torch.float32)
val_target_points = torch.zeros((len(val_patient_list), N_out, 3), dtype=torch.float32)
val_target_labels = torch.zeros((len(val_patient_list), N_out), dtype=torch.int64)
for k_val, patient_id in enumerate(val_patient_list):
	patient_data = data[patient_id]
	for idx, (label, organ) in enumerate(label_organs.items()):
		val_target_points[k_val, idx * N_points:(idx + 1) * N_points] = torch.from_numpy(patient_data[organ])
		val_target_labels[k_val, idx * N_points:(idx + 1) * N_points] = int(label)
	val_input_points[k_val] = torch.from_numpy(patient_data['input_points'])

conversion_path = args.conversion_path

if os.path.exists(conversion_path):
	conversion = np.load(conversion_path, allow_pickle=True)
	
from torch.utils.data import Dataset

class NAKO_10k_All_Dataset(Dataset):
	def __init__(self, input_points, output_points, output_labels, init_mean, init_labels, conversion, patient_ids, img_path):
		
		self.input_points = input_points
		self.output_points = output_points
		self.output_labels = output_labels
		self.init_mean = init_mean
		self.init_mean_labels = init_labels
		self.conversion = conversion
		self.patients = patient_ids

		self.img_path_main = img_path
	
	def __len__(self):
		return len(self.input_points)

	def __getitem__(self, idx):

		input_point_cloud = self.input_points[idx]
		output_point_cloud = self.output_points[idx]
		output_label = self.output_labels[idx]

		# Patient Img Transform File
		patient = self.patients[idx]
		patient_path = os.path.join(self.img_path_main, patient, "wat.nii.gz")
		transform = []
		if patient in self.conversion:
			transform.append(self.conversion[patient])
		else:
			raise ValueError(f"Patient {patient} not found in conversion data.")
		
		# Init Shape for the organ (noise added)
		init_points = self.init_mean + np.random.normal(0, 0.001, self.init_mean.shape)
		init_labels = self.init_mean_labels

		# Convert to torch tensors if they are numpy arrays
		if isinstance(init_points, np.ndarray):
			init_points = torch.tensor(init_points, dtype=torch.float32)
		elif isinstance(init_points, torch.Tensor):
			init_points = init_points.float()
			
		if isinstance(init_labels, np.ndarray):
			init_labels = torch.tensor(init_labels, dtype=torch.long)
		elif isinstance(init_labels, torch.Tensor):
			init_labels = init_labels.long()

		if isinstance(input_point_cloud, np.ndarray):
			input_point_cloud = torch.tensor(input_point_cloud, dtype=torch.float32)
		elif isinstance(input_point_cloud, torch.Tensor):
			input_point_cloud = input_point_cloud.float()

		if isinstance(output_point_cloud, np.ndarray):
			output_point_cloud = torch.tensor(output_point_cloud, dtype=torch.float32)
		elif isinstance(output_point_cloud, torch.Tensor):
			output_point_cloud = output_point_cloud.float()

		if isinstance(output_label, np.ndarray):
			output_label = torch.tensor(output_label, dtype=torch.long)
		elif isinstance(output_label, torch.Tensor):
			output_label = output_label.long()

		return input_point_cloud, init_points, init_labels, output_point_cloud, output_label, transform, patient_path

def collate_fn(batch):
	body_coord, mean_coord, mean_labels, target_coord, target_label, transform_matrix, patient_path = list(zip(*batch))
	body_offset, count = [], 0
	for item in body_coord:
		count += item.shape[0]
		body_offset.append(count)
	mean_offset, count = [], 0
	for item in mean_coord:
		count += item.shape[0]
		mean_offset.append(count)
	target_offset, count = [], 0
	for item in target_coord:
		count += item.shape[0]
		target_offset.append(count)
				
	return torch.cat(body_coord), torch.IntTensor(body_offset), torch.cat(mean_coord), torch.cat(mean_labels), torch.IntTensor(mean_offset), torch.cat(target_coord), torch.cat(target_label), torch.IntTensor(target_offset), transform_matrix, patient_path

save_path = args.save_path

if not os.path.exists(save_path):
	os.makedirs(save_path, exist_ok=True)

model_save_path = os.path.join(save_path, "models")
os.makedirs(model_save_path, exist_ok=True)
csv_save_path = os.path.join(save_path, "csv")
os.makedirs(csv_save_path, exist_ok=True)
fig_save_path = os.path.join(save_path, "figures")
os.makedirs(fig_save_path, exist_ok=True)

train_data = NAKO_10k_All_Dataset(input_points=train_input_points,	output_points=train_target_points,	output_labels=train_target_labels,	init_mean=init_mean,	init_labels=init_mean_labels,	conversion=conversion,	patient_ids=train_patient_list,	img_path="../../../../eytankats/data/nako_10k/images_mri_stitched/")
train_loader = torch.utils.data.DataLoader(train_data, batch_size=B, shuffle=True, drop_last=True, collate_fn=collate_fn)

test_data = NAKO_10k_All_Dataset(input_points=test_input_points,	output_points=test_target_points,	output_labels=test_target_labels,	init_mean=init_mean,	init_labels=init_mean_labels,	conversion=conversion,	patient_ids=test_patient_list,	img_path="../../../../eytankats/data/nako_10k/images_mri_stitched/")
test_loader = torch.utils.data.DataLoader(test_data, batch_size=B, shuffle=False, drop_last=False, collate_fn=collate_fn)
val_loader = None
if val_patient_list:
	val_data = NAKO_10k_All_Dataset(input_points=val_input_points, output_points=val_target_points, output_labels=val_target_labels, init_mean=init_mean, init_labels=init_mean_labels, conversion=conversion, patient_ids=val_patient_list, img_path="")
	val_loader = torch.utils.data.DataLoader(val_data, batch_size=B, shuffle=False, drop_last=False, collate_fn=collate_fn)
	print(f"validation set: {len(val_patient_list)} samples - checkpoint and early stopping on validation loss")

from util.chamfer_loss import calc_chamfer_stacked_objectwise as criterion
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

optimizer = Adam(model.parameters(), lr=args.lr)
scheduler = MultiStepLR(optimizer, milestones=[int(m) for m in args.milestones.split(",") if m.strip()], gamma=args.gamma)

from tqdm import tqdm
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
from util.evaluation_functions import evaluate

def monitor(loader):
	"""Loss and mean CD (mm) of the model and of the template on a loader. BatchNorm uses each batch's
	own statistics, as in training; the running statistics are left untouched (momentum 0)."""
	bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
	saved = [m.momentum for m in bns]
	for m in bns:
		m.momentum = 0.0
	model.train()
	loss, cd, cd_template, n = 0.0, 0.0, 0.0, 0
	with torch.no_grad():
		for body_coord, body_offset, mean_coord, mean_labels, mean_offset, target_coord, target_labels, target_offset, transform_matrix, patient_path in loader:
			body_coord, body_offset = body_coord.to(device), body_offset.to(device)
			mean_coord, mean_labels, mean_offset = mean_coord.to(device), mean_labels.to(device), mean_offset.to(device)
			target_coord, target_labels = target_coord.to(device), target_labels.to(device)
			nb = len(body_offset)
			output_coord = predict(body_coord, body_offset, mean_coord, mean_offset)
			loss += nb * criterion(output_coord, target_coord, mean_labels, target_labels, nb).item()
			cd += nb * evaluate(output_coord, target_coord, mean_labels, target_labels, batch_size=nb, affine=transform_matrix[0], device=device)["total_cd"]
			cd_template += nb * evaluate(mean_coord, target_coord, mean_labels, target_labels, batch_size=nb, affine=transform_matrix[0], device=device)["total_cd"]
			n += nb
	for m, momentum in zip(bns, saved):
		m.momentum = momentum
	return loss / n, cd / n, cd_template / n

if not args.eval_only:
	select = "val_loss" if val_loader is not None else "train_loss"
	best_loss = float("inf")
	total_epoch = args.epochs
	history = []
	early_stop_counter = 0
	for epoch in range(1,total_epoch+1):
		model.train()
		avg_loss = 0.0
		for i, (body_coord, body_offset, mean_coord, mean_labels, mean_offset, target_coord, target_labels, target_offset, transform_matrix, patient_path) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{total_epoch}", leave=False, disable=not sys.stderr.isatty())): 
			body_coord, body_offset = body_coord.to(device), body_offset.to(device)
			mean_coord, mean_labels, mean_offset = mean_coord.to(device), mean_labels.to(device), mean_offset.to(device)
			target_coord, target_labels = target_coord.to(device), target_labels.to(device)
		
			output_coord = predict(body_coord, body_offset, mean_coord, mean_offset)

			loss = criterion(output_coord, target_coord, mean_labels, target_labels, B)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		
			avg_loss += loss.item()
	
		avg_loss /= len(train_loader)
		row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train_loss": avg_loss}
		line = f"Epoch {epoch}/{total_epoch} | lr {row['lr']:.2e} | Avg Loss - {avg_loss:.6f}"
		if val_loader is not None:
			row["val_loss"], row["val_cd_mm"], row["val_template_cd_mm"] = monitor(val_loader)
			line += (f" | val loss {row['val_loss']:.6f} | val CD {row['val_cd_mm']:.3f} mm"
					 f" (template {row['val_template_cd_mm']:.3f} mm)")
		if args.monitor_test:
			row["test_loss"], row["test_cd_mm"], row["template_cd_mm"] = monitor(test_loader)
			line += (f" | test loss {row['test_loss']:.6f} | test CD {row['test_cd_mm']:.3f} mm"
					 f" (template {row['template_cd_mm']:.3f} mm)")
		if row[select] <= best_loss:
			best_loss = row[select]
			best_epoch = epoch
			torch.save(model.state_dict(), f"{model_save_path}/model_weights_best.pth")
			early_stop_counter = 0
			line += " | New best result"
		else:
			early_stop_counter += 1
			line += f" | Early stop counter: {early_stop_counter}"
		print(line, flush=True)
		history.append(row)
		pd.DataFrame(history).to_csv(f"{csv_save_path}/loss_history.csv", index=False)

		if args.patience and early_stop_counter >= args.patience:
			print("Early stopping triggered.")
			break
		
		scheduler.step()

	torch.save(model.state_dict(), f"{model_save_path}/model_weights_last.pth")
	print(f"Best checkpoint: epoch {best_epoch} ({select} {best_loss:.6f})")
	with open(f"{csv_save_path}/best_epoch.txt", "w") as f:
		f.write(f"{best_epoch} {select} {best_loss:.8f}\n")

	hist = pd.DataFrame(history)
	fig, ax = plt.subplots(1, 2 if args.monitor_test else 1, figsize=(14 if args.monitor_test else 8, 6), squeeze=False)
	ax[0, 0].plot(hist.epoch, hist.train_loss, label="train")
	if "val_loss" in hist:
		ax[0, 0].plot(hist.epoch, hist.val_loss, label="validation")
		ax[0, 0].axvline(best_epoch, color="0.5", linestyle=":", linewidth=1)
	if args.monitor_test:
		ax[0, 0].plot(hist.epoch, hist.test_loss, label="test")
		ax[0, 1].plot(hist.epoch, hist.test_cd_mm, label="model (test)")
		ax[0, 1].plot(hist.epoch, hist.template_cd_mm, "--", label="template (test)")
		ax[0, 1].set_xlabel("Epoch")
		ax[0, 1].set_ylabel("CD (mm)")
		ax[0, 1].legend()
	ax[0, 0].set_yscale("log")
	ax[0, 0].set_title("Running Loss over Epoch")
	ax[0, 0].set_xlabel("Epoch")
	ax[0, 0].set_ylabel("CD Loss")
	ax[0, 0].legend()
	plt.savefig(f"{fig_save_path}/Running Loss")

import pandas as pd
from util.evaluation_functions import evaluate

gc.collect()
torch.cuda.empty_cache()

model.load_state_dict(torch.load(f"{model_save_path}/model_weights_best.pth"))

def recalibrate_bn(loader):
	"""Recompute BatchNorm running statistics with the final weights, as a cumulative average over
	the training batches. With small batches the running averages kept during training can drift far
	from the batch statistics the weights were trained with, which corrupts eval-mode predictions."""
	for m in model.modules():
		if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
			m.reset_running_stats()
			m.momentum = None
	model.train()
	with torch.no_grad():
		for body_coord, body_offset, mean_coord, mean_labels, mean_offset, *_ in loader:
			predict(body_coord.to(device), body_offset.to(device), mean_coord.to(device), mean_offset.to(device))

if args.bn_recalibrate:
	recalibrate_bn(train_loader)

all_metrics = {"cd": {}, "hd95": {}, "max_width": {}, "min_width": {}, "max_depth": {}, "min_depth": {}, "max_height": {}, "min_height": {}}
for metric, metric_results in all_metrics.items():
	for label, organ in label_organs.items():
		metric_results[organ] = 0.0

cd, hd95 = 0.0, 0.0
predictions = []
model.eval()
    
with torch.no_grad():
	for i, (body_coord, body_offset, mean_coord, mean_labels, mean_offset, target_coord, target_labels, target_offset, transform_matrix, patient_path) in enumerate(tqdm(test_loader)): 
		body_coord, body_offset = body_coord.to(device), body_offset.to(device)
		mean_coord, mean_labels, mean_offset = mean_coord.to(device), mean_labels.to(device), mean_offset.to(device)
		target_coord, target_labels = target_coord.to(device), target_labels.to(device)

		output_coord = predict(body_coord, body_offset, mean_coord, mean_offset)
			
		nb = len(body_offset)
		for r in range(nb):
			b0 = 0 if r == 0 else mean_offset[r - 1].item()
			predictions.append(output_coord[b0:mean_offset[r].item()].cpu().numpy())
		results = evaluate(output_coord.detach(), target_coord, mean_labels, target_labels, batch_size=nb, affine=transform_matrix[0], device=device)

		cd += nb * results["total_cd"]
		hd95 += nb * results["total_hd95"]

		for idx, (label, organ) in enumerate(label_organs.items()):
			all_metrics["cd"][organ] += nb * results["per_organ_cd"][idx]
			all_metrics["hd95"][organ] += nb * results["per_organ_hd95"][idx]
			all_metrics["max_width"][organ] += nb * results["max_error_mm"][idx, 0]
			all_metrics["max_depth"][organ] += nb * results["max_error_mm"][idx, 1]
			all_metrics["max_height"][organ] += nb * results["max_error_mm"][idx, 2]
			all_metrics["min_width"][organ] += nb * results["min_error_mm"][idx, 0]
			all_metrics["min_depth"][organ] += nb * results["min_error_mm"][idx, 1]
			all_metrics["min_height"][organ] += nb * results["min_error_mm"][idx, 2]

	cd /= len(test_data)
	hd95 /= len(test_data)

	for metric, metric_results in all_metrics.items():
		for label, organ in label_organs.items():
			metric_results[organ] /= len(test_data)

	all_metrics = {metric: {organ: value.item() for organ, value in metric_results.items()} for metric, metric_results in all_metrics.items()}
	metrics_df = pd.DataFrame(all_metrics)
	metrics_df.to_csv(f"{csv_save_path}/trained_vs_target_{which_label}.csv")
	np.savez(f"{save_path}/predictions.npz", **dict(zip(test_patient_list, predictions)))

	import random

	for r in range(len(mean_offset)-1):

		offset_b, offset_e = mean_offset[r], mean_offset[r+1]
		mean_coord_np = mean_coord[offset_b:offset_e].detach().cpu().numpy()
		mean_label_np = mean_labels[offset_b:offset_e].detach().cpu().numpy()
		target_coord_np = target_coord[offset_b:offset_e].detach().cpu().numpy()
		target_labels_np = target_labels[offset_b:offset_e].detach().cpu().numpy()
		output_coord_np = output_coord[offset_b:offset_e].detach().cpu().numpy()

		fig = plt.figure(figsize=(8,8))
		ax = fig.add_subplot(1,3,1, projection='3d')
		ax.scatter(mean_coord_np[:,2], mean_coord_np[:,1], mean_coord_np[:, 0], c=mean_label_np, cmap='plasma')
		ax.set_xlim([-1,1])
		ax.set_ylim([-1,1])
		ax.set_zlim([-1,1])
		ax.set_title("Mean Coordinates")
		ax = fig.add_subplot(1,3,2, projection='3d')
		ax.scatter(output_coord_np[:,2], output_coord_np[:,1], output_coord_np[:, 0], c=mean_label_np, cmap='plasma')
		ax.set_xlim([-1,1])
		ax.set_ylim([-1,1])
		ax.set_zlim([-1,1])
		ax.set_title("Output Coordinates")
		ax = fig.add_subplot(1,3,3, projection='3d')
		ax.scatter(target_coord_np[:,2], target_coord_np[:,1], target_coord_np[:, 0], c=target_labels_np, cmap='plasma')
		ax.set_xlim([-1,1])
		ax.set_ylim([-1,1])
		ax.set_zlim([-1,1])
		ax.set_title("Target Coordinates")

		plt.savefig(f"{fig_save_path}/Output Results - {r+1}")

# reset accumulators: otherwise the template metrics include half of the trained-model metrics
all_metrics = {metric: {organ: 0.0 for organ in label_organs.values()} for metric in all_metrics}
cd, hd95 = 0.0, 0.0
for i, (body_coord, body_offset, mean_coord, mean_labels, mean_offset, target_coord, target_labels, target_offset, transform_matrix, patient_path) in enumerate(tqdm(test_loader)): 
	body_coord, body_offset = body_coord.to(device), body_offset.to(device)
	mean_coord, mean_labels, mean_offset = mean_coord.to(device), mean_labels.to(device), mean_offset.to(device)
	target_coord, target_labels = target_coord.to(device), target_labels.to(device)
		
	nb = len(body_offset)
	results = evaluate(mean_coord, target_coord, mean_labels, target_labels, batch_size=nb, affine=transform_matrix[0], device=device)

	cd += nb * results["total_cd"]
	hd95 += nb * results["total_hd95"]

	for idx, (label, organ) in enumerate(label_organs.items()):
		all_metrics["cd"][organ] += nb * results["per_organ_cd"][idx]
		all_metrics["hd95"][organ] += nb * results["per_organ_hd95"][idx]
		all_metrics["max_width"][organ] += nb * results["max_error_mm"][idx, 0]
		all_metrics["max_depth"][organ] += nb * results["max_error_mm"][idx, 1]
		all_metrics["max_height"][organ] += nb * results["max_error_mm"][idx, 2]
		all_metrics["min_width"][organ] += nb * results["min_error_mm"][idx, 0]
		all_metrics["min_depth"][organ] += nb * results["min_error_mm"][idx, 1]
		all_metrics["min_height"][organ] += nb * results["min_error_mm"][idx, 2]

cd /= len(test_data)
hd95 /= len(test_data)

for metric, metric_results in all_metrics.items():
	for label, organ in label_organs.items():
		metric_results[organ] /= len(test_data)

all_metrics = {metric: {organ: value.item() for organ, value in metric_results.items()} for metric, metric_results in all_metrics.items()}
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(f"{csv_save_path}/init_vs_target_{which_label}.csv")