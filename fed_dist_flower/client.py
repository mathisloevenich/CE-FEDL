import flwr as fl
import random

import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from utils import (
	compute_soft_labels,
	train,
	evaluate,
	train_on_soft_labels,
	set_parameters,
	values_to_bytes,
	bytes_to_values
)
from models import create_model

from flwr.common import FitIns, FitRes, GetParametersIns, GetParametersRes, Code, Status, EvaluateIns, EvaluateRes, \
	parameters_to_ndarrays, Parameters


class FlowerClient(fl.client.Client):

	def __init__(
			self,
			cid,
			train_loader: DataLoader,
			val_loader: DataLoader,
			x_pub,
			model_architecture="cnn500k",
			dataset_name="cifar",
			optimiser="Adam"
	):
		self.cid = cid
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.x_pub = x_pub  # fixed public data
		self.soft_labels = None
		self.distilled_soft_labels = None
		self.model_architecture = model_architecture
		self.model = None
		self.dataset_name = dataset_name
		self.optimiser = optimiser

		# initialize model
		self.initialize_model()

	def initialize_model(self, rand=True):
		seed = random.randint(0, 1000) if rand else None
		self.model = create_model(self.model_architecture, self.dataset_name, seed=seed)

	def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
		"""Rewrite to return soft_labels"""

		return GetParametersRes(
			status=Status(code=Code.OK, message="Success"),
			parameters=self.soft_labels
		)

	def set_parameters(self, parameters):
		self.distilled_soft_labels = parameters

	def train(self, train_loader, epochs=1, train_fn="train"):
		history = {
			"losses": [],
			"accuracies": []
		}

		methods = {
			"train": train,
			"train_sl": train_on_soft_labels
		}

		for epoch in range(epochs):
			loss, acc = methods[train_fn](self.model, train_loader, optimiser=self.optimiser)
			history["losses"].append(loss)
			history["accuracies"].append(acc)

		return sum(history["losses"]) / epochs, sum(history["accuracies"]) / epochs

	def fit(self, ins: FitIns) -> FitRes:
		# set_parameters(self.model, parameters)

		# only train on soft labels after first round
		if ins.config["server_round"] > 1:
			# only allocate once for every client
			client_loader = DataLoader(
				TensorDataset(self.x_pub, self.distilled_soft_labels), batch_size=32
			)
			self.train(client_loader, train_fn="train_sl", epochs=ins.config["client_dist_epochs"])

		loss, accuracy = self.train(self.train_loader, epochs=ins.config["epochs"])
		self.soft_labels = compute_soft_labels(self.model, self.x_pub)  # predict soft labels

		soft_label_tensor = self.soft_labels
		if soft_label_tensor.is_cuda:
			soft_label_tensor = self.soft_labels.cpu()  # Move tensor to CPU if it's on GPU

		parameters = Parameters(soft_label_tensor.numpy(), tensor_type="params")
		return FitRes(
			status=Status(code=Code.OK, message="Success"),
			parameters=parameters,
			num_examples=len(self.train_loader),
			metrics={"accuracy": float(accuracy)}
		)

	def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
		set_parameters(self.model, ins.parameters)
		loss, accuracy = evaluate(self.model, self.val_loader)

		return EvaluateRes(
			status=Status(code=Code.OK, message="Success"),
			loss=float(loss),
			num_examples=len(self.val_loader),
			metrics={"accuracy": float(accuracy)}
		)
