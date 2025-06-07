import torch
from tqdm import tqdm
import models
from loss.utils import get_loss_function
from .utils import get_optimizer
from predictors.get_predictor import get_predictor
class Trainer:
    """
    Trainer class that implement all the functions regarding training.
    All the arguments are passed through args."""
    def __init__(self, args, num_classes):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = models.utils.build_model(args.model)
        self.batch_size = args.batch_size

        self.optimizer = get_optimizer(args, self.model)

        self.predictor = get_predictor(args, self.model)

        self.num_classes = num_classes
        self.loss_function = get_loss_function(args, self.predictor)

    def train_batch(self, images, texts):
        # Move data to device
        images = images.to(self.device)

        # CLIP's text processor typically gives you input_ids, attention_mask etc.
        # Assuming 'texts' is already a dictionary of tokenized inputs
        input_ids = texts['input_ids'].to(self.device)
        attention_mask = texts['attention_mask'].to(self.device) if 'attention_mask' in texts else None

        # Forward pass - CLIP returns image and text features
        image_features, text_features = self.model(
            images,
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Compute CLIP's contrastive loss
        loss = self.loss_function(image_features, text_features)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def train(self, data_loader, epochs):
        self.model.train()

        for epoch in range(epochs):
            for data, target in tqdm(data_loader, desc=f"Epoch: {epoch} / {epochs}"):
                self.train_batch(data, target)

        if self.args.save_model == "True":
            models.utils.save_model(self.args, self.model)

