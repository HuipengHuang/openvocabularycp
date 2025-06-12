import torch
from tqdm import tqdm
import models
from loss.utils import get_loss_function
from .utils import get_optimizer
from predictors.get_predictor import get_predictor
import clip

class Trainer:
    """
    Trainer class that implement all the functions regarding training.
    All the arguments are passed through args."""
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = models.utils.build_model(args)
        self.batch_size = args.batch_size

        self.optimizer = get_optimizer(args, self.model)

        self.predictor = get_predictor(args, self.model)
        self.loss_function = get_loss_function(args, self.predictor)
        self.text_inputs = None


    def train_batch(self, images, labels):
        images, labels = images.to(self.device), labels.to(self.device)

        logits_per_image, logits_per_text = self.model(images, self.text_inputs)

        image_loss = self.loss_function(logits_per_image, labels)
        text_loss = self.loss_function(logits_per_text.T, labels)

        loss = (image_loss + text_loss) / 2
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



    def train(self, data_loader, epochs=None):
        self.model.train()

        self.text_inputs = clip.tokenize([f"a photo of a {self.args.label2class[i]}" for i in range(self.args.num_classes)]).to(
            "cuda")

        if epochs is None:
            epochs = self.args.epochs
        for epoch in range(epochs):
            for images, labels in tqdm(data_loader, desc=f"Epoch: {epoch} / {epochs}"):
                self.train_batch(images, labels)

        if self.args.save_model == "True":
            models.utils.save_model(self.args, self.model)

