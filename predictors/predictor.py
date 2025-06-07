import numpy as np
from scores.utils import get_score
import torch
import math
import torch.nn as nn
import clip


class Predictor:
    def __init__(self, args, model):
        self.score_function = get_score(args)
        self.model = model
        self.threshold = None
        self.alpha = args.alpha
        self.device = next(model.parameters()).device
        self.args = args
        self.label2class = args.label2class
        self.template_embedding = self.model.token_embedding(clip.tokenize([f"A photo of a dog"]).to("cuda"))
        self.normalized_token_embedding = self.model.token_embedding.weight.data.clone().detach()
        self.normalized_token_embedding = self.normalized_token_embedding / self.normalized_token_embedding.norm(dim=-1, keepdim=True)


    def learn_v(self, images):
        v = nn.Parameter(torch.ones(size=(1, self.template_embedding.shape[-1]), device="cuda"), requires_grad=True)
        optimizer = torch.optim.Adam([v], lr=1e-1)

        image_features = self.model.encode_image(images).clone().detach().requires_grad_(False)

        for i in range(200):
            template_embedding = self.template_embedding.clone().detach().requires_grad_(False)
            template_embedding[0, 5] = v[0]
            text_inputs = template_embedding
            #print(text_inputs[0, 5].requires_grad, text_inputs.requires_grad)

            text_features = self.model.encode_x(text_inputs)
            cos = torch.sum(image_features * text_features) / (torch.norm(image_features) * torch.norm(text_features))
            loss = -cos
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
        return v

    def calibrate(self, cal_loader, alpha=None):
        """ Input calibration dataloader.
            Compute scores for all the calibration data and take the (1 - alpha) quantile."""
        self.model.eval()

        if alpha is None:
            alpha = self.alpha

        cal_score = torch.tensor([], device=self.device)
        #  Assume batch size is 1
        for images, target in cal_loader:
            images = images.to(self.device)
            target = target.to(self.device)

            v = self.learn_v(images)

            class_name = self.label2class[target.item()]
            class_id = clip.tokenize(class_name)[0, 1]
            v_class = self.model.token_embedding(torch.tensor(class_id, device="cuda")).clone().detach()
            score = 1 - torch.sum(v * v_class) / (torch.norm(v) * torch.norm(v_class))

            cal_score = torch.cat((cal_score, score.view(1, -1)), 0)
        N = cal_score.shape[0]
        threshold = torch.quantile(cal_score, math.ceil((1 - alpha) * (N + 1)) / N, dim=0)
        self.threshold = threshold
        return threshold

    def calibrate_batch_logit(self, logits, target, alpha):
        """Design for conformal training, which needs to compute threshold in every batch"""
        prob = torch.softmax(logits, dim=-1)
        batch_score = self.score_function.compute_target_score(prob, target)
        N = target.shape[0]
        return torch.quantile(batch_score, math.ceil((1 - alpha) * (N + 1)) / N, dim=0)

    def evaluate(self, test_loader):
        """Must be called after calibration.
        Output a dictionary containing Top1 Accuracy, Coverage and Average Prediction Set Size."""
        self.model.eval()
        if self.args.algorithm == "cp":
            total_coverage = 0
            total_prediction_set_size = 0
            total_samples = 0

            for image, target in test_loader:
                image, target = image.to(self.device), target.to(self.device)
                total_samples += target.shape[0]

                v = self.learn_v(image)

                cos_tensor = self.normalized_token_embedding @ (v / v.norm(dim=-1, keepdim=True)).view(-1)
                score_tensor = 1 - cos_tensor

                prediction_set = (score_tensor <= self.threshold).to(torch.int)

                class_name = self.label2class[target.item()]
                class_id = clip.tokenize(class_name)[0, 1]

                if prediction_set[class_id] == 1:
                    total_coverage += 1

                total_prediction_set_size += prediction_set.sum().item()


            coverage = total_coverage / total_samples
            avg_set_size = total_prediction_set_size / total_samples
            result_dict = {
                f"AverageSetSize": avg_set_size,
                f"Coverage": coverage,
                }
        else:
            total_samples = 0
            total_accuracy = 0
            with torch.no_grad():
                for image, target in test_loader:
                    image, target = image.to(self.device), target.to(self.device)
                    batch_size = target.shape[0]
                    total_samples += batch_size

                    logit = self.model(image)
                    prob = torch.softmax(logit, dim=-1)
                    prediction = torch.argmax(prob, dim=-1)
                    total_accuracy += (prediction == target).sum().item()

                accuracy = total_accuracy / total_samples
                result_dict = {
                    f"{self.args.score}_Top1Accuracy": accuracy,
                }

        return result_dict

