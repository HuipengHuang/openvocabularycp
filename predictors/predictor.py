from tqdm import tqdm
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
        print(self.device)
        self.args = args
        self.template_embedding = self.model.token_embedding(clip.tokenize([f"A photo of a dog"]).to("cuda"))
        self.batch_template_embedding = torch.cat([self.template_embedding for _ in range(args.batch_size)])
        self.normalized_token_embedding = self.model.token_embedding.weight.data.clone().detach()
        self.normalized_token_embedding = self.normalized_token_embedding / self.normalized_token_embedding.norm(dim=-1, keepdim=True)


    def learn_v(self, images):
        v = nn.Parameter(torch.rand(size=(self.args.batch_size, self.template_embedding.shape[-1]), device="cuda", requires_grad=True), requires_grad=True)
        optimizer = torch.optim.SGD([v], lr=1e-1)

        image_features = self.model.encode_image(images).clone().detach().requires_grad_(False)

        for i in range(100):
            text_inputs = self.batch_template_embedding.clone().detach().requires_grad_(False)
            text_inputs[:, 5] = v

            text_features = self.model.encode_x(text_inputs)
            diagonal_cos_value = [torch.cosine_similarity(image_features[i], text_features[i], dim=-1) for i in range(text_features.size(0))]

            loss = -sum(diagonal_cos_value)
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

        if self.args.algorithm == "cp":
            cal_score = torch.tensor([], device=self.device)
            #  Assume batch size is 1
            for images, target in tqdm(cal_loader, desc="Calibrating"):
                images = images.to(self.device)
                target = target.to(self.device)

                v = self.learn_v(images)

                class_name = self.args.label2class[target.detach().cpu().numpy()]
                class_id = torch.tensor([clip.tokenize(cls).to("cuda")[0, 1] for cls in class_name]).to("cuda")

                v_class = self.model.token_embedding(class_id)

                batch_score = 1 - torch.tensor([torch.cosine_similarity(v[i], v_class[i], dim=-1) for i in range(v.size(0))], device="cuda")

                cal_score = torch.cat((cal_score, batch_score), 0)
        else:
            with torch.no_grad():
                text_feature = self.model.token_embedding(clip.tokenize([[f"A photo of a {self.args.label2class[i]}"] for i in range(self.args.num_classes) ]).to("cuda"))
                text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)

                cal_score = torch.tensor([], device=self.device)
                for images, target in tqdm(cal_loader, desc="Calibrating"):
                    images = images.to(self.device)
                    target = target.to(self.device)
                    image_features = self.model.encode_image(images)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                    logits_per_image = (image_features @ text_feature.t()) * self.model.logit_scale.exp()
                    probabilities = logits_per_image.softmax(dim=-1)
                    batch_score = 1 - probabilities[torch.arange(images.shape[0]), target]
                    cal_score = torch.cat([cal_score, batch_score], 0)

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

            for image, target in tqdm(test_loader, desc="Evaluating"):
                image, target = image.to(self.device), target.to(self.device)
                total_samples += target.shape[0]

                v = self.learn_v(image)
                cos_tensor = self.normalized_token_embedding @ (v / v.norm(dim=-1, keepdim=True)).view(-1, self.args.batch_size)
                score_tensor = 1 - cos_tensor.T

                prediction_set = (score_tensor <= self.threshold).to(torch.int)

                class_name = self.args.label2class[target.clone().detach().cpu().numpy()]
                class_id = torch.tensor([clip.tokenize(cls).to("cuda")[0, 1] for cls in class_name]).to("cuda")

                total_coverage += torch.sum(prediction_set[torch.arange(self.args.batch_size), class_id]).item()


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
                text_feature = self.model.encode_text(clip.tokenize(
                    [f"A photo of a {self.args.label2class[i]}" for i in range(self.args.num_classes)]).to("cuda"))
                text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)

                for images, target in tqdm(test_loader, desc="Evaluating"):
                    images = images.to(self.device)
                    target = target.to(self.device)
                    total_samples += target.shape[0]

                    image_features = self.model.encode_image(images)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                    logits_per_image = (image_features @ text_feature.t()) * self.model.logit_scale.exp()
                    probabilities = logits_per_image.softmax(dim=-1)
                    total_accuracy += torch.argmax(probabilities, dim=-1).eq(target).sum().item()
                result_dict = {"Accuracy": total_accuracy / total_samples,}
        return result_dict

