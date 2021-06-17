import torch
from transformers import Trainer, Adafactor, AdamW
from transformers.trainer_pt_utils import get_parameter_names


class CRFTrainer(Trainer):
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            crf_parameters = [name for name in decay_parameters if "crf" in name]
            decay_parameters = [name for name in decay_parameters if "crf" not in name]


            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if
                               n not in decay_parameters and n not in crf_parameters],
                    "weight_decay": 0.0,
                }
            ]
            if hasattr(self.model, 'crf'):
                optimizer_grouped_parameters.append({
                    "params": self.model.crf.parameters(),
                    'lr': self.model.config.crf_lr,
                })

            optimizer_cls = Adafactor if self.args.adafactor else AdamW
            if self.args.adafactor:
                optimizer_cls = Adafactor
                optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
            else:
                optimizer_cls = AdamW
                optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                }
            optimizer_kwargs["lr"] = self.args.learning_rate
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
