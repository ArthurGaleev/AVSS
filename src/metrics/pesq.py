import torch
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from src.metrics.base_metric import BaseMetric
class Pesq(BaseMetric):
    def __init__(self, sample_rate, compare="first", *args, **kwargs):
        super().__init__(*args, **kwargs)
        if(sample_rate==16000):
            self.pesqMetric = PerceptualEvaluationSpeechQuality(16000,"wb")
        elif(sample_rate==8000):
            self.pesqMetric = PerceptualEvaluationSpeechQuality(8000,"nb")
        else:
            raise NotImplementedError("Pesq metric is not implemented for sample rates not 16kHz or 8kHz")
        if(compare!="first" and compare!="second" and compare!="average"):
            raise NotImplementedError("You should compare only the first and the second wavs pred and initial")
        self.compare=compare
    def __call__(self, audio_pred_first, audio_first, audio_pred_second, audio_second , **batch):
        # audio_pred_first=[audio_pred_first[i, ...] for i in range(audio_pred_first.shape[0])]
        # audio_pred_second=[audio_pred_second[i, ...] for i in range(audio_pred_second.shape[0])]
        #TODO Uncomment when fixed reconstruct
        pesqs=[]
        if(self.compare=="first"):
            ests, targets=audio_pred_first, audio_first
        elif(self.compare=="second"):
            ests, targets=audio_pred_second, audio_second
        else:
            ests, targets=(audio_pred_first+audio_pred_second, audio_first+audio_second)
        for est, target in zip(ests, targets):
            pesqs.append(self.pesqMetric(est, target).item())
        return sum(pesqs)/len(pesqs)