from pathlib import Path

import pandas as pd

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster
        batch.update(self.transform_batch(self.get_spectrogram(batch)))
        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()
        outputs = self.model(**batch)
        batch.update(outputs)
        batch.update(self.reconstruct_wav(batch))
        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_audio(batch["audio_first"], audio_name="audio_first")
            self.log_audio(batch["audio_second"], audio_name="audio_second")
            self.log_audio(batch["audio_mix"], audio_name="audio_mix")
            self.log_spectrogram(
                batch["spectrogram_first"], spectrogram_name="spectrogram_first"
            )
            self.log_spectrogram(
                batch["spectrogram_second"], spectrogram_name="spectrogram_second"
            )
            self.log_spectrogram(
                batch["spectrogram_mix"], spectrogram_name="spectrogram_mix"
            )
            self.log_audio(batch["audio_pred_second"], audio_name="audio_pred_second")
            self.log_audio(batch["audio_pred_first"], audio_name="audio_pred_first")
            self.log_spectrogram(
                batch["spectrogram_pred_first"],
                spectrogram_name="spectrogram_pred_first",
            )
            self.log_spectrogram(
                batch["spectrogram_pred_second"],
                spectrogram_name="spectrogram_pred_second",
            )
        else:
            # Log Stuff
            self.log_audio(batch["audio_first"], audio_name="audio_first")
            self.log_audio(batch["audio_second"], audio_name="audio_second")
            self.log_audio(batch["audio_mix"], audio_name="audio_mix")
            self.log_audio(batch["audio_pred_second"], audio_name="audio_pred_second")
            self.log_audio(batch["audio_pred_first"], audio_name="audio_pred_first")

    def log_spectrogram(self, spectrogram, spectrogram_name="spectrogram"):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot, self.config)
        self.writer.add_image(spectrogram_name, image)

    def log_audio(self, audio, audio_name="audio"):
        audio = audio[0].detach().cpu()
        self.writer.add_audio(audio_name, audio, sample_rate=self.config.sample_rate)
