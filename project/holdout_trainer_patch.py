import os

import pytorch_lightning as pl
from dental_data import PatchBasedCompleteDentalDataModule
from dental_net import DentalNetModule

epochs = 2000
num_samples = 2048

official_experiment = True

loi = ['IP', 'CR', 'CL', '6R_ER', '6L_EL', 'CM', 'MM']
for landmark in loi:
    # init dataset
    dataset = PatchBasedCompleteDentalDataModule(batch_size=64,
                                                 num_point_samples=num_samples,
                                                 landmark=landmark,
                                                 patch_size=8.)

    # init model, pass dataset for labels information and preprocessing steps
    model = DentalNetModule(dataset=dataset)

    # # train the model on the dataset
    if not official_experiment:
        trainer = pl.Trainer(num_sanity_val_steps=0,
                             logger=False,
                             checkpoint_callback=False,
                             gpus=[1],
                             max_epochs=epochs)
        trainer.fit(model, dataset)
        trainer.test(verbose=False,
                     model=model,
                     test_dataloaders=dataset.inference_val_dataloader())
    else:
        trainer = pl.Trainer(gpus=[1], max_epochs=epochs)
        trainer.fit(model, dataset)
        trainer.save_checkpoint(
            os.path.join(model.logger.log_dir,
                         "checkpoints/final_checkpoint.ckpt"))

        # infer on train, validation and test set and write results
        for i in range(5):
            for (name, data_loader) in dataset.inference_sets().items():
                outs = trainer.test(test_dataloaders=data_loader(),
                                    verbose=False,
                                    model=model)
                model.write_outputs_in_logdir(outs,
                                              '{}_results_{}.csv'.format(
                                                  name, i),
                                              log_dir=None)
