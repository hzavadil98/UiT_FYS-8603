import os  # Added for path manipulation and checks
import sys  # For detailed exception info
import traceback

# --- Start of new pre-flight checks ---
print("[TRAIN_SYNT_DEBUG] TOP OF SCRIPT: Initial sys.path:", sys.path)
print(
    "[TRAIN_SYNT_DEBUG] TOP OF SCRIPT: Current working directory (os.getcwd()):",
    os.getcwd(),
)
script_dir = os.path.dirname(os.path.abspath(__file__))
print(
    "[TRAIN_SYNT_DEBUG] TOP OF SCRIPT: Script directory (os.path.dirname(os.path.abspath(__file__))):",
    script_dir,
)
if script_dir not in sys.path:
    print(
        f"[TRAIN_SYNT_DEBUG] TOP OF SCRIPT: Adding script directory {script_dir} to sys.path."
    )
    sys.path.insert(0, script_dir)  # Add to front for priority
print("[TRAIN_SYNT_DEBUG] TOP OF SCRIPT: Modified sys.path:", sys.path)
sys.stdout.flush()  # Force flush before any risky imports
# --- End of new pre-flight checks ---

print("[TRAIN_SYNT_DEBUG] Script execution started.")

try:
    print("[TRAIN_SYNT_DEBUG] Attempting to import standard modules...")
    import pytorch_lightning as pl
    import torch
    import torchvision.transforms.v2 as T
    from pytorch_lightning.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
    )
    from pytorch_lightning.loggers import WandbLogger

    import wandb

    print("[TRAIN_SYNT_DEBUG] Standard modules imported successfully.")
    sys.stdout.flush()

    print("[TRAIN_SYNT_DEBUG] Attempting to import from 'src'...")
    # Check if src directory exists relative to script_dir, just for sanity
    src_path_check = os.path.join(script_dir, "src")
    print(
        f"[TRAIN_SYNT_DEBUG] Checking for src directory at: {src_path_check}, Exists: {os.path.isdir(src_path_check)}"
    )
    sys.stdout.flush()

    from src import (  # This is a potential point of failure if src or its contents are not found
        Synthetic_2v_Dataloader,
        TwoViewCNN,
    )

    print("[TRAIN_SYNT_DEBUG] Successfully imported from 'src'.")
    sys.stdout.flush()

    print("[TRAIN_SYNT_DEBUG] All modules imported successfully.")
    sys.stdout.flush()  # Added flush

    print("Using custom TwoViewCNN model for synthetic data.")  # Original print
    sys.stdout.flush()  # Added flush

    def main():
        """
        Training single view featurizer models - 4 of them each on a specific "modality"
        """
        print("[TRAIN_SYNT_DEBUG] main() function started.")
        # Recognizes if running on my mac or on the server - sets the root_folder and accelerator
        if torch.backends.mps.is_available():
            accelerator = "mps"
            devices = 1
            print("[TRAIN_SYNT_DEBUG] MPS accelerator detected.")
        elif torch.cuda.is_available():
            accelerator = "gpu"
            devices = torch.cuda.device_count()
            print(
                f"[TRAIN_SYNT_DEBUG] CUDA accelerator detected with {devices} devices."
            )
            torch.set_float32_matmul_precision("high")
            # torch.cuda.empty_cache()
        else:
            accelerator = "cpu"  # Default to CPU if neither is available
            devices = 1
            print(
                "[TRAIN_SYNT_DEBUG] No MPS or CUDA accelerator detected, defaulting to CPU."
            )

        print(
            f"[TRAIN_SYNT_DEBUG] Using {accelerator} with {devices} devices for training."
        )

        print("[TRAIN_SYNT_DEBUG] Initializing transforms...")
        transform = T.Compose(
            [
                T.RandomHorizontalFlip(0.5),
                T.RandomVerticalFlip(0.5),
                T.GaussianNoise(0.1, 0.1),
            ]
        )
        print("[TRAIN_SYNT_DEBUG] Transforms initialized.")

        # Initialize DataLoader once before the loop if data is the same for all runs
        print("[TRAIN_SYNT_DEBUG] Initializing Synthetic_2v_Dataloader...")
        dataloader = Synthetic_2v_Dataloader(
            n_samples=[5000, 1000, 1000], transform=transform, batch_size=16
        )
        print("[TRAIN_SYNT_DEBUG] Synthetic_2v_Dataloader initialized.")
        ##########################################################################################################
        """
        train_transform = T.Compose(
            [
                T.ToImage(),
                # T.RandomRotation(degrees=10),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(
                    mean=[781.0543],
                    std=[1537.8235],
                ),
                # T.RandomAdjustSharpness(sharpness_factor=1, p=1),
                # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                # T.RandomRotation(degrees=10),
                # T.Normalize(
                #    mean=[0.5, 0.5, 0.5],
                #    std=[0.7, 0.7, 0.7],
                # ),
            ]
        )

        transform = T.Compose(
            [
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(
                    mean=[781.0543],
                    std=[1537.8235],
                ),
                # T.RandomAdjustSharpness(sharpness_factor=1, p=1),
                # T.Normalize(
                #    mean=[0.5, 0.5, 0.5],
                #    std=[0.7, 0.7, 0.7],
                # ),
            ]
        )

        dataloader = Breast_Cancer_Dataloader(
            root_folder="/storage/Mammo/",
            annotation_csv="modified_breast-level_annotations.csv",
            imagefolder_path="New_512",
            batch_size=16,
            num_workers=8,
            train_transform=train_transform,
            transform=transform,
        )
        """
        ##########################################################################################################
        print("DataLoader initialized.")  # Original print
        print("[TRAIN_SYNT_DEBUG] Initializing TwoViewCNN model...")
        model = TwoViewCNN(
            num_classes=3, task=1, num_views=2, input_channels=1, resnext_inplanes=16
        )
        print("Model initialized.")  # Original print
        print("[TRAIN_SYNT_DEBUG] TwoViewCNN model initialized.")

        print("[TRAIN_SYNT_DEBUG] Initializing WandbLogger...")
        wandb_logger = WandbLogger(project="Synthetic data", log_model="best")
        print("[TRAIN_SYNT_DEBUG] WandbLogger initialized.")
        # wandb_logger.watch(model, log="all", log_freq=1) # Temporarily disable watch

        print("[TRAIN_SYNT_DEBUG] Initializing ModelCheckpoint callback...")
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints/",
            filename="best_epoch",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            save_last=True,
        )
        print("[TRAIN_SYNT_DEBUG] ModelCheckpoint callback initialized.")

        print("[TRAIN_SYNT_DEBUG] Initializing LearningRateMonitor callback...")
        lr_monitor = LearningRateMonitor(logging_interval="step")
        print("[TRAIN_SYNT_DEBUG] LearningRateMonitor callback initialized.")

        print("[TRAIN_SYNT_DEBUG] Initializing EarlyStopping callback...")
        early_stopping = EarlyStopping(monitor="val_loss", patience=8, mode="min")
        print("[TRAIN_SYNT_DEBUG] EarlyStopping callback initialized.")

        # Add profiler
        # profiler = PyTorchProfiler(dirpath="./profiler_logs", filename="profile") # Temporarily disable profiler
        print("[TRAIN_SYNT_DEBUG] Initializing pl.Trainer...")
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator=accelerator,
            devices=devices,
            logger=wandb_logger,
            callbacks=[checkpoint_callback, lr_monitor, early_stopping],
            log_every_n_steps=10,
            # profiler=profiler,  # Temporarily disable profiler
        )
        print("[TRAIN_SYNT_DEBUG] pl.Trainer initialized.")

        # Train
        print("[TRAIN_SYNT_DEBUG] Starting trainer.fit()...")
        trainer.fit(model, dataloader)
        print("[TRAIN_SYNT_DEBUG] trainer.fit() completed.")

        # Load best weights
        print(
            f"[TRAIN_SYNT_DEBUG] Finished training, loading the best epoch: {checkpoint_callback.best_model_path}"
        )
        print("[TRAIN_SYNT_DEBUG] Loading model from checkpoint...")
        model = TwoViewCNN.load_from_checkpoint(checkpoint_callback.best_model_path)
        print("[TRAIN_SYNT_DEBUG] Model loaded from checkpoint.")

        # Test
        print("[TRAIN_SYNT_DEBUG] Starting trainer.test()...")
        trainer.test(model, dataloader)
        print("[TRAIN_SYNT_DEBUG] trainer.test() completed.")

        # Finish wandb run
        print("[TRAIN_SYNT_DEBUG] Finishing wandb run...")
        wandb.finish()
        print("[TRAIN_SYNT_DEBUG] wandb run finished.")

        print("[TRAIN_SYNT_DEBUG] main() function completed.")
        return 0

    if __name__ == "__main__":
        print("[TRAIN_SYNT_DEBUG] __main__ block started.")
        sys.stdout.flush()  # Added flush
        main()
        print("[TRAIN_SYNT_DEBUG] __main__ block completed.")

except ImportError as e:
    print(f"[TRAIN_SYNT_ERROR] ImportError occurred: {e}")
    print(f"[TRAIN_SYNT_ERROR] Name of missing module: {e.name}")
    if hasattr(e, "path") and e.path is not None:
        print(f"[TRAIN_SYNT_ERROR] Path hint for missing module: {e.path}")
    print(f"[TRAIN_SYNT_ERROR] Current Python path (sys.path): {sys.path}")
    print("[TRAIN_SYNT_ERROR] Traceback:")
    traceback.print_exc()
    sys.exit(1)  # Exit with an error code
except Exception as e:
    print(f"[TRAIN_SYNT_ERROR] An unexpected error occurred: {e}")
    print("[TRAIN_SYNT_ERROR] Traceback:")
    traceback.print_exc()
    sys.exit(1)  # Exit with an error code

print("[TRAIN_SYNT_DEBUG] Script execution nominally finished.")
