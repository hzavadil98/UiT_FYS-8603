from src import Four_view_two_branch_model, Patient_Cancer_Dataloader


def check_dataloader_passes_model(dataloader, model):
    batch = next(iter(dataloader.train_dataloader()))
    x, y = batch

    y_left, y_right = model.forward(x)
    print(f"data passed ok, outputs: {y_left.shape}, {y_right.shape}")
    return 0


def main():
    dataloader = Patient_Cancer_Dataloader(
        root_folder="/Users/jazav7774/Library/CloudStorage/OneDrive-UiTOffice365/Data/Mammo/",
        annotation_csv="modified_breast-level_annotations.csv",
        imagefolder_path="New_512",
        batch_size=32,
        num_workers=4,
    )
    # dataloader.train_dataset.plot(0)

    model = Four_view_two_branch_model(num_class=5)

    check_dataloader_passes_model(dataloader, model)
    return 0


if __name__ == "__main__":
    main()
