"""
evaluation_simif.py — Standalone evaluation script for SIMIF.

Loads a trained SIMIF checkpoint and runs inference on a specified data split,
computing per-image metrics (MAE, MSE, RMSE, PSNR, SSIM, Pearson correlation)
and saving the generated stain patches as .npy files.

Usage:
    Edit the configuration block in __main__ and run:
        python evaluation_simif.py
"""

from trainer_simif import read_json_from_txt
from train_simif import TrainerCGAN


if __name__ == '__main__':
    # ---- Marker panel -------------------------------------------------------
    # Load the full marker panel from the JSON-formatted output.txt file.
    # The order must match the channel order in the .npy image files.
    stain_panel = read_json_from_txt("./output.txt")

    # Markers that are always available as input and never imputed
    fixed_stain = ["dapi", "autofluorescence"]

    # All markers that can potentially be imputed (excludes fixed markers)
    potential_output = ["cd8", "pd-l1"]

    # ---- Paths --------------------------------------------------------------
    data_csv_path = "try_scale.csv"   # CSV with Image_Paths and Split_Name columns
    results_dir = "./results_SIMIF"   # Directory containing the saved checkpoint

    # ---- Build the trainer object -------------------------------------------
    # target_marker is required by the base Trainer constructor but is not used
    # during evaluation; set it to any marker in potential_output.
    obj = TrainerCGAN(marker_panel=stain_panel,
                      fixed_markers=fixed_stain,
                      potential_output_markers=potential_output,
                      results_dir=results_dir,
                      target_marker=["cd8"],
                      lr=0.0002,
                      seed=1)

    # ---- Run evaluation -----------------------------------------------------
    obj.eval(data_csv_path,
             split_name='test',
             img_size=224,
             batch_size=32,
             num_workers=4,
             required_stains=["cd8"],
             checkpoint_name='checkpoint.pt')
