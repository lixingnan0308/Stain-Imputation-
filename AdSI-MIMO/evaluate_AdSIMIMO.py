"""
evaluate_AdSIMIMO.py — Standalone evaluation script for AdSI-MIMO.

Loads a trained AdSI-MIMO checkpoint and runs inference on a held-out data
split. For each non-fixed output marker the script:
  - Zeros out the target channel in the encoder input (simulating an absent stain)
  - Reconstructs it via the MultiMAE decoder
  - Saves a .npy file containing [real, generated] channels for each image
  - Records MAE, MSE, RMSE, PSNR, SSIM, and Pearson correlation per image

Usage:
    Edit the configuration block below and run:
        python evaluate_AdSIMIMO.py
"""

from trainer_AdSIMIMO import read_json_from_txt
from train_AdSIMIMO import TrainerMMAE


if __name__ == '__main__':
    # ---- Marker panel -------------------------------------------------------
    # Load the full marker panel from the JSON-formatted output.txt file.
    # The order must match the channel order in the .npy image files.
    stain_panel = read_json_from_txt('./output.txt')

    # Markers that are always available as input and are never imputed
    fixed_stain = ['dapi', 'autofluorescence']

    # ---- Paths --------------------------------------------------------------
    data_csv_path = './try_scale.csv'   # CSV with Image_Paths and Split_Name columns
    results_dir = './results_AdSIMIMO'  # Directory containing the saved checkpoint

    # ---- Build the trainer object -------------------------------------------
    obj = TrainerMMAE(
        marker_panel=stain_panel,
        fixed_stain=fixed_stain,
        results_dir=results_dir,
        lr=0.0001,
        seed=1,
    )

    mask_biomarker = [
                                "pd-l1",
                                "cd16",
                                "cd8",
                                "cd68",
                                "sox10"
                                
                                ]
    # ---- Run evaluation -----------------------------------------------------
    obj.eval(
        data_csv_path,
        split_name='test',
        img_size=224,
        batch_size=32,
        num_workers=4,
        checkpoint_name='checkpoint.pt',
        mask_biomarker = mask_biomarker
    )
