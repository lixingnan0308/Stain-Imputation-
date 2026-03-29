"""
AdSIMIMO — Model components for the AdSI-MIMO stain imputation architecture.

This package provides the MultiMAE backbone and its associated adapters,
adapted for single-channel multiplexed immunofluorescence (MxIF) data.

Modules:
    multimae1          : Training-time MultiMAE with Dirichlet + curriculum masking.
    multimae_e         : Evaluation-time MultiMAE with selective deterministic masking.
    multimae_utils     : Core transformer blocks (Block, trunc_normal_, etc.).
    input_adapters     : PatchedInputAdapter for tokenising per-domain images.
    output_adapters    : SpatialOutputAdapter for reconstructing spatial outputs.
    criterion          : Masked pixel-level loss functions (L1, MSE, CrossEntropy).
    output_adapter_utils: Shared utilities for output adapter decoder blocks.
"""
