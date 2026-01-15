import torch
from src.models.maize_model import MaizeDiseaseModel

def run_model_check():
    print("--- Model Sanity Check ---")
    
    # 1. Initialize the model
    model = MaizeDiseaseModel(num_classes=8)
    model.eval() # Set to evaluation mode
    
    # 2. Create a dummy batch: [Batch Size, Channels, Height, Width]
    # This simulates 1 image of 224x224 pixels
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # 3. Run a forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    # 4. Verification logic
    print(f"Output shape: {output.shape}") # Expected: torch.Size([1, 8])
    
    if output.shape == (1, 8):
        print("✅ Success: Model correctly outputs 8 class logits.")
    else:
        print(f"❌ Failure: Unexpected output shape {output.shape}")

    # 5. Check MPS (GPU) transfer
    try:
        model.to("mps")
        dummy_input_mps = dummy_input.to("mps")
        output_mps = model(dummy_input_mps)
        print("✅ Success: Model successfully moved to Mac GPU (MPS).")
    except Exception as e:
        print(f"❌ Failure: Could not run on MPS. Error: {e}")

if __name__ == "__main__":
    run_model_check()