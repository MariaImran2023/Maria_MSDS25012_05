Diffusion Model - Image Generation
Deep Learning Spring 2025 - Assignment 5 (Bonus)
Student: Maria | Roll No: MSDS25012

RUNNING THE CODE

Train Model
python MSDS25012_05.py --data_path /path/to/data --save_path models/

Arguments
--data_path          Path to dataset (default: data/)
--save_path          Save directory (default: models/)
--epochs             Training epochs (default: 600)
--batch_size         Batch size (default: 16)
--lr                 Learning rate (default: 1e-4)
--image_size         Image size (default: 64)
--T                  Diffusion timesteps (default: 1000)
--images_per_class   Images per class (default: 120)

Examples
# Default training
python MSDS25012_05.py

Custom settings
python MSDS25012_05.py --data_path ./animals/ --epochs 300 --batch_size 32

Testing / Generation
Open and run: test_single_sample.ipynb
Update MODEL_PATH to point to your trained model.

FILE STRUCTURE
MSDS25012_05/
├── MSDS25012_05.py              # Main training script
├── MSDS25012_05_AllCode.py      # All code combined
├── test_single_sample.ipynb     # Testing notebook
├── Report.pdf                   # Report
├── README.txt                   # This file
├── models/                      # Saved models & plots
│   ├── best_model.pth           # Best model checkpoint
│   ├── loss_graph.png           # Training loss plot
│   ├── forward_process.png      # Forward diffusion visualization
│   ├── sampling_process.png     # Reverse diffusion visualization
│   └── samples_epoch_*.png      # Generated samples
└── data/                        # Dataset folder (5 classes)

DEPENDENCIES
pip install torch torchvision matplotlib numpy pillow tqdm

## DATASET FORMAT
data/
├── class_1/   (images)
├── class_2/   (images)
├── class_3/   (images)
├── class_4/   (images)
└── class_5/   (images)
