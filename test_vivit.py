# coding=utf-8
"""
Evaluation script for multi-class video classification (e.g., SSv2 with 174 classes)
"""
import argparse
import torch
from tqdm import tqdm

from utils.data_utils import get_loader
import numpy as np
import torch.nn.functional as F
from models.modeling import MyViViT
from os import listdir, makedirs
from os.path import join, basename
from pandas import DataFrame


def simple_accuracy(preds, labels):
    """Top-1 accuracy"""
    return (preds == labels).mean()


def top_k_accuracy(logits, labels, k=5):
    """Top-k accuracy"""
    top_k_preds = np.argsort(logits, axis=1)[:, -k:]
    correct = 0
    for i, label in enumerate(labels):
        if label in top_k_preds[i]:
            correct += 1
    return correct / len(labels)


def evaluate_with_labels(args, model, test_loader):
    """
    Evaluate model when ground truth labels are available.
    Prints Top-1 and Top-5 accuracy.
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_logits = []
    
    epoch_iterator = tqdm(test_loader,
                          desc="Evaluating...",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
    
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        
        with torch.no_grad():
            logits = model(x)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())
            all_logits.append(logits.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = np.concatenate(all_logits, axis=0)
    
    # Calculate metrics
    top1_acc = simple_accuracy(all_preds, all_labels)
    top5_acc = top_k_accuracy(all_logits, all_labels, k=5)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total samples: {len(all_labels)}")
    print(f"Top-1 Accuracy: {top1_acc*100:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc*100:.2f}%")
    print("="*50)
    
    # Save detailed results
    makedirs("results", exist_ok=True)
    results_df = DataFrame({
        'ground_truth': all_labels,
        'prediction': all_preds,
        'correct': all_preds == all_labels
    })
    
    output_file = join("results", f"{args.name}_evaluation.csv")
    results_df.to_csv(output_file, sep='\t', index=False)
    print(f"\nDetailed results saved to: {output_file}")
    
    return top1_acc, top5_acc


def predict_without_labels(args, model, test_loader):
    """
    Generate predictions when no ground truth labels are available.
    Outputs predicted class and confidence for each video.
    """
    model.eval()
    
    all_predictions = []
    all_confidences = []
    all_top5_preds = []
    all_top5_confs = []
    
    epoch_iterator = tqdm(test_loader,
                          desc="Predicting...",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
    
    for step, batch in enumerate(epoch_iterator):
        batch = batch.to(args.device)
        x = batch
        
        with torch.no_grad():
            logits = model(x)
            probabilities = F.softmax(logits, dim=1)
            
            # Top-1 prediction
            confidence, predicted_class = torch.max(probabilities, dim=1)
            all_predictions.extend(predicted_class.cpu().numpy().tolist())
            all_confidences.extend(confidence.cpu().numpy().tolist())
            
            # Top-5 predictions
            top5_conf, top5_pred = torch.topk(probabilities, k=min(5, probabilities.shape[1]), dim=1)
            all_top5_preds.extend(top5_pred.cpu().numpy().tolist())
            all_top5_confs.extend(top5_conf.cpu().numpy().tolist())
    
    # Get video names
    videos = sorted(listdir(args.test_dir + "/videos"))
    
    # Create DataFrame
    results_df = DataFrame({
        'fname': videos,
        'predicted_class': all_predictions,
        'confidence': all_confidences,
        'top5_predictions': [str(p) for p in all_top5_preds],
        'top5_confidences': [str([f"{c:.4f}" for c in conf]) for conf in all_top5_confs]
    })
    
    # Save results
    makedirs("results", exist_ok=True)
    output_file = join("results", f"{args.name}_predictions.csv")
    results_df.to_csv(output_file, sep='\t', index=False)
    
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(f"Total videos: {len(videos)}")
    print(f"\nSample predictions:")
    print(results_df.head(10).to_string(index=False))
    print("="*50)
    print(f"\nResults saved to: {output_file}")
    
    return results_df


def test(args):
    """Main test function"""
    
    # Load model from checkpoint
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, weights_only=False)
        config = checkpoint["config"]
        
        print(f"Model config:")
        print(f"  - num_classes: {config.num_classes}")
        print(f"  - img_size: {config.img_size}")
        print(f"  - num_frames: {config.num_frames}")
        
        model = MyViViT(config, config.img_size, config.num_classes, config.num_frames)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(args.device)
        
        args.img_size = config.img_size
        args.num_frames = config.num_frames
        args.num_classes = config.num_classes

    # Get data loader
    train_loader, test_loader = get_loader(args)
    
    # Check if we have labels (for evaluation) or not (for prediction)
    # Try to get a batch and see if it has labels
    sample_batch = next(iter(test_loader))
    
    if isinstance(sample_batch, (list, tuple)) and len(sample_batch) == 2:
        # Has labels - run evaluation
        print("\nLabels detected - running evaluation mode")
        evaluate_with_labels(args, model, test_loader)
    else:
        # No labels - run prediction
        print("\nNo labels detected - running prediction mode")
        predict_without_labels(args, model, test_loader)


def main():
    parser = argparse.ArgumentParser()
    
    # Required parameters
    parser.add_argument("--name", required=False, default="test",
                        help="Name of this run. Used for output files.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "custom"], default="custom",
                        help="Which dataset type.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.bin or .pt file)")
    parser.add_argument("--test_dir", required=True, type=str,
                        help="Path to the test data folder (should contain 'videos/' subfolder)")
    parser.add_argument("--data_dir", default=None, type=str,
                        help="Path to training data (not needed for testing)")
    
    # Optional parameters
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training")
    
    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print(f"Using device: {device}")
    
    # Run test
    test(args)


if __name__ == "__main__":
    main()