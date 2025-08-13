#!/usr/bin/env python3
"""
VT1/VT2 Prediction Pipeline Runner

This script runs the complete VT1/VT2 prediction pipeline
with the specified configuration.
"""

import sys
import os
import argparse
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from mock_vt1_vt2.pipeline import VTPredictionPipeline


def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(description='Run VT1/VT2 prediction pipeline')
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/pipeline_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='outputs',
        help='Output directory for results'
    )
    parser.add_argument(
        '--create-plots', 
        action='store_true',
        help='Create evaluation plots'
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Configuration file not found: {args.config}")
        print("Please make sure the config file exists or specify a different path.")
        return 1
    
    try:
        # Initialize pipeline
        print("Initializing VT1/VT2 prediction pipeline...")
        pipeline = VTPredictionPipeline(args.config)
        
        # Run the complete pipeline
        print("Running pipeline...")
        results = pipeline.run_pipeline()
        
        # Print results summary
        print("\n" + "="*50)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*50)
        
        print(f"\nTrained {len(results['models'])} models:")
        for target, model_info in results['models'].items():
            print(f"  - {target}: {model_info['algorithm']}")
        
        print("\nEvaluation Results:")
        for target, metrics in results['evaluation_results'].items():
            print(f"\n{target.upper()}:")
            for metric, value in metrics['metrics'].items():
                print(f"  {metric.upper()}: {value:.4f}")
        
        # Create plots if requested
        if args.create_plots:
            print("\nCreating evaluation plots...")
            pipeline.evaluator.create_evaluation_plots(f"{args.output_dir}/plots/")
        
        # Generate evaluation report
        print("\nGenerating evaluation report...")
        pipeline.evaluator.generate_evaluation_report(f"{args.output_dir}/evaluation_report.txt")
        
        print(f"\nAll results saved to: {args.output_dir}/")
        print("Pipeline completed successfully!")
        
        return 0
        
    except Exception as e:
        print(f"Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
