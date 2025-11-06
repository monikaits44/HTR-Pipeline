#!/usr/bin/env python3
"""
Validation Script for HTR-Pipeline Setup and Execution
This script validates the entire environment, dependencies, data, and configuration
to ensure the pipeline is ready to run.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json
from pathlib import Path
from typing import Dict, List, Tuple

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}\n")

def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_info(text: str):
    """Print info message."""
    print(f"  {text}")


class SetupValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.root_dir = Path(__file__).parent.parent.parent
        os.chdir(self.root_dir)
    
    def validate_directory_structure(self) -> bool:
        """Validate the project directory structure."""
        print_header("1. Validating Directory Structure")
        
        required_dirs = [
            "scripts",
            "scripts/preprocessing",
            "scripts/postprocessing",
            "utils",
            "saved_models",
            "saved_models/experiments",
            "data"
        ]
        
        all_valid = True
        for dir_path in required_dirs:
            full_path = self.root_dir / dir_path
            if full_path.exists():
                print_success(f"Directory exists: {dir_path}/")
            else:
                print_error(f"Missing directory: {dir_path}/")
                self.errors.append(f"Missing directory: {dir_path}/")
                all_valid = False
        
        return all_valid
    
    def validate_scripts(self) -> bool:
        """Validate that all required scripts exist."""
        print_header("2. Validating Scripts")
        
        required_scripts = [
            "scripts/trainer.py",
            "scripts/preprocessing/prepare_iam.py",
            "scripts/postprocessing/evaluate.py",
            "scripts/postprocessing/demo.py",
            "models.py"
        ]
        
        all_valid = True
        for script_path in required_scripts:
            full_path = self.root_dir / script_path
            if full_path.exists():
                size = full_path.stat().st_size
                print_success(f"Script found: {script_path} ({size} bytes)")
            else:
                print_error(f"Missing script: {script_path}")
                self.errors.append(f"Missing script: {script_path}")
                all_valid = False
        
        return all_valid
    
    def validate_python_environment(self) -> bool:
        """Validate Python version and virtual environment."""
        print_header("3. Validating Python Environment")
        
        all_valid = True
        
        # Check Python version
        python_version = sys.version_info
        print_info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        if python_version.major == 3 and python_version.minor >= 8:
            print_success(f"Python version is compatible (3.{python_version.minor}+)")
        else:
            print_error(f"Python version should be 3.8 or higher")
            self.errors.append("Incompatible Python version")
            all_valid = False
        
        # Check if running in virtual environment
        in_venv = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )
        
        if in_venv:
            print_success("Running in virtual environment")
            print_info(f"  Virtual env path: {sys.prefix}")
        else:
            print_warning("Not running in virtual environment (recommended to use venv)")
            self.warnings.append("Not in virtual environment")
        
        return all_valid
    
    def validate_dependencies(self) -> bool:
        """Validate that all required Python packages are installed."""
        print_header("4. Validating Dependencies")
        
        required_packages = {
            'torch': 'PyTorch',
            'numpy': 'NumPy',
            'omegaconf': 'OmegaConf',
            'tqdm': 'tqdm',
            'PIL': 'Pillow',
            'albumentations': 'Albumentations',
            'nltk': 'NLTK',
            'editdistance': 'Edit Distance',
            'scipy': 'SciPy',
            'skimage': 'scikit-image'
        }
        
        all_valid = True
        for package, name in required_packages.items():
            try:
                if package == 'PIL':
                    import PIL
                elif package == 'skimage':
                    import skimage
                else:
                    __import__(package)
                
                # Get version if available
                try:
                    if package == 'PIL':
                        version = PIL.__version__
                    elif package == 'skimage':
                        version = skimage.__version__
                    else:
                        mod = __import__(package)
                        version = getattr(mod, '__version__', 'unknown')
                    print_success(f"{name} installed (version: {version})")
                except:
                    print_success(f"{name} installed")
            except ImportError:
                print_error(f"{name} not installed (pip install {package if package != 'PIL' else 'Pillow'})")
                self.errors.append(f"Missing dependency: {name}")
                all_valid = False
        
        return all_valid
    
    def validate_config_file(self) -> bool:
        """Validate the configuration file."""
        print_header("5. Validating Configuration File")
        
        config_path = self.root_dir / "config.yaml"
        
        if not config_path.exists():
            print_error("config.yaml not found")
            self.errors.append("Missing config.yaml")
            return False
        
        print_success("config.yaml found")
        
        # Try to parse config
        try:
            from omegaconf import OmegaConf
            config = OmegaConf.load(config_path)
            print_success("config.yaml is valid YAML")
            
            # Check required fields
            required_fields = ['device', 'data', 'preproc', 'arch', 'train', 'eval']
            for field in required_fields:
                if field in config:
                    print_success(f"  Config section '{field}' found")
                else:
                    print_error(f"  Missing config section: {field}")
                    self.errors.append(f"Missing config section: {field}")
                    return False
            
            # Display key config values
            print_info(f"\n  Key Configuration:")
            print_info(f"    Device: {config.device}")
            print_info(f"    Data path: {config.data.path}")
            print_info(f"    Epochs: {config.train.num_epochs}")
            print_info(f"    Batch size: {config.train.batch_size}")
            print_info(f"    Learning rate: {config.train.lr}")
            
            return True
            
        except Exception as e:
            print_error(f"Error parsing config.yaml: {e}")
            self.errors.append(f"Invalid config.yaml: {e}")
            return False
    
    def validate_data(self) -> bool:
        """Validate that required data is present."""
        print_header("6. Validating Data")
        
        all_valid = True
        
        # Check for IAM data directory
        data_dir = self.root_dir / "data" / "IAM" / "processed_lines"
        
        if not data_dir.exists():
            print_error(f"Processed data directory not found: {data_dir}")
            print_info("  Run data preparation first:")
            print_info("  python scripts/preprocessing/prepare_iam.py <forms_path> <xml_path> <splits_path> <output_path>")
            self.errors.append("Missing processed data")
            return False
        
        print_success(f"Data directory found: {data_dir}")
        
        # Check for train/val/test splits
        splits = ['train', 'val', 'test']
        for split in splits:
            split_dir = data_dir / split
            if split_dir.exists():
                num_files = len(list(split_dir.glob('*.png')))
                if num_files > 0:
                    print_success(f"  {split} split: {num_files} images")
                else:
                    print_warning(f"  {split} split directory exists but is empty")
                    self.warnings.append(f"{split} split is empty")
            else:
                print_error(f"  Missing {split} split directory")
                self.errors.append(f"Missing {split} split")
                all_valid = False
        
        # Check for classes file
        classes_file = data_dir / "classes.npy"
        if classes_file.exists():
            try:
                import numpy as np
                classes = np.load(classes_file)
                print_success(f"  classes.npy found ({len(classes)} character classes)")
            except Exception as e:
                print_warning(f"  classes.npy exists but couldn't load: {e}")
        else:
            print_info("  classes.npy not found (will be created on first training run)")
        
        return all_valid
    
    def validate_models_module(self) -> bool:
        """Validate that models.py can be imported."""
        print_header("7. Validating Models Module")
        
        try:
            import models
            print_success("models.py can be imported")
            
            # Check for HTRNet class
            if hasattr(models, 'HTRNet'):
                print_success("  HTRNet class found")
            else:
                print_error("  HTRNet class not found in models.py")
                self.errors.append("HTRNet class missing")
                return False
            
            return True
            
        except Exception as e:
            print_error(f"Cannot import models.py: {e}")
            self.errors.append(f"Cannot import models: {e}")
            return False
    
    def validate_utils_module(self) -> bool:
        """Validate that utils module can be imported."""
        print_header("8. Validating Utils Module")
        
        utils_modules = [
            ('utils.htr_dataset', 'HTRDataset'),
            ('utils.metrics', 'CER'),
            ('utils.metrics', 'WER'),
            ('utils.preprocessing', 'load_image'),
            ('utils.transforms', 'aug_transforms')
        ]
        
        all_valid = True
        for module_name, class_name in utils_modules:
            try:
                module = __import__(module_name, fromlist=[class_name])
                if hasattr(module, class_name):
                    print_success(f"{module_name}.{class_name} found")
                else:
                    print_error(f"{class_name} not found in {module_name}")
                    self.errors.append(f"{module_name}.{class_name} missing")
                    all_valid = False
            except Exception as e:
                print_error(f"Cannot import {module_name}: {e}")
                self.errors.append(f"Cannot import {module_name}")
                all_valid = False
        
        return all_valid
    
    def validate_experiment_structure(self) -> bool:
        """Validate experiment directory structure."""
        print_header("9. Validating Experiment Structure")
        
        experiments_dir = self.root_dir / "saved_models" / "experiments"
        
        if not experiments_dir.exists():
            print_warning("No experiments directory yet (will be created on first training run)")
            return True
        
        print_success(f"Experiments directory exists: {experiments_dir}")
        
        # List existing runs
        run_dirs = sorted([d for d in experiments_dir.iterdir() if d.is_dir() and d.name.startswith('run_')])
        
        if not run_dirs:
            print_info("  No previous experiment runs found")
            return True
        
        print_info(f"\n  Found {len(run_dirs)} previous experiment run(s):")
        
        for run_dir in run_dirs[-5:]:  # Show last 5 runs
            config_file = run_dir / "config.json"
            log_file = run_dir / "training.log"
            model_file = run_dir / "model.pt"
            
            status_parts = []
            if config_file.exists():
                status_parts.append("config✓")
            if log_file.exists():
                status_parts.append("log✓")
            if model_file.exists():
                model_size = model_file.stat().st_size / (1024 * 1024)
                status_parts.append(f"model✓({model_size:.1f}MB)")
            
            status = ", ".join(status_parts) if status_parts else "incomplete"
            print_info(f"    {run_dir.name}: {status}")
        
        if len(run_dirs) > 5:
            print_info(f"    ... and {len(run_dirs) - 5} more")
        
        return True
    
    def validate_pytorch_device(self) -> bool:
        """Validate PyTorch and available devices."""
        print_header("10. Validating PyTorch Device")
        
        try:
            import torch
            print_success(f"PyTorch version: {torch.__version__}")
            
            # Check CUDA availability
            if torch.cuda.is_available():
                print_success(f"CUDA is available")
                print_info(f"  CUDA version: {torch.version.cuda}")
                print_info(f"  Available GPUs: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    print_info(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
            else:
                print_warning("CUDA not available, will use CPU")
                print_info("  Training on CPU will be significantly slower")
                self.warnings.append("CUDA not available")
            
            # Test tensor creation
            try:
                test_tensor = torch.zeros(1, 1)
                print_success("PyTorch tensor creation works")
            except Exception as e:
                print_error(f"PyTorch tensor creation failed: {e}")
                self.errors.append("PyTorch not working properly")
                return False
            
            return True
            
        except Exception as e:
            print_error(f"PyTorch validation failed: {e}")
            self.errors.append(f"PyTorch error: {e}")
            return False
    
    def print_summary(self):
        """Print validation summary."""
        print_header("Validation Summary")
        
        if not self.errors and not self.warnings:
            print_success(f"{Colors.BOLD}All validations passed! ✓{Colors.END}")
            print_info("\nYou can now run the training pipeline:")
            print_info(f"{Colors.BOLD}  python scripts/trainer.py config.yaml{Colors.END}")
        else:
            if self.errors:
                print_error(f"{Colors.BOLD}Found {len(self.errors)} error(s):{Colors.END}")
                for error in self.errors:
                    print_error(f"  - {error}")
                print()
            
            if self.warnings:
                print_warning(f"{Colors.BOLD}Found {len(self.warnings)} warning(s):{Colors.END}")
                for warning in self.warnings:
                    print_warning(f"  - {warning}")
                print()
            
            if self.errors:
                print_error(f"{Colors.BOLD}Please fix the errors before running the pipeline.{Colors.END}")
                return False
            else:
                print_warning(f"{Colors.BOLD}Warnings detected but pipeline can run.{Colors.END}")
                print_info("\nYou can run the training pipeline:")
                print_info(f"{Colors.BOLD}  python scripts/trainer.py config.yaml{Colors.END}")
        
        return len(self.errors) == 0
    
    def run_all_validations(self) -> bool:
        """Run all validation checks."""
        validations = [
            self.validate_directory_structure,
            self.validate_scripts,
            self.validate_python_environment,
            self.validate_dependencies,
            self.validate_config_file,
            self.validate_data,
            self.validate_models_module,
            self.validate_utils_module,
            self.validate_experiment_structure,
            self.validate_pytorch_device
        ]
        
        all_passed = True
        for validation in validations:
            try:
                if not validation():
                    all_passed = False
            except Exception as e:
                print_error(f"Validation failed with exception: {e}")
                self.errors.append(f"Validation exception: {e}")
                all_passed = False
        
        return all_passed and self.print_summary()


def main():
    """Main entry point."""
    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("="*80)
    print(" HTR-Pipeline Setup Validation")
    print("="*80)
    print(f"{Colors.END}")
    
    validator = SetupValidator()
    success = validator.run_all_validations()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
