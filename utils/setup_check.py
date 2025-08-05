"""
Setup and dependency checking utilities
"""

import importlib
import sys
from typing import List, Dict, Tuple


def check_dependencies() -> bool:
    """
    Check if all required dependencies are available
    Returns True if all dependencies are satisfied
    """
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Hugging Face Transformers',
        'sentence_transformers': 'Sentence Transformers',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'sklearn': 'Scikit-learn',
        'faiss': 'FAISS (faiss-cpu)',
        'nltk': 'NLTK',
        'flask': 'Flask',
        'requests': 'Requests',
        'tqdm': 'TQDM',
        'psutil': 'PSUtil',
        'pandas': 'Pandas',
        'joblib': 'Joblib',
        'yaml': 'PyYAML',
        'cryptography': 'Cryptography'
    }
    
    missing_packages = []
    available_packages = []
    
    for package, description in required_packages.items():
        try:
            if package == 'sklearn':
                importlib.import_module('sklearn')
            elif package == 'yaml':
                importlib.import_module('yaml')
            elif package == 'faiss':
                try:
                    importlib.import_module('faiss')
                except ImportError:
                    importlib.import_module('faiss-cpu')
            else:
                importlib.import_module(package)
            available_packages.append((package, description))
        except ImportError:
            missing_packages.append((package, description))
    
    # Check built-in modules
    try:
        import sqlite3
        available_packages.append(('sqlite3', 'SQLite3 (built-in)'))
    except ImportError:
        missing_packages.append(('sqlite3', 'SQLite3'))
    
    # Report results
    if available_packages:
        print("✅ Available packages:")
        for package, desc in available_packages:
            print(f"   • {package} ({desc})")
    
    if missing_packages:
        print("\n❌ Missing packages:")
        for package, desc in missing_packages:
            print(f"   • {package} ({desc})")
        
        print("\n💡 To install missing packages:")
        print("   pip install -r requirements.txt")
        print("   or")
        print("   pip install torch transformers sentence-transformers numpy scipy scikit-learn faiss-cpu nltk flask requests tqdm psutil pandas joblib pyyaml cryptography")
        
        return False
    
    print(f"\n✅ All {len(available_packages)} required dependencies are available!")
    return True


def check_system_requirements() -> Dict[str, bool]:
    """
    Check system requirements like memory, disk space, etc.
    """
    import psutil
    import os
    
    requirements = {
        'memory_gb': 4,  # Minimum 4GB RAM
        'disk_gb': 2,    # Minimum 2GB free space
        'python_version': (3, 8)  # Minimum Python 3.8
    }
    
    results = {}
    
    # Check Python version
    python_version = sys.version_info[:2]
    results['python_version'] = python_version >= requirements['python_version']
    
    # Check available memory
    memory_gb = psutil.virtual_memory().total / (1024**3)
    results['memory'] = memory_gb >= requirements['memory_gb']
    
    # Check available disk space
    disk_usage = psutil.disk_usage('.')
    free_gb = disk_usage.free / (1024**3)
    results['disk_space'] = free_gb >= requirements['disk_gb']
    
    return results


def get_system_info() -> Dict:
    """
    Get detailed system information
    """
    import platform
    import psutil
    
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'disk_free_gb': round(psutil.disk_usage('.').free / (1024**3), 2)
    }
    
    return info


if __name__ == "__main__":
    print("🔍 Checking dependencies...")
    if check_dependencies():
        print("\n🔍 Checking system requirements...")
        req_results = check_system_requirements()
        
        for requirement, satisfied in req_results.items():
            status = "✅" if satisfied else "❌"
            print(f"{status} {requirement}: {satisfied}")
        
        print("\n📊 System Information:")
        sys_info = get_system_info()
        for key, value in sys_info.items():
            print(f"   {key}: {value}")
    else:
        sys.exit(1)
