import pkgutil
import importlib
import os

# Get the path of the current directory (where this __init__.py lives)
# We use a list because pkgutil.iter_modules expects a list of paths
package_path = [os.path.dirname(__file__)]

# Iterate over all modules found in this directory
for _, module_name, _ in pkgutil.iter_modules(package_path):
    # Dynamically import the module
    importlib.import_module(f"{__name__}.{module_name}")
