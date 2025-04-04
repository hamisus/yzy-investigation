# Module Renaming: puzzle_cracking → image_cracking

The `puzzle_cracking` module has been renamed to `image_cracking` for improved clarity and to better reflect its purpose within the project. This document outlines the key changes and how to update your code if you're currently using this module.

## Changes Made

1. Directory renamed: `yzy_investigation/projects/puzzle_cracking` → `yzy_investigation/projects/image_cracking`
2. Class renamed: `PuzzleCrackingPipeline` → `ImageCrackingPipeline`
3. Command renamed: `puzzle-crack` → `image-crack`
4. Configuration key renamed: `PUZZLE_CRACKING` → `IMAGE_CRACKING`
5. All imports and internal references updated accordingly

## Using the Updated Module

### CLI Commands

If you were previously using:
```
python -m yzy_investigation.main puzzle-crack --input-dir ./data/raw/yews
```

You should now use:
```
python -m yzy_investigation.main image-crack --input-dir ./data/raw/yews
```

### Module Imports

If you were importing from the module in your code:

```python
# Old import
from yzy_investigation.projects.puzzle_cracking import StegStrategy

# New import
from yzy_investigation.projects.image_cracking import StegStrategy
```

### Configuration

If you were accessing configuration:

```python
# Old configuration access
from yzy_investigation.config import PUZZLE_CRACKING

# New configuration access
from yzy_investigation.config import IMAGE_CRACKING
```

## Motivation for the Change

The rename was done to more accurately reflect the module's purpose, which is primarily focused on image steganography analysis. The term "image cracking" better communicates this specific focus compared to the more general "puzzle cracking" label.

All functionality remains identical - only the naming has been updated for clarity.

If you encounter any issues with the renamed module, please file an issue or contact the development team. 