# MedImageAugment.py

## Overview
This module provides a collection of image augmentation techniques that can be applied to 2D or 3D images. Image augmentation is commonly used in deep learning and computer vision tasks to increase the diversity of training data and improve the generalization of machine learning models. The module supports various augmentations, including rotations, flips, scaling, noise addition, and more.

## Requirements
To use this module, you need to have the following Python packages installed:

- `numpy`
- `cv2` (OpenCV)
- `scipy`
- `scikit-image`
- `sklearn`

You can install them using `pip`:

```bash
pip install numpy opencv-python scipy scikit-image scikit-learn
```

## Usage

### Import the module and create an instance of MedImageAugment:

```python
import numpy as np
import cv2
from typing import Optional, Callable, List, Tuple
from scipy.ndimage import gaussian_filter, map_coordinates, shift, zoom
from sklearn.model_selection import ParameterGrid
from skimage.transform import AffineTransform, warp
from skimage.util import random_noise
import random
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

class MedImageAugment:
    # Class implementation...
```

### Instantiate the `MedImageAugment` class:

```python
augmenter = MedImageAugment(seed=42, modality='MRI', random_rotation_3d=True, random_scaling_3d=True,
                         random_crop_3d=True, random_horizontal_flip_3d=True, random_vertical_flip_3d=True)
```

### Add augmentations:

You can add specific augmentations to the `MedImageAugment` instance using the provided methods. For example:

```python
# Add some augmentations
augmenter.add_random_brightness(max_delta=30)
augmenter.add_random_contrast(lower=0.7, upper=1.3)
augmenter.add_elastic_deformation(alpha=9, sigma=0.7)
augmenter.add_speckle_noise(mean=0, std=10)
augmenter.add_random_rotation_3d(max_angle=30)
# Add more augmentations as needed...
```

### Augment a single image:

```python
import cv2

# Load an image (example)
image_path = "path_to_your_image.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Augment the image
augmented_image = augmenter.augment_image(image)
```

### Augment a batch of images:

```python
import glob

# Load a batch of images (example)
image_paths = glob.glob("path_to_folder/*.png")
images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

# Augment the batch of images
augmented_images = augmenter.augment_batch(images)
```

### Modality-specific augmentations:

The module provides modality-specific augmentations for different medical imaging modalities, such as 'CT', 'X-ray', 'MRI', and 'Ultrasound'. When creating the `MedImageAugment` instance, you can specify the modality, and the augmentations will be automatically set accordingly.

### Additional Notes:

- Each augmentation function performs a specific image transformation.
- You can add multiple augmentations to the `MedImageAugment` instance to apply them sequentially.
- The augmentations are randomly shuffled before applying to introduce diversity.
- Some augmentations support 3D images, but make sure to set the corresponding parameters when creating the `MedImageAugment` instance.

## Example:
Here's an example of using the `MedImageAugment` class to augment a single 2D image:

```python
import cv2

# Load an image
image_path = "path_to_your_image.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Instantiate the MedImageAugment class
augmenter = MedImageAugment(seed=42, modality='X-ray')

# Add some augmentations
augmenter.add_random_brightness(max_delta=30)
augmenter.add_random_contrast(lower=0.7, upper=1.3)
augmenter.add_speckle_noise(mean=0, std=10)
augmenter.add_random_horizontal_flip()

# Augment the image
augmented_image = augmenter.augment_image(image)

# Display the original and augmented images (example)
cv2.imshow("Original Image", image)
cv2.imshow("Augmented Image", augmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Adding ParameterGrid for Hyperparameter Search

In some cases, you may want to perform hyperparameter search to find the best combination of augmentation parameters for your specific use case. To facilitate this, you can add a method to the `MedImageAugment` class that generates a parameter grid containing all possible combinations of the augmentation parameters. Here's how you can do it:

```python
from itertools import product

class MedImageAugment:
    # Previous class implementation...

    def generate_parameter_grid(self) -> List[dict]:
        """Generate a parameter grid for hyperparameter search.

        Returns:
            List[dict]: A list of dictionaries representing all possible combinations of augmentation parameters.
        """
        param_grid = []

        # Define the hyperparameter ranges
        max_angle_values = [15, 30, 45]
        scale_range_values = [(0.8, 1.2), (0.9, 1.1), (0.7, 1.3)]
        crop_size_values = [(100, 100), (120, 120), (80, 80)]
        # Add more hyperparameter ranges as needed...

        # Generate all possible combinations of hyperparameters
        for max_angle, scale_range, crop_size in product(max_angle_values, scale_range_values, crop_size_values):
            params = {
                'max_angle': max_angle,
                'scale_range': scale_range,
                'crop_size': crop_size,
                # Add more hyperparameters here...
            }
            param_grid.append(params)

        return param_grid
```

Now, you can use this method to generate the parameter grid and then loop through all the combinations to evaluate the performance of your model with different augmentations during hyperparameter search.

```python
augmenter = MedImageAugment(seed=42, modality='X-ray')
param_grid = augmenter.generate_parameter_grid()

for params in param_grid:
    # Instantiate the MedImageAugment class with specific parameters
    augmenter = MedImageAugment(seed=42, modality='X-ray', random_rotation_3d=True, random_scaling_3d=True,
                             random_crop_3d=True, random_horizontal_flip_3d=True, random_vertical_flip_3d=True)
    
    # Add augmentations with the specific hyperparameters
    augmenter.add_random_rotation_3d(max_angle=params['max_angle'])
    augmenter.add_random_scaling_3d(scale_range=params['scale_range'])
    augmenter.add_random_crop_3d(crop_size=params['crop_size'])

    # Train and evaluate your model with the current augmentation settings
    # Perform hyperparameter search using different combinations of augmentations
    # ...

```

## Functions

1. `__init__(self, seed: Optional[int] = None, modality: str = 'general', ...)`:
   - Constructor of the class that initializes the augmentation list, seed, and modality.
   - Inputs:
     - `seed` (Optional[int]): Seed value for reproducible randomness. Defaults to `None`.
     - `modality` (str): The modality of the images to be augmented (e.g., 'CT', 'X-ray', 'MRI', 'Ultrasound'). Defaults to `'general'`.
     - Parameters for enabling specific 3D augmentations:
       - `random_rotation_3d` (bool): Enable random 3D rotation. Defaults to `False`.
       - `random_scaling_3d` (bool): Enable random 3D scaling. Defaults to `False`.
       - `random_crop_3d` (bool): Enable random 3D cropping. Defaults to `False`.
       - `random_horizontal_flip_3d` (bool): Enable random 3D horizontal flipping. Defaults to `False`.
       - `random_vertical_flip_3d` (bool): Enable random 3D vertical flipping. Defaults to `False`.

2. `add_modality_specific_augmentations(self)`:
   - Adds modality-specific augmentations based on the specified `modality`.
   - Raises a `ValueError` if an invalid modality is provided.

3. `validate_range(self, value, min_val, max_val, value_name)`:
   - Helper method to validate if a value is within a specified range.
   - Inputs:
     - `value`: The value to be validated.
     - `min_val`: Minimum allowed value.
     - `max_val`: Maximum allowed value.
     - `value_name`: Name of the value for error messages.

4. `validate_positive(self, value, value_name)`:
   - Helper method to validate if a value is positive.
   - Inputs:
     - `value`: The value to be validated.
     - `value_name`: Name of the value for error messages.

5. `add_random_rotation(self, max_angle: float)`:
   - Adds random rotation augmentation to the list of augmentations.
   - Inputs:
     - `max_angle` (float): Maximum rotation angle in degrees.

6. `add_random_horizontal_flip(self)`:
   - Adds random horizontal flip augmentation to the list of augmentations.

7. `add_random_vertical_flip(self)`:
   - Adds random vertical flip augmentation to the list of augmentations.

8. `add_random_brightness(self, max_delta: float)`:
   - Adds random brightness augmentation to the list of augmentations.
   - Inputs:
     - `max_delta` (float): Maximum value for brightness adjustment.

9. `add_random_contrast(self, lower: float, upper: float)`:
   - Adds random contrast augmentation to the list of augmentations.
   - Inputs:
     - `lower` (float): Lower bound for the contrast adjustment factor.
     - `upper` (float): Upper bound for the contrast adjustment factor.

10. `add_elastic_deformation(self, alpha: float, sigma: float)`:
    - Adds elastic deformation augmentation to the list of augmentations.
    - Inputs:
      - `alpha` (float): Elastic deformation intensity.
      - `sigma` (float): Standard deviation of the Gaussian filter.

11. `add_random_shear(self, intensity: float)`:
    - Adds random shear augmentation to the list of augmentations.
    - Inputs:
      - `intensity` (float): Maximum shear intensity.

12. `add_random_shift(self, max_shift: float)`:
    - Adds random shift augmentation to the list of augmentations.
    - Inputs:
      - `max_shift` (float): Maximum shift value.

13. `add_random_scaling(self, scale_range: Tuple[float, float])`:
    - Adds random scaling augmentation to the list of augmentations.
    - Inputs:
      - `scale_range` (Tuple[float, float]): Range of scaling factors.

14. `add_random_crop(self, crop_size: Tuple[int, int])`:
    - Adds random crop augmentation to the list of augmentations.
    - Inputs:
      - `crop_size` (Tuple[int, int]): Size of the cropped region.

15. `add_random_blur(self, min_sigma: float, max_sigma: float)`:
    - Adds random blur augmentation to the list of augmentations.
    - Inputs:
      - `min_sigma` (float): Minimum value for the Gaussian blur sigma.
      - `max_sigma` (float): Maximum value for the Gaussian blur sigma.

16. `add_motion_blur(self, max_angle: int, max_kernel_size: int)`:
    - Adds motion blur augmentation to the list of augmentations.
    - Inputs:
      - `max_angle` (int): Maximum angle for motion blur.
      - `max_kernel_size` (int): Maximum kernel size for motion blur.

17. `add_random_rotation_3d(self, max_angle: float)`:
    - Adds random 3D rotation augmentation to the list of augmentations.
    - Inputs:
      - `max_angle` (float): Maximum rotation angle in degrees for each axis.

18. `add_random_scaling_3d(self, scale_range: Tuple[float, float])`:
    - Adds random 3D scaling augmentation to the list of augmentations.
    - Inputs:
      - `scale_range` (Tuple[float, float]): Range of scaling factors for each axis.

19. `add_speckle_noise(self, mean: float, std: float)`:
    - Adds speckle noise augmentation to the list of augmentations.
    - Inputs:
      - `mean` (float): Mean value for the noise.
      - `std` (float): Standard deviation of the noise.

20. `add_rician_noise(self, mean: float, std: float)`:
    - Adds Rician noise augmentation to the list of augmentations.
    - Inputs:
      - `mean` (float): Mean value for the noise.
      - `std` (float): Standard deviation of the noise.

21. `add_random_crop_3d(self, crop_size: Tuple[int, int, int])`:
    - Adds random 3D crop augmentation to the list of augmentations.
    - Inputs:
      - `crop_size` (Tuple[int, int, int]): Size of the cropped region for each axis.

22. `add_random_horizontal_flip_3d(self)`:
    - Adds random 3D horizontal flip augmentation to the list of augmentations.

23. `add_random_vertical_flip_3d(self)`:
    - Adds random 3D vertical flip augmentation to the list of augmentations.

24. `augment_image(self, image: np.ndarray) -> np.ndarray`:
    - Applies the list of augmentations to a single image and returns the augmented image.
    - Inputs:
      - `image` (np.ndarray): The input image to be augmented.
    - Output:
      - (np.ndarray): The augmented image.

25. `augment_batch(self, images: List[np.ndarray]) -> List[np.ndarray]`:
    - Applies the list of augmentations to a batch of images and returns the augmented batch.


    - Inputs:
      - `images` (List[np.ndarray]): List of input images to be augmented.
    - Output:
      - (List[np.ndarray]): List of augmented images.

26. `add_gaussian_noise(self, mean: float = 0., std: float = 1.)`:
    - Adds Gaussian noise augmentation to the list of augmentations.
    - Inputs:
      - `mean` (float): Mean value for the Gaussian noise. Defaults to 0.
      - `std` (float): Standard deviation of the Gaussian noise. Defaults to 1.

27. `add_salt_pepper_noise(self, salt_prob: float = 0.05, pepper_prob: float = 0.05)`:
    - Adds salt and pepper noise augmentation to the list of augmentations.
    - Inputs:
      - `salt_prob` (float): Probability of adding salt noise.
      - `pepper_prob` (float): Probability of adding pepper noise.

28. `add_poisson_noise(self, lam: float = 1.)`:
    - Adds Poisson noise augmentation to the list of augmentations.
    - Inputs:
      - `lam` (float): Poisson parameter (mean) for the noise. Defaults to 1.

29. `add_anisotropic_diffusion(self, max_iter=10, dt=0.1, kappa=50)`:
    - Adds anisotropic diffusion augmentation to the list of augmentations.
    - Inputs:
      - `max_iter` (int): Maximum number of iterations for anisotropic diffusion. Defaults to 10.
      - `dt` (float): Time step parameter for anisotropic diffusion. Defaults to 0.1.
      - `kappa` (float): Conductance parameter for anisotropic diffusion. Larger values reduce diffusion across edges. Defaults to 50.

## Conclusion

With the `MedImageAugment` class and the ability to generate a parameter grid, you have a powerful tool to augment your 2D and 3D images and perform hyperparameter search to find the best augmentation settings for your particular use case. Experiment with different combinations of augmentations and hyperparameters to optimize the performance of your deep learning models and enhance their generalization capabilities.

## Note:
Remember that the module contains both 2D and 3D augmentations. When working with 3D images, ensure that the image dimensions are correct for 3D-specific augmentations. Additionally, this documentation does not include the description of the 'add_random_shear', 'add_random_shift', 'add_random_blur', 'add_motion_blur', 'add_gaussian_noise', 'add_salt_pepper_noise', and 'add_poisson_noise' methods, as their usages are quite similar to other provided augmentations.

Feel free to experiment with different combinations of augmentations to suit your specific needs and data.

## License

This project is licensed under the MIT License - see LICENSE for details.