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
from scipy.sparse import diags

class ImageAugment:
    def __init__(self, seed: Optional[int] = None, modality: str = 'general',
                 random_rotation_3d: bool = False, random_scaling_3d: bool = False,
                 random_crop_3d: bool = False, random_horizontal_flip_3d: bool = False,
                 random_vertical_flip_3d: bool = False):
        self.augmentations = []
        self.seed = seed
        self.modality = modality
        np.random.seed(self.seed)

        # Enable specific 3D augmentations based on the provided parameters
        if random_rotation_3d:
            self.add_random_rotation_3d(max_angle=30)
        if random_scaling_3d:
            self.add_random_scaling_3d(scale_range=(0.8, 1.2))
        if random_crop_3d:
            self.add_random_crop_3d(crop_size=(100, 100, 100))
        if random_horizontal_flip_3d:
            self.add_random_horizontal_flip_3d()
        if random_vertical_flip_3d:
            self.add_random_vertical_flip_3d()

        # Add modality-specific augmentations
        self.add_modality_specific_augmentations()

    def add_modality_specific_augmentations(self):
        if self.modality not in ['CT', 'X-ray', 'MRI', 'Ultrasound']:
            raise ValueError("Invalid modality. Please select among 'CT', 'X-ray', 'MRI', 'Ultrasound'.")

        if self.modality == 'CT' or self.modality == 'X-ray':
            self.add_random_brightness(max_delta=30)
            self.add_random_contrast(lower=0.7, upper=1.3)
        elif self.modality == 'MRI' or self.modality == 'Ultrasound':
            self.add_elastic_deformation(alpha=9, sigma=0.7)

    def validate_range(self, value, min_val, max_val, value_name):
        if value < min_val or value > max_val:
            raise ValueError(f"Invalid {value_name}. It should be in range [{min_val}, {max_val}].")

    def validate_positive(self, value, value_name):
        if value < 0:
            raise ValueError(f"Invalid {value_name}. It should be greater than 0.")

    def add_random_rotation(self, max_angle: float):
        self.validate_range(max_angle, 0, 180, "max_angle")

        def random_rotation(image: np.ndarray):
            angle = np.random.uniform(-max_angle, max_angle)
            rows, cols = image.shape[:2]
            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            return cv2.warpAffine(image, M, (cols, rows))

        self.augmentations.append(random_rotation)

    def add_random_horizontal_flip(self):
        def random_horizontal_flip(image: np.ndarray):
            if np.random.random() < 0.5:
                return cv2.flip(image, 1)
            return image

        self.augmentations.append(random_horizontal_flip)

    def add_random_vertical_flip(self):
        def random_vertical_flip(image: np.ndarray):
            if np.random.random() < 0.5:
                return cv2.flip(image, 0)
            return image

        self.augmentations.append(random_vertical_flip)

    def add_random_brightness(self, max_delta: float):
        self.validate_range(max_delta, 0, 255, "max_delta")

        def random_brightness(image: np.ndarray):
            delta = np.random.uniform(-max_delta, max_delta)
            return np.clip(image + delta, 0, 255).astype(np.uint8)

        self.augmentations.append(random_brightness)

    def add_random_contrast(self, lower: float, upper: float):
        if lower < 0 or upper < lower:
            raise ValueError("Invalid contrast range. 'lower' should be less than 'upper', and both should be greater than 0.")

        def random_contrast(image: np.ndarray):
            contrast_factor = np.random.uniform(lower, upper)
            return np.clip(127.5 + contrast_factor * image - 127.5, 0, 255).astype(np.uint8)

        self.augmentations.append(random_contrast)

    def add_elastic_deformation(self, alpha: float, sigma: float):
        self.validate_positive(alpha, 'alpha')
        self.validate_positive(sigma, 'sigma')

        def elastic_deformation(image: np.ndarray):
            shape = image.shape
            dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
            dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
            dz = np.zeros_like(dx)

            if len(shape) == 2:  # 2D image
                y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
                indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
            elif len(shape) == 3:  # 3D image
                z, y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
                indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z+dz, (-1, 1))

            return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

        self.augmentations.append(elastic_deformation)


    def add_random_shear(self, intensity: float):
        self.validate_positive(intensity, 'intensity')

        def random_shear(image: np.ndarray):
            shear = AffineTransform(shear=np.random.uniform(-intensity, intensity))
            return warp(image, shear, mode='reflect')

        self.augmentations.append(random_shear)

    def add_random_shift(self, max_shift: float):
        self.validate_positive(max_shift, 'max_shift')

        def random_shift(image: np.ndarray):
            shift_values = np.random.uniform(-max_shift, max_shift, size=image.ndim)
            return shift(image, shift_values, mode='reflect')

        self.augmentations.append(random_shift)

    def add_random_scaling(self, scale_range: Tuple[float, float]):
        if scale_range[0] < 0 or scale_range[1] < 0 or scale_range[0] > scale_range[1]:
            raise ValueError("Invalid 'scale_range'. Both values should be greater than 0 and first value should be less than the second value.")

        def random_scaling(image: np.ndarray):
            scale_value = np.random.uniform(scale_range[0], scale_range[1])
            return zoom(image, scale_value)

        self.augmentations.append(random_scaling)

    def add_random_crop(self, crop_size: Tuple[int, int]):
        if crop_size[0] <= 0 or crop_size[1] <= 0:
            raise ValueError("Invalid 'crop_size'. Both values should be greater than 0.")
        if crop_size[0] >= self.image_shape[0] or crop_size[1] >= self.image_shape[1]:
            raise ValueError("Invalid 'crop_size'. Both values should be smaller than the dimensions of the image.")

        self.crop_size = crop_size  # Store crop_size in the object for later use in augmentations

        def random_crop(image: np.ndarray):
            start_x = np.random.randint(0, image.shape[0] - self.crop_size[0] + 1)
            start_y = np.random.randint(0, image.shape[1] - self.crop_size[1] + 1)

            cropped_image = image[start_x : start_x + self.crop_size[0], start_y : start_y + self.crop_size[1]]
            return cropped_image

        self.augmentations.append(random_crop)


    def add_random_blur(self, min_sigma: float, max_sigma: float):
        if min_sigma < 0 or max_sigma < min_sigma:
            raise ValueError("Invalid sigma range. 'min_sigma' should be greater than or equal to 0, and 'max_sigma' should be greater than 'min_sigma'.")

        def random_blur(image: np.ndarray):
            sigma = np.random.uniform(min_sigma, max_sigma)
            return gaussian_filter(image, sigma=sigma)

        self.augmentations.append(random_blur)

    def add_motion_blur(self, max_angle: int, max_kernel_size: int):
        self.validate_range(max_angle, 0, 180, "max_angle")
        self.validate_positive(max_kernel_size, 'max_kernel_size')

        def motion_blur(image: np.ndarray):
            angle = np.random.uniform(-max_angle, max_angle)
            kernel_size = random.choice(range(1, max_kernel_size + 1, 2))
            kernel = np.zeros((kernel_size, kernel_size))
            x_center, y_center = kernel_size // 2, kernel_size // 2
            cv2.ellipse(kernel, (x_center, y_center), (x_center, y_center), angle, 0, 360, 1, -1)
            kernel /= kernel.sum()
            return cv2.filter2D(image, -1, kernel)

        self.augmentations.append(motion_blur)

    def add_random_rotation_3d(self, max_angle: float):
        self.validate_range(max_angle, 0, 180, "max_angle")

        def random_rotation_3d(image: np.ndarray):
            # Check if image is 3D
            if len(image.shape) != 3:
                raise ValueError("Image for 3D rotation is not 3D. It has {len(image.shape)} dimensions.")
            # Random angles for each axis
            angles = np.random.uniform(-max_angle, max_angle, size=(3,))
            # Perform 3D rotation using OpenCV's Rodrigues' rotation formula
            rotation_matrix = cv2.Rodrigues(angles)[0]
            return cv2.warpAffine(image, rotation_matrix, image.shape[::-1], flags=cv2.INTER_NEAREST)

        self.augmentations.append(random_rotation_3d)

    def add_random_scaling_3d(self, scale_range: Tuple[float, float]):
        if scale_range[0] < 0 or scale_range[1] < 0 or scale_range[0] > scale_range[1]:
            raise ValueError("Invalid 'scale_range'. Both values should be greater than 0 and the first value should be less than the second value.")

        def random_scaling_3d(image: np.ndarray):
            # Check if image is 3D
            if len(image.shape) != 3:
                raise ValueError("Image for 3D scaling is not 3D. It has {len(image.shape)} dimensions.")
            scales = np.random.uniform(scale_range[0], scale_range[1], size=(3,))
            return zoom(image, scales, order=1)

        self.augmentations.append(random_scaling_3d)

    def add_speckle_noise(self, mean: float, std: float):
        self.validate_positive(std, 'std')

        def speckle_noise(image: np.ndarray):
            noise = image + image * np.random.randn(*image.shape)
            return np.clip(noise, 0, 255).astype(np.uint8)

        self.augmentations.append(speckle_noise)

    def add_rician_noise(self, mean: float, std: float):
        self.validate_positive(std, 'std')

        def rician_noise(image: np.ndarray):
            noise = np.random.normal(mean, std, image.shape)
            noise = np.sqrt(image + noise)**2
            return np.clip(noise, 0, 255).astype(np.uint8)

        self.augmentations.append(rician_noise)

    def add_random_crop_3d(self, crop_size: Tuple[int, int, int]):
        def random_crop_3d(image: np.ndarray):
            # Check if image is 3D
            if len(image.shape) != 3:
                raise ValueError("Image for 3D crop is not 3D. It has {len(image.shape)} dimensions.")
            if any(image.shape[i] < crop_size[i] for i in range(3)):
                raise ValueError("Invalid 'crop_size'. Each value should be less than the respective dimension of the image.")
            start_x = np.random.randint(0, image.shape[0] - crop_size[0] + 1)
            start_y = np.random.randint(0, image.shape[1] - crop_size[1] + 1)
            start_z = np.random.randint(0, image.shape[2] - crop_size[2] + 1)
            return image[start_x : start_x + crop_size[0],
                        start_y : start_y + crop_size[1],
                        start_z : start_z + crop_size[2]]

        self.augmentations.append(random_crop_3d)

    def add_random_horizontal_flip_3d(self):
        def random_horizontal_flip_3d(image: np.ndarray):
            # Check if image is 3D
            if len(image.shape) != 3:
                raise ValueError("Image for 3D horizontal flip is not 3D. It has {len(image.shape)} dimensions.")
            if np.random.random() < 0.5:
                return np.flip(image, axis=0)
            return image

        self.augmentations.append(random_horizontal_flip_3d)

    def add_random_vertical_flip_3d(self):
        def random_vertical_flip_3d(image: np.ndarray):
            # Check if image is 3D
            if len(image.shape) != 3:
                raise ValueError("Image for 3D vertical flip is not 3D. It has {len(image.shape)} dimensions.")
            if np.random.random() < 0.5:
                return np.flip(image, axis=1)
            return image

        self.augmentations.append(random_vertical_flip_3d)

    def augment_image(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            raise ValueError("Image is None. Please provide a valid image.")
        if len(image.shape) < 2:
            raise ValueError("Invalid image dimensions. Please provide a 2D or 3D image.")

        np.random.shuffle(self.augmentations)  # Randomize the order of augmentations

        for augmentation in self.augmentations:
            image = augmentation(image)
        return image

    def augment_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        if images is None or len(images) == 0:
            raise ValueError("Image batch is None or empty. Please provide a valid batch of images.")
        return [self.augment_image(image) for image in images]

    def add_gaussian_noise(self, mean: float = 0., std: float = 1.):
        self.validate_positive(std, 'std')

        def gaussian_noise(image: np.ndarray):
            noise = np.random.normal(mean, std, image.shape)
            return np.clip(image + noise, 0, 255).astype(np.uint8)

        self.augmentations.append(gaussian_noise)

    def add_salt_pepper_noise(self, salt_prob: float = 0.05, pepper_prob: float = 0.05):
        self.validate_range(salt_prob, 0, 1, 'salt_prob')
        self.validate_range(pepper_prob, 0, 1, 'pepper_prob')

        def salt_pepper_noise(image: np.ndarray):
            total_pixels = image.size
            num_salt = np.ceil(total_pixels * salt_prob)
            num_pepper = np.ceil(total_pixels * pepper_prob)

            # Add Salt noise
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
            image[coords] = 255

            # Add Pepper noise
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            image[coords] = 0

            return image

        self.augmentations.append(salt_pepper_noise)

    def add_poisson_noise(self, lam: float = 1.):
        self.validate_positive(lam, 'lambda')

        def poisson_noise(image: np.ndarray):
            noise = np.random.poisson(lam, image.shape)
            return np.clip(image + noise, 0, 255).astype(np.uint8)
        self.augmentations.append(poisson_noise)

    def add_anisotropic_diffusion(self, max_iter=10, dt=0.1, kappa=50):
        def anisotropic_diff(image):
            # Initialize the diffusion kernel
            dx = np.array([[-1, 1]])
            dy = dx.T

            for _ in range(max_iter):
                # Compute gradient components
                grad_x = cv2.filter2D(image, cv2.CV_64F, dx)
                grad_y = cv2.filter2D(image, cv2.CV_64F, dy)

                # Compute the diffusivity (conductance)
                diffusivity = 1 / (1 + (grad_x**2 + grad_y**2) / kappa**2)

                # Compute the Laplacian using a sparse matrix solver (Chambolle's method)
                laplacian = cv2.Laplacian(image, cv2.CV_64F)

                # Perform the update using the solver
                image_flat = image.reshape(-1)
                update_term_flat = dt * diffusivity * laplacian.reshape(-1)

                diag_main = diags([1 + 4 * dt * diffusivity], [0], shape=(image_flat.size, image_flat.size))
                diag_off = diags([dt * diffusivity], [-1, 1], shape=(image_flat.size, image_flat.size))

                updated_image_flat = spsolve(diag_main - diag_off, image_flat + update_term_flat)
                updated_image = np.clip(updated_image_flat, 0, 255).reshape(image.shape).astype(np.uint8)

                image = updated_image

            return image

        self.augmentations.append(anisotropic_diff)

