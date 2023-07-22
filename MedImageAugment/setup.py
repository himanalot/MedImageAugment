from setuptools import setup, find_packages

setup(
    name='medimageaugment',
    version='1.3.4.2',
    description='A Python class for image augmentation',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Ishan Ramrakhiani',
    author_email='ishanramrakhiani@gmail.com',
    url='https://github.com/himanalot/medimageaugment',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'scikit-image',
        'scikit-learn'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],
)