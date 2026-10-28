try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(
    name='your-repo-name',
    version='1.0.0',
    url='https://github.com/DecodEPFL/your-repo-name',
    license='CC-BY-4.0 License',
    author='Your_name',
    author_email='name.surname@epfl.ch',
    description='Add a description here',
    install_requires=['torch>=2.3',
                      'numpy>=1.24.4',
                      'scipy>=1.13.1',
                      'matplotlib==3.8.4'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.10.12',
)
