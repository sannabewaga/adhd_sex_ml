from setuptools import setup, find_packages

def get_requirements(file_path):
    with open(file_path, 'r') as f:
        req = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        if '-e .' in req:
            req.remove('-e .')

        return req

setup(
    name="adhdpred",
    version="0.1",
    description="prediction for adhd and sex",
    author="Sarthak Gaurav",
    author_email="gauravsarthak00@gmail.com",
    packages=find_packages(),  # Automatically finds all packages in the directory
    install_requires=get_requirements('requirements.txt'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
