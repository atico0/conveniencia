from setuptools import setup, find_packages

setup(
    name="conveniencia",
    version="1.0.0",
    description="Um pacote Python de exemplo",
    author="Luis Gabriel",
    author_email="luisgabrielmarques12@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",     # Requer a versão 1.19.0 ou superior do numpy
        "pandas>=1.3.0"     # Exige a versão exata 1.3.0 do pandas
        "matplotlib>=3.8.4",
        "seaborn>=0.12.0",
        "sklearn>=1.4.2",
        
    ],
)