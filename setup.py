from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='A ML project that scrape the comments from the current video using youtube API,and analyze them and give them sentiment {positive,negative,neutral} using a ml model from flask_app which operates as backend.',
    author='Saurav',
    license='MIT',
)
